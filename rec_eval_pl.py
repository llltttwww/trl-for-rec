#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import re
from typing import List, Optional, Tuple, Dict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 直接复用你训练脚本里的数据构造
from rec_utils.train_init import create_dataset, RankingPrompter

K_FIXED = 5
DIGITK_RE = re.compile(r"^[1-5]{5}$")

VALID_MODES = ["one_forward", "five_forward", "generate"]


# --------------------------
# parsing utils
# --------------------------
def parse_perm_digits(text: str, K: int = K_FIXED) -> Optional[List[int]]:
    """
    解析严格的纯 digits 输出，例如 '41532'
    返回 0-based: [3,0,4,2,1]
    """
    s = (text or "").strip()
    s = s[:K]  # 防止多吐
    if len(s) != K:
        return None
    if K == 5:
        if not DIGITK_RE.fullmatch(s):
            return None
    else:
        if not re.fullmatch(rf"^[1-{K}]{{{K}}}$", s):
            return None
    if len(set(s)) != K:
        return None
    return [int(ch) - 1 for ch in s]


def ndcg_single_positive(perm0: List[int], pos_idx: int) -> float:
    # rank 从 1 开始
    rank = perm0.index(pos_idx) + 1
    return 1.0 / math.log2(rank + 1.0)


def _get_last_token_index(attn_mask: torch.Tensor) -> torch.Tensor:
    """
    attn_mask: [B, L] (0/1)
    返回每个样本最后一个非 padding token 的 index（对 left/right padding 都稳健）
    """
    rev = torch.flip(attn_mask, dims=[1])
    first_one_from_right = rev.float().argmax(dim=1)  # [B]
    last_idx = attn_mask.size(1) - 1 - first_one_from_right
    return last_idx  # [B]


def _digit_token_ids(tokenizer, K: int, device: str) -> torch.Tensor:
    """
    返回 digits '1'..'K' 对应的 token ids 张量 [K]
    要求每个 digit 是单 token
    """
    ids = []
    for d in range(1, K + 1):
        enc = tokenizer.encode(str(d), add_special_tokens=False)
        assert len(enc) == 1, f"Digit '{d}' is not a single token: {enc} -> {tokenizer.convert_ids_to_tokens(enc)}"
        ids.append(enc[0])
    return torch.tensor(ids, device=device)


# --------------------------
# dataset build (aligned with training)
# --------------------------
def build_rec_hf_dataset_pl_aligned(
    dataset_path: str,
    tokenizer,
    split: str = "train",
    K: int = 5,
    user_win_size: int = 10,
    seed: int = 42,
):
    """
    和你训练脚本一致：
      - create_dataset(...)
      - RankingPrompter(tokenizer)
      - prompter.build_prompt(..., apply_chat_template=False, pl_grpo=True)
      - prompt 存 messages
    """
    ranking_ds = create_dataset(
        dataset_path=dataset_path,
        split=split,
        K=K,
        user_win_size=user_win_size,
        seed=seed,
    )
    prompter = RankingPrompter(tokenizer)

    rows = []
    for i in range(len(ranking_ds)):
        ex = ranking_ds[i]
        prompt_text = prompter.build_prompt(
            user_text=ex["user_text"],
            candidates=ex["candidates"],
            apply_chat_template=False,
            pl_grpo=True,
        )
        rows.append(
            {
                "prompt": [{"role": "user", "content": prompt_text}],
                "solution": str(ex["target_idx"]),  # 0-based
            }
        )

    from datasets import Dataset as HFDataset
    return HFDataset.from_list(rows)


# --------------------------
# forward-based methods
# --------------------------
@torch.inference_mode()
def rank_from_first_token_logits(
    model,
    tokenizer,
    prompt_texts: List[str],
    digit_ids_t: torch.Tensor,  # [K]
    max_prompt_length: int = 4096,
    device: str = "cuda",
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """
    单次 forward：
      - 取 next-token logits（prompt 最后一个 token 位置）
      - 只看 digits '1'..'K' 的 logits
      - logits 降序 -> ranking 字符串，比如 "25134"
    """
    enc = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    ).to(device)

    out = model(**enc)  # logits: [B, L, V]

    last_idx = _get_last_token_index(enc["attention_mask"])  # [B]
    next_logits = out.logits[torch.arange(out.logits.size(0), device=device), last_idx, :]  # [B, V]

    digit_logits = next_logits.index_select(dim=-1, index=digit_ids_t)   # [B, K]
    digit_probs = torch.softmax(digit_logits.float(), dim=-1)            # [B, K]

    order = torch.argsort(digit_logits, dim=-1, descending=True)         # [B, K] values in 0..K-1
    rankings: List[str] = []
    for i in range(order.size(0)):
        ranks = [str(int(x.item()) + 1) for x in order[i]]  # 0..K-1 -> '1'..'K'
        rankings.append("".join(ranks))

    return rankings, digit_logits, digit_probs


@torch.inference_mode()
def greedy_permutation_by_iterative_forward(
    model,
    tokenizer,
    prompt_texts: List[str],
    digit_ids_t: torch.Tensor,  # [K]
    max_prompt_length: int = 4096,
    device: str = "cuda",
) -> List[str]:
    """
    five_forward：
    - 做 K 次迭代：每次对 (prompt + 已选 digits) forward
    - 在剩余 digits 上取 argmax
    - 无放回生成一个 permutation 字符串
    """
    enc = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    ).to(device)

    input_ids = enc["input_ids"]            # [B, L]
    attention_mask = enc["attention_mask"]  # [B, L]
    B = input_ids.size(0)
    K = digit_ids_t.numel()

    remaining = torch.ones((B, K), dtype=torch.bool, device=device)
    chosen_digits = [[] for _ in range(B)]

    cur_input_ids = input_ids
    cur_attn = attention_mask

    for _ in range(K):
        out = model(input_ids=cur_input_ids, attention_mask=cur_attn)
        last_idx = _get_last_token_index(cur_attn)  # [B]
        next_logits = out.logits[torch.arange(B, device=device), last_idx, :]  # [B, V]
        digit_logits = next_logits.index_select(dim=-1, index=digit_ids_t)     # [B, K]

        masked = digit_logits.masked_fill(~remaining, float("-inf"))  # [B, K]
        pick = torch.argmax(masked, dim=-1)  # [B] in 0..K-1

        for b in range(B):
            chosen_digits[b].append(int(pick[b].item()))
        remaining[torch.arange(B, device=device), pick] = False

        next_token_ids = digit_ids_t[pick].view(B, 1)  # [B,1]
        cur_input_ids = torch.cat([cur_input_ids, next_token_ids], dim=1)
        cur_attn = torch.cat([cur_attn, torch.ones((B, 1), dtype=cur_attn.dtype, device=device)], dim=1)

    rankings = []
    for b in range(B):
        rankings.append("".join(str(x + 1) for x in chosen_digits[b]))
    return rankings


# --------------------------
# batching helper
# --------------------------
def _iter_batches(ds, tok, batch_size: int):
    for start in range(0, len(ds), batch_size):
        batch = ds[start: start + batch_size]
        prompts = batch["prompt"]
        sols = batch["solution"]

        prompt_texts = [
            tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in prompts
        ]
        pos = [int(s) for s in sols]
        yield start, prompt_texts, pos


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / max(len(xs), 1))


# --------------------------
# per-mode evaluation
# --------------------------
@torch.inference_mode()
def eval_one_forward(
    model, tok, ds, digit_ids_t: torch.Tensor,
    batch_size: int, max_prompt_length: int, device: str,
    print_every: int = 0, print_examples: int = 1,
) -> Dict[str, float]:
    total = valid = hit1 = 0
    ndcgs: List[float] = []
    global_idx = 0

    pbar = tqdm(total=len(ds), desc="Mode=one_forward")
    for batch_id, (start, prompt_texts, pos_list) in enumerate(_iter_batches(ds, tok, batch_size)):
        rankings, _, probs = rank_from_first_token_logits(
            model, tok, prompt_texts, digit_ids_t=digit_ids_t,
            max_prompt_length=max_prompt_length, device=device,
        )

        should_print_batch = (print_every > 0 and (batch_id % print_every == 0))
        printed = 0

        for i in range(len(prompt_texts)):
            perm0 = parse_perm_digits(rankings[i], K=K_FIXED)  # 一定合法
            pos = pos_list[i]

            total += 1
            valid += 1
            if perm0[0] == pos:
                hit1 += 1
            ndcgs.append(ndcg_single_positive(perm0, pos))

            if should_print_batch and printed < print_examples:
                print("\n========== Debug (one_forward) ==========")
                print(f"[Global Sample Idx] {global_idx}")
                print(f"[Gold pos_idx]      {pos}")
                print(f"[Ranking str]       '{rankings[i]}'")
                p = probs[i].detach().cpu().tolist()
                print(f"[First-token probs] 1..5 = {['%.4f'%x for x in p]}")
                print("========================================\n")
                printed += 1

            global_idx += 1

        pbar.update(len(prompt_texts))
    pbar.close()

    return {
        "samples": float(total),
        "valid_rate": valid / max(total, 1),
        "hit@1": hit1 / max(valid, 1),
        "ndcg@5": _mean(ndcgs),
    }


@torch.inference_mode()
def eval_five_forward(
    model, tok, ds, digit_ids_t: torch.Tensor,
    batch_size: int, max_prompt_length: int, device: str,
    print_every: int = 0, print_examples: int = 1,
) -> Dict[str, float]:
    total = valid = hit1 = 0
    ndcgs: List[float] = []
    global_idx = 0

    pbar = tqdm(total=len(ds), desc="Mode=five_forward")
    for batch_id, (start, prompt_texts, pos_list) in enumerate(_iter_batches(ds, tok, batch_size)):
        rankings = greedy_permutation_by_iterative_forward(
            model, tok, prompt_texts, digit_ids_t=digit_ids_t,
            max_prompt_length=max_prompt_length, device=device,
        )

        should_print_batch = (print_every > 0 and (batch_id % print_every == 0))
        printed = 0

        for i in range(len(prompt_texts)):
            perm0 = parse_perm_digits(rankings[i], K=K_FIXED)  # 一定合法
            pos = pos_list[i]

            total += 1
            valid += 1
            if perm0[0] == pos:
                hit1 += 1
            ndcgs.append(ndcg_single_positive(perm0, pos))

            if should_print_batch and printed < print_examples:
                print("\n========== Debug (five_forward) ==========")
                print(f"[Global Sample Idx] {global_idx}")
                print(f"[Gold pos_idx]      {pos}")
                print(f"[Ranking str]       '{rankings[i]}'")
                print("=========================================\n")
                printed += 1

            global_idx += 1

        pbar.update(len(prompt_texts))
    pbar.close()

    return {
        "samples": float(total),
        "valid_rate": valid / max(total, 1),
        "hit@1": hit1 / max(valid, 1),
        "ndcg@5": _mean(ndcgs),
    }


@torch.inference_mode()
def eval_generate(
    model, tok, ds,
    batch_size: int, max_prompt_length: int, device: str,
    K: int,
    gen_do_sample: bool, gen_temperature: float, gen_top_p: float,
    print_every: int = 0, print_examples: int = 1,
) -> Dict[str, float]:
    total = valid = hit1 = 0
    ndcgs: List[float] = []
    global_idx = 0

    pbar = tqdm(total=len(ds), desc="Mode=generate")
    for batch_id, (start, prompt_texts, pos_list) in enumerate(_iter_batches(ds, tok, batch_size)):
        enc = tok(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
        ).to(device)

        gen = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=K,
            do_sample=gen_do_sample,
            temperature=gen_temperature,
            top_p=gen_top_p,
            pad_token_id=tok.pad_token_id,
            eos_token_id=None,
        )

        should_print_batch = (print_every > 0 and (batch_id % print_every == 0))
        printed = 0

        input_len = enc["input_ids"].shape[1]

        for i in range(gen.size(0)):
            out_ids = gen[i, input_len: input_len + K]
            gen_text = tok.decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            perm0 = parse_perm_digits(gen_text, K=K)
            pos = pos_list[i]

            total += 1
            if perm0 is not None:
                valid += 1
                if perm0[0] == pos:
                    hit1 += 1
                ndcgs.append(ndcg_single_positive(perm0, pos))

            if should_print_batch and printed < print_examples:
                print("\n========== Debug (generate) ==========")
                print(f"[Global Sample Idx] {global_idx}")
                print(f"[Gold pos_idx]      {pos}")
                print(f"[Generate text]     '{gen_text}'   valid={perm0 is not None}")
                print("=====================================\n")
                printed += 1

            global_idx += 1

        pbar.update(len(prompt_texts))
    pbar.close()

    return {
        "samples": float(total),
        "valid_rate": valid / max(total, 1),
        "hit@1": hit1 / max(valid, 1),   # on valid only
        "ndcg@5": _mean(ndcgs),          # on valid only
    }


# --------------------------
# main
# --------------------------
@torch.inference_mode()
def run_all_modes(
    model_dir: str,
    dataset_path: str,
    split: str,
    batch_size: int,
    max_prompt_length: int,
    device: str,
    user_win_size: int = 10,
    K: int = K_FIXED,
    modes: List[str] = None,
    # generate params
    gen_do_sample: bool = True,
    gen_temperature: float = 1.0,
    gen_top_p: float = 1.0,
    # debug
    print_every: int = 0,
    print_examples: int = 1,
):
    assert K == 5, "当前脚本默认 K=5（如需扩展我也可以改成通用）"
    modes = modes or VALID_MODES
    for m in modes:
        assert m in VALID_MODES, f"Unknown mode: {m}, must be one of {VALID_MODES}"

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map=None,
    ).to(device)
    model.eval()

    digit_ids_t = _digit_token_ids(tok, K=K, device=device)

    ds = build_rec_hf_dataset_pl_aligned(
        dataset_path=dataset_path,
        tokenizer=tok,
        split=split,
        K=K,
        user_win_size=user_win_size,
        seed=42,
    )

    print("\n========== Eval Compare (separate runs) ==========")
    print(f"modes       : {modes}")
    print(f"samples     : {len(ds)}")
    print(f"batch_size  : {batch_size}")
    print("==================================================\n")

    results: Dict[str, Dict[str, float]] = {}

    if "one_forward" in modes:
        r = eval_one_forward(
            model, tok, ds, digit_ids_t=digit_ids_t,
            batch_size=batch_size, max_prompt_length=max_prompt_length, device=device,
            print_every=print_every, print_examples=print_examples,
        )
        results["one_forward"] = r

    if "five_forward" in modes:
        r = eval_five_forward(
            model, tok, ds, digit_ids_t=digit_ids_t,
            batch_size=batch_size, max_prompt_length=max_prompt_length, device=device,
            print_every=print_every, print_examples=print_examples,
        )
        results["five_forward"] = r

    if "generate" in modes:
        r = eval_generate(
            model, tok, ds,
            batch_size=batch_size, max_prompt_length=max_prompt_length, device=device,
            K=K,
            gen_do_sample=gen_do_sample, gen_temperature=gen_temperature, gen_top_p=gen_top_p,
            print_every=print_every, print_examples=print_examples,
        )
        results["generate"] = r

    # pretty print
    def _print_block(name: str, r: Dict[str, float]):
        print(f"\n[{name}]")
        print(f"samples     : {int(r['samples'])}")
        print(f"valid_rate  : {r['valid_rate']:.4f}")
        print(f"hit@1       : {r['hit@1']:.4f}")
        print(f"ndcg@5      : {r['ndcg@5']:.4f}")

    for m in modes:
        _print_block(m, results[m])

    print("\n===============================================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_prompt_length", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--user_win_size", type=int, default=10)

    # which modes to run (default all three)
    ap.add_argument(
        "--modes",
        nargs="+",
        choices=VALID_MODES,
        default=VALID_MODES,
        help="Which modes to run. Default: run all three: one_forward five_forward generate",
    )

    # generate params
    ap.add_argument("--gen_greedy", action="store_true", help="force greedy decoding for generate")
    ap.add_argument("--gen_do_sample", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--gen_temperature", type=float, default=1.0)
    ap.add_argument("--gen_top_p", type=float, default=1.0)

    # debug
    ap.add_argument("--print_every", type=int, default=0)
    ap.add_argument("--print_examples", type=int, default=1)

    args = ap.parse_args()

    # greedy overrides sampling knobs
    if args.gen_greedy:
        gen_do_sample = False
        gen_temperature = 1.0
        gen_top_p = 1.0
    else:
        gen_do_sample = bool(args.gen_do_sample)
        gen_temperature = float(args.gen_temperature)
        gen_top_p = float(args.gen_top_p)

    run_all_modes(
        model_dir=args.model_dir,
        dataset_path=args.dataset_path,
        split=args.split,
        batch_size=args.batch_size,
        max_prompt_length=args.max_prompt_length,
        device=args.device,
        user_win_size=args.user_win_size,
        K=K_FIXED,
        modes=args.modes,
        gen_do_sample=gen_do_sample,
        gen_temperature=gen_temperature,
        gen_top_p=gen_top_p,
        print_every=args.print_every,
        print_examples=args.print_examples,
    )


if __name__ == "__main__":
    main()
