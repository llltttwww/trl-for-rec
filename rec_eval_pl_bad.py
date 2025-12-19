#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import re
from typing import List, Dict, Any, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from rec_utils.rec_grpo_data import build_rec_hf_dataset

# vLLM client（来自本地的 trl 源码）
try:
    from trl.extras.vllm_client import VLLMClient
except ImportError:
    VLLMClient = None


def _build_completions_from_vllm(raw_outputs: Any, tokenizer) -> List[List[Dict[str, str]]]:
    """同你原来的版本：从 vllm_client.generate 的输出抽 completion 文本。"""
    if isinstance(raw_outputs, dict) and "completion_ids" in raw_outputs:
        completion_ids_list = raw_outputs["completion_ids"]
        normalized_ids: List[List[int]] = []
        for ids in completion_ids_list:
            if isinstance(ids, torch.Tensor):
                normalized_ids.append(ids.tolist())
            else:
                normalized_ids.append(list(ids))
        texts = tokenizer.batch_decode(
            normalized_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return [[{"role": "assistant", "content": t}] for t in texts]

    completions: List[List[Dict[str, str]]] = []
    if isinstance(raw_outputs, list):
        for out in raw_outputs:
            if isinstance(out, str):
                text = out
            elif isinstance(out, dict):
                text = out.get("content") or out.get("text") or str(out)
            elif isinstance(out, list) and len(out) > 0:
                first = out[0]
                if isinstance(first, dict):
                    text = first.get("content") or first.get("text") or str(first)
                else:
                    text = str(first)
            else:
                text = str(out)
            completions.append([{"role": "assistant", "content": text}])
    else:
        completions.append([{"role": "assistant", "content": str(raw_outputs)}])

    return completions


def parse_perm_digits(text: str, K: int) -> Optional[List[int]]:
    """
    从 completion 文本中抽取一个长度为 K 的“纯数字排列”，例如 K=5 时抽 '21354'。
    允许输出里带别的字符：会在文本中扫描所有连续数字串，然后在每个串里滑窗找 K 位子串。
    返回 0-based ranking（每位减 1）：[1,0,2,4,3] 这种。
    """
    if not text:
        return None

    digit_block_re = re.compile(r"\d+")
    for m in digit_block_re.finditer(text):
        block = m.group(0)
        if len(block) < K:
            continue
        # 在 block 里滑窗找长度 K 的片段
        for i in range(0, len(block) - K + 1):
            s = block[i : i + K]
            # 必须每位都在 1..K 且无重复
            ok = True
            seen = set()
            for ch in s:
                if ch < "0" or ch > "9":
                    ok = False
                    break
                d = ord(ch) - ord("0")
                if d < 1 or d > K:
                    ok = False
                    break
                if d in seen:
                    ok = False
                    break
                seen.add(d)
            if ok and len(seen) == K:
                # 转成 0-based 索引
                return [ (ord(ch) - ord("0")) - 1 for ch in s ]

    return None


def format_reward_perm_digits(completions: List[List[Dict[str, str]]], K: int) -> List[float]:
    """合法排列=1，否则=0。"""
    out: List[float] = []
    for comp in completions:
        text = comp[0].get("content", "") if comp else ""
        perm0 = parse_perm_digits(text, K)
        out.append(1.0 if perm0 is not None else 0.0)
    return out


def ndcg_reward_perm_digits(completions: List[List[Dict[str, str]]], solution: List[str], K: int) -> List[Optional[float]]:
    """
    单正样本的 NDCG@K：
      rank = 正样本在预测排列中的名次(1..K)
      DCG = 1/log2(rank+1)
      IDCG = 1/log2(1+1) = 1
      NDCG = DCG
    若输出非法排列，则返回 None（你也可以改成 0.0）。
    """
    out: List[Optional[float]] = []
    for comp, sol in zip(completions, solution):
        text = comp[0].get("content", "") if comp else ""
        perm0 = parse_perm_digits(text, K)
        if perm0 is None:
            out.append(None)
            continue
        try:
            pos_idx = int(sol)  # 0..K-1
        except Exception:
            out.append(None)
            continue
        if pos_idx < 0 or pos_idx >= K:
            out.append(None)
            continue
        try:
            rank = perm0.index(pos_idx) + 1  # 1..K
        except ValueError:
            out.append(None)
            continue
        out.append(1.0 / math.log2(rank + 1.0))
    return out


@torch.inference_mode()
def evaluate_checkpoint(
    model_dir: str,
    dataset_path: str,
    split: str = "test",
    K: int = 5,
    user_win_size: int = 10,
    max_prompt_length: int = 4096,
    max_new_tokens: int = 32,
    batch_size: int = 4,
    device: str = "cuda",
    use_vllm: bool = False,
    vllm_base_url: str = "http://0.0.0.0:8000",
    vllm_temperature: float = 0.0,
    vllm_top_p: float = 1.0,
    print_every: int = 0,
    print_examples: int = 1,
):
    print(f"[Eval] Loading tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side="left"

    model = None
    vllm_client = None
    if use_vllm:
        assert VLLMClient is not None, "trl.extras.vllm_client 未安装，无法 use_vllm=True"
        print(f"[Eval] Using vLLM server at {vllm_base_url}")
        vllm_client = VLLMClient(base_url=vllm_base_url)
    else:
        print(f"[Eval] Loading local model from {model_dir} to {device}")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map=None,
        ).to(device)
        model.eval()

    print(f"[Eval] Building HF dataset from {dataset_path}, split={split}")
    ds = build_rec_hf_dataset(
        dataset_path=dataset_path,
        model_name_or_path=model_dir,
        split=split,
        K=K,
        user_win_size=user_win_size,
        seed=42,
    )
    print(f"[Eval] Dataset size: {len(ds)} samples")

    all_format_rewards: List[float] = []
    all_ndcg_rewards: List[float] = []
    all_hit1: List[float] = []
    all_valid: List[float] = []

    global_sample_idx = 0

    for start in tqdm(range(0, len(ds), batch_size), desc="Evaluating"):
        batch = ds[start : start + batch_size]

        prompts = batch["prompt"]      # list of list[{"role","content"}]
        solutions = batch["solution"]  # list[str] (pos_idx 0-based)

        prompt_texts = [
            tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            for msgs in prompts
        ]

        batch_idx = start // batch_size
        should_print_batch = (print_every > 0 and (batch_idx % print_every == 0))

        # ===== generate =====
        if use_vllm:
            sampling_params = {
                "n": 1,
                "repetition_penalty": 1.0,
                "temperature": vllm_temperature,
                "top_p": vllm_top_p,
                "top_k": -1,
                "min_p": 0.0,
                "max_tokens": max_new_tokens,
                "truncate_prompt_tokens": max_prompt_length,
                "guided_decoding_regex": None,
                "generation_kwargs": None,
            }
            raw_outputs = vllm_client.generate(prompts=prompt_texts, **sampling_params)
            completions_batch = _build_completions_from_vllm(raw_outputs, tokenizer)
        else:
            enc = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_length,
            ).to(device)

            gen_ids = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            completions_batch = []
            for i in range(gen_ids.size(0)):
                prompt_len = enc["attention_mask"][i].sum().item()
                generated_ids = gen_ids[i, prompt_len:]
                text = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                print(f'@@@ Gen_text: {text}')
                completions_batch.append([{"role": "assistant", "content": text}])

        n_samples = min(len(completions_batch), len(solutions))
        if n_samples == 0:
            continue
        completions_used = completions_batch[:n_samples]
        solutions_used = solutions[:n_samples]

        # ===== reward: digits permutation =====
        fmt_batch = format_reward_perm_digits(completions_used, K=K)
        ndcg_batch = ndcg_reward_perm_digits(completions_used, solutions_used, K=K)

        printed_in_this_batch = 0
        for comp, sol, fmt, ndcg in zip(completions_used, solutions_used, fmt_batch, ndcg_batch):
            text = comp[0].get("content", "") if comp else ""

            all_format_rewards.append(float(fmt))
            all_valid.append(1.0 if fmt > 0.5 else 0.0)
            all_ndcg_rewards.append(0.0 if ndcg is None else float(ndcg))

            # hit@1
            hit1 = 0.0
            perm0 = parse_perm_digits(text, K)
            try:
                pos_idx = int(sol)
            except Exception:
                pos_idx = -1
            if perm0 is not None and 0 <= pos_idx < K:
                if perm0[0] == pos_idx:
                    hit1 = 1.0
            all_hit1.append(hit1)

            if should_print_batch and printed_in_this_batch < print_examples:
                prompt_str = prompt_texts[printed_in_this_batch]

                print("\n========== Sample Debug ==========")
                print(f"[Global Sample Idx] {global_sample_idx}")
                print(f"[Batch Idx]        {batch_idx}")
                print(f"[Gold Pos Idx]     {pos_idx}")
                print(f"[Valid Perm]       {perm0 is not None}")
                print(f"[Parsed Perm0]     {perm0}")
                print(f"[Format Reward]    {float(fmt):.4f}")
                print(f"[NDCG@K]           {0.0 if ndcg is None else float(ndcg):.4f}")
                print(f"[Hit@1]            {hit1:.4f}")
                print("----------- Completion -----------")
                print(text)
                print("=================================\n")

                printed_in_this_batch += 1

            global_sample_idx += 1

    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / max(len(xs), 1))

    print("\n========== Evaluation Results ==========")
    print(f"#Samples            : {len(ds)}")
    print(f"Valid rate          : {_mean(all_valid):.4f}")
    print(f"Format reward mean  : {_mean(all_format_rewards):.4f}")
    print(f"NDCG@K mean         : {_mean(all_ndcg_rewards):.4f}")
    print(f"Hit@1               : {_mean(all_hit1):.4f}")
    print("========================================\n")

    return {
        "valid_rate": _mean(all_valid),
        "format_reward_mean": _mean(all_format_rewards),
        "ndcg_mean": _mean(all_ndcg_rewards),
        "hit1": _mean(all_hit1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--user_win_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--vllm_base_url", type=str, default="http://0.0.0.0:8000")
    parser.add_argument("--vllm_temperature", type=float, default=0.0)
    parser.add_argument("--vllm_top_p", type=float, default=1.0)

    parser.add_argument("--print_every", type=int, default=0)
    parser.add_argument("--print_examples", type=int, default=1)

    args = parser.parse_args()

    evaluate_checkpoint(
        model_dir=args.model_dir,
        dataset_path=args.dataset_path,
        split=args.split,
        K=args.K,
        user_win_size=args.user_win_size,
        batch_size=args.batch_size,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        use_vllm=args.use_vllm,
        vllm_base_url=args.vllm_base_url,
        vllm_temperature=args.vllm_temperature,
        vllm_top_p=args.vllm_top_p,
        print_every=args.print_every,
        print_examples=args.print_examples,
    )


if __name__ == "__main__":
    main()
