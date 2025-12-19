#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, random, re, math
from typing import Optional, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from rec_utils.train_init import create_dataset, RankingPrompter

K = 5
ITEM_RE = re.compile(r"<I([1-5])>")

def parse_perm(text: str, K: int = 5) -> Optional[List[int]]:
    perm = [int(m.group(1)) - 1 for m in ITEM_RE.finditer(text)][:K]
    if len(perm) != K: return None
    if sorted(perm) != list(range(K)): return None
    return perm

def ndcg_reward_single(pred_text: str, pos_idx: int) -> float:
    perm = parse_perm(pred_text, K=K)
    if perm is None: return 0.0
    rank_pos = perm.index(pos_idx)
    return 1.0 / math.log2(rank_pos + 2)

@torch.inference_mode()
def generate_from_text(model, tok, prompt_text: str, max_new_tokens: int):
    enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model.generate(
        **enc,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    gen_ids = out[0, enc["input_ids"].shape[1]:]
    gen_text = tok.decode(gen_ids, skip_special_tokens=False)
    return gen_text, gen_ids.tolist(), enc["input_ids"].shape[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--dataset_path", type=str, required=True)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--num_cases", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--user_win_size", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    args = ap.parse_args()

    random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.ckpt, use_fast=False)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # sanity: <I*> ids
    for i in range(1, 6):
        t = f"<I{i}>"
        tid = tok.convert_tokens_to_ids(t)
        print(t, "->", tid, " token_str=", tok.convert_ids_to_tokens(tid))

    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    ).eval()

    ds = create_dataset(args.dataset_path, split=args.split, K=K, user_win_size=args.user_win_size, seed=args.seed)
    prompter = RankingPrompter(tok)
    idxs = random.sample(range(len(ds)), k=min(args.num_cases, len(ds)))

    for n, idx in enumerate(idxs, 1):
        ex = ds[idx]
        pos_idx = int(ex["target_idx"])

        # 1) RAW：完全不套 chat template（有些训练就是这么喂的）
        raw_prompt = prompter.build_prompt(
            user_text=ex["user_text"],
            candidates=ex["candidates"],
            apply_chat_template=False,
            pl_grpo=True,
        )

        # 2) CHAT：用 tokenizer 的 chat template，并加 generation prompt
        chat_prompt = tok.apply_chat_template(
            [{"role": "user", "content": raw_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

        for tag, prompt_text in [("RAW", raw_prompt), ("CHAT", chat_prompt)]:
            gen_text, gen_ids, in_len = generate_from_text(model, tok, prompt_text, args.max_new_tokens)
            hits = [m.group(0) for m in ITEM_RE.finditer(gen_text)]
            perm = parse_perm(gen_text)
            r = ndcg_reward_single(gen_text, pos_idx)

            print("\n" + "=" * 110)
            print(f"[{n}/{len(idxs)}] idx={idx}  target_idx={pos_idx}  MODE={tag}")
            print(f"prompt_len_tokens={in_len}")
            print("gen_text:", repr(gen_text))
            print("gen_ids:", gen_ids[:40], "..." if len(gen_ids) > 40 else "")
            print("hits:", hits)
            print("perm:", perm, "reward:", f"{r:.4f}")

if __name__ == "__main__":
    main()
