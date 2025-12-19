#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import random
import re
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from rec_utils.train_init import create_dataset, RankingPrompter

K = 5
ITEM_RE = re.compile(r"<I([1-5])>")

def parse_perm(text: str, K: int = 5) -> Optional[List[int]]:
    perm = [int(m.group(1)) - 1 for m in ITEM_RE.finditer(text)]
    perm = perm[:K]
    if len(perm) != K:
        return None
    if sorted(perm) != list(range(K)):
        return None
    return perm

def ndcg_reward_single(pred_text: str, pos_idx: int, K: int = 5) -> float:
    perm = parse_perm(pred_text, K=K)
    if perm is None:
        return 0.0
    rank_pos = perm.index(pos_idx)
    return 1.0 / math.log2(rank_pos + 2)

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="SFT checkpoint dir (model + tokenizer)")
    ap.add_argument("--dataset_path", type=str, required=True)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--user_win_size", type=int, default=10)
    ap.add_argument("--num_cases", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_prompt_length", type=int, default=4096)

    # generation knobs (no constraint)
    ap.add_argument("--max_new_tokens", type=int, default=32, help="give it room to fail / ramble")
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--print_prompt", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.ckpt, use_fast=False)
    tok.padding_side = "left"
    tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # just for sanity (optional)
    missing = [f"<I{i}>" for i in range(1, 6) if tok.convert_tokens_to_ids(f"<I{i}>") in (None, -1)]
    if missing:
        print(f"[WARN] tokenizer missing tokens: {missing} (then SFT can't possibly learn that format)")

    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    ds = create_dataset(
        dataset_path=args.dataset_path,
        split=args.split,
        K=K,
        user_win_size=args.user_win_size,
        seed=args.seed,
    )
    prompter = RankingPrompter(tok)

    indices = random.sample(range(len(ds)), k=min(args.num_cases, len(ds)))

    fmt_ok = 0
    perm_ok = 0
    rewards = []

    print("=" * 100)
    print(f"ckpt: {args.ckpt}")
    print(f"split={args.split}  cases={len(indices)}  max_new_tokens={args.max_new_tokens}  do_sample={args.do_sample}")
    print("=" * 100)

    for j, idx in enumerate(indices, start=1):
        ex = ds[idx]
        pos_idx = int(ex["target_idx"])

        prompt_text = prompter.build_prompt(
            user_text=ex["user_text"],
            candidates=ex["candidates"],
            apply_chat_template=True,
            pl_grpo=True,
        )

        enc = tok(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_prompt_length,
            add_special_tokens=False,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        out = model.generate(
            **enc,
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else None,
            top_p=args.top_p if args.do_sample else None,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tok.eos_token_id,   # allow early stop if it emits eos
            pad_token_id=tok.pad_token_id,
        )

        gen_ids = out[0, enc["input_ids"].shape[1]:]
        gen_text = tok.decode(gen_ids, skip_special_tokens=False)

        hits = [m.group(0) for m in ITEM_RE.finditer(gen_text)]
        is_fmt = (len(hits) == 5) and (gen_text.strip().replace("\n", "").replace(" ", "") == "".join(hits))
        # 上面这个 is_fmt 很严格：要求输出“只有”5个<I*>且无其它字符（含空格/换行）

        perm = parse_perm(gen_text, K=K)
        r = ndcg_reward_single(gen_text, pos_idx, K=K)

        fmt_ok += int(is_fmt)
        perm_ok += int(perm is not None)
        rewards.append(r)

        print(f"\n[{j}/{len(indices)}] idx={idx}  target_idx(pos)={pos_idx}")
        if args.print_prompt:
            print("-" * 40 + " PROMPT " + "-" * 40)
            print(prompt_text)
            print("-" * 99)
        print(f"raw_gen: {repr(gen_text)}")
        print(f"extracted: {hits}")
        print(f"fmt_only_5_tokens: {is_fmt}")
        print(f"perm: {perm}")
        print(f"reward: {r:.4f}")

    n = len(indices)
    mean_r = sum(rewards) / max(1, n)
    print("\n" + "=" * 100)
    print(f"fmt_only_5_tokens acc: {fmt_ok}/{n} = {fmt_ok/n:.3f}")
    print(f"perm_valid acc       : {perm_ok}/{n} = {perm_ok/n:.3f}")
    print(f"mean reward          : {mean_r:.6f}")
    print("=" * 100)

if __name__ == "__main__":
    main()
