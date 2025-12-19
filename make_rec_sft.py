#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import random
import argparse
from typing import List, Dict, Any

from transformers import AutoTokenizer
from rec_utils.train_init import create_dataset, RankingPrompter


def build_teacher_completion(pos_idx: int, K: int, rng: random.Random) -> str:
    """
    teacher policy: 把正样本放在第 1 位，其余 K-1 个随机排列
    输出格式：<I?><I?><I?><I?><I?>（无空格/无换行）
    """
    assert 0 <= pos_idx < K
    ids = list(range(K))
    ids.remove(pos_idx)
    rng.shuffle(ids)
    perm = [pos_idx] + ids
    return "".join([f"<I{i+1}>" for i in perm])


def dump_jsonl(rows: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", type=str, required=True)
    ap.add_argument("--model_name_or_path", type=str, required=True)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--user_win_size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)

    # 我建议 cold-start：5e4 条就够把输出模式“教会”，再上 GRPO
    ap.add_argument("--num_samples", type=int, default=50000)
    ap.add_argument("--eval_size", type=int, default=2000)
    ap.add_argument("--output_dir", type=str, required=True)

    # 是否把 prompt 走 tokenizer.apply_chat_template
    ap.add_argument("--apply_chat_template", action="store_true")

    args = ap.parse_args()

    assert args.K == 5, "这个脚本按你当前设定固定 K=5（<I1>..<I5>）。"

    rng = random.Random(args.seed)

    # tokenizer（只用于可选 chat_template；prompt 文本里 <I*> 只是字符串）
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    prompter = RankingPrompter(tok)

    # 你的 RankingDataset（会在 __getitem__ 里按 base_seed+idx 固定采样候选）
    ds = create_dataset(
        dataset_path=args.dataset_path,
        split=args.split,
        K=args.K,
        user_win_size=args.user_win_size,
        seed=args.seed,
    )

    n = len(ds)
    take = min(args.num_samples, n)
    all_indices = list(range(n))
    rng.shuffle(all_indices)
    picked = all_indices[:take]

    # 划分 train/eval（eval_size 从 picked 里切）
    eval_size = min(args.eval_size, max(0, take // 20), take)  # 默认最多取 5% 且不超过 args.eval_size
    eval_indices = picked[:eval_size]
    train_indices = picked[eval_size:]

    def make_rows(indices: List[int]) -> List[Dict[str, Any]]:
        rows = []
        for idx in indices:
            ex = ds[idx]
            prompt_text = prompter.build_prompt(
                user_text=ex["user_text"],
                candidates=ex["candidates"],
                apply_chat_template=args.apply_chat_template,
                pl_grpo=True,  # ✅ 关键：PL prompt（候选带 <I*>）
            )
            completion = build_teacher_completion(ex["target_idx"], args.K, rng)

            rows.append(
                {
                    "prompt": prompt_text,
                    "completion": completion,
                    # 下面这些留着你 debug / 之后做离线评估很有用
                    "target_idx": ex["target_idx"],
                    "candidate_ids": ex["candidate_ids"],
                    "target_item_id": ex["target_item_id"],
                    "user_id": ex.get("user_id"),
                }
            )
        return rows

    train_rows = make_rows(train_indices)
    eval_rows = make_rows(eval_indices)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    dump_jsonl(train_rows, os.path.join(out_dir, "train.jsonl"))
    dump_jsonl(eval_rows, os.path.join(out_dir, "eval.jsonl"))

    meta = {
        "dataset_path": args.dataset_path,
        "model_name_or_path": args.model_name_or_path,
        "split": args.split,
        "K": args.K,
        "user_win_size": args.user_win_size,
        "seed": args.seed,
        "num_samples_requested": args.num_samples,
        "num_samples_taken": take,
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "apply_chat_template": bool(args.apply_chat_template),
        "teacher_policy": "put positive (<I(pos)>) at rank-1, shuffle the rest",
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote: {out_dir}/train.jsonl ({len(train_rows)})")
    print(f"[OK] wrote: {out_dir}/eval.jsonl ({len(eval_rows)})")
    print(f"[OK] wrote: {out_dir}/meta.json")


if __name__ == "__main__":
    main()
