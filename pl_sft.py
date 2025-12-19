#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_name_or_path", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True)  # train.jsonl / eval.jsonl
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--max_length", type=int, default=4096)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--logging_steps", type=int, default=10)

    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--save_total_limit", type=int, default=2)

    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report_to", type=str, default="tensorboard")
    ap.add_argument("--run_name", type=str, default="pl-sft-chatfmt")

    args = ap.parse_args()

    train_path = os.path.join(args.data_dir, "train.jsonl")
    eval_path = os.path.join(args.data_dir, "eval.jsonl")

    if not os.path.exists(train_path):
        raise FileNotFoundError(train_path)
    if not os.path.exists(eval_path):
        print(f"[WARN] eval.jsonl not found at {eval_path}, will train without eval.")
        eval_path = None

    # -------- tokenizer --------
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    tok.padding_side = "right"
    tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 你的 5 个 token（如果本来就有，不会新增）
    special = [f"<I{i}>" for i in range(1, 6)]
    num_added = tok.add_tokens(special, special_tokens=False)
    if num_added:
        print(f"[tokenizer] added {num_added} tokens: {special}")

    # -------- model --------
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
        device_map=None,
    )
    model.resize_token_embeddings(len(tok))

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # -------- dataset --------
    data_files = {"train": train_path}
    if eval_path is not None:
        data_files["eval"] = eval_path
    ds = load_dataset("json", data_files=data_files)

    def to_conversation_format(ex):
        # 把原来的字符串 prompt/completion 转成对话消息列表
        # 保留其他字段（target_idx / user_id / candidate_ids ...）不动
        p = ex["prompt"]
        c = ex["completion"]
        if isinstance(p, str):
            ex["prompt"] = [{"role": "user", "content": p}]
        if isinstance(c, str):
            ex["completion"] = [{"role": "assistant", "content": c}]
        return ex

    train_ds = ds["train"].map(to_conversation_format, desc="Converting train to chat format")
    eval_ds = ds["eval"].map(to_conversation_format, desc="Converting eval to chat format") if "eval" in ds else None

    # -------- SFT config --------
    sft_args = SFTConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        seed=args.seed,

        completion_only_loss=True,
        max_length=args.max_length,
        packing=False,

        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",

        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps if eval_ds is not None else None,

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        report_to=args.report_to,
        bf16=args.bf16,
        fp16=args.fp16,

        remove_unused_columns=False,
    )

    # -------- trainer --------
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,  # 关键：这里用 tokenizer，才能 apply_chat_template
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "sft_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(f"[OK] saved to {args.output_dir}")

def maybe_attach_debugger():
    """
    用法：
      - 默认只在 rank0 开 debugpy，端口 5678，并等待 VSCode attach 后再继续。
      - 多进程想每个 rank 都能 attach：DEBUGPY_RANK0_ONLY=0，则端口=5678+LOCAL_RANK
    环境变量：
      DEBUGPY=1                  开启
      DEBUGPY_WAIT=1             是否 wait_for_client（默认 1）
      DEBUGPY_HOST=0.0.0.0       监听地址（默认 0.0.0.0）
      DEBUGPY_PORT=5678          基础端口（默认 5678）
      DEBUGPY_RANK0_ONLY=1       只在 rank0 启动（默认 1）
    """
    import os

    if os.environ.get("DEBUGPY", "1") != "1":
        return

    import debugpy

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    rank0_only = os.environ.get("DEBUGPY_RANK0_ONLY", "1") == "1"
    if rank0_only and rank != 0:
        return

    host = os.environ.get("DEBUGPY_HOST", "0.0.0.0")
    base_port = int(os.environ.get("DEBUGPY_PORT", "5679"))
    port = base_port if rank0_only else (base_port + local_rank)

    # 防止重复 listen（例如某些 launcher 可能二次 import）
    try:
        debugpy.listen((host, port))
    except Exception as e:
        print(f"[debugpy] listen failed on {host}:{port} (rank={rank}): {e}", flush=True)
        return

    print(f"[debugpy] listening on {host}:{port} (rank={rank}, local_rank={local_rank})", flush=True)

    if os.environ.get("DEBUGPY_WAIT", "1") == "1":
        debugpy.wait_for_client()
        debugpy.breakpoint()  # attach 后自动停在这里


if __name__ == "__main__":
    maybe_attach_debugger()
    main()
