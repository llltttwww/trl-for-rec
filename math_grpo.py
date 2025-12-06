#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""
GRPO on DeepMath-103K with Qwen2.5-3B-Instruct

- 支持 CLI 配置 (TrlParser + GRPOConfig + ModelConfig)
- 支持量化 / LoRA（通过 ModelConfig 参数控制）
- 支持 train/eval 切分
- 使用 accuracy_reward 做可验证数学奖励
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset

from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.rewards import accuracy_reward

# 可选：在 Space 或 TrackIO 里记录日志
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


# ============= 自定义脚本参数 =============
@dataclass
class DeepMathScriptArguments(ScriptArguments):
    # 可以是 HF Hub 名（trl-lib/DeepMath-103K），也可以是本地目录
    dataset_path: str = field(
        default="trl-lib/DeepMath-103K",
        metadata={"help": "HF dataset id 或本地 DeepMath-103K 路径"},
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "使用数据集的哪个 split，默认 train"},
    )
    eval_size: int = field(
        default=1000,
        metadata={"help": "从 train 中划出多少条做 eval；<=0 则不做 eval"},
    )


def main():
    # 解析 CLI 参数：顺序固定 (ScriptArgs, GRPOConfig, ModelConfig)
    script_args, training_args, model_args = TrlParser(
        (DeepMathScriptArguments, GRPOConfig, ModelConfig)
    ).parse_args_and_config()

    # ============= Model / dtype / quantization =============
    # dtype: "auto" 或 None 交给 transformers 自己判断
    if model_args.dtype in ["auto", None]:
        dtype = None
    else:
        dtype = getattr(torch, model_args.dtype)

    # 这些会传给 AutoModelForCausalLM.from_pretrained(...)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=dtype,
    )

    # 量化（4bit / 8bit 等）
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        training_args.model_init_kwargs["quantization_config"] = quantization_config
        training_args.model_init_kwargs["device_map"] = get_kbit_device_map()

    # ============= Dataset =============
    # DeepMath-103K 默认列：
    # - "prompt": 题目
    # - "solution": 最终答案（给 accuracy_reward 用）
    dataset = load_dataset(script_args.dataset_path, split=script_args.dataset_split)

    if script_args.eval_size and script_args.eval_size > 0:
        dataset = dataset.train_test_split(test_size=script_args.eval_size, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    # ============= Trainer =============
    # accuracy_reward 会自动从 dataset 的 "solution" 列读取答案
    # （要求列名就是 solution，DeepMath-103K 已满足）
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,         # 可以是 HF 名，也可以是本地路径
        args=training_args,
        reward_funcs=accuracy_reward,                # 也可以换成 [accuracy_reward] 或多个 reward 组成的 list
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),     # 如果 --use_peft 就会启用 LoRA 等
    )

    trainer.train()

    # Save / push
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        # dataset_name 只是 Hub 上的名字，随便填
        trainer.push_to_hub(dataset_name=script_args.dataset_name or "DeepMath-103K-GRPO")


if __name__ == "__main__":
    main()
