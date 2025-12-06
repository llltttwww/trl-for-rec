#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
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

from rec_utils.rec_grpo_data import build_rec_hf_dataset
from rec_utils.rec_grpo_reward import format_reward_rec, ndcg_reward_rec  # ✅ 两个 reward 函数

os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


# ============= 自定义脚本参数 =============
@dataclass
class RecScriptArguments(ScriptArguments):
    dataset_path: str = field(
        default="/mnt/shared-storage-user/p1-shared/luotianwei/trl-for-rec/data/Musical_Instruments_0_2022-10-2023-10",
        metadata={"help": "HF load_from_disk 的推荐数据路径"},
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "train / val 等"},
    )
    eval_size: int = field(
        default=1000,
        metadata={"help": "从 train 中切多少条做 eval；<=0 表示不做 eval"},
    )
    topk: int = field(
        default=5,
        metadata={"help": "每个样本的候选 item 数（K），比如 5"},
    )
    user_win_size: int = field(
        default=10,
        metadata={"help": "用户历史窗口大小"},
    )
    format_weight: float = field(
        default=0.3,
        metadata={"help": "格式奖励的权重"},
    )
    ndcg_weight: float = field(
        default=0.7,
        metadata={"help": "NDCG 排序奖励的权重"},
    )


def main():
    # 解析 CLI 参数
    script_args, training_args, model_args = TrlParser(
        (RecScriptArguments, GRPOConfig, ModelConfig)
    ).parse_args_and_config()

    # ============= Model / dtype / quant =============
    if model_args.dtype in ["auto", None]:
        dtype = None
    else:
        dtype = getattr(torch, model_args.dtype)

    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        # attn_implementation="flash_attention_2",
        torch_dtype=dtype,
        device_map=None,
    )

    # ============= Dataset =============
    full_train_ds = build_rec_hf_dataset(
        dataset_path=script_args.dataset_path,
        model_name_or_path=model_args.model_name_or_path,
        split=script_args.dataset_split,
        K=script_args.topk,
        user_win_size=script_args.user_win_size,
    )

    if script_args.eval_size and script_args.eval_size > 0 and len(full_train_ds) > script_args.eval_size:
        ds_dict = full_train_ds.train_test_split(test_size=script_args.eval_size, seed=42)
        train_dataset = ds_dict["train"]
        eval_dataset = ds_dict["test"]
    else:
        train_dataset = full_train_ds
        eval_dataset = None

    # ============= Reward 权重 =============
    # GRPOConfig 里自带 reward_weights 字段，顺序对应 reward_funcs 里的顺序
    training_args.reward_weights = [script_args.format_weight, script_args.ndcg_weight]

    # ============= Trainer =============
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        # ✅ 两个 reward 函数按顺序传进去
        reward_funcs=[format_reward_rec, ndcg_reward_rec],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name or "Rec-GRPO-NDCG")


if __name__ == "__main__":
    main()
