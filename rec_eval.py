#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from typing import List, Dict, Optional, Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from rec_utils.rec_grpo_data import build_rec_hf_dataset
from rec_utils.rec_grpo_reward import (
    format_reward_rec,
    ndcg_reward_rec,
    _parse_ranking,
    _normalize_ranking_1_to_0_based,
)

# vLLM client（来自本地的 trl 源码）
try:
    from trl.extras.vllm_client import VLLMClient
except ImportError:
    VLLMClient = None


def _build_completions_from_vllm(raw_outputs: Any, tokenizer) -> List[List[Dict[str, str]]]:
    """
    处理 trl.extras.vllm_client.VLLMClient.generate 的返回结果。

    正常情况（对齐 GRPOTrainer）：
        raw_outputs 是一个 dict，至少包含：
            - "completion_ids": List[List[int]]
        我们用 tokenizer.batch_decode 把它转成文本，再包装成：
            List[List[{"role": "assistant", "content": text}]]

    如果 raw_outputs 不是 dict 或缺少 "completion_ids"，
    则回退到比较宽松的 heuristic 逻辑，尽量从中抽出字符串。
    """
    # === 正常路径：dict + completion_ids ===
    if isinstance(raw_outputs, dict) and "completion_ids" in raw_outputs:
        completion_ids_list = raw_outputs["completion_ids"]  # List[List[int]] 或 tensor

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

    # === 回退路径：兼容“奇怪结构”的输出（例如 List[...]）===
    completions: List[List[Dict[str, str]]] = []

    if isinstance(raw_outputs, list):
        for out in raw_outputs:
            if isinstance(out, str):
                text = out
            elif isinstance(out, dict):
                # 尝试 content / text 字段
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
        # 实在不认识，整个转成字符串
        text = str(raw_outputs)
        completions.append([{"role": "assistant", "content": text}])

    return completions


@torch.inference_mode()
def evaluate_checkpoint(
    model_dir: str,
    dataset_path: str,
    split: str = "test",
    K: int = 5,
    user_win_size: int = 10,
    max_prompt_length: int = 4096,
    max_new_tokens: int = 512,
    batch_size: int = 4,
    device: str = "cuda",
    use_vllm: bool = False,
    vllm_base_url: str = "http://0.0.0.0:8000",
    vllm_temperature: float = 0.0,
    vllm_top_p: float = 1.0,
    print_every: int = 0,
    print_examples: int = 1,
):
    """
    在给定 checkpoint 上，对指定 split 做 rollout 评测。
    - 如果 use_vllm=False：本地加载 HF 模型，model.generate
    - 如果 use_vllm=True ：只加载 tokenizer，调用远端 vLLM server 生成

    print_every:
        每隔多少个 batch 打印一次模型回答（0 表示不打印）。
    print_examples:
        每次打印该 batch 里前多少个样本。
    """

    # 1) 加载 tokenizer
    print(f"[Eval] Loading tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1.1 模型 / vLLM client
    model = None
    vllm_client = None

    if use_vllm:
        assert VLLMClient is not None, "trl.extras.vllm_client 未安装，无法 use_vllm=True"
        print(f"[Eval] Using vLLM server at {vllm_base_url}")
        # 这里只做客户端，不会占用本机 GPU
        vllm_client = VLLMClient(base_url=vllm_base_url)
    else:
        print(f"[Eval] Loading local model from {model_dir} to {device}")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map=None,
        ).to(device)
        model.eval()

    # 2) 构建 HF Dataset（和训练时一样）
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

    # 3) 开始 batch rollout
    all_format_rewards: List[float] = []
    all_ndcg_rewards: List[float] = []
    all_hit1: List[float] = []

    # 用于打印时给样本编号
    global_sample_idx = 0

    for start in tqdm(range(0, len(ds), batch_size), desc="Evaluating"):
        batch = ds[start : start + batch_size]

        prompts = batch["prompt"]      # list of list[{"role","content"}]
        solutions = batch["solution"]  # list[str]，里面是 target_idx（0-based）

        # 3.1 构造 chat prompt 文本（和训练时一样）
        prompt_texts = [
            tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            for msgs in prompts
        ]

        # 当前是第几个 batch（从 0 开始）
        batch_idx = start // batch_size
        should_print_batch = (
            print_every > 0 and (batch_idx % print_every == 0)
        )

        # ============ 模型生成部分：两种模式 ============

        if use_vllm:
            # ---- vLLM server 生成 ----
            # 对齐 GRPOTrainer._generate_single_turn 的 sampling 参数
            sampling_params = {
                "n": 1,  # eval 一般只要 1 个 completion
                "repetition_penalty": 1.0,
                "temperature": vllm_temperature,
                "top_p": vllm_top_p,
                "top_k": -1,  # 不用 top_k
                "min_p": 0.0,
                "max_tokens": max_new_tokens,
                "truncate_prompt_tokens": max_prompt_length,
                "guided_decoding_regex": None,
                "generation_kwargs": None,
            }

            raw_outputs = vllm_client.generate(
                prompts=prompt_texts,
                **sampling_params,
            )

            # 从 dict["completion_ids"] 里解码出文本
            completions_batch: List[List[Dict[str, str]]] = _build_completions_from_vllm(
                raw_outputs, tokenizer
            )

        else:
            # ---- 本地 HF 模型 generate ----
            enc = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_length,
            ).to(device)

            input_ids = enc["input_ids"]
            attn_mask = enc["attention_mask"]

            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            completions_batch: List[List[Dict[str, str]]] = []
            for i in range(gen_ids.size(0)):
                prompt_len = attn_mask[i].sum().item()
                generated_ids = gen_ids[i, prompt_len:]

                text = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

                completions_batch.append(
                    [{"role": "assistant", "content": text}]
                )

        # ============ 长度对齐，避免 strict=True 抛错 ============
        n_samples = min(len(completions_batch), len(solutions))
        if n_samples == 0:
            continue
        if n_samples != len(solutions) or n_samples != len(completions_batch):
            print(
                f"[Eval][WARN] completions_batch({len(completions_batch)}) "
                f"!= solutions({len(solutions)}), only using first {n_samples} samples in this batch."
            )

        completions_used = completions_batch[:n_samples]
        solutions_used = solutions[:n_samples]

        # 统一批量算 reward
        fmt_batch = format_reward_rec(
            completions=completions_used,
            solution=solutions_used,
            K=K,
        )  # List[float]

        ndcg_batch = ndcg_reward_rec(
            completions=completions_used,
            solution=solutions_used,
            K=K,
        )  # List[Optional[float]]

        # 再计算 hit@1 & 汇总 + 按需打印
        printed_in_this_batch = 0

        for comp, sol, fmt, ndcg in zip(
            completions_used, solutions_used, fmt_batch, ndcg_batch, strict=True
        ):
            text = comp[0].get("content", "") if comp else ""

            all_format_rewards.append(float(fmt))
            all_ndcg_rewards.append(0.0 if ndcg is None else float(ndcg))

            # Hit@1
            ranking_1 = _parse_ranking(text)
            hit1 = 0.0
            try:
                pos_idx = int(sol)
            except Exception:
                pos_idx = -1

            ranking_0 = None
            if ranking_1 is not None and pos_idx >= 0:
                ranking_0 = _normalize_ranking_1_to_0_based(ranking_1, K)
                if ranking_0 is not None and len(ranking_0) > 0:
                    if ranking_0[0] == pos_idx:
                        hit1 = 1.0

            all_hit1.append(hit1)

            # ====== 按需打印当前样本 ======
            if should_print_batch and printed_in_this_batch < print_examples:
                # prompt 原始是 messages，我们就把 chat template 文本拿来，用于 debug
                prompt_str = prompt_texts[printed_in_this_batch]
                # 简单截断一下，避免太长刷屏
                max_prompt_show = 10000
                max_answer_show = 10000

                def _truncate(s: str, max_len: int) -> str:
                    return (s[:max_len] + "... [TRUNCATED]") if len(s) > max_len else s

                print("\n========== Sample Debug ==========")
                print(f"[Global Sample Idx] {global_sample_idx}")
                print(f"[Batch Idx]        {batch_idx}")
                print(f"[Gold Pos Idx]     {pos_idx}")
                print(f"[Format Reward]    {float(fmt):.4f}")
                print(f"[NDCG@K]           {0.0 if ndcg is None else float(ndcg):.4f}")
                print(f"[Hit@1]            {hit1:.4f}")
                print(f"[Parsed Ranking-1] {ranking_1}")
                print(f"[Parsed Ranking-0] {ranking_0}")
                print("----------- Prompt (truncated) -----------")
                print(_truncate(prompt_str, max_prompt_show))
                print("----------- Completion (truncated) ------")
                print(_truncate(text, max_answer_show))
                print("=========================================\n")

                printed_in_this_batch += 1

            global_sample_idx += 1

    # 4) 汇总
    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / max(len(xs), 1))

    fmt_mean = _mean(all_format_rewards)
    ndcg_mean = _mean(all_ndcg_rewards)
    hit1_mean = _mean(all_hit1)

    print("\n========== Evaluation Results ==========")
    print(f"#Samples            : {len(ds)}")
    print(f"Format reward mean  : {fmt_mean:.4f}")
    print(f"NDCG@K mean         : {ndcg_mean:.4f}")
    print(f"Hit@1               : {hit1_mean:.4f}")
    print("========================================\n")

    return {
        "format_reward_mean": fmt_mean,
        "ndcg_mean": ndcg_mean,
        "hit1": hit1_mean,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="要评测的 checkpoint 目录（trainer.save_model 输出）",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="训练时用的 HF load_from_disk 数据路径",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="评测使用的数据集 split，例如 test / val",
    )
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--user_win_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")

    # vLLM 相关
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="是否使用远端 vLLM server 做生成",
    )
    parser.add_argument(
        "--vllm_base_url",
        type=str,
        default="http://0.0.0.0:8000",
        help="vLLM server 的 base_url，例如 http://127.0.0.1:8000",
    )
    parser.add_argument("--vllm_temperature", type=float, default=0.0)
    parser.add_argument("--vllm_top_p", type=float, default=1.0)

    # 打印 debug 用
    parser.add_argument(
        "--print_every",
        type=int,
        default=0,
        help="每隔多少个 batch 打印一次模型回答（0 表示不打印）",
    )
    parser.add_argument(
        "--print_examples",
        type=int,
        default=1,
        help="每次打印该 batch 中的多少个样本",
    )

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
