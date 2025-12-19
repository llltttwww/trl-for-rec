#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList

from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_peft_config,
)

# 你已有的数据构造组件
from rec_utils.train_init import create_dataset, RankingPrompter


os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

K_FIXED = 5


# =========================
# LogitsProcessor: 只在生成段遇到 <answer> 后开始无放回约束
# =========================
class PermutationOnlyProcessor(LogitsProcessor):
    """
    强制生成段只输出 K 个数字 1..K，且无放回（permutation）。
    - step 0..K-1：只允许剩余的数字 token
    - step >= K：全部禁掉（靠 max_new_tokens=K 停止）
    """

    def __init__(self, tok, K: int = 5):
        self.tok = tok
        self.K = K

        # 关键：必须是“裸数字”在 vocab 里是单 token，否则你会约束失败
        self.item_ids = []
        for i in range(1, K + 1):
            ids = tok.encode(str(i), add_special_tokens=False)
            assert len(ids) == 1, f"Digit '{i}' is not a single token: {tok.convert_ids_to_tokens(ids)}"
            self.item_ids.append(ids[0])

        self.item_set = set(self.item_ids)
        self._prompt_lens: Optional[List[int]] = None

    def _mask_except_keep_logits(self, scores_row: torch.FloatTensor, allowed_ids: List[int]):
        masked = torch.full_like(scores_row, float("-inf"))
        masked[allowed_ids] = scores_row[allowed_ids]
        return masked

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # bs = input_ids.size(0)
        # if self._prompt_lens is None or len(self._prompt_lens) != bs:
        #     self._prompt_lens = [input_ids.size(1)] * bs  # generate 初始长度（含 padding）

        # for b in range(bs):
        #     prompt_len = self._prompt_lens[b]
        #     gen = input_ids[b, prompt_len:].tolist()
        #     step = len(gen)

        #     if step < self.K:
        #         used = set([t for t in gen if t in self.item_set])
        #         remaining = [t for t in self.item_ids if t not in used]
        #         scores[b] = self._mask_except_keep_logits(scores[b], remaining)
        #     else:
        #         scores[b].fill_(float("-inf"))

        return scores


def _extract_contents(completions: List) -> List[str]:
    """
    TRL/GRPO 的 completions 可能是：
      - str
      - dict {content: ...}
      - List[dict] 或 List[str]
    这里统一提取出文本 content。
    """
    contents: List[str] = []
    for msgs in completions:
        if not msgs:
            contents.append("")
            continue
        if isinstance(msgs, str):
            contents.append(msgs)
            continue
        if isinstance(msgs, dict):
            contents.append(msgs.get("content", "") or "")
            continue
        # list-like
        first = msgs[0]
        if isinstance(first, dict):
            contents.append(first.get("content", "") or "")
        elif isinstance(first, str):
            contents.append(first)
        else:
            contents.append(str(first))
    return contents

DIGIT5_RE = re.compile(r"^[1-5]{5}$")

def _parse_perm(text: str, K: int = K_FIXED) -> Optional[List[int]]:
    """
    解析 5 位数字 permutation，例如 '41532'
    返回 0-based item id list: [3,0,4,2,1]
    """
    s = (text or "").strip()
    s = s[:K]  # 只取前 5 个字符，防止模型多吐
    if len(s) != K:
        return None
    if not DIGIT5_RE.fullmatch(s):
        return None
    if len(set(s)) != K:
        return None
    return [int(ch) - 1 for ch in s]

def format_reward_pl(
    completions: List[List[Dict[str, str]]],
    solution: List[str],
    **kwargs,
) -> List[float]:
    """
    仅做格式检查：能解析出合法 permutation -> 1，否则 0。
    有了强约束 logits_processor 后，这个一般可以不用（权重=0 或直接不传）。
    """
    contents = _extract_contents(completions)
    return [1.0 if _parse_perm(t) is not None else 0.0 for t in contents]

def ndcg_reward_pl(
    completions: List[List[Dict[str, str]]],
    solution: List[str],
    **kwargs,
) -> List[float]:
    """
    你的旧逻辑：solution 是正样本在候选中的位置 pos_idx（0..K-1），
    perm 给出预测排序（perm 的 index 表示 rank）。
    reward = 1/log2(rank+2) （相当于 DCG@K 的单点版本）
    """
    contents = _extract_contents(completions)
    out: List[float] = []

    for t, sol in zip(contents, solution, strict=True):
        try:
            pos_idx = int(sol)
        except Exception:
            out.append(0.0)
            continue
        if pos_idx < 0 or pos_idx >= K_FIXED:
            out.append(0.0)
            continue

        perm = _parse_perm(t, K=K_FIXED)
        if perm is None:
            out.append(0.0)
            continue

        # perm 里存的是 item 的 id（0..K-1），它在 perm 里的位置就是 rank
        rank_pos = perm.index(pos_idx)
        out.append(1.0 / math.log2(rank_pos + 2))

    return out


# =========================
# 自定义脚本参数
# =========================
@dataclass
class RecScriptArguments(ScriptArguments):
    dataset_path: str = field(
        default="/mnt/shared-storage-user/p1-shared/luotianwei/trl-for-rec/data/Musical_Instruments_0_2022-10-2023-10",
        metadata={"help": "HF load_from_disk 的推荐数据路径"},
    )
    dataset_split: str = field(default="train", metadata={"help": "train / val 等"})
    eval_size: int = field(default=1000, metadata={"help": "从 train 中切多少条做 eval；<=0 表示不做 eval"})
    topk: int = field(default=5, metadata={"help": "每个样本的候选 item 数（K），固定 5"})
    user_win_size: int = field(default=10, metadata={"help": "用户历史窗口大小"})
    format_weight: float = field(default=0.0, metadata={"help": "格式奖励权重（有约束解码后可设 0）"})
    ndcg_weight: float = field(default=1.0, metadata={"help": "NDCG 奖励权重"})


def build_rec_hf_dataset_pl(
    dataset_path: str,
    tokenizer,
    split: str = "train",
    K: int = 5,
    user_win_size: int = 10,
    seed: int = 42,
):
    """
    复用 create_dataset + RankingPrompter；
    prompt 用 pl_grpo=True（候选 <I#> 标号，answer 输出 <I#> permutation）。
    """
    assert K == 5, f"PL-GRPO version fixes K=5, got K={K}"

    ranking_ds = create_dataset(
        dataset_path=dataset_path,
        split=split,
        K=K,
        user_win_size=user_win_size,
        seed=seed,
    )

    prompter = RankingPrompter(tokenizer)

    all_rows = []
    for i in range(len(ranking_ds)):
        ex = ranking_ds[i]

        prompt_text = prompter.build_prompt(
            user_text=ex["user_text"],
            candidates=ex["candidates"],
            apply_chat_template=False,
            pl_grpo=True,  # ✅ 关键：PL prompt
        )
        prompt_msg = [{"role": "user", "content": prompt_text}]
        solution = str(ex["target_idx"])  # 0-based 正样本位置

        all_rows.append(
            {
                "prompt": prompt_msg,
                "solution": solution,
                "candidate_ids": ex["candidate_ids"],
                "target_idx": ex["target_idx"],
                "user_id": ex.get("user_id"),
                "target_item_id": ex["target_item_id"],
            }
        )

    from datasets import Dataset as HFDataset
    return HFDataset.from_list(all_rows)


def main():
    script_args, training_args, model_args = TrlParser(
        (RecScriptArguments, GRPOConfig, ModelConfig)
    ).parse_args_and_config()

    assert script_args.topk == 5, f"PL-GRPO fixes topk=5, got topk={script_args.topk}"

    # -------- tokenizer：先建、先加 special，再交给 GRPOTrainer --------
    tok = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)
    tok.padding_side = "left"
    tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # -------- dataset --------
    full_train_ds = build_rec_hf_dataset_pl(
        dataset_path=script_args.dataset_path,
        tokenizer=tok,
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

    # Reward 权重
    training_args.reward_weights = [1]

    # -------- model_init_kwargs：device_map 不能是 auto（尤其 deepspeed）--------
    if model_args.dtype in ["auto", None]:
        dtype = None
    else:
        dtype = getattr(torch, model_args.dtype)

    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        torch_dtype=dtype,
        device_map=None,  # ✅ 关键：不能是 'auto'
    )

    # 如果你要用约束 logits_processor，vLLM / paged 这两条路径默认吃不到
    if getattr(training_args, "use_vllm", False):
        raise ValueError("extra_logits_processor 目前只在 transformers.generate 路径生效，请设置 use_vllm=False。")
    if getattr(training_args, "use_transformers_paged", False):
        raise ValueError("extra_logits_processor 目前未接入 paged generate_batch 路径，请设置 use_transformers_paged=False。")

    # -------- trainer --------
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[ndcg_reward_pl],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
        processing_class=tok,  # ✅ 强制用同一个 tokenizer
    )

    # resize embeddings（policy + ref）
    trainer.model.resize_token_embeddings(len(tok))
    if getattr(trainer, "ref_model", None) is not None:
        trainer.ref_model.resize_token_embeddings(len(tok))

    # -------- generation config：只改 trainer.generation_config，别再写 training_args.generation_kwargs --------
    trainer.generation_config.do_sample = True
    trainer.generation_config.max_new_tokens = 5
    trainer.generation_config.eos_token_id = None
    trainer.generation_config.pad_token_id = tok.pad_token_id

    # -------- 注入“每次生成都 new 一个”的 logits processor（避免跨 batch 复用 prompt_len）--------
    # 依赖你修改后的 GRPOTrainer：在 generate 前如果 extra_logits_processor 可调用，则 extra_lp = extra_lp()
    trainer.extra_logits_processor = lambda: LogitsProcessorList(
        [PermutationOnlyProcessor(tok, K=5)]
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name or "Rec-PL-GRPO")

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
    base_port = int(os.environ.get("DEBUGPY_PORT", "5678"))
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
    # maybe_attach_debugger()
    main()
