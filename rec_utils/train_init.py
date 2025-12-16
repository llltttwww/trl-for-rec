"""
train_init.py - 推荐排序数据处理模块

功能：将用户历史和K个候选物品整合为一个prompt，让LLM输出排序列表

使用示例见文件末尾的 demo() 函数
"""

from __future__ import annotations
import random
from datetime import datetime
from typing import Any, Dict, List, Tuple
from torch.utils.data import Dataset
import datasets as hfds


# =========================
# Prompt 模板
# =========================

RANKING_PROMPT_TMPL = """You are a recommendation assistant. Based on the user's purchase history, rank the candidate items by how likely the user would want to buy them.

# USER HISTORY
Contains purchase times, purchased items, and ratings (out of 5).
{user_text}

# CANDIDATE ITEMS
{candidates_text}

# INSTRUCTION
You MUST respond with the following format:

<think>
Write your reasoning here.
</think>
<answer>
RANKING: number1,number2,number3,number4,number5
</answer>

Rules:
- The <answer> must contain exactly ONE line starting with "RANKING:".
- The ranking must be a permutation of 1..5 with no repeats.
- Do not output anything outside the <think> and <answer> tags.

Example:
<think>...</think>
<answer>
RANKING: 2, 4, 5, 1, 3
</answer>
"""

# =========================
# 文本构建函数
# =========================

def format_time_ago(delta) -> str:
    """时间差格式化"""
    days, hours = delta.days, delta.seconds // 3600
    minutes = (delta.seconds % 3600) / 60
    if days > 0:
        return f"{days}d {hours + minutes/60:.1f}h ago"
    return f"{hours}h {minutes:.1f}min ago" if hours > 0 else f"{minutes:.1f}min ago"


def build_user_text(sequence: Dict[str, Any], id2title: Dict[int, str], win_size: int = 10) -> str:
    """构建用户历史文本"""
    titles = sequence.get("history_item_title") or [
        id2title.get(int(i), f"Item#{i}") for i in (sequence.get("history_item_id") or [])
    ]
    ratings = sequence.get("history_rating") or []
    
    # 时间处理
    history_ts = sequence.get("history_timestamp")
    ts = sequence.get("timestamp")
    if history_ts and ts:
        base = datetime.fromtimestamp(ts / 1000)
        times = [format_time_ago(base - datetime.fromtimestamp(t / 1000)) for t in history_ts]
    else:
        times = ["t0"] * len(titles)
    
    # 取最近 win_size 条
    start = max(0, len(titles) - win_size)
    lines = [
        f"{times[i]}: [{titles[i]}] ({ratings[i] if i < len(ratings) else ''})"
        for i in range(start, len(titles))
    ]
    return "\n".join(lines)


def build_item_text(item: Dict[str, Any], idx: int) -> str:
    """构建单个候选物品文本（带编号）"""
    title = item.get("title") or item.get("item_title") or "Unknown Item"
    desc = item.get("description", "")
    if isinstance(desc, list):
        desc = " ".join(desc[::-1]) if desc else ""
    return f"[{idx}] {title} | Rating: {item.get('average_rating', 0):.1f} | Buyers: {item.get('rating_number', 0)}\n    Description: {desc[:200]}{'...' if len(desc) > 200 else ''}"


def build_candidates_text(candidates: List[Dict[str, Any]]) -> str:
    """构建所有候选物品的文本"""
    return "\n".join(build_item_text(c, i) for i, c in enumerate(candidates, start=1))


# =========================
# Prompt 构建器
# =========================

class RankingPrompter:
    """排序任务Prompt构建器"""
    
    def __init__(self, tokenizer=None):
        self.tok = tokenizer

    def build_prompt(self, user_text: str, candidates: List[Dict[str, Any]], 
                    apply_chat_template: bool = True) -> str:
        """构建排序prompt"""
        raw = RANKING_PROMPT_TMPL.format(
            user_text=user_text,
            candidates_text=build_candidates_text(candidates)
        )
        if apply_chat_template and self.tok and hasattr(self.tok, "apply_chat_template"):
            return self.tok.apply_chat_template(
                [{"role": "user", "content": raw}],
                tokenize=False, add_generation_prompt=True
            )
        return raw


# =========================
# 数据集
# =========================

class RankingDataset(Dataset):
    """排序任务数据集：每个样本一个prompt，包含所有候选物品"""
    
    def __init__(self, split_ds: hfds.Dataset, item_info: hfds.Dataset,
                 K: int = 20, user_win_size: int = 10, seed: int = 42):
        self.split_ds = split_ds
        self.K = K
        self.user_win_size = user_win_size
        self.base_seed = seed
        
        # 构建物品索引
        self.id2row = {int(r["item_id"]): dict(r) for r in item_info}
        self.id2title = {
            int(r["item_id"]): (r.get("title") or r.get("item_title") or f"Item#{r['item_id']}").strip()
            for r in item_info
        }
        self.valid_ids = [i for i in self.id2row if i != 0]

    def __len__(self) -> int:
        return len(self.split_ds)

    def _sample_candidates(self, pos_id: int) -> Tuple[List[int], int]:
        """采样K个候选（含1个正样本）"""
        pool = [i for i in self.valid_ids if i != pos_id]
        negs = random.sample(pool, self.K - 1)
        pos_idx = random.randint(0, self.K - 1)
        negs.insert(pos_idx, pos_id)
        return negs, pos_idx

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        random.seed(self.base_seed + idx)
        ex = self.split_ds[idx]
        pos_id = int(ex["item_id"])
        
        user_text = build_user_text(ex, self.id2title, self.user_win_size)
        cand_ids, pos_idx = self._sample_candidates(pos_id)
        candidates = [self.id2row[i] for i in cand_ids]
        
        return {
            "user_text": user_text,
            "candidates": candidates,
            "candidate_ids": cand_ids,
            "target_idx": pos_idx,  # 正样本在候选中的位置
            "target_item_id": pos_id,
            "user_id": ex.get("user_id"),
        }


# =========================
# 数据加载
# =========================

def load_dataset(dataset_path: str, split: str = "train") -> Tuple[hfds.Dataset, hfds.Dataset]:
    """加载数据集，返回 (split_ds, item_info)"""
    ds = hfds.load_from_disk(dataset_path)
    return ds[split], ds["item_info"]


def create_dataset(dataset_path: str, split: str = "train", 
                   K: int = 20, user_win_size: int = 10, seed: int = 42) -> RankingDataset:
    """创建排序数据集"""
    split_ds, item_info = load_dataset(dataset_path, split)
    return RankingDataset(split_ds, item_info, K, user_win_size, seed)


# =========================
# 使用示例
# =========================

def demo(dataset_path: str = None, model_name: str = None):
    """演示使用方法"""
    print("=" * 70)
    print("排序推荐数据处理模块")
    print("=" * 70)
    
    # 示例1：模拟数据
    print("\n【示例1】使用模拟数据构建Prompt")
    print("-" * 50)
    
    user_text = """2d 3.5h ago: [Wireless Gaming Mouse] (5)
1d 12.0h ago: [Mechanical Keyboard RGB] (4)
5h 30.0min ago: [USB-C Hub Multiport] (5)"""
    
    candidates = [
        {"title": "Gaming Headset", "average_rating": 4.5, "rating_number": 1234, "description": "High-quality headset"},
        {"title": "Mouse Pad XL", "average_rating": 4.2, "rating_number": 567, "description": "Large gaming mouse pad"},
        {"title": "Webcam 1080P", "average_rating": 4.0, "rating_number": 890, "description": "HD webcam for streaming"},
    ]
    
    prompter = RankingPrompter()
    prompt = prompter.build_prompt(user_text, candidates, apply_chat_template=False)
    print(prompt)
    
    # 示例2：真实数据
    if dataset_path:
        print("\n\n【示例2】使用真实数据集")
        print("-" * 50)
        
        ds = create_dataset(dataset_path, split="train", K=5, user_win_size=5)
        sample = ds[0]
        
        print(f"数据集大小: {len(ds)}")
        print(f"user_id: {sample['user_id']}")
        print(f"target_idx: {sample['target_idx']} (正样本位置)")
        print(f"candidate_ids: {sample['candidate_ids']}")
        
        prompter = RankingPrompter()
        if model_name:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            prompter = RankingPrompter(tok)
        
        prompt = prompter.build_prompt(sample["user_text"], sample["candidates"])
        print(f"\n生成的Prompt:\n{prompt}")
    
    print("\n" + "=" * 70)
    print("""
主要接口:
  - create_dataset(path, split, K, win_size, seed) -> RankingDataset
  - RankingPrompter(tokenizer).build_prompt(user_text, candidates) -> str
  - dataset[idx] -> {user_text, candidates, candidate_ids, target_idx, ...}
""")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/mnt/shared-storage-user/p1-shared/luotianwei/PACPO/data/Musical_Instruments_0_2022-10-2023-10")
    parser.add_argument("--model_name", type=str, default="/mnt/shared-storage-user/p1-shared/luotianwei/hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1/")
    args = parser.parse_args()
    demo(args.dataset_path, args.model_name)
