# rec_rewards.py
import math
import re
from typing import List, Dict, Optional


RANKING_PATTERN = re.compile(r"RANKING\s*:\s*([0-9,\s]+)", re.IGNORECASE)


def _extract_contents(completions: List) -> List[str]:
    """把各种 completion 结构抽成纯文本 content。

    支持：
    - List[List[{"role":..., "content":...}]]  （训练时原生格式）
    - List[{"role":..., "content":...}]
    - List[str]
    - List[List[str]]
    """
    contents: List[str] = []
    for msgs in completions:
        # 空的情况
        if not msgs:
            contents.append("")
            continue

        # 1) 直接就是一个字符串：["...","..."] 或 completions=["..."]
        if isinstance(msgs, str):
            contents.append(msgs)
            continue

        # 2) 直接就是一个 dict：{"role":..., "content":...}
        if isinstance(msgs, dict):
            contents.append(msgs.get("content", "") or "")
            continue

        # 3) 其他情况，假定是 list[...]，取第一个元素
        first = msgs[0]

        if isinstance(first, dict):
            contents.append(first.get("content", "") or "")
        elif isinstance(first, str):
            contents.append(first)
        else:
            # 实在不知道是啥，就直接转成字符串
            contents.append(str(first))

    return contents


def _parse_ranking(text: str) -> Optional[List[int]]:
    """
    从模型输出中解析出 ranking 序列。
    期望格式类似：RANKING: 2, 5, 1, 4, 3
    """
    m = RANKING_PATTERN.search(text)
    if not m:
        return None
    raw = m.group(1)
    try:
        nums = [int(x) for x in raw.replace(" ", "").split(",") if x]
        if not nums:
            return None
        return nums
    except ValueError:
        return None


def _normalize_ranking_1_to_0_based(
    ranking_1_based: List[int],
    K: int,
) -> Optional[List[int]]:
    """
    把 [1..K] 的排序转换成 [0..K-1]，并校验是否是一个合法排列。
    """
    ranking_0 = [x - 1 for x in ranking_1_based]
    if len(ranking_0) < K:
        return None
    ranking_0 = ranking_0[:K]

    if sorted(ranking_0) != list(range(K)):
        # 不是 0..K-1 的一个排列，认为无效
        return None
    return ranking_0


def _ndcg_at_k_single(pos_idx: int, ranking_0: List[int]) -> float:
    """
    单个样本的 NDCG@K。
    这里只假设只有一个正样本（位置 pos_idx），其它都是 0 相关性。

    DCG = 1 / log2(rank_pos + 2)
    IDCG = 1 / log2(1 + 1) = 1
    所以 NDCG = DCG
    """
    K = len(ranking_0)
    if pos_idx < 0 or pos_idx >= K:
        return 0.0

    try:
        rank_pos = ranking_0.index(pos_idx)  # 0-based 名次
    except ValueError:
        # 正样本不在预测的前 K 里
        return 0.0

    # rank_pos=0 → DCG=1/log2(2)=1，rank_pos 越大越小
    return 1.0 / math.log2(rank_pos + 2)


# ============================================================
# 1. 格式奖励：对齐 accuracy_reward 的接口
# ============================================================

def format_reward_rec(
    completions: List[List[Dict[str, str]]],
    solution: List[str],          # 仅为保持接口一致，这里不用
    **kwargs,
) -> List[float]:
    """
    格式奖励：
    - 能解析出 RANKING 行，且是 [1..K] 的一个合法排列 → 1.0
    - 否则 → 0.0
    """
    contents = _extract_contents(completions)
    K = int(kwargs.get("K", 5))  # 可以固定 5，也可以从 kwargs 里传

    rewards: List[float] = []
    for text in contents:
        ranking_1 = _parse_ranking(text)
        if ranking_1 is None:
            rewards.append(0.0)
            continue

        norm = _normalize_ranking_1_to_0_based(ranking_1, K)
        rewards.append(1.0 if norm is not None else 0.0)

    return rewards


# ============================================================
# 2. NDCG 奖励：第二个参数用 solution（存 target_idx）
# ============================================================

def ndcg_reward_rec(
    completions: List[List[Dict[str, str]]],
    solution: List[str],   # dataset["solution"]，每个元素是正样本 index 的字符串，比如 "2"
    **kwargs,
) -> List[Optional[float]]:
    """
    NDCG 奖励：
    - 假设 dataset 里 `solution[i]` 是正样本在候选中的位置（0-based），以字符串形式存，比如 "2"
    - 模型输出 RANKING: 2, 5, 1, 4, 3 （1-based）
    - 先转成 0-based 排列，再用单点 NDCG 计算
    - 解析失败就返回 None（该样本会被跳过）
    """
    contents = _extract_contents(completions)
    K = int(kwargs.get("K", 5))

    rewards: List[Optional[float]] = []
    for text, sol in zip(contents, solution, strict=True):
        # 从 solution 里解析出 pos_idx（0-based）
        try:
            pos_idx = int(sol)
        except (TypeError, ValueError):
            rewards.append(None)
            continue

        ranking_1 = _parse_ranking(text)
        if ranking_1 is None:
            rewards.append(None)
            continue

        ranking_0 = _normalize_ranking_1_to_0_based(ranking_1, K)
        if ranking_0 is None:
            rewards.append(None)
            continue

        rewards.append(float(_ndcg_at_k_single(pos_idx, ranking_0)))

    return rewards


# ============================================================
# 3. 合成奖励：格式 + NDCG
# ============================================================

def combined_reward_rec(
    completions: List[List[Dict[str, str]]],
    solution: List[str],
    **kwargs,
) -> List[Optional[float]]:
    """
    合成奖励：
      - 先算格式奖励 f_i ∈ {0,1}
      - 再算 NDCG 奖励 n_i ∈ [0,1] 或 None
      - 默认组合方式：reward_i = f_i * n_i
        - 格式不对 → f_i=0 → reward_i=0
        - NDCG 无法解析 → None（让 Trainer 跳过）
    你也可以改成加权和，比如 0.1 * f_i + 0.9 * n_i。
    """
    fmt = format_reward_rec(completions, solution, **kwargs)       # list[float]
    ndcg = ndcg_reward_rec(completions, solution, **kwargs)        # list[Optional[float]]

    rewards: List[Optional[float]] = []
    for f, n in zip(fmt, ndcg, strict=True):
        if n is None:
            rewards.append(None)   # 完全不可用，跳过
        else:
            # 简单乘法：格式正确时才给到 NDCG 奖励
            rewards.append(f * n)

            # 如果你想要“格式 + 排序”同时都有信号，可以改成：
            # rewards.append(0.1 * f + 0.9 * n)

    return rewards
