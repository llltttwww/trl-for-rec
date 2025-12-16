# rec_rewards.py
import math
import re
from typing import List, Dict, Optional

K_FIXED = 5

ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
RANKING_PATTERN = re.compile(r"RANKING\s*:\s*([0-9,\s]+)", re.IGNORECASE)


def _extract_contents(completions: List) -> List[str]:
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

        first = msgs[0]
        if isinstance(first, dict):
            contents.append(first.get("content", "") or "")
        elif isinstance(first, str):
            contents.append(first)
        else:
            contents.append(str(first))

    return contents


def _extract_answer_block(text: str) -> str:
    m = ANSWER_BLOCK_RE.search(text)
    return m.group(1) if m else ""  # ✅ 没有 <answer> 直接视为失败


def _parse_ranking_from_answer(text: str) -> Optional[List[int]]:
    ans = _extract_answer_block(text)
    if not ans:
        return None

    # 可选：如果你想更严格，确保 <answer> 内只有一个 RANKING:
    # if len(RANKING_PATTERN.findall(ans)) != 1:
    #     return None

    m = RANKING_PATTERN.search(ans)
    if not m:
        return None

    raw = m.group(1)
    try:
        nums = [int(x) for x in raw.replace(" ", "").split(",") if x]
        return nums if nums else None
    except ValueError:
        return None


def _normalize_ranking_1_to_0_based(ranking_1_based: List[int], K: int) -> Optional[List[int]]:
    ranking_0 = [x - 1 for x in ranking_1_based]
    if len(ranking_0) < K:
        return None
    ranking_0 = ranking_0[:K]
    if sorted(ranking_0) != list(range(K)):
        return None
    return ranking_0


def _ndcg_at_k_single(pos_idx: int, ranking_0: List[int]) -> float:
    try:
        rank_pos = ranking_0.index(pos_idx)
    except ValueError:
        return 0.0
    return 1.0 / math.log2(rank_pos + 2)


# ============================================================
# 1) format reward: 0/1
# ============================================================
def format_reward_rec(
    completions: List[List[Dict[str, str]]],
    solution: List[str],
    **kwargs,
) -> List[float]:
    contents = _extract_contents(completions)
    rewards: List[float] = []

    for text in contents:
        ranking_1 = _parse_ranking_from_answer(text)
        if ranking_1 is None:
            rewards.append(0.0)
            continue

        ranking_0 = _normalize_ranking_1_to_0_based(ranking_1, K_FIXED)
        rewards.append(1.0 if ranking_0 is not None else 0.0)

    return rewards


# ============================================================
# 2) ndcg reward: 永远 float，失败 -> 0.0
# ============================================================
def ndcg_reward_rec(
    completions: List[List[Dict[str, str]]],
    solution: List[str],
    **kwargs,
) -> List[float]:
    contents = _extract_contents(completions)
    rewards: List[float] = []

    for text, sol in zip(contents, solution, strict=True):
        # solution 解析失败 -> 0
        try:
            pos_idx = int(sol)
        except (TypeError, ValueError):
            rewards.append(0.0)
            continue

        # target 越界 -> 0
        if pos_idx < 0 or pos_idx >= K_FIXED:
            rewards.append(0.0)
            continue

        ranking_1 = _parse_ranking_from_answer(text)
        if ranking_1 is None:
            rewards.append(0.0)
            continue

        ranking_0 = _normalize_ranking_1_to_0_based(ranking_1, K_FIXED)
        if ranking_0 is None:
            rewards.append(0.0)
            continue

        rewards.append(float(_ndcg_at_k_single(pos_idx, ranking_0)))

    return rewards
