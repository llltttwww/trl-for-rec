from datasets import Dataset as HFDataset
from transformers import AutoTokenizer

from .train_init import create_dataset, RankingPrompter  # 你自己那个文件


def build_rec_hf_dataset(
    dataset_path: str,
    model_name_or_path: str,
    split: str = "train",
    K: int = 5,
    user_win_size: int = 10,
    seed: int = 42,
) -> HFDataset:
    """
    把 RankingDataset 转成 HF Dataset，列至少包含：
      - prompt: list[{"role": "user", "content": "..."}]
      - solution: str（你后面用来算 reward 的 GT）
      - candidate_ids / target_idx 等你自己的字段
    """
    # 1) 创建你定义的 RankingDataset
    ranking_ds = create_dataset(
        dataset_path=dataset_path,
        split=split,
        K=K,
        user_win_size=user_win_size,
        seed=seed,
    )

    # 2) Tokenizer & Prompter
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    prompter = RankingPrompter(tokenizer)

    all_rows = []
    for i in range(len(ranking_ds)):
        ex = ranking_ds[i]

        # 注意：这里 **不要** apply_chat_template，保持 content 是纯文本，
        # 让 GRPOTrainer / tokenizer 自己去套 chat template（如果需要）。
        prompt_text = prompter.build_prompt(
            user_text=ex["user_text"],
            candidates=ex["candidates"],
            apply_chat_template=False,
            pl_grpo=True,
        )

        # chat 格式的 prompt（推荐用 list）
        prompt_msg = [{"role": "user", "content": prompt_text}]

        # 这里先放一个占位的 solution，你后面会用来做 format_reward + ndcg
        # 比如可以约定：solution 是正样本在候选中的 index，或者是一个 NDCG 的“理想排序串”。
        # 举例：只训练把正样本排第一：
        solution = str(ex["target_idx"])  # 或者你想要的任何可 parse 的格式

        all_rows.append(
            {
                "prompt": prompt_msg,          # ✅ 现在是 {"role":..., "content":...} 结构（外面包一层 list）
                "solution": solution,          # ✅ 给 reward 函数用
                "candidate_ids": ex["candidate_ids"],
                "target_idx": ex["target_idx"],
                "user_id": ex.get("user_id"),
                "target_item_id": ex["target_item_id"],
            }
        )

    return HFDataset.from_list(all_rows)
