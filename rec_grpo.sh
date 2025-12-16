export VLLM_ALLOW_INSECURE_SERIALIZATION=1

export CUDA_VISIBLE_DEVICES=4,5,6,7

unset http_proxy
unset https_proxy

torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --master-port 29507 \
  rec_grpo.py \
  \
  --model_name_or_path /mnt/shared-storage-user/p1-shared/luotianwei/hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1 \
  --output_dir /mnt/shared-storage-user/p1-shared/luotianwei/trl-for-rec/rec_checkpoints_new \
  \
  --dataset_path /mnt/shared-storage-user/p1-shared/luotianwei/trl-for-rec/data/Musical_Instruments_0_2022-10-2023-10 \
  --dataset_split train \
  --eval_size 1000 \
  --topk 5 \
  --user_win_size 10 \
  \
  --format_weight 0.3 \
  --ndcg_weight 0.7 \
  \
  --learning_rate 1e-6 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_generations 8 \
  --max_prompt_length 4096 \
  --max_completion_length 2048 \
  --bf16 True \
  --use-vllm \
  --report_to tensorboard