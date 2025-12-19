unset http_proxy
unset https_proxy
export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --master-port 29509 \
  pl_grpo.py \
  --model_name_or_path /mnt/shared-storage-user/p1-shared/luotianwei/trl-for-rec/sft_ckpt_pl \
  --output_dir /mnt/shared-storage-user/p1-shared/luotianwei/trl-for-rec/rec_checkpoints_pl \
  --dataset_path /mnt/shared-storage-user/p1-shared/luotianwei/trl-for-rec/data/Musical_Instruments_0_2022-10-2023-10 \
  --dataset_split train \
  --eval_size 1000 \
  --topk 5 \
  --user_win_size 10 \
  --format_weight 0.0 \
  --ndcg_weight 1.0 \
  --learning_rate 1e-6 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_generations 8 \
  --max_prompt_length 4096 \
  --max_completion_length 2048 \
  --save_steps 1000 \
  --bf16 True \
  --report_to tensorboard
