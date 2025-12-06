torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  --master-port=29505 \
  /mnt/shared-storage-user/p1-shared/luotianwei/trl-for-rec/math_grpo.py \
  \
  --model_name_or_path /mnt/shared-storage-user/p1-shared/luotianwei/hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1 \
  --output_dir /mnt/shared-storage-user/p1-shared/luotianwei/outputs/grpo-Qwen2.5-3B-DeepMath \
  \
  --dataset_path /mnt/shared-storage-user/p1-shared/luotianwei/hf_cache/hub/datasets--trl-lib--DeepMath-103K/snapshots/066c50a88d4e14cefc056e31111db2dba17f6c68 \
  --dataset_split train \
  --eval_size 1000 \
  \
  --learning_rate 1e-6 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_generations 8 \
  --max_prompt_length 4096 \
  --max_completion_length 2048 \
  --bf16 True \
  --deepspeed /mnt/shared-storage-user/p1-shared/luotianwei/trl-for-rec/configs/ds_z2_config.json \
  # --use-vllm
  \
#   --use_peft \
#   --lora_target_modules "q_proj" "v_proj"
