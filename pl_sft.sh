unset http_proxy
unset https_proxy
export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --nnodes=1 --nproc_per_node=4 --master-port 29511 \
  pl_sft.py \
  --model_name_or_path /mnt/shared-storage-user/p1-shared/luotianwei/hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1 \
  --data_dir /mnt/shared-storage-user/p1-shared/luotianwei/trl-for-rec/sft_pl_warmup \
  --output_dir /mnt/shared-storage-user/p1-shared/luotianwei/trl-for-rec/sft_ckpt_pl \
  --max_length 4096 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --bf16 \
  --gradient_checkpointing \
  --save_steps 500 \
  --eval_steps 500 \
  --report_to tensorboard \
  --num_train_epochs 5