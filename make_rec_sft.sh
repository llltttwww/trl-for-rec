python make_rec_sft.py \
  --dataset_path /mnt/shared-storage-user/p1-shared/luotianwei/trl-for-rec/data/Musical_Instruments_0_2022-10-2023-10 \
  --model_name_or_path /mnt/shared-storage-user/p1-shared/luotianwei/hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1 \
  --output_dir /mnt/shared-storage-user/p1-shared/luotianwei/trl-for-rec/sft_pl_warmup \
  --num_samples 2000 \
  --eval_size 300
