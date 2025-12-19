python rec_eval_pl.py \
  --model_dir /mnt/shared-storage-user/p1-shared/luotianwei/trl-for-rec/rec_checkpoints_pl/checkpoint-3000 \
  --dataset_path /mnt/shared-storage-user/p1-shared/luotianwei/trl-for-rec/data/Musical_Instruments_0_2022-10-2023-10 \
  --split test \
  --batch_size 8 \
  --print_every 50 \
  --print_examples 2 \
  --modes one_forward