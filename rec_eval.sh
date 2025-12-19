export CUDA_VISIBLE_DEVICES=4

unset http_proxy
unset https_proxy

python rec_eval.py \
  --model_dir /mnt/shared-storage-user/p1-shared/luotianwei/trl-for-rec/rec_checkpoints_new/checkpoint-2000 \
  --dataset_path /mnt/shared-storage-user/p1-shared/luotianwei/trl-for-rec/data/Musical_Instruments_0_2022-10-2023-10 \
  --split test \
  --K 5 \
  --user_win_size 10 \
  --batch_size 128 \
  --max_prompt_length 4096 \
  --max_new_tokens 4096 \
  --use_vllm \
  --vllm_base_url http://0.0.0.0:8000 \
  --print_every 50 \
  --print_examples 2
