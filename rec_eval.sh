export CUDA_VISIBLE_DEVICES=4

unset http_proxy
unset https_proxy

python rec_eval.py \
  --model_dir /mnt/shared-storage-user/p1-shared/luotianwei/hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1 \
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
