export PARTITION=${GROUP}
# 修改成你的路径！！！！！！！！！！！"""
export CFSCTL=/mnt/shared-storage-user/p1-shared/fanyuchen/cfs/bin/cfsctl
export CFG=/mnt/shared-storage-user/p1-shared/fanyuchen/cfs/cfsd.cfg
export TORCH_CUDA_ARCH_LIST="9.0"

cd /mnt/shared-storage-user/p1-shared/luotianwei/trl-for-rec

source /mnt/shared-storage-user/p1-shared/luotianwei/init_script_slime_11_15.sh

export PATH=/mnt/shared-storage-user/fanyuchen/miniconda3/bin:$PATH
source /mnt/shared-storage-user/fanyuchen/miniconda3/etc/profile.d/conda.sh
conda activate trl

CUDA_VISIBLE_DEVICES=0,1,2,3 trl vllm-serve --model /mnt/shared-storage-user/p1-shared/luotianwei/hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1 --tensor-parallel-size 1 --data-parallel-size 4 > /mnt/shared-storage-user/p1-shared/luotianwei/trl-for-rec/logs/vllm.out 2>&1 &

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# 等 20 秒让 server 起起来
sleep 20

# 可选：脚本退出时自动清理 vLLM
trap 'echo "[INFO] killing vLLM ${VLLM_PID}"; kill ${VLLM_PID} 2>/dev/null || true' EXIT

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