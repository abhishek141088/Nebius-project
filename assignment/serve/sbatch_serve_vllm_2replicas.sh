#!/bin/bash
#SBATCH --job-name=serve-vllm-2x
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/data/assignment/serve/serve_vllm_2x_%j.log
#SBATCH --time=00:30:00
set -euo pipefail

export PYLIBS_VLLM="/mnt/data/assignment/pylibs_vllm"
export ADAPTER_PATH="/mnt/data/assignment/train/lora_adapter_v2_2000rows"
export VLLM_ATTENTION_BACKEND="FLASHINFER"
export VLLM_USE_FLASHINFER_SAMPLER=0

# 2 replicas, NOT tensor parallelism: --nodes=2 + --ntasks-per-node=1 means
# srun launches one independent task per node, each running its own
# self-contained vLLM instance on its own GPU - no cross-node
# communication at all (unlike tensor parallelism, which would need a
# network round-trip per layer per token, exactly what this no-IB setup
# can't do cheaply). Each replica serves the same model+adapter
# independently; a client load-balances across both (e.g. aiperf's
# --url http://worker-0:8001 --url http://worker-1:8001).
srun --container-image=nvcr.io/nvidia/pytorch:24.10-py3 \
     --container-mounts=/mnt/data:/mnt/data \
     --export=ALL \
     bash -c '
set -euo pipefail
export PYTHONPATH="${PYLIBS_VLLM}"
echo "Serving replica on $(hostname)"
mv /usr/local/lib/python3.10/dist-packages/flash_attn* /tmp/ 2>/dev/null || true
rm -rf /root/.cache/flashinfer 2>/dev/null || true
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --enable-lora \
  --lora-modules finetuned="${ADAPTER_PATH}" \
  --max-lora-rank 16 \
  --enforce-eager \
  --port 8001
'
