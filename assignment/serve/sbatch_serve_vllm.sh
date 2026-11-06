#!/bin/bash
#SBATCH --job-name=serve-vllm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/data/assignment/serve/serve_vllm_%j.log
#SBATCH --time=00:30:00
set -euo pipefail

export PYLIBS_VLLM="/mnt/data/assignment/pylibs_vllm"
export ADAPTER_PATH="/mnt/data/assignment/train/lora_adapter_v2_2000rows"
# The container's pre-installed flash_attn .so is compiled against its
# original torch build, but our isolated pylibs_vllm env loads its own
# torch 2.13.0 (self-contained, separate from the training environment's
# torch) - mismatched ABI, undefined symbol on import. Force a backend
# that doesn't need the container's flash_attn at all; flashinfer was
# installed fresh as one of vLLM's own dependencies into this same
# isolated env, so it matches torch 2.13.0 correctly.
export VLLM_ATTENTION_BACKEND="FLASHINFER"
# FlashInfer's sampling kernel JIT-compiles on first use and got stuck in
# what looked like a non-terminating ninja rebuild loop (15+ min, same
# target repeatedly) rather than genuine progress. Force vLLM's simpler
# native (non-FlashInfer) top-k/top-p sampler instead - no JIT compile
# needed, at a small throughput cost we can just measure honestly.
export VLLM_USE_FLASHINFER_SAMPLER=0

srun --container-image=nvcr.io/nvidia/pytorch:24.10-py3 \
     --container-mounts=/mnt/data:/mnt/data \
     --export=ALL \
     bash -c '
set -euo pipefail
export PYTHONPATH="${PYLIBS_VLLM}"
echo "Serving on $(hostname)"
# The container own flash_attn is compiled against its original torch,
# not the self-contained torch our isolated pylibs_vllm env loads -
# undefined symbol on import, and vLLM does not catch this particular
# ImportError. Hide it from this (ephemeral, container-local) process so
# vLLM correctly detects it as unavailable and skips to a different
# code path instead of hitting the broken import.
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
