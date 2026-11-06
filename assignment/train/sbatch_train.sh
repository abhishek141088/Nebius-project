#!/bin/bash
#SBATCH --job-name=lora-sft
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/data/assignment/train/sbatch_train_%j.log
#SBATCH --time=02:00:00
set -euo pipefail

export PYLIBS_DIR="/mnt/data/assignment/pylibs"
export TRAIN_SCRIPT="/mnt/data/assignment/train/train_lora_sft.py"
export MONITOR_DIR="/mnt/data/assignment/monitoring"
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${MONITOR_DIR}" /mnt/data/assignment/train/lora_adapter /mnt/data/assignment/train/runs

# Resolve the rendezvous address from Slurm's own node list - the
# standard pattern for multi-node torchrun under Slurm.
NODES=$(scontrol show hostnames "${SLURM_JOB_NODELIST}")
export MASTER_ADDR=$(echo "${NODES}" | head -n1)
export MASTER_PORT=29500

echo "Nodes: ${NODES}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

# Variables are exported above so they're inherited by the remote shell
# through srun - avoids interpolating them into the --container-image
# bash -c string, which gets messy and error-prone with nested quoting.
srun --container-image=nvcr.io/nvidia/pytorch:24.10-py3 \
     --container-mounts=/mnt/data:/mnt/data \
     --export=ALL \
     bash -c '
set -euo pipefail
export PYTHONPATH="${PYLIBS_DIR}:${PYTHONPATH:-}"

LOGFILE="${MONITOR_DIR}/gpu_util_$(hostname).log"
nvidia-smi dmon -s u -o DT -f "${LOGFILE}" &
DMON_PID=$!

echo "Node $(hostname), SLURM_NODEID=${SLURM_NODEID}, starting torchrun"
torchrun --nnodes=2 --nproc_per_node=1 \
         --node_rank="${SLURM_NODEID}" \
         --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
         "${TRAIN_SCRIPT}"

kill "${DMON_PID}" || true
'
