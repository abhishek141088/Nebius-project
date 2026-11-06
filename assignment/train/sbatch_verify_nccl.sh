#!/bin/bash
#SBATCH --job-name=verify-nccl
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/data/assignment/train/verify_nccl_%j.log
#SBATCH --time=00:10:00
set -euo pipefail

export PYLIBS_DIR="/mnt/data/assignment/pylibs"
export VERIFY_SCRIPT="/mnt/data/assignment/train/verify_nccl.py"
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

NODES=$(scontrol show hostnames "${SLURM_JOB_NODELIST}")
export MASTER_ADDR=$(echo "${NODES}" | head -n1)
export MASTER_PORT=29501

echo "Nodes: ${NODES}"

srun --container-image=nvcr.io/nvidia/pytorch:24.10-py3 \
     --container-mounts=/mnt/data:/mnt/data \
     --export=ALL \
     bash -c '
set -euo pipefail
export PYTHONPATH="${PYLIBS_DIR}:${PYTHONPATH:-}"
echo "Node $(hostname), SLURM_NODEID=${SLURM_NODEID}"
torchrun --nnodes=2 --nproc_per_node=1 \
         --node_rank="${SLURM_NODEID}" \
         --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
         "${VERIFY_SCRIPT}"
'
