#!/bin/bash
#SBATCH --job-name=bandwidth-test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/data/assignment/train/bandwidth_test_%j.log
#SBATCH --time=00:05:00
set -euo pipefail

export PYLIBS_DIR="/mnt/data/assignment/pylibs"
export SCRIPT="/mnt/data/assignment/train/bandwidth_test.py"

NODES=$(scontrol show hostnames "${SLURM_JOB_NODELIST}")
export MASTER_ADDR=$(echo "${NODES}" | head -n1)

echo "Nodes: ${NODES}"
echo "Server (rank 0): ${MASTER_ADDR}"

srun --container-image=nvcr.io/nvidia/pytorch:24.10-py3 \
     --container-mounts=/mnt/data:/mnt/data \
     --export=ALL \
     bash -c '
set -euo pipefail
export RANK="${SLURM_NODEID}"
echo "Node $(hostname), RANK=${RANK}"
python3 "${SCRIPT}"
'
