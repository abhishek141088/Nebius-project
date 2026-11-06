#!/bin/bash
#SBATCH --job-name=serve-model
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/data/assignment/serve/serve_%j.log
#SBATCH --time=00:15:00
set -euo pipefail

export PYLIBS_DIR="/mnt/data/assignment/pylibs"

srun --container-image=nvcr.io/nvidia/pytorch:24.10-py3 \
     --container-mounts=/mnt/data:/mnt/data \
     --export=ALL \
     bash -c '
set -euo pipefail
export PYTHONPATH="${PYLIBS_DIR}:${PYTHONPATH:-}"
echo "Serving on $(hostname)"
python3 /mnt/data/assignment/serve/serve_model.py
'
