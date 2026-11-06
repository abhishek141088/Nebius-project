#!/bin/bash
# One-time environment verification/setup for the fine-tuning assignment.
#
# Installs the training-side Python packages into a shared location on
# /mnt/data (not into the container's own ephemeral filesystem, which
# disappears when the job ends) so later jobs don't need to reinstall them.
#
# Run this SCRIPT ITSELF via srun with --container-image, e.g.:
#   srun -N1 --gres=gpu:1 --container-image=nvcr.io/nvidia/pytorch:24.10-py3 \
#        --container-mounts=/mnt/data:/mnt/data bash /mnt/data/.../setup_env.sh
# Do NOT also call srun from inside this script - it already runs inside
# the container via the outer srun's --container-image, so an inner srun
# would try to re-acquire the same already-allocated GPU from within its
# own job step and deadlock.
set -euo pipefail

PYLIBS_DIR="/mnt/data/assignment/pylibs"

mkdir -p "${PYLIBS_DIR}"

# Two compounding problems, both now understood and both handled below:
#
# 1. `pip install --target=...` resolves dependencies as if starting from
#    a totally empty environment - it does NOT look at what's already
#    installed in the container's own site-packages (that's how --target
#    is documented to behave: self-contained trees, not layering on top
#    of an existing environment). So even with nothing pulling in a new
#    torch version on purpose, pip still tries to fetch its OWN torch
#    into the target dir, which then mismatches the container's
#    already-compiled torchvision. --no-deps is the only way to stop
#    pip from doing this at all.
# 2. Separately, unpinned/latest `transformers` (5.x) also has its own
#    RUNTIME check (independent of pip) that disables torch support if
#    the installed torch is older than 2.6 - so even sidestepping
#    problem 1, a too-new transformers would still reject the
#    container's 2.5.0a0 torch at import time.
#
# Fix: pin the 5 top-level packages to a mutually compatible, same-era
# release line (~Nov 2024, well before transformers' torch>=2.6 floor),
# and install everything (top-level + their actual transitive deps,
# determined by actually letting pip resolve this combination once)
# with --no-deps, so torch itself is never touched either way.
# huggingface-hub and tokenizers are pinned too (not left latest, unlike
# the other small deps) because --no-deps means nothing constrains them
# to a transformers-4.46.3-compatible range - left unpinned, pip grabs
# the absolute latest huggingface-hub, which uses a different internal
# HTTP stack (needs httpcore2/httpx2) than the requests/aiohttp-based one
# transformers 4.46.3 actually expects. These exact versions are the ones
# pip's own resolver picked for this exact stack in an earlier full
# (non --no-deps) resolution, so they're a known-working pair.
pip install --target="${PYLIBS_DIR}" --no-cache-dir --no-deps \
  "transformers==4.46.3" "accelerate==1.1.1" "peft==0.13.2" \
  "trl==0.11.4" "datasets==3.1.0" \
  "huggingface-hub==0.36.2" "tokenizers==0.20.3" \
  MarkupSafe aiohappyeyeballs aiohttp aiosignal async-timeout attrs \
  certifi charset_normalizer dill docstring-parser filelock frozenlist \
  fsspec hf-xet idna jinja2 multidict multiprocess \
  packaging pandas propcache psutil pyarrow python-dateutil pytz pyyaml \
  regex requests safetensors six tqdm typeguard tyro tzdata \
  urllib3 xxhash yarl

export PYTHONPATH="${PYLIBS_DIR}:${PYTHONPATH:-}"
python3 -c "
import torch, transformers, peft, trl
print('torch:', torch.__version__, '| cuda available:', torch.cuda.is_available())
print('transformers:', transformers.__version__)
print('peft:', peft.__version__)
print('trl:', trl.__version__)
"
