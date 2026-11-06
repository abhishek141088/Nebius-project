# Task 1: Distributed fine-tuning through Soperator

- **Git commit used**: `5c2c70b` (assignment code as currently committed) — but note the
  Configuration Mismatch section below: the *scripts* committed at that hash reflect a
  *later* job (73, full ~50k-row dataset, 1 epoch), not job 55. Job 55's actual
  configuration is reconstructed separately in this folder (see below).
- **Soperator release**: v4.1.17 (the tag this whole sandbox deployment is built from)
- **Slurm job ID**: 55
- **Worker nodes**: `worker-0`, `worker-1`
- **GPUs**: 2 × H100, one per node (confirmed: `AllocTRES` shows `gres/gpu=2`, `NNodes=2`)
- **Distributed launch**: two nodes, one `torchrun` process per node (`--nnodes=2
  --nproc_per_node=1`), launched via a single `srun` job step within one `sbatch`
  allocation — see `../../train/sbatch_train.sh` (unchanged since job 55; only the
  training script's own hyperparameters/dataset differed at that time, see below)
- **Network**: InfiniBand disabled (`NCCL_IB_DISABLE=1`); NCCL used plain Ethernet
  sockets between the two pods — independently verified earlier in this session via a
  dedicated NCCL all-reduce test with `NCCL_DEBUG=INFO`, confirming genuine cross-node
  communication over `eth0` (not a hidden InfiniBand path)
- **Base model and revision**: `Qwen/Qwen2.5-7B-Instruct` — revision not pinned (loaded
  via plain repo name, no explicit commit hash); see Limitations
- **Dataset and revision**: `bigcode/self-oss-instruct-sc2-exec-filter-50k` — revision
  not pinned, same caveat; subsampled with `random.seed(42)` after filtering to
  instruction+response pairs ≤6000 chars combined (see
  `historical_prepare_dataset_job55.py`)
- **Training examples**: 2000 (subsampled from the filtered dataset, not the full
  ~50k rows)
- **Epochs**: 2
- **Final Slurm state**: `COMPLETED`, `ExitCode=0:0` (both the job and its `.0` step)
- **Adapter location**: `../../train/lora_adapter_v2_2000rows/` — **not**
  `../../train/lora_adapter/`, which was overwritten by a later job (73) on
  2026-11-06T02:10:55. The `_v2_2000rows` copy's own mtime (2026-11-05T21:43:29)
  matches job 55's `End` time (21:43:37) from `slurm-job-55.txt`.
- **Adapter checksum file**: `adapter-sha256.txt`

## Evidence

- `slurm-job-55.txt` — `sacct` record for job 55 (state, exit code, timing, node/GPU allocation)
- `../../train/sbatch_train_55.log` — job 55's own training log (both ranks' NCCL init, training progress, final loss, adapter-save message)
- `../../monitoring/gpu_util_worker-0.log`, `../../monitoring/gpu_util_worker-1.log` — `nvidia-smi dmon` samples; **verified to correspond to job 55 specifically**, see "Monitoring log correspondence" below
- `adapter_config.json` — copied from `lora_adapter_v2_2000rows/`, confirms LoRA `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, and the exact target module list — independently cross-checks the reconstructed training script below
- `adapter-sha256.txt` — SHA-256 of `lora_adapter_v2_2000rows/adapter_model.safetensors`
- `historical_train_lora_sft_job55.py` — reconstructed training script as it existed when job 55 ran (see Configuration Mismatch)
- `historical_prepare_dataset_job55.py` — reconstructed dataset-prep script as it existed for job 55's dataset

## Monitoring log correspondence (explicit, per the identified gap)

`sacct` gives job 55's window as `Start=2026-11-05T21:38:28`, `End=2026-11-05T21:43:37`.

The committed `gpu_util_worker-0.log` / `gpu_util_worker-1.log` files' own first/last
samples:

| File | First sample | Last sample |
|---|---|---|
| `gpu_util_worker-0.log` | `2026-11-05 21:38:52` | `2026-11-05 21:43:30` |
| `gpu_util_worker-1.log` | `2026-11-05 21:38:53` | `2026-11-05 21:43:31` |

Both fall inside job 55's `sacct` window (21:38:28–21:43:37), confirming these specific
log files are job 55's own `dmon` output, not a later job's. (This matters because the
live path on the cluster — `/mnt/data/assignment/monitoring/gpu_util_worker-*.log` — is
a **shared, non-job-namespaced** path that later training runs overwrite; the
*committed* copies in this repo were pulled before job 73 overwrote the live cluster
copy, and are the ones referenced here.)

Recomputed from these exact files: average SM utilization during the active training
window was **89.9% (worker-0)** and **89.0% (worker-1)** — both over the 80% bonus
target, though per the task description this is a separate bonus, not part of the core
Task 1 demonstration.

## Configuration mismatch — explicitly resolved, not glossed over

The committed `assignment/train/train_lora_sft.py` and
`assignment/data/prepare_dataset.py` reflect a **later** job (73): the full
~50,657-row dataset and 1 epoch. Job 55 itself ran with a 2000-row subsample and 2
epochs — this is visible directly in job 55's own log
(`../../train/sbatch_train_55.log`, which records `'epoch': 2.0`).

Rather than represent today's script as job 55's, the two scripts actually used for
job 55 are reconstructed here from this session's own record:
`historical_train_lora_sft_job55.py` and `historical_prepare_dataset_job55.py`. The
reconstruction is cross-checked, not just recalled from memory: the LoRA
hyperparameters in `historical_train_lora_sft_job55.py` (`r=16`, `lora_alpha=32`,
`lora_dropout=0.05`, the exact 7-module `target_modules` list) match
`adapter_config.json`, which is a value only PEFT itself could have written, baked
into the actual saved checkpoint — not something that could be faked after the fact.

`assignment/train/sbatch_train.sh` (the `sbatch` launcher) and
`assignment/environment/setup_env.sh` (package versions) were **not** changed between
job 55 and job 73, so the currently committed copies of those two files are accurate,
unmodified evidence for job 55 as well.

## Limitations

- **No model/dataset revision pinning**: both `Qwen/Qwen2.5-7B-Instruct` and
  `bigcode/self-oss-instruct-sc2-exec-filter-50k` were loaded via their plain repo
  names (implicit `main`/latest), not an explicit commit hash. If either repo has
  changed upstream since the run, the exact bytes used can't be re-derived from the
  name alone.
- **No container image digest recorded**: the job ran in
  `nvcr.io/nvidia/pytorch:24.10-py3` (tag only); no image digest was captured at
  the time, and it could not be recovered afterward from this session's tooling
  access.
- **Installed Python package versions** are the ones pinned in
  `assignment/environment/setup_env.sh` (`transformers==4.46.3`, `peft==0.13.2`,
  `trl==0.11.4`, `accelerate==1.1.1`, `datasets==3.1.0`, `huggingface-hub==0.36.2`,
  `tokenizers==0.20.3`, installed `--no-deps` on top of the container's own
  `torch==2.5.0a0+nv24.10`) — confirmed unchanged between job 55 and later jobs, but
  not re-verified by re-running `pip freeze` inside job 55's own now-gone container
  instance (that instance no longer exists to query).
- **`lora_adapter/` (the un-suffixed path)** is **not** job 55's checkpoint — it was
  overwritten by job 73. Anyone re-running inference against the live cluster path
  needs to use `lora_adapter_v2_2000rows/` specifically for job-55-equivalent results.
