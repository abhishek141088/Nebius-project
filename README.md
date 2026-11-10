# Soperator on Nebius: reproduction guide

This repo deploys a Slurm-on-Kubernetes cluster (Soperator 4.1.17) on Nebius with two single-GPU H100 workers and no InfiniBand, then runs distributed fine-tuning, inference and evaluation on it as Slurm jobs.

This guide lists every step in order so that a person or an AI agent can repeat the work. Each step says whether it was **validated** (run on the live cluster) or **reconstructed** (written afterwards from logs and not re-run).

Related documents:

- `CHANGES.md`: what was changed and why, corrections, runbook.
- `ARCHITECTURE.svg`: architecture diagram.
- `assignment/Evidence/Task1/README.md`: evidence for the submitted training run (job 55).

## Rules for anyone repeating this, including AI agents

1. Never write a secret into a file. Credentials live in shell environment variables only.
2. Do not print or commit `candidate-soperator.env` or the backend secret file.
3. Do not run `terraform destroy` on a cluster that is being kept for a demo.
4. Run Slurm commands inside the jail: over SSH to the login node, or with `chroot /mnt/jail` when using `kubectl exec`. Outside the jail, `srun` fails with `dlopen(chroot.so)`.
5. A full deployment from a clean clone has **not** been validated. Steps 4 and 5 below describe the manual fixes the first deployment needed.

## What you get

| Item | Value |
|---|---|
| Region | `eu-north1` |
| Soperator | `4.1.17` (recipe tag `soperator-v4.1.17-1`) |
| GPU workers | 2 x `gpu-h100-sxm`, preset `1gpu-16vcpu-200gb`, no GPU cluster |
| CPU node groups | System 4 to 24 x `8vcpu-32gb`; Login 1 x `16vcpu-64gb`; Controller 1 x `4vcpu-16gb`; Accounting 1 x `8vcpu-32gb`; NFS 1 x `4vcpu-16gb` (all `cpu-d3`) |
| Shared storage | Jail filesystem 2 TiB; `/mnt/data` filesystem 2 TiB |
| Settings | `public_o11y_enabled = false`, backups `force_disable` |

## Step 0. Prerequisites

Install on the workstation: `terraform`, the `nebius` CLI (logged in), `kubectl`, `jq`, `yq`, `git`, `ssh`.

You also need, from Nebius:

- Tenant ID, project ID and region.
- The Terraform state bucket name and key.
- A backend access key ID and secret (delivered separately).

Create an SSH key pair for login-node access and put only the **public** key in `terraform.tfvars` (`slurm_login_ssh_root_public_keys`):

```bash
ssh-keygen -t ed25519 -f ~/.ssh/nebius_soperator_ed25519
```

## Step 1. Get the code

```bash
git clone https://github.com/abhishek141088/Nebius-project.git
```

```bash
cd Nebius-project/nebius-solutions-library-soperator-v4.1.17-1/soperator/installations/example
```

The recipe in this repo already contains these changes to the stock `soperator-v4.1.17-1` tag:

| File | Change |
|---|---|
| `modules/k8s/gpu_fabric_validation.tf` | The InfiniBand check applies only to presets that support a GPU cluster. |
| `modules/available_resources/gres.tf`, `outputs.tf` | Per-preset GRES override for the single-GPU H100 preset (`File=/dev/nvidia0 Cores=0-7`). |
| `installations/example/main.tf` | Looks up the per-preset GRES override before the platform default. |
| `modules/k8s/locals.tf`, `k8s_ng_workers_v2.tf` | Adds a topology label to non-InfiniBand GPU workers. **The key is wrong and has no effect** (see step 5). |
| `installations/example/variables.tf` | Unused tenant data block removed, as the sandbox instructions require. |
| `installations/example/.envrc` | Uses the assigned state bucket; backend credentials come from `AWS_*` environment variables. |
| `installations/example/terraform.tfvars` | Node sizes, two fixed workers, observability and backup settings. |

## Step 2. Set the environment

Create `candidate-soperator.env` in `installations/example/` yourself. It is gitignored. It must export these variables (values from Nebius; do not commit them):

```bash
export NEBIUS_TENANT_ID=...
export NEBIUS_PROJECT_ID=...
export NEBIUS_REGION=eu-north1
export TF_BACKEND_BUCKET=...
export TF_BACKEND_KEY=...
export TF_BACKEND_AWS_ACCESS_KEY_ID=...
export TF_BACKEND_AWS_SECRET_ACCESS_KEY=...
```

Then load it and the recipe's own setup script:

```bash
source ./candidate-soperator.env
```

```bash
source .envrc
```

`.envrc` gets an IAM token, discovers the first subnet in the project, exports the `AWS_*` and `TF_VAR_*` variables and writes `terraform_backend_override.tf` (which contains no credentials).

## Step 3. Deploy with Terraform

**Validated** (on the first deployment, followed by several incremental applies).

```bash
terraform init -reconfigure
```

```bash
terraform plan -input=false -out=candidate-soperator.tfplan
```

```bash
terraform apply candidate-soperator.tfplan
```

Notes:

- `.terraform.lock.hcl` is gitignored, so a clean clone resolves provider versions fresh. The versions used here were nebius 0.6.80, kubernetes 3.4.0, flux 1.10.0 and helm 2.17.0.
- A successful apply means the desired state was handed to Flux. It does not mean Slurm is healthy. Continue with step 4.

## Step 4. Check the cluster

The recipe adds a kubectl context named `nebius-<company_name>-slurm` (here `nebius-abishek-jain-slurm`).

```bash
kubectl get helmreleases -A
```

```bash
kubectl get pods -n soperator
```

Expected: every HelmRelease `Ready`, and `controller-0`, `login-0`, `worker-0`, `worker-1` running.

**If `controller-0` crash-loops with `fatal: No Assoc usage file ... to recover`:** the state volume has a `clustername` file but no state files, because an earlier start died within its first 5 minutes. On the first deployment this was caused by the GRES crash that step 1's fix removes, so a fresh deployment may not hit it. Recovery that was validated:

1. `kubectl scale deploy soperator-controller-manager -n soperator-system --replicas=0`
2. Patch the `controller-0` container command to add `-i` to `slurmctld` and let it start.
3. Wait at least 5 minutes so the first periodic state save happens.
4. Scale the operator back to 1. It restores the stock command, which now starts normally.

See `CHANGES.md` section 7 for details.

## Step 5. Give the workers topology data (manual, required)

**Validated.** Without this, worker pods wait for per-node topology data and fail after 180 seconds. On a cluster without InfiniBand nothing populates it.

Find the Kubernetes node names of the two GPU workers:

```bash
kubectl get pods -n soperator -o wide | grep worker
```

Patch the data in, replacing the two node names:

```bash
kubectl patch resourcedistribution.apps.kruise.io topology-soperator --type merge -p '{"spec":{"resource":{"data":{"<node-1>":"{\"tier-1\":\"unknown\"}","<node-2>":"{\"tier-1\":\"unknown\"}"}}}}'
```

Delete the worker pods so they retry immediately:

```bash
kubectl delete pod worker-0 worker-1 -n soperator
```

**Proposed permanent fix, not applied and not tested:** in `modules/k8s/locals.tf`, change the label key `topologyconf.slurm.nebius.ai/tier-1` to `topology.nebius.com/tier-1`, which is the prefix the operator watches.

## Step 6. Log in and smoke-test Slurm

Get the login node's public address:

```bash
kubectl get svc soperator-login-svc -n soperator
```

```bash
ssh -i ~/.ssh/nebius_soperator_ed25519 root@<EXTERNAL-IP>
```

On the login node:

```bash
sinfo
```

```bash
srun -N2 --gres=gpu:1 nvidia-smi -L
```

Expected: both workers `idle`, and one H100 listed per node. **Validated.**

## Step 7. Copy the assignment code to shared storage

All scripts use absolute paths under `/mnt/data/assignment`. From the repo root on the workstation:

```bash
scp -i ~/.ssh/nebius_soperator_ed25519 -r assignment root@<EXTERNAL-IP>:/mnt/data/
```

All remaining steps run on the login node.

## Step 8. Install the training environment

**Validated.** Installs pinned packages into `/mnt/data/assignment/pylibs` on top of the container's own PyTorch.

```bash
srun -N1 --gres=gpu:1 --container-image=nvcr.io/nvidia/pytorch:24.10-py3 --container-mounts=/mnt/data:/mnt/data bash /mnt/data/assignment/environment/setup_env.sh
```

Expected last lines: `cuda available: True` and the versions `transformers 4.46.3`, `peft 0.13.2`, `trl 0.11.4`.

## Step 9. Optional network checks

**Validated.** Confirms NCCL works over Ethernet and measures throughput between workers (about 13.5 Gbps here).

```bash
sbatch /mnt/data/assignment/train/sbatch_verify_nccl.sh
```

```bash
sbatch /mnt/data/assignment/train/sbatch_bandwidth_test.sh
```

## Step 10. Prepare the dataset

Two configurations exist. Pick one and use it for steps 10 and 11.

| Configuration | Scripts | Data | Epochs | Runtime |
|---|---|---|---|---|
| **A: job 55** (submitted run) | `assignment/Evidence/Task1/historical_*_job55.py` | 2,000 rows | 2 | about 5 minutes |
| **B: job 73** (full dataset) | `assignment/data/prepare_dataset.py`, `assignment/train/train_lora_sft.py` | about 50,000 rows | 1 | about 45 minutes |

For configuration A, copy the historical scripts over the current ones first (they were reconstructed from the session record and cross-checked against the saved adapter configuration, but not re-run):

```bash
cp /mnt/data/assignment/Evidence/Task1/historical_prepare_dataset_job55.py /mnt/data/assignment/data/prepare_dataset.py
```

```bash
cp /mnt/data/assignment/Evidence/Task1/historical_train_lora_sft_job55.py /mnt/data/assignment/train/train_lora_sft.py
```

Then, for either configuration:

```bash
srun -N1 --gres=gpu:1 --container-image=nvcr.io/nvidia/pytorch:24.10-py3 --container-mounts=/mnt/data:/mnt/data python3 /mnt/data/assignment/data/prepare_dataset.py
```

Model: `Qwen/Qwen2.5-7B-Instruct`. Dataset: `bigcode/self-oss-instruct-sc2-exec-filter-50k`. Neither revision is pinned. The docstring at the top of `prepare_dataset.py` names an older dataset and model; the constants in the code are correct.

## Step 11. Task 1: distributed fine-tuning

**Validated** (jobs 55 and 73).

```bash
sbatch /mnt/data/assignment/train/sbatch_train.sh
```

The job takes both nodes, starts one `torchrun` process per node with `NCCL_IB_DISABLE=1`, and records GPU utilization on each node with `nvidia-smi dmon`.

Check the result (replace the job ID):

```bash
sacct -j <jobid> -X --format=JobID,State,ExitCode,Elapsed,NNodes,AllocTRES%50
```

Expected: `COMPLETED`, `0:0`, 2 nodes, `gres/gpu=2`. The log is `/mnt/data/assignment/train/sbatch_train_<jobid>.log` and the adapter is written to `/mnt/data/assignment/train/lora_adapter`.

Each training run overwrites `lora_adapter` and the two monitoring logs. After a configuration A run, keep a copy under the name the serving scripts expect:

```bash
cp -r /mnt/data/assignment/train/lora_adapter /mnt/data/assignment/train/lora_adapter_v2_2000rows
```

Reference results: job 55 had 209 s training runtime and loss 0.517; job 73 had 2,611 s and loss 0.516.

## Step 12. Task 4: GPU utilization

The training job writes `/mnt/data/assignment/monitoring/gpu_util_worker-0.log` and `gpu_util_worker-1.log`. Column 4 is GPU utilization in percent, one sample per second.

Average between the first and last non-zero sample:

```bash
awk '!/^#/{n++; v[n]=$4; if($4>0){if(!f)f=n; l=n}} END{for(i=f;i<=l;i++)s+=v[i]; print s/(l-f+1)}' /mnt/data/assignment/monitoring/gpu_util_worker-0.log
```

Reference for job 55: about 90% on each GPU over that window, and 69.5% over the whole job including model loading. Inference utilization was not measured.

## Step 13. Task 3: evaluate the base and fine-tuned models

The original launch commands were not saved, so these are **reconstructed** from the script's arguments. `run_eval.py` generates code for each of the 42 prompts, runs it against the tests with a 5-second timeout and writes a JSON report.

Base model:

```bash
srun -N1 --gres=gpu:1 --container-image=nvcr.io/nvidia/pytorch:24.10-py3 --container-mounts=/mnt/data:/mnt/data python3 /mnt/data/assignment/eval/run_eval.py --model Qwen/Qwen2.5-7B-Instruct --out /mnt/data/assignment/eval/baseline_results.json --label baseline
```

Fine-tuned model:

```bash
srun -N1 --gres=gpu:1 --container-image=nvcr.io/nvidia/pytorch:24.10-py3 --container-mounts=/mnt/data:/mnt/data python3 /mnt/data/assignment/eval/run_eval.py --model Qwen/Qwen2.5-7B-Instruct --adapter /mnt/data/assignment/train/lora_adapter --out /mnt/data/assignment/eval/finetuned_results.json --label fine-tuned
```

Reference results: base 41/42; job 73 adapter 36/42. On an earlier 30-prompt version of the harness: base 27/30, job 55 adapter 28/30. The job 55 adapter was not scored on the 42-prompt set.

## Step 14. Install the serving environment

**Reconstructed. No install script was saved.** The serving environment is a separate directory, `/mnt/data/assignment/pylibs_vllm`, with its own PyTorch. The versions found in it are `vllm 0.31.0`, `torch 2.13.0`, `flashinfer-python 0.7.0.post1` and `aiperf 0.11.0`. An equivalent install, not re-run:

```bash
srun -N1 --gres=gpu:1 --container-image=nvcr.io/nvidia/pytorch:24.10-py3 --container-mounts=/mnt/data:/mnt/data pip install --target=/mnt/data/assignment/pylibs_vllm --no-cache-dir vllm==0.31.0 aiperf==0.11.0
```

The serving scripts already contain the workarounds this environment needed: they hide the container's own `flash_attn`, set `VLLM_ATTENTION_BACKEND=FLASHINFER` and disable the FlashInfer sampler.

## Step 15. Task 2: serve the fine-tuned model

**Validated.** Both scripts serve `Qwen/Qwen2.5-7B-Instruct` with the adapter at `train/lora_adapter_v2_2000rows` under the model name `finetuned`, on port 8001.

One GPU:

```bash
sbatch /mnt/data/assignment/serve/sbatch_serve_vllm.sh
```

Or one independent replica per GPU:

```bash
sbatch /mnt/data/assignment/serve/sbatch_serve_vllm_2replicas.sh
```

Wait until the log `/mnt/data/assignment/serve/serve_vllm_<jobid>.log` shows the server is up, then test from the login node:

```bash
curl -s http://worker-0:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"finetuned","messages":[{"role":"user","content":"Write a Python function that reverses a string."}],"max_tokens":128}'
```

Use `"model":"Qwen/Qwen2.5-7B-Instruct"` in the same request to query the base model for comparison. Both jobs have a 30-minute time limit.

## Step 16. Benchmark inference

**Validated.** While the two-replica serving job holds both nodes, a separate benchmark job cannot get a node. Run the client as a step inside the serving job instead (replace the job ID):

```bash
srun --overlap --jobid=<serve-jobid> --nodes=1 -w worker-0 --container-image=nvcr.io/nvidia/pytorch:24.10-py3 --container-mounts=/mnt/data:/mnt/data bash -c 'PYTHONPATH=/mnt/data/assignment/pylibs_vllm python3 -m aiperf profile --model finetuned --tokenizer Qwen/Qwen2.5-7B-Instruct --url http://worker-0:8001 --url http://worker-1:8001 --url-strategy round-robin --endpoint-type chat --streaming --concurrency 8 --request-count 80 --synthetic-input-tokens-mean 256 --synthetic-input-tokens-stddev 0 --output-tokens-mean 128 --output-tokens-stddev 0 --artifact-dir /mnt/data/assignment/serve/aiperf_results_2replica'
```

Two details matter: use `python3 -m aiperf` (the `aiperf` command is not on the path), and pass `--tokenizer`, because the model name `finetuned` is not a tokenizer name.

For the single-GPU benchmark use one `--url`, `--concurrency 4` and `--request-count 40`.

Reference results: 93.6 output tokens/s on one GPU; 145.3 output tokens/s across two replicas (80 of 80 requests, no errors). The two runs used different concurrency, so the ratio is not a clean scaling figure.

## Step 17. Stop jobs and clean up

```bash
squeue
```

```bash
scancel <jobid>
```

Decommission the cluster only when it is no longer needed for a demo. `terraform destroy` was not run in this project.

## Known gaps

- A clean destroy-and-apply has not been tested, and steps 4 and 5 are manual.
- The topology label fix and a guard against a stale `clustername` file are proposed in `CHANGES.md` section 8 but not applied.
- Steps 13 and 14 are reconstructed, not re-run.
- Model, dataset and container image revisions are not pinned.
- Training does not checkpoint (`save_strategy="no"`).
- `assignment/serve/serve_model.py`, `generate_once.py`, `sbatch_serve.sh` and the `diagnose_*.py` scripts are an earlier serving attempt, replaced by vLLM. They are not needed to reproduce the results.
- Security is at sandbox level: SSH is open to any source address, jobs run as root, backups are disabled.
