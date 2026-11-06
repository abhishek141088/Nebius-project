"""Download, format, and subsample the coding-instruction dataset.

Source: iamtarun/python_code_instructions_18k_alpaca (Alpaca-style
instruction/input/output triples, Python-focused). Formats each example
into Qwen2.5-Coder-3B-Instruct's own chat template so the saved text is
exactly what the model will be trained on - no templating logic needed
again in the training script.

Run inside the same container/pylibs environment as setup_env.sh:
  srun -N1 --gres=gpu:1 --container-image=nvcr.io/nvidia/pytorch:24.10-py3 \
       --container-mounts=/mnt/data:/mnt/data \
       python3 /mnt/data/assignment/data/prepare_dataset.py
"""
import json
import os
import random
import sys

sys.path.insert(0, "/mnt/data/assignment/pylibs")

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# Execution-verified (every reference solution was actually run and checked
# to pass) - the dataset used to train StarCoder2-Instruct. Chosen after the
# first dataset (generic GPT-generated Alpaca-style) produced a net
# regression on the held-out eval - this one has a real quality bar behind
# it instead of being unverified model output.
DATASET_NAME = "bigcode/self-oss-instruct-sc2-exec-filter-50k"
OUT_DIR = "/mnt/data/assignment/data"
OUT_PATH = os.path.join(OUT_DIR, "train.jsonl")
# Scaling up from the 2000-row trial run to the full dataset (~50,661 rows
# before filtering) now that training speed is confirmed fast. A number
# higher than the actual row count just means "take everything" below.
TARGET_ROWS = 100_000
# Responses here are verbose (reasoning + implementation + explanation +
# tests, not just a bare function), so the char budget is much higher than
# the previous dataset's simple instruction/output pairs.
MAX_CHARS = 6000
SEED = 42

SYSTEM_PROMPT = "You are a helpful coding assistant that writes correct, clean Python code."


def build_messages(example):
    instruction = example["instruction"].strip()
    response = example["response"].strip()

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ], len(instruction) + len(response)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Loading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME, split="train")
    print(f"Raw dataset size: {len(ds)}")
    print(f"Columns: {ds.column_names}")

    rows = []
    for example in ds:
        messages, length = build_messages(example)
        if length == 0 or length > MAX_CHARS:
            continue
        rows.append(messages)

    print(f"After length filtering (<= {MAX_CHARS} chars): {len(rows)}")

    random.seed(SEED)
    random.shuffle(rows)
    rows = rows[:TARGET_ROWS]
    print(f"Subsampled to: {len(rows)}")

    with open(OUT_PATH, "w") as f:
        for messages in rows:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            f.write(json.dumps({"text": text}) + "\n")

    print(f"Wrote {len(rows)} rows to {OUT_PATH}")
    print("\n--- Sample formatted record ---")
    with open(OUT_PATH) as f:
        sample = json.loads(f.readline())
    print(sample["text"])


if __name__ == "__main__":
    main()
