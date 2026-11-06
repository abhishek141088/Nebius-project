"""HISTORICAL RECONSTRUCTION - NOT the current committed script.

This is the exact content of assignment/data/prepare_dataset.py AS IT
EXISTED when the dataset used by job 55 was generated: a 2000-row
subsample (TARGET_ROWS=2000), not the full ~50k-row dataset the current
committed script uses (TARGET_ROWS=100_000, i.e. "take everything").
Reconstructed from this session's own conversation record.
"""
import json
import os
import random
import sys

sys.path.insert(0, "/mnt/data/assignment/pylibs")

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATASET_NAME = "bigcode/self-oss-instruct-sc2-exec-filter-50k"
OUT_DIR = "/mnt/data/assignment/data"
OUT_PATH = os.path.join(OUT_DIR, "train.jsonl")
TARGET_ROWS = 2000
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


if __name__ == "__main__":
    main()
