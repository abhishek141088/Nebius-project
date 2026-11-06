"""HISTORICAL RECONSTRUCTION - NOT the current committed script.

This is the exact content of assignment/train/train_lora_sft.py AS IT
EXISTED when job 55 was submitted. The file has since been edited
(switched to the full ~50k-row dataset, 1 epoch, for a later job - see
job 73) - that LATER version is what's committed at
assignment/train/train_lora_sft.py today. This historical copy is
reconstructed from this session's own conversation record (the exact
edits applied are known precisely) and cross-checked against job 55's
own saved adapter_config.json (r=16, lora_alpha=32, lora_dropout=0.05,
target_modules matching exactly - see adapter_config.json in this
folder), which could only have been produced by LoraConfig with these
exact values.

Do not confuse this with the current assignment/train/train_lora_sft.py.
"""
import sys

sys.path.insert(0, "/mnt/data/assignment/pylibs")

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATA_PATH = "/mnt/data/assignment/data/train.jsonl"
ADAPTER_OUTPUT_DIR = "/mnt/data/assignment/train/lora_adapter"
RUN_DIR = "/mnt/data/assignment/train/runs"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)

    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=RUN_DIR,
        num_train_epochs=2,
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        report_to=[],
        ddp_find_unused_parameters=False,
        dataset_text_field="text",
        max_seq_length=2048,
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
    )

    trainer.train()

    if trainer.is_world_process_zero():
        trainer.model.save_pretrained(ADAPTER_OUTPUT_DIR)
        tokenizer.save_pretrained(ADAPTER_OUTPUT_DIR)
        print(f"Saved LoRA adapter to {ADAPTER_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
