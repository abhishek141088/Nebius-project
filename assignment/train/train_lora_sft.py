"""LoRA SFT fine-tuning script, run under torchrun across both nodes.

Each GPU holds a full copy of the base model; DDP syncs only the LoRA
adapter gradients across nodes, which keeps cross-node traffic small -
the right call given there's no InfiniBand, only standard Ethernet
between the two worker nodes.
"""
import sys

sys.path.insert(0, "/mnt/data/assignment/pylibs")

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

# Switched from Qwen2.5-Coder-3B-Instruct (already near-ceiling on simple
# Python tasks, leaving no headroom to show improvement) and the generic
# unverified dataset (caused a net regression) to a bigger, general-purpose
# (not code-specialized) model + an execution-verified dataset.
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
        # Scaling up to the full ~50k-row dataset (from the 2000-row
        # trial) - 1 epoch now gives far more gradient updates than 2
        # epochs did on the small subsample, so pulling epochs back down.
        num_train_epochs=1,
        learning_rate=2e-4,
        # Starting conservative on batch size: the 7B model is ~2.3x the
        # param count of the 3B one, and responses in the new dataset are
        # longer (verbose reasoning + implementation + tests), raising
        # per-example activation memory too. gradient_checkpointing keeps
        # peak memory down (and keeps the GPU doing more work per step,
        # which also helps utilization) - tune batch size up from here if
        # there's headroom once this doesn't OOM.
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
