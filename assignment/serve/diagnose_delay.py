"""Does a delay between model load and the first generate() call cause a
hang, independent of any server framework? Tests the idle-GPU theory
directly."""
import sys
import time

sys.path.insert(0, "/mnt/data/assignment/pylibs")

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "/mnt/data/assignment/train/lora_adapter_v2_2000rows"

print("Loading...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to("cuda")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()
print("Model ready. Sleeping 45s to mimic server idle wait...", flush=True)
time.sleep(45)

print("Now calling generate()...", flush=True)
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write a Python function is_leap_year(year)."},
]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
print("Tokenized and moved to device.", flush=True)

with torch.no_grad():
    output_ids = model.generate(input_ids, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)

print("GENERATE RETURNED SUCCESSFULLY", flush=True)
print(tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True), flush=True)
