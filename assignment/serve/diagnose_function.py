"""Call generate() through a wrapper function (same shape as the server's
generate_completion) but with NO http server / socketserver involved at
all - isolates whether function-wrapping itself is the trigger."""
import sys

sys.path.insert(0, "/mnt/data/assignment/pylibs")

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "/mnt/data/assignment/train/lora_adapter_v2_2000rows"
SYSTEM_PROMPT = "You are a helpful coding assistant."

print("Loading...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to("cuda")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()
print("Model ready.", flush=True)


def generate_completion(prompt, max_new_tokens=100):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    print("  [in function] tokenized, calling generate()...", flush=True)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    print("  [in function] generate() returned", flush=True)
    return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)


print("Calling generate_completion()...", flush=True)
result = generate_completion("Write a Python function is_leap_year(year).")
print("RESULT:", result, flush=True)
