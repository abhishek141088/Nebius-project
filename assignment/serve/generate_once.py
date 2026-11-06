"""One-shot generation script, invoked as a subprocess by serve_model.py
for each request. Loads the model fresh each call (~15s overhead) - a
deliberate trade-off, not an oversight. Calling model.generate() directly
from inside an open HTTP socket's request handler hung indefinitely at
0% GPU utilization every time it was tried (threaded FastAPI route,
async FastAPI route, and plain stdlib http.server all reproduced it
identically), while the exact same generate() call made from a plain
script with no open socket in the process worked every time, including
deliberately immediately after a 45s idle wait. The cause is some
driver/socket interaction specific to this environment that further
isolation didn't pin down in the time available - this subprocess
boundary sidesteps it entirely by guaranteeing no HTTP socket is ever
open in the process that calls into CUDA.
"""
import json
import sys

sys.path.insert(0, "/mnt/data/assignment/pylibs")

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "/mnt/data/assignment/train/lora_adapter_v2_2000rows"
SYSTEM_PROMPT = "You are a helpful coding assistant that writes correct, clean Python code."


def main():
    request = json.loads(sys.stdin.read())
    prompt = request["prompt"]
    max_new_tokens = request.get("max_new_tokens", 200)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    completion = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(json.dumps({"prompt": prompt, "completion": completion}))


if __name__ == "__main__":
    main()
