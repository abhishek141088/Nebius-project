"""Generate + execute + score code for each eval prompt, against a given model.

Used for BOTH the untrained baseline and the fine-tuned model - same
harness, same prompts, same scoring - so the two runs are directly
comparable.

Usage (inside the container, with pylibs on PYTHONPATH):
  python3 run_eval.py --model Qwen/Qwen2.5-Coder-3B-Instruct \
                       --out /mnt/data/assignment/eval/baseline_results.json \
                       --label baseline

  python3 run_eval.py --model Qwen/Qwen2.5-Coder-3B-Instruct \
                       --adapter /mnt/data/assignment/train/lora_adapter \
                       --out /mnt/data/assignment/eval/finetuned_results.json \
                       --label fine-tuned
"""
import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, "/mnt/data/assignment/pylibs")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

SYSTEM_PROMPT = "You are a helpful coding assistant that writes correct, clean Python code."
PROMPTS_PATH = "/mnt/data/assignment/eval/prompts.jsonl"
TIMEOUT_SECONDS = 5

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(generated_text):
    matches = CODE_BLOCK_RE.findall(generated_text)
    if matches:
        # The model sometimes splits its answer into multiple fenced code
        # blocks (e.g. a class definition, then a separate function that
        # uses it) - concatenate all of them, not just the first.
        candidate = "\n\n".join(m.strip() for m in matches)
    else:
        def_idx = generated_text.find("def ")
        candidate = generated_text[def_idx:].strip() if def_idx != -1 else generated_text.strip()

    return _clean_code(candidate)


def _clean_code(candidate):
    """Keep only top-level def/class/import/assign statements, dropping any
    trailing demo/example invocations (print(...), bare calls, etc.) - the
    eval harness supplies its own test calls, so the model's own "Example
    usage" snippets are never needed and sometimes truncated mid-statement
    by the token limit, which would otherwise cause a spurious SyntaxError
    unrelated to whether the actual function is correct."""
    lines = candidate.splitlines()
    tree = None
    # If generation got cut off mid-statement, the tail is an incomplete
    # line - trim from the end until what's left parses.
    while lines:
        source = "\n".join(lines)
        try:
            tree = ast.parse(source)
            break
        except SyntaxError:
            lines.pop()
    if tree is None:
        return candidate  # nothing parsed; let it fail downstream as-is

    keep_types = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                  ast.Import, ast.ImportFrom, ast.Assign)
    kept_source = "\n".join(lines)
    segments = []
    for node in tree.body:
        if isinstance(node, keep_types):
            segment = ast.get_source_segment(kept_source, node)
            if segment:
                segments.append(segment)
    return "\n\n".join(segments) if segments else candidate


def run_test(code, test_code):
    full_script = code + "\n\n" + test_code
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_script)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        if result.returncode == 0:
            return True, None
        return False, (result.stderr or result.stdout).strip()[-500:]
    except subprocess.TimeoutExpired:
        return False, f"timed out after {TIMEOUT_SECONDS}s"
    finally:
        import os

        os.unlink(path)


def load_prompts():
    prompts = []
    with open(PROMPTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base model name or path")
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter path to apply on top")
    parser.add_argument("--out", required=True, help="Path to write the JSON results report")
    parser.add_argument("--label", default="run", help="Label for this run, e.g. baseline / fine-tuned")
    # Raised from 300: class-based, multi-method solutions (LRU cache,
    # Trie, Union-Find, etc.) in the harder eval tier need more room.
    parser.add_argument("--max-new-tokens", type=int, default=500)
    args = parser.parse_args()

    print(f"Loading tokenizer + model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )

    if args.adapter:
        print(f"Applying LoRA adapter: {args.adapter}")
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()

    prompts = load_prompts()
    print(f"Loaded {len(prompts)} eval prompts")

    results = []
    for p in prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": p["instruction"]},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(
            output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
        )
        code = extract_code(generated_text)
        passed, error = run_test(code, p["test_code"])

        results.append(
            {
                "id": p["id"],
                "passed": passed,
                "error": error,
                "generated_text": generated_text,
                "extracted_code": code,
            }
        )
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {p['id']}")

    pass_count = sum(r["passed"] for r in results)
    total = len(results)
    print(f"\n{args.label}: {pass_count}/{total} passed")

    report = {
        "label": args.label,
        "model": args.model,
        "adapter": args.adapter,
        "pass_count": pass_count,
        "total": total,
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote report to {args.out}")


if __name__ == "__main__":
    main()
