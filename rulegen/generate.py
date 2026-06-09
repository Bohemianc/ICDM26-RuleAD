import json
import re
from pathlib import Path

import torch
from rdkit import Chem
from transformers import AutoModelForCausalLM, AutoTokenizer

from rule_schema import validate_structured_rule_object

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

JOBS = [
    ("BBBP", "BBBP", None),
    ("ClinTox_task1", "ClinTox", 1),
    ("HIV", "HIV", None),
] + [(f"Tox21_task{i}", "Tox21", i) for i in range(12)] \
  + [(f"SIDER_task{i}", "SIDER", i) for i in range(27)]


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    return tokenizer, model


def run_llm(tokenizer, model, prompt):
    messages = [
        {"role": "system", "content": "You output strict JSON only."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=4096, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def extract_rules(text):
    obj, decoder = None, json.JSONDecoder()
    fenced = re.findall(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", text, flags=re.S)
    for cand in fenced + [text]:
        for i, ch in enumerate(cand):
            if ch in "[{":
                try:
                    obj, _ = decoder.raw_decode(cand[i:])
                    break
                except json.JSONDecodeError:
                    continue
        if obj is not None:
            break
    rules = obj.get("rules", []) if isinstance(obj, dict) else (obj or [])
    kept = []
    for rule in rules:
        try:
            validated = validate_structured_rule_object(rule)
        except Exception:
            continue
        if validated["type"].startswith("smarts") and Chem.MolFromSmarts(validated["smarts"]) is None:
            continue
        kept.append(validated)
    return kept


def cache_name(dataset, task_idx):
    task = f"_task{task_idx}" if task_idx is not None else ""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", MODEL)
    return f"{dataset}{task}__{safe}.json"


def main():
    out_dir = Path("rules") / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_model()
    for stem, dataset, task_idx in JOBS:
        prompt = (PROMPTS_DIR / f"{stem}.txt").read_text()
        rules = extract_rules(run_llm(tokenizer, model, prompt))
        out = out_dir / cache_name(dataset, task_idx)
        out.write_text(json.dumps({"rules": rules}, ensure_ascii=False, indent=2))
        print(f"{stem} -> {out}  ({len(rules)} rules)", flush=True)


if __name__ == "__main__":
    main()
