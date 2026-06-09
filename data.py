import json
from pathlib import Path

DATA_DIR = Path("data")
RULES_DIR = Path("rules/filtered")


def set_data_dir(path):
    global DATA_DIR
    DATA_DIR = Path(path)


def set_rules_dir(path):
    global RULES_DIR
    RULES_DIR = Path(path)


def rulebook_path(dataset, task_key):
    return RULES_DIR / f"{dataset}_{task_key}.json"


def load_split(dataset, task_key, seed):
    p = DATA_DIR / f"{dataset}_{task_key}_seed{seed}.json"
    if not p.exists():
        raise FileNotFoundError(f"split not found: {p}")
    return json.loads(p.read_text())


def load_rulebook(dataset, task_key):
    p = rulebook_path(dataset, task_key)
    if not p.exists():
        raise FileNotFoundError(f"rulebook not found: {p}")
    return json.loads(p.read_text())
