import json
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
FILTER_SEED = 42
FIRING_LO = 0.05
FIRING_HI = 0.70

SPLITS_DIR = Path("data")
RULE_CACHE_DIR = Path("rules/raw")
RULEBANK_DIR = Path("rules/filtered")


def now() -> str:
    return datetime.now().isoformat()


def load_split(dataset, task_key, seed):
    p = SPLITS_DIR / f"{dataset}_{task_key}_seed{seed}.json"
    if not p.exists():
        raise FileNotFoundError(f"split not found: {p}")
    return json.loads(p.read_text())


def _cache_rule_file(dataset, task_idx):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", MODEL)
    task_part = f"_task{task_idx}" if task_idx is not None else ""
    return RULE_CACHE_DIR / f"{dataset}{task_part}__{safe}.json"


DESCRIPTOR_HANDLES = {
    "logp": lambda m: Descriptors.MolLogP(m),
    "mol_wt": lambda m: Descriptors.MolWt(m),
    "molwt": lambda m: Descriptors.MolWt(m),
    "mw": lambda m: Descriptors.MolWt(m),
    "tpsa": lambda m: Descriptors.TPSA(m),
    "hbd": lambda m: Lipinski.NumHDonors(m),
    "hba": lambda m: Lipinski.NumHAcceptors(m),
    "num_rotatable_bonds": lambda m: Lipinski.NumRotatableBonds(m),
    "rotatable_bonds": lambda m: Lipinski.NumRotatableBonds(m),
    "rot_bonds": lambda m: Lipinski.NumRotatableBonds(m),
    "num_aromatic_rings": lambda m: rdMolDescriptors.CalcNumAromaticRings(m),
    "aromatic_rings": lambda m: rdMolDescriptors.CalcNumAromaticRings(m),
    "num_aliphatic_rings": lambda m: rdMolDescriptors.CalcNumAliphaticRings(m),
    "aliphatic_rings": lambda m: rdMolDescriptors.CalcNumAliphaticRings(m),
    "num_rings": lambda m: rdMolDescriptors.CalcNumRings(m),
    "rings": lambda m: rdMolDescriptors.CalcNumRings(m),
    "ring_count": lambda m: rdMolDescriptors.CalcNumRings(m),
    "num_heterocycles": lambda m: rdMolDescriptors.CalcNumHeterocycles(m),
    "heterocycles": lambda m: rdMolDescriptors.CalcNumHeterocycles(m),
    "num_amide_bonds": lambda m: rdMolDescriptors.CalcNumAmideBonds(m),
    "amide_bonds": lambda m: rdMolDescriptors.CalcNumAmideBonds(m),
    "fraction_csp3": lambda m: rdMolDescriptors.CalcFractionCSP3(m),
    "fraction_sp3": lambda m: rdMolDescriptors.CalcFractionCSP3(m),
    "csp3": lambda m: rdMolDescriptors.CalcFractionCSP3(m),
    "num_spiro_atoms": lambda m: rdMolDescriptors.CalcNumSpiroAtoms(m),
    "num_bridgehead_atoms": lambda m: rdMolDescriptors.CalcNumBridgeheadAtoms(m),
    "heavy_atoms": lambda m: m.GetNumHeavyAtoms(),
    "num_heavy_atoms": lambda m: m.GetNumHeavyAtoms(),
}


def _normalize(mol, name: str):
    name = (name or "heavy_atoms").lower()
    if name in ("heavy_atoms", "num_heavy_atoms"):
        return mol.GetNumHeavyAtoms()
    if name in ("atoms", "num_atoms"):
        return mol.GetNumAtoms()
    if name in ("num_rings", "rings", "ring_count"):
        return max(rdMolDescriptors.CalcNumRings(mol), 1)
    return mol.GetNumHeavyAtoms()


def _recover_direction(rule: dict):
    d = (rule.get("direction") or "").strip().lower()
    if d in {"higher", "lower"}:
        return d
    if rule.get("type") in {"smarts_count", "smarts_ratio"}:
        return "higher"
    return None


def _compile_ok(rule: dict):
    t = rule.get("type", "")
    if t in {"smarts_count", "smarts_ratio"}:
        smarts = rule.get("smarts")
        if not smarts:
            return False
        return Chem.MolFromSmarts(smarts) is not None
    if t in {"descriptor", "desirability_window"}:
        desc = (rule.get("descriptor") or "").lower()
        if desc not in DESCRIPTOR_HANDLES:
            return False
        if t == "desirability_window":
            for k in ("a", "b", "c", "d"):
                if rule.get(k) is None:
                    return False
        return True
    return False


def _eval_numeric(rule: dict, mol):
    t = rule.get("type", "")
    try:
        if t == "smarts_count":
            pat = Chem.MolFromSmarts(rule["smarts"])
            if pat is None:
                return None
            return float(len(mol.GetSubstructMatches(pat)))
        if t == "smarts_ratio":
            pat = Chem.MolFromSmarts(rule["smarts"])
            if pat is None:
                return None
            n_match = len(mol.GetSubstructMatches(pat))
            denom = _normalize(mol, rule.get("normalizer", "heavy_atoms"))
            if denom <= 0:
                return 0.0
            return float(n_match) / float(denom)
        if t in ("descriptor", "desirability_window"):
            desc = (rule.get("descriptor") or "").lower()
            fn = DESCRIPTOR_HANDLES.get(desc)
            if fn is None:
                return None
            return float(fn(mol))
    except Exception:
        return None
    return None


def _firing_from_values(rule: dict, values):
    t = rule.get("type", "")
    direction = _recover_direction(rule)
    vals = [v for v in values if v is not None]
    info = {"type": t, "direction": direction, "n_valid": len(vals)}
    if not vals:
        return [0] * len(values), {**info, "threshold": None,
                                    "threshold_kind": "no_valid_values"}
    if t == "smarts_count":
        thr = 1.0
        firings = [1 if (v is not None and v >= thr) else 0 for v in values]
        info.update({"threshold": thr, "threshold_kind": "count>=1"})
        return firings, info
    if t == "smarts_ratio":
        med = statistics.median(vals)
        if med <= 0.0:
            firings = [1 if (v is not None and v > 0.0) else 0 for v in values]
            info.update({"threshold": 0.0, "threshold_kind": "ratio>0"})
        else:
            firings = [1 if (v is not None and v > med) else 0 for v in values]
            info.update({"threshold": med, "threshold_kind": "ratio>median"})
        return firings, info
    if t == "descriptor":
        med = statistics.median(vals)
        if direction == "lower":
            firings = [1 if (v is not None and v < med) else 0 for v in values]
            info.update({"threshold": med, "threshold_kind": "desc<median"})
        else:
            firings = [1 if (v is not None and v > med) else 0 for v in values]
            info.update({"threshold": med, "threshold_kind": "desc>median"})
        return firings, info
    if t == "desirability_window":
        c = float(rule.get("c"))
        firings = [1 if (v is not None and v >= c) else 0 for v in values]
        info.update({"threshold": c, "threshold_kind": "desc>=c"})
        return firings, info
    return [0] * len(values), {**info, "threshold": None,
                                "threshold_kind": "unhandled_type"}


def build_filtered_rulebank(dataset: str, task_idx: Optional[int],
                            task_key: str, label: str,
                            force: bool = False) -> dict:
    out_path = RULEBANK_DIR / f"{label}.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text())

    cache_path = _cache_rule_file(dataset, task_idx)
    if not cache_path.exists():
        raise FileNotFoundError(f"missing rule cache: {cache_path}")
    bundle = json.loads(cache_path.read_text())
    raw_rules = bundle.get("rules", []) or []

    split = load_split(dataset, task_key, FILTER_SEED)
    train_normal = list(split["train_normal"])
    mols = [Chem.MolFromSmiles(s) for s in train_normal]
    n_normals = sum(1 for m in mols if m is not None)

    kept, rejects = [], []
    for i, rule in enumerate(raw_rules):
        if not _compile_ok(rule):
            rejects.append({"index": i, "name": rule.get("name"),
                             "reason": "not_compilable"})
            continue
        direction = _recover_direction(rule)
        if direction is None:
            rejects.append({"index": i, "name": rule.get("name"),
                             "reason": "direction_unrecoverable"})
            continue
        values = []
        for m in mols:
            if m is None:
                values.append(None)
                continue
            values.append(_eval_numeric(rule, m))
        firings, fire_info = _firing_from_values(rule, values)
        fired = sum(firings)
        rate = fired / n_normals if n_normals else 0.0
        if rate < FIRING_LO or rate > FIRING_HI:
            rejects.append({"index": i, "name": rule.get("name"),
                             "reason": f"fire_rate:{rate:.4f}"})
            continue
        r2 = dict(rule)
        r2["_filter"] = {
            "direction_resolved": direction,
            "fire_rate_train_normal": rate,
            "fired": fired,
            "n_train_normal": n_normals,
            "threshold": fire_info.get("threshold"),
            "threshold_kind": fire_info.get("threshold_kind"),
            "seed": FILTER_SEED,
        }
        kept.append(r2)

    result = {
        "task_key": label,
        "split_seed": FILTER_SEED,
        "firing_lo": FIRING_LO,
        "firing_hi": FIRING_HI,
        "n_train_normal": n_normals,
        "n_input_rules": len(raw_rules),
        "n_kept": len(kept),
        "n_rejected": len(rejects),
        "rejects": rejects[:50],
        "rules": kept,
        "source_rule_cache": str(cache_path),
        "generated_at": now(),
    }
    out_path.write_text(json.dumps(result, indent=2))
    return result


FILTER_JOBS = [
    ("BBBP",    None, "task0", "BBBP_task0"),
    ("ClinTox", 1,    "task1", "ClinTox_task1"),
    ("HIV",     None, "task0", "HIV_task0"),
] + [("Tox21", i, f"task{i}", f"Tox21_task{i}") for i in range(12)] \
  + [("SIDER", i, f"task{i}", f"SIDER_task{i}") for i in range(27)]



def cmd_filter(data_dir):
    global SPLITS_DIR, RULE_CACHE_DIR, RULEBANK_DIR
    SPLITS_DIR = data_dir
    RULE_CACHE_DIR = Path("rules") / "raw"
    RULEBANK_DIR = Path("rules") / "filtered"
    RULEBANK_DIR.mkdir(parents=True, exist_ok=True)
    for dataset, task_idx, task_key, label in FILTER_JOBS:
        try:
            res = build_filtered_rulebank(dataset, task_idx, task_key, label, force=True)
            print(f"{label}: kept {res['n_kept']}/{res['n_input_rules']} "
                  f"(rejected {res['n_rejected']})", flush=True)
        except FileNotFoundError as exc:
            print(f"{label}: skipped ({exc})", flush=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Filter the raw rule cache into rulebooks.")
    ap.add_argument("--data-dir", default="./data")
    args = ap.parse_args()
    cmd_filter(Path(args.data_dir))
