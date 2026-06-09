import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import data
from model import score_task

SPLIT_RE = re.compile(r"([A-Za-z0-9]+)_(task\d+)_seed(\d+)\.json$")
METRICS = ["auroc_ranksum", "auroc_graph_only", "auroc_rule_only"]


def enumerate_jobs(seeds, datasets):
    jobs = []
    for p in sorted(data.DATA_DIR.glob("*.json")):
        m = SPLIT_RE.match(p.name)
        if not m:
            continue
        ds, tk, sd = m.group(1), m.group(2), int(m.group(3))
        if seeds and sd not in seeds:
            continue
        if datasets and ds not in datasets:
            continue
        jobs.append((ds, tk, sd))
    return jobs


def aggregate(rows):
    """Per dataset, average over tasks within each seed then over seeds."""
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)
    out = {}
    for ds, rs in by_ds.items():
        seeds = sorted({r["seed"] for r in rs})
        block = {}
        for metric in METRICS:
            per_seed = []
            for sd in seeds:
                vals = [r[metric] for r in rs if r["seed"] == sd and r[metric] == r[metric]]
                if vals:
                    per_seed.append(float(np.mean(vals)))
            if per_seed:
                block[metric] = {"mean": float(np.mean(per_seed)),
                                 "std": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0}
        out[ds] = block
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--rules", default=None,
                    help="rulebook dir (e.g. rules/generic_rules/k_match for the generic ablation)")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.data_dir:
        data.set_data_dir(args.data_dir)
    if args.rules:
        data.set_rules_dir(args.rules)

    jobs = enumerate_jobs(args.seeds, args.datasets)
    print(f"[chemrulead] data_dir={data.DATA_DIR}  jobs={len(jobs)}", flush=True)

    banks, rows = {}, []
    for i, (ds, tk, sd) in enumerate(jobs, start=1):
        if (ds, tk) not in banks:
            try:
                banks[(ds, tk)] = data.load_rulebook(ds, tk)
            except FileNotFoundError as exc:
                banks[(ds, tk)] = None
                print(f"[skip] {ds}/{tk}: {exc}", flush=True)
        if banks[(ds, tk)] is None:
            continue
        res = score_task(ds, tk, sd, banks[(ds, tk)])
        rows.append(res)
        print(f"[{i}/{len(jobs)}] {ds}/{tk} seed={sd}  graph={res['auroc_graph_only']:.4f}  "
              f"rule={res['auroc_rule_only']:.4f}  chemrulead={res['auroc_ranksum']:.4f}", flush=True)

    summary = aggregate(rows)
    print("\n" + "=" * 56)
    print(f"{'dataset':<10}{'graph':>10}{'rule':>10}{'ChemRuleAD':>12}")
    for ds in sorted(summary):
        blk = summary[ds]
        g = blk.get("auroc_graph_only", {}).get("mean", float("nan"))
        r = blk.get("auroc_rule_only", {}).get("mean", float("nan"))
        c = blk.get("auroc_ranksum", {}).get("mean", float("nan"))
        print(f"{ds:<10}{g:>10.4f}{r:>10.4f}{c:>12.4f}")
    print("=" * 56)

    if args.out:
        payload = {"per_dataset": summary,
                   "per_run": [{k: v for k, v in r.items() if k != "edge_stats"} for r in rows]}
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"[chemrulead] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
