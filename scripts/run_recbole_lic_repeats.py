# scripts/run_recbole_lic_repeats.py
import json
from pathlib import Path
from statistics import mean, stdev

import torch
from recbole.quick_start import run_recbole


_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat


SEEDS = [42, 2023, 2024]
METRICS = ["hit@10", "ndcg@10", "mrr@10"]


def main():
    runs = []
    for seed in SEEDS:
        print(f"\n[INFO] Running seed={seed}")
        result = run_recbole(
            model="LICRec",
            dataset="ml-100k",
            config_file_list=[
                "recbole_config/ml-100k.yaml",
                "recbole_config/lic.yaml",
            ],
            config_dict={"seed": seed},
        )
        best_test = result.get("test_result", {})
        best_test_norm = {str(k).lower(): v for k, v in best_test.items()}
        run_metrics = {k: best_test_norm.get(k) for k in METRICS}
        run_metrics["seed"] = seed
        runs.append(run_metrics)

    summary = {"runs": runs, "mean": {}, "std": {}}
    for k in METRICS:
        vals = [r[k] for r in runs if r.get(k) is not None]
        summary["mean"][k] = mean(vals) if vals else None
        summary["std"][k] = stdev(vals) if len(vals) > 1 else 0.0

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "recbole_lic_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("\n[SUMMARY]")
    for k in METRICS:
        m = summary["mean"][k]
        s = summary["std"][k]
        if m is None:
            print(f"{k}: mean=None, std=None")
        else:
            print(f"{k}: mean={m:.6f}, std={s:.6f}")
    print(f"[INFO] Saved to {out_path}")


if __name__ == "__main__":
    main()
