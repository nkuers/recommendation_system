import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from experiments.run_comparative_experiment import set_seed, train_and_evaluate
DEFAULT_MODELS = ["mfseq", "dt_attn", "dt_seq", "hour", "interest_clock", "lic_v1"]
DEFAULT_DATASETS = ["movielens", "amazon_books", "amazon_electronics", "steam"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run recommendation experiments.")
    parser.add_argument("--dataset", default="all",
                        choices=DEFAULT_DATASETS + ["all"],
                        help="Dataset to run (or 'all').")
    parser.add_argument("--model", default="all",
                        choices=DEFAULT_MODELS + ["all"],
                        help="Model to run (or 'all').")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Number of repeats per model/dataset.")
    parser.add_argument("--output", type=str, default="",
                        help="Optional output json path.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    datasets = DEFAULT_DATASETS if args.dataset == "all" else [args.dataset]
    models = DEFAULT_MODELS if args.model == "all" else [args.model]

    results = {}
    for dataset_name in datasets:
        results[dataset_name] = {}
        for model_type in models:
            print(f"\n{'='*60}")
            print(f"Dataset={dataset_name}, Model={model_type}")
            print(f"{'='*60}")
            try:
                run_metrics = {"HR@10": [], "NDCG@10": [], "MRR": [], "AUC": []}
                for rep in range(args.repeats):
                    run_seed = args.seed + rep
                    set_seed(run_seed)
                    hr_k, ndcg_k, mrr, auc = train_and_evaluate(
                        dataset_name,
                        model_type,
                        num_epochs=None,
                        window_size=10
                    )
                    run_metrics["HR@10"].append(hr_k)
                    run_metrics["NDCG@10"].append(ndcg_k)
                    run_metrics["MRR"].append(mrr)
                    run_metrics["AUC"].append(auc)

                results[dataset_name][model_type] = {
                    "runs": run_metrics,
                    "mean": {k: mean(v) for k, v in run_metrics.items()},
                    "std": {k: (stdev(v) if len(v) > 1 else 0.0) for k, v in run_metrics.items()}
                }
            except Exception as exc:
                print(f"[ERROR] Failed: {exc}")
                results[dataset_name][model_type] = {"error": str(exc)}

    output_path = args.output
    if not output_path:
        out_dir = ROOT / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        if len(datasets) == 1 and len(models) == 1:
            output_path = str(out_dir / f"{datasets[0]}_{models[0]}.json")
        else:
            output_path = str(out_dir / "summary.json")

    if Path(output_path).name == "summary.json" and Path(output_path).exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = {}

    for dataset_name, dataset_results in results.items():
        existing[dataset_name] = dataset_results

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, sort_keys=True)

    if Path(output_path).name == "summary.json":
        out_dir = Path(output_path).parent
        for dataset_name, dataset_results in results.items():
            dataset_path = out_dir / f"{dataset_name}.json"
            with open(dataset_path, "w", encoding="utf-8") as f:
                json.dump({dataset_name: dataset_results}, f, indent=2, sort_keys=True)
    print(f"[INFO] Results saved to {output_path}")


if __name__ == "__main__":
    main()
