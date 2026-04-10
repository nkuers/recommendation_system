import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASETS = {
    "beauty_longseq": Path("data/amazon/amazon-beauty-5core-longseq/amazon-beauty-5core-longseq.inter"),
    "sports_longseq": Path("data/amazon/amazon-sports-5core-longseq/amazon-sports-5core-longseq.inter"),
    "toys_longseq": Path("data/amazon/amazon-toys-5core-longseq/amazon-toys-5core-longseq.inter"),
}


def load_intervals(inter_path: Path) -> np.ndarray:
    user_ts = {}
    with inter_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        uid_idx = header.index("user_id:token")
        ts_idx = header.index("timestamp:float")
        for row in reader:
            if not row:
                continue
            uid = row[uid_idx]
            ts = float(row[ts_idx])
            user_ts.setdefault(uid, []).append(ts)

    intervals = []
    for timestamps in user_ts.values():
        timestamps.sort()
        if len(timestamps) < 2:
            continue
        diffs = np.diff(np.asarray(timestamps, dtype=np.float64))
        diffs = np.maximum(diffs, 0.0)
        intervals.extend(diffs.tolist())
    return np.asarray(intervals, dtype=np.float64)


def summarize_intervals(values: np.ndarray) -> dict:
    if values.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def save_csv(rows: list, out_path: Path) -> None:
    fieldnames = ["dataset", "count", "mean", "median", "std", "min", "p25", "p50", "p75", "p90", "p95", "max"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_boxplot(intervals_by_dataset: dict, out_path: Path) -> None:
    labels = list(intervals_by_dataset.keys())
    log_values = [np.log1p(v) for v in intervals_by_dataset.values()]

    plt.figure(figsize=(8, 5), dpi=300)
    plt.boxplot(log_values, labels=labels, showfliers=False)
    plt.ylabel("log(1 + interval)")
    plt.xlabel("Dataset")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze time interval distributions for longseq datasets.")
    parser.add_argument("--output-dir", default="results/time_interval_analysis", help="Directory for csv and plot output.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    intervals_by_dataset = {}

    for dataset, path in DATASETS.items():
        values = load_intervals(path)
        intervals_by_dataset[dataset] = values
        stats = summarize_intervals(values)
        row = {"dataset": dataset}
        row.update(stats)
        rows.append(row)

    csv_path = output_dir / "time_interval_stats.csv"
    fig_path = output_dir / "time_interval_boxplot.png"

    save_csv(rows, csv_path)
    plot_boxplot(intervals_by_dataset, fig_path)

    print(f"saved: {csv_path}")
    print(f"saved: {fig_path}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
