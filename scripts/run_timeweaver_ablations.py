import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_GRID = ROOT / "scripts" / "run_grid.py"


DEFAULT_VARIANTS = ["base", "cont", "gate_discrete", "gate", "aug_only", "aug"]


def parse_list(value):
    return [x.strip() for x in value.split(",") if x.strip()]


DATASETS_ALL = [
    "ml-100k",
    "amazon-beauty-5core-subset",
    "amazon-beauty-5core-longseq",
    "amazon-electronics-5core-subset",
    "amazon-sports-5core-subset",
    "amazon-sports-5core-longseq",
    "amazon-toys-5core-subset",
    "amazon-toys-5core-longseq",
    "amazon-toys-5core-subset-50k",
]

DATASETS_ALL_FULL = [
    "ml-100k-full",
    "amazon-beauty-5core-subset-full",
    "amazon-beauty-5core-longseq-full",
    "amazon-electronics-5core-subset-full",
    "amazon-sports-5core-subset-full",
    "amazon-sports-5core-longseq-full",
    "amazon-toys-5core-subset-full",
    "amazon-toys-5core-longseq-full",
    "amazon-toys-5core-subset-50k-full",
]


def main():
    parser = argparse.ArgumentParser(
        description="Run all TimeWeaver ablations on one dataset via run_grid.py"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset key, or 'all' (uni100 set), or 'all-full' (full-sort set).",
    )
    parser.add_argument("--seeds", default="42,2023,2024")
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS), help="Comma-separated variants")
    parser.add_argument("--output", default=None, help="Shared output json path")
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--skip-if-done", action="store_true")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    variants = parse_list(args.variants)
    if not variants:
        raise ValueError("No variants provided.")

    key = args.dataset.strip().lower()
    if key == "all":
        datasets = DATASETS_ALL
    elif key == "all-full":
        datasets = DATASETS_ALL_FULL
    else:
        datasets = [args.dataset]

    output = args.output
    if output is None:
        safe_dataset = args.dataset.replace("/", "_")
        output = f"results/timeweaver_ablations_{safe_dataset}.json"

    for dataset in datasets:
        for variant in variants:
            cmd = [
                sys.executable,
                str(RUN_GRID),
                "--models",
                "timeweaver",
                "--datasets",
                dataset,
                "--timeweaver-variant",
                variant,
                "--seeds",
                args.seeds,
                "--device",
                args.device,
                "--output",
                output,
            ]
            if args.epochs_override is not None:
                cmd.extend(["--epochs-override", str(args.epochs_override)])
            if args.skip_if_done:
                cmd.append("--skip-if-done")
            if args.tag:
                cmd.extend(["--tag", args.tag])

            print(f"[RUN] timeweaver ablation variant={variant} dataset={dataset}")
            proc = subprocess.run(cmd, cwd=str(ROOT), text=True)
            if proc.returncode != 0:
                raise SystemExit(proc.returncode)

    print(f"[DONE] all variants finished. results: {output}")


if __name__ == "__main__":
    main()

