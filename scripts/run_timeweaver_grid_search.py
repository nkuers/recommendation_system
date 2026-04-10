import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_list_int(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_list_float(value):
    return [float(x.strip()) for x in value.split(",") if x.strip()]

def parse_list(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_summary_key(key):
    parts = key.split("::")
    if len(parts) < 3:
        return None
    model = parts[0]
    dataset = parts[1]
    variant = parts[2]
    tag = "::".join(parts[3:]) if len(parts) > 3 else ""
    return model, dataset, variant, tag


def print_topk(summary_path, model, variant, metric, topk):
    if not summary_path.exists():
        print(f"[WARN] summary not found: {summary_path}")
        return
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    by_dataset = {}
    for key, metrics in data.items():
        parsed = parse_summary_key(key)
        if not parsed:
            continue
        key_model, dataset, key_variant, tag = parsed
        if key_model != model:
            continue
        if key_variant != variant:
            continue
        if metric not in metrics or "mean" not in metrics[metric]:
            continue
        by_dataset.setdefault(dataset, []).append((tag, metrics))

    if not by_dataset:
        print("[WARN] no matched entries in summary")
        return

    for dataset, rows in sorted(by_dataset.items()):
        rows.sort(key=lambda x: x[1][metric]["mean"], reverse=True)
        print(f"\n[TOP {topk}] {dataset} by {metric}")
        for i, (tag, m) in enumerate(rows[:topk], start=1):
            hit = m.get("hit@10", {}).get("mean", float("nan"))
            mrr = m.get("mrr@10", {}).get("mean", float("nan"))
            ndcg = m.get("ndcg@10", {}).get("mean", float("nan"))
            print(
                f"{i}. tag={tag or '(no-tag)'} | {metric}={m[metric]['mean']:.6f} "
                f"(std={m[metric]['std']:.6f}, n={m[metric]['n']}) | "
                f"hit@10={hit:.6f} mrr@10={mrr:.6f} ndcg@10={ndcg:.6f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Run TimeWeaver hyper-parameter grid search via run_grid.py.")
    parser.add_argument("--model", default="timeweaver", choices=["timeweaver"], help="Model to tune.")
    parser.add_argument("--datasets", required=True, help="Comma-separated datasets.")
    parser.add_argument("--seeds", default="42,2023,2024", help="Comma-separated seeds.")
    parser.add_argument("--device", default="gpu", help="gpu or cpu")
    parser.add_argument("--variant", default="aug", choices=["base", "cont", "gate_discrete", "gate", "aug_only", "aug"])
    parser.add_argument("--gate-hiddens", default="8,16", help="Comma-separated time_gate_hidden values.")
    parser.add_argument("--aug-strengths", default="0.1,0.2,0.3", help="Comma-separated time_aug_strength values.")
    parser.add_argument("--contrastive-weights", default="0.03,0.05,0.1", help="Comma-separated contrastive_weight values.")
    parser.add_argument("--aug-prob", type=float, default=1.0, help="time_aug_prob")
    parser.add_argument("--contrastive-temp", type=float, default=0.2, help="contrastive_temp")
    parser.add_argument("--ema-decays", default="0.05,0.1,0.2", help="Comma-separated ema_decay values (base).")
    parser.add_argument("--tcn-kernels", default="-1,5,7,9", help="Comma-separated tcn_kernel_size values (base).")
    parser.add_argument("--hidden-dropouts", default="0.2,0.5", help="Comma-separated hidden_dropout_prob values (base).")
    parser.add_argument("--num-layers", default="1,2", help="Comma-separated num_layers values (base).")
    parser.add_argument("--ffn-hiddens", default="256,512", help="Comma-separated ffn_hidden_size values (base).")
    parser.add_argument("--time-buckets", default="1000", help="Comma-separated time_bucket_size values (base).")
    parser.add_argument("--output", default="results/run_grid_timeweaver_tune.json", help="run_grid output path.")
    parser.add_argument("--skip-if-done", action="store_true", help="Pass --skip-if-done to run_grid.")
    parser.add_argument("--no-summary", action="store_true", help="Pass --no-summary to run_grid.")
    parser.add_argument("--topk", type=int, default=3, help="Print top-k after search.")
    parser.add_argument("--metric", default="ndcg@10", help="Primary metric for top-k ranking.")
    args = parser.parse_args()

    if args.variant == "base":
        ema_decays = parse_list_float(args.ema_decays)
        tcn_kernels = parse_list_int(args.tcn_kernels)
        hidden_dropouts = parse_list_float(args.hidden_dropouts)
        num_layers = parse_list_int(args.num_layers)
        ffn_hiddens = parse_list_int(args.ffn_hiddens)
        time_buckets = parse_list_int(args.time_buckets)
        combos = list(
            itertools.product(ema_decays, tcn_kernels, hidden_dropouts, num_layers, ffn_hiddens, time_buckets)
        )
    else:
        gate_hiddens = parse_list_int(args.gate_hiddens)
        aug_strengths = parse_list_float(args.aug_strengths)
        contrastive_weights = parse_list_float(args.contrastive_weights)
        combos = list(itertools.product(gate_hiddens, aug_strengths, contrastive_weights))
    print(f"[INFO] total combos: {len(combos)}")

    for idx, combo in enumerate(combos, start=1):
        if args.variant == "base":
            ema_decay, tcn_kernel, dropout, layers, ffn_hidden, time_bucket = combo
            tag = (
                f"e{str(ema_decay).replace('.', 'p')}_k{tcn_kernel}_d{str(dropout).replace('.', 'p')}"
                f"_l{layers}_f{ffn_hidden}_b{time_bucket}"
            )
        else:
            g, s, c = combo
            tag = f"g{g}_s{str(s).replace('.', 'p')}_c{str(c).replace('.', 'p')}"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_grid.py"),
            "--models",
            args.model,
            "--datasets",
            args.datasets,
            "--seeds",
            args.seeds,
            "--device",
            args.device,
            "--timeweaver-variant",
            args.variant,
            "--tag",
            tag,
            "--output",
            args.output,
        ]
        if args.variant == "base":
            cmd += [
                "--tw-ema-decay",
                str(ema_decay),
                "--tw-tcn-kernel",
                str(tcn_kernel),
                "--tw-hidden-dropout",
                str(dropout),
                "--tw-num-layers",
                str(layers),
                "--tw-ffn-hidden",
                str(ffn_hidden),
                "--tw-time-bucket",
                str(time_bucket),
            ]
        else:
            cmd += [
                "--tw-gate-hidden",
                str(g),
                "--tw-aug-strength",
                str(s),
                "--tw-aug-prob",
                str(args.aug_prob),
                "--tw-contrastive-weight",
                str(c),
                "--tw-contrastive-temp",
                str(args.contrastive_temp),
            ]
        if args.skip_if_done:
            cmd.append("--skip-if-done")
        if args.no_summary:
            cmd.append("--no-summary")

        print(f"[{idx}/{len(combos)}] run tag={tag}")
        subprocess.run(cmd, check=True, cwd=str(ROOT))

    print("[INFO] grid search finished")
    if not args.no_summary:
        out_path = ROOT / args.output
        summary_path = out_path.with_name(out_path.stem + "_summary.json")
        print_topk(summary_path, model=args.model, variant=args.variant, metric=args.metric, topk=args.topk)


if __name__ == "__main__":
    main()
