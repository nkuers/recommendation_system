import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def plot_time_aug_prob(output_dir: Path) -> Path:
    x = [0.05, 0.10, 0.15, 0.30]
    hit = [0.022500, 0.024033, 0.025233, 0.024433]
    ndcg = [0.010267, 0.011400, 0.012200, 0.011567]
    mrr = [0.006533, 0.007633, 0.008233, 0.007633]

    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(x, hit, marker="o", linewidth=2, markersize=7, label="Hit@10")
    plt.plot(x, ndcg, marker="s", linewidth=2, markersize=7, label="NDCG@10")
    plt.plot(x, mrr, marker="D", linewidth=2, markersize=7, label="MRR@10")
    plt.axvline(x=0.15, color="gray", linestyle="--", linewidth=1)

    for xv, yv in zip(x, hit):
        plt.text(xv, yv + 0.00045, f"{yv:.4f}", ha="center", va="bottom", fontsize=10)
    for xv, yv in zip(x, ndcg):
        plt.text(xv, yv + 0.00035, f"{yv:.4f}", ha="center", va="bottom", fontsize=10)
    for xv, yv in zip(x, mrr):
        plt.text(xv, yv + 0.00025, f"{yv:.4f}", ha="center", va="bottom", fontsize=10)

    plt.xlabel("time_aug_prob", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.xticks(x, [f"{v:.2f}" for v in x], fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylim(0.005, 0.0265)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(frameon=True, fontsize=11, loc="upper right")
    plt.tight_layout()

    out = output_dir / "time_aug_prob_sensitivity.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_time_gate_hidden(output_dir: Path) -> Path:
    x = [8, 16, 32, 64]
    ndcg = [0.010433, 0.010633, 0.010600, 0.010300]

    colors = ["#7da0fa", "#2f5fd0", "#7da0fa", "#7da0fa"]

    plt.figure(figsize=(7.5, 5), dpi=300)
    bars = plt.bar(range(len(x)), ndcg, color=colors, width=0.6, edgecolor="black", linewidth=0.8)

    for bar, yv in zip(bars, ndcg):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yv + 0.00008,
            f"{yv:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.xlabel("time_gate_hidden", fontsize=12)
    plt.ylabel("NDCG@10", fontsize=12)
    plt.xticks(range(len(x)), [str(v) for v in x], fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylim(0.0100, 0.0109)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    out = output_dir / "time_gate_hidden_sensitivity.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot parameter sensitivity figures for thesis.")
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Directory to save figures. Default: figures",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p1 = plot_time_aug_prob(output_dir)
    p2 = plot_time_gate_hidden(output_dir)

    print(f"saved: {p1}")
    print(f"saved: {p2}")


if __name__ == "__main__":
    main()
