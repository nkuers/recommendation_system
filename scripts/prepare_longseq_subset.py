import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Build long-sequence subset from an existing .inter file.")
    parser.add_argument(
        "--src",
        default="data/amazon/amazon-beauty-5core-subset/amazon-beauty-5core-subset.inter",
        help="Source .inter path",
    )
    parser.add_argument(
        "--out-dir",
        default="data/amazon/amazon-beauty-5core-longseq",
        help="Output dataset directory",
    )
    parser.add_argument(
        "--out-name",
        default="amazon-beauty-5core-longseq",
        help="Output dataset name (used as .inter filename stem)",
    )
    parser.add_argument("--min-len", type=int, default=30, help="Keep users with sequence length >= min-len")
    args = parser.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out_dir)
    out_name = args.out_name
    out_path = out_dir / f"{out_name}.inter"

    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    df = pd.read_csv(src, sep="\t")
    raw_cols = list(df.columns)
    col_map = {c: c.split(":")[0] for c in raw_cols}
    df = df.rename(columns=col_map)
    required = {"user_id", "item_id", "rating", "timestamp"}
    miss = required.difference(set(df.columns))
    if miss:
        raise ValueError(f"Missing required columns: {sorted(miss)}")

    # Stable temporal order.
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df["timestamp"] = df["timestamp"].astype("int64")
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    user_len = df.groupby("user_id").size()
    keep_users = user_len[user_len >= args.min_len].index
    sub = df[df["user_id"].isin(keep_users)].copy()

    # Keep original RecBole header style.
    out_dir.mkdir(parents=True, exist_ok=True)
    out_cols = [
        ("user_id", "token"),
        ("item_id", "token"),
        ("rating", "float"),
        ("timestamp", "float"),
    ]
    out_df = pd.DataFrame(
        {
            f"{name}:{dtype}": sub[name]
            for name, dtype in out_cols
        }
    )
    out_df.to_csv(out_path, sep="\t", index=False)

    def _stats(d):
        g = d.groupby("user_id").size()
        return {
            "users": int(g.shape[0]),
            "items": int(d["item_id"].nunique()),
            "inters": int(d.shape[0]),
            "seq_mean": float(g.mean()) if g.shape[0] else 0.0,
            "seq_median": float(g.median()) if g.shape[0] else 0.0,
            "seq_p90": float(g.quantile(0.9)) if g.shape[0] else 0.0,
            "seq_max": int(g.max()) if g.shape[0] else 0,
        }

    print("[SRC]", _stats(df))
    print("[SUB]", _stats(sub))
    print(f"[DONE] wrote: {out_path}")


if __name__ == "__main__":
    main()
