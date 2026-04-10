# scripts/prepare_bmlp_ml100k.py
from pathlib import Path
import argparse

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare ml-100k for BMLP.")
    parser.add_argument("--min_purchase_user", type=int, default=0, help="Min purchase count per user")
    parser.add_argument("--min_purchase_item", type=int, default=0, help="Min purchase count per item")
    parser.add_argument("--min_user_total", type=int, default=0, help="Min total interactions per user")
    return parser.parse_args()


def main():
    args = parse_args()
    src = Path("data/movielens/ml-100k/ml-100k.inter")
    dst_dir = Path("data/movielens/ml-100k-bmlp")
    dst = dst_dir / "ml-100k-bmlp.inter"

    if not src.exists():
        raise SystemExit(f"source not found: {src}")

    df = pd.read_csv(src, sep="\t")
    # normalize column names from RecBole format (e.g., rating:float)
    df.columns = [c.split(":")[0] for c in df.columns]
    if "rating" not in df.columns:
        raise SystemExit(f"rating column not found in {src}")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])

    # Behavior mapping: rating >= 4 -> purchase(1), else auxiliary(2)
    df["behavior_type"] = (df["rating"] >= 4).astype(int).map({1: 1, 0: 2})

    if args.min_purchase_user > 0 or args.min_purchase_item > 0:
        purchase = df[df["behavior_type"] == 1]
        if args.min_purchase_user > 0:
            user_buy_cnt = purchase.groupby("user_id").size()
            valid_users = set(user_buy_cnt[user_buy_cnt >= args.min_purchase_user].index)
            df = df[df["user_id"].isin(valid_users)]
        if args.min_purchase_item > 0:
            item_buy_cnt = purchase.groupby("item_id").size()
            valid_items = set(item_buy_cnt[item_buy_cnt >= args.min_purchase_item].index)
            df = df[df["item_id"].isin(valid_items)]

    if args.min_user_total > 0:
        user_total_cnt = df.groupby("user_id").size()
        valid_users = set(user_total_cnt[user_total_cnt >= args.min_user_total].index)
        df = df[df["user_id"].isin(valid_users)]

    # Order by time for reproducibility (RecBole will re-order by time)
    df = df.sort_values(["user_id", "timestamp"])

    dst_dir.mkdir(parents=True, exist_ok=True)
    df = df[["user_id", "item_id", "rating", "timestamp", "behavior_type"]]
    header = "user_id:token\titem_id:token\trating:float\ttimestamp:float\tbehavior_type:token"
    with open(dst, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        df.to_csv(f, sep="\t", header=False, index=False)

    print(f"[INFO] saved: {dst}")
    print(f"[INFO] interactions: {len(df)}")
    print(f"[INFO] users: {df['user_id'].nunique()}, items: {df['item_id'].nunique()}")


if __name__ == "__main__":
    main()
