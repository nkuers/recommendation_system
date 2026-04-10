import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_INFO = {
    "ml-100k": {
        "src": Path("data/movielens/ml-100k/ml-100k.inter"),
        "dst_dir": Path("data/movielens/ml-100k-idurl"),
        "dst_name": "ml-100k-idurl",
    },
    "amazon-beauty-5core-subset": {
        "src": Path("data/amazon/amazon-beauty-5core-subset/amazon-beauty-5core-subset.inter"),
        "dst_dir": Path("data/amazon/amazon-beauty-5core-subset-idurl"),
        "dst_name": "amazon-beauty-5core-subset-idurl",
    },
    "amazon-electronics-5core-subset": {
        "src": Path("data/amazon/amazon-electronics-5core-subset/amazon-electronics-5core-subset.inter"),
        "dst_dir": Path("data/amazon/amazon-electronics-5core-subset-idurl"),
        "dst_name": "amazon-electronics-5core-subset-idurl",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare IDURL offline fields for RecBole datasets.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_INFO.keys()))
    parser.add_argument("--max-seq-len", type=int, default=32)
    parser.add_argument("--replace-ratio", type=float, default=0.2)
    parser.add_argument("--n-facet", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def normalize_columns(df):
    df = df.copy()
    df.columns = [c.split(":")[0] for c in df.columns]
    required = {"user_id", "item_id", "rating", "timestamp"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    return df


def build_train_mask(df):
    sizes = df.groupby("user_id")["item_id"].transform("size")
    pos = df.groupby("user_id").cumcount()
    # Match RecBole LS valid_and_test: leave last two interactions for valid/test when possible.
    train_cut = np.where(sizes >= 3, sizes - 2, np.maximum(sizes - 1, 0))
    return pos < train_cut


def make_newc_degree(df, train_df, n_facet):
    train_cnt = train_df["item_id"].value_counts()
    if train_cnt.empty:
        return np.zeros(len(df), dtype=np.int64)

    train_rarity = 1.0 / train_cnt.astype(float)
    train_rarity_values = train_rarity.to_numpy()
    # Learn novelty bins only from train interactions to avoid leakage.
    try:
        _, bin_edges = pd.qcut(train_rarity_values, q=n_facet, retbins=True, duplicates="drop")
    except Exception:
        # Fall back to uniform bins on train rarity range.
        lo = float(np.min(train_rarity_values))
        hi = float(np.max(train_rarity_values))
        if hi <= lo:
            return np.zeros(len(df), dtype=np.int64)
        bin_edges = np.linspace(lo, hi, num=n_facet + 1)

    rarity_map = train_rarity.to_dict()
    unseen_rarity = float(np.max(train_rarity_values)) + 1.0
    all_rarity = np.array([float(rarity_map.get(i, unseen_rarity)) for i in df["item_id"]], dtype=np.float64)

    if len(bin_edges) <= 2:
        return np.zeros(len(df), dtype=np.int64)

    labels = np.digitize(all_rarity, bin_edges[1:-1], right=True).astype(np.int64)
    max_label = min(n_facet - 1, len(bin_edges) - 2)
    labels = np.clip(labels, 0, max_label)
    return labels


def build_semantic_aug(df, train_df, max_seq_len, replace_ratio, seed):
    train_hist_map = train_df.groupby("user_id")["item_id"].apply(list).to_dict()
    item_pool = train_df["item_id"].unique()
    if len(item_pool) == 0:
        item_pool = df["item_id"].unique()

    sem_aug = []
    sem_aug_lengths = []
    rng = np.random.default_rng(seed)

    for row in df.itertuples(index=False):
        uid = row.user_id
        iid = row.item_id
        hist_clip = train_hist_map.get(uid, [])[-max_seq_len:]

        if not hist_clip:
            aug = [iid]
        else:
            aug = list(hist_clip)
            n_replace = max(1, int(len(aug) * replace_ratio))
            rep_pos = rng.choice(len(aug), size=n_replace, replace=False)
            rep_items = rng.choice(item_pool, size=n_replace, replace=True)
            for p, new_item in zip(rep_pos, rep_items):
                aug[p] = new_item

        sem_aug.append(" ".join(str(x) for x in aug))
        sem_aug_lengths.append(len(aug))

    return sem_aug, sem_aug_lengths


def main():
    args = parse_args()
    info = DATASET_INFO[args.dataset]
    src = info["src"]
    if not src.exists():
        raise SystemExit(f"source not found: {src}")

    df = pd.read_csv(src, sep="\t")
    df = normalize_columns(df)
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    df["item_id"] = df["item_id"].astype(str)
    train_mask = build_train_mask(df)
    train_df = df[train_mask].copy()

    df["newc_degree"] = make_newc_degree(df=df, train_df=train_df, n_facet=args.n_facet)
    sem_aug, sem_aug_lengths = build_semantic_aug(
        df=df,
        train_df=train_df,
        max_seq_len=args.max_seq_len,
        replace_ratio=args.replace_ratio,
        seed=args.seed,
    )
    df["sem_aug"] = sem_aug
    df["sem_aug_lengths"] = sem_aug_lengths

    dst_dir = info["dst_dir"]
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{info['dst_name']}.inter"

    out_cols = [
        "user_id",
        "item_id",
        "rating",
        "timestamp",
        "newc_degree",
        "sem_aug",
        "sem_aug_lengths",
    ]
    header = (
        "user_id:token\titem_id:token\trating:float\ttimestamp:float\t"
        "newc_degree:float\tsem_aug:token_seq\tsem_aug_lengths:float"
    )
    with open(dst, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        df[out_cols].to_csv(f, sep="\t", header=False, index=False)

    print(f"[INFO] saved: {dst}")
    print(f"[INFO] interactions: {len(df)}")
    print(f"[INFO] users: {df['user_id'].nunique()}, items: {df['item_id'].nunique()}")


if __name__ == "__main__":
    main()
