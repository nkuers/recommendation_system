from pathlib import Path
import pandas as pd

PAIRS = [
    (
        "data/amazon/amazon-beauty-5core-subset/amazon-beauty-5core-subset.inter",
        "IDURL-main/data/Beauty.txt",
    ),
    (
        "data/amazon/amazon-electronics-5core-subset/amazon-electronics-5core-subset.inter",
        "IDURL-main/data/Electronics.txt",
    ),
]


def convert_one(src: Path, dst: Path):
    if not src.exists():
        raise FileNotFoundError(f"source not found: {src}")

    df = pd.read_csv(src, sep="\t")
    df.columns = [c.split(":")[0] for c in df.columns]
    req = {"user_id", "item_id", "timestamp"}
    miss = req.difference(df.columns)
    if miss:
        raise ValueError(f"missing columns in {src}: {sorted(miss)}")

    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    u_map = {u: i + 1 for i, u in enumerate(df["user_id"].unique())}
    i_map = {it: i + 1 for i, it in enumerate(df["item_id"].unique())}
    df["u"] = df["user_id"].map(u_map)
    df["i"] = df["item_id"].map(i_map)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for u, g in df.groupby("u", sort=False):
            seq = " ".join(str(x) for x in g["i"].tolist())
            f.write(f"{u} {seq}\n")

    print(f"[DONE] {dst}")
    print(f"  users={len(u_map)} items={len(i_map)} inters={len(df)}")


if __name__ == "__main__":
    for src_s, dst_s in PAIRS:
        convert_one(Path(src_s), Path(dst_s))
