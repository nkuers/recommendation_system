# preprocess/steam.py
import pandas as pd


def load_steam(path: str) -> pd.DataFrame:
    """
    Steam-200k (CSV 逗号分隔) - 方案B：基于 playtime 累积构造软 timestamp
    输出三列：userId, itemId, timestamp(int)，itemId 从 1 开始（0 padding）
    """
    df = pd.read_csv(
        path,
        header=None,
        names=["userId", "game", "action", "value", "timestamp_raw"],
        engine="python"
    )

    df["action"] = df["action"].astype(str).str.strip().str.lower()
    df = df[df["action"] == "play"].copy()

    # 连续编码
    df["userId"] = df["userId"].astype("category").cat.codes
    # itemId 从 1 开始，0 预留给 padding
    df["itemId"] = df["game"].astype("category").cat.codes + 1

    # playtime -> numeric
    df["playtime"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)

    # 保留原始行顺序作为用户内事件顺序（稳定）
    df = df.reset_index(drop=False).rename(columns={"index": "row_id"})
    df = df.sort_values(["userId", "row_id"], kind="mergesort")

    # 方案B：同一用户内 playtime 累积作为软 timestamp
    df["timestamp"] = df.groupby("userId")["playtime"].cumsum()

    # 统一 timestamp 为 int（便于 Δt 与 bucket）
    df["timestamp"] = df["timestamp"].round().astype(int)

    out = df[["userId", "itemId", "timestamp"]].copy()

    print(f"[INFO] Steam loaded: {out.shape}, "
          f"num_users={out['userId'].nunique()}, "
          f"num_items={out['itemId'].nunique()}")
    return out
