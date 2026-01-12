# preprocess/amazon.py
import pandas as pd


def load_amazon_csv(
    path: str,
    min_rating: float | None = None,
    keep_rating: bool = True
) -> pd.DataFrame:
    """
    读取 Amazon 采样后的 CSV（books_sample.csv / electronics_sample.csv）
    期望列：userId,itemId,timestamp,rating

    参数：
    - min_rating: 评分过滤阈值（例如 4.0）。None 表示不过滤
    - keep_rating: 是否保留 rating 列（后续可用于过滤/分析）
    返回：
    - DataFrame，至少包含 userId,itemId,timestamp（以及可选 rating）
    """

    df = pd.read_csv(path)
    expected = {"userId", "itemId", "timestamp", "rating"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Amazon CSV missing columns: {missing}. "
                         f"Expected columns: {expected}")

    # 可选：评分过滤（很多论文/工程里会只保留 positive feedback）
    if min_rating is not None:
        df = df[df["rating"] >= min_rating].copy()

    # ID 离散化（字符串 -> int），与 Steam 的 itemName 编码同理
    df["userId"] = df["userId"].astype("category").cat.codes
    df["itemId"] = df["itemId"].astype("category").cat.codes

    # timestamp 确保为 int
    df["timestamp"] = df["timestamp"].astype(int)

    # 输出列控制
    if keep_rating:
        out = df[["userId", "itemId", "timestamp", "rating"]]
    else:
        out = df[["userId", "itemId", "timestamp"]]

    print(f"[INFO] Amazon loaded: {path}, shape: {out.shape}, "
          f"min_rating={min_rating}")
    return out
