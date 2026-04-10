# preprocess/movielens_100k.py
import pandas as pd


def load_movielens_100k(path: str) -> pd.DataFrame:
    """
    MovieLens 100K 数据加载与字段规范化
    u.data 格式: userId, itemId, rating, timestamp (tab 分隔)
    输出字段：
        userId: 连续整数 [0, num_users)
        itemId: 连续整数 [1, num_items]，0 预留给 padding
        timestamp: int
    """
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["userId", "itemId", "rating", "timestamp"]
    )

    df = df[["userId", "itemId", "timestamp"]]

    df["userId"] = df["userId"].astype("category").cat.codes
    df["itemId"] = df["itemId"].astype("category").cat.codes + 1
    df["timestamp"] = df["timestamp"].astype(int)

    print(f"[INFO] MovieLens-100K loaded: {df.shape}, "
          f"num_users={df['userId'].nunique()}, "
          f"num_items={df['itemId'].nunique()}")

    return df
