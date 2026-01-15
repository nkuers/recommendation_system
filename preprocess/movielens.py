# preprocess/movielens.py
import pandas as pd


def load_movielens(path: str) -> pd.DataFrame:
    """
    MovieLens-small 数据加载与字段规范化
    输出字段：
        userId: 连续整数 [0, num_users)
        itemId: 连续整数 [1, num_items]，0 预留给 padding
        timestamp: int
    """
    df = pd.read_csv(path)

    # 字段统一
    df = df.rename(columns={"movieId": "itemId"})
    df = df[["userId", "itemId", "timestamp"]]

    # 关键步骤：重新编码（而不是 astype(int)）
    df["userId"] = df["userId"].astype("category").cat.codes
    # itemId 从 1 开始，0 预留给 padding
    df["itemId"] = df["itemId"].astype("category").cat.codes + 1

    # 时间戳
    df["timestamp"] = df["timestamp"].astype(int)

    print(f"[INFO] MovieLens loaded: {df.shape}, "
          f"num_users={df['userId'].nunique()}, "
          f"num_items={df['itemId'].nunique()}")

    return df
