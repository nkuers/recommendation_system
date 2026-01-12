import pandas as pd
from typing import List, Dict


# ------------------------------
# 1. 数据加载与字段统一
# ------------------------------
def load_dataset(path: str, dataset_name: str = "MovieLens") -> pd.DataFrame:
    """
    通用数据集加载函数
    dataset_name: 用于不同数据集字段处理
    """
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {dataset_name} dataset: {path}, shape: {df.shape}")

    # 字段统一
    if dataset_name.lower() == "movielens":
        df = df.rename(columns={"movieId": "itemId"})
    elif dataset_name.lower() == "steam":
        df = df.rename(columns={"gameId": "itemId"})
    elif dataset_name.lower() == "amazon":
        df = df.rename(columns={"asin": "itemId",
                                "reviewerID": "userId",
                                "unixReviewTime": "timestamp"})

    # 类型转换
    df["userId"] = df["userId"].astype(int)
    df["itemId"] = df["itemId"].astype(int)
    df["timestamp"] = df["timestamp"].astype(int)

    return df


# ------------------------------
# 2. 时间排序
# ------------------------------
def sort_by_user_and_time(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(by=["userId", "timestamp"], ascending=True)


# ------------------------------
# 3. 用户序列 + Δt 构建
# ------------------------------
def build_user_sequences(df: pd.DataFrame):
    """
    构建基础用户序列和 Δt
    """
    user_sequences = {}
    user_delta_t = {}

    for user, group in df.groupby("userId"):
        items = group["itemId"].tolist()
        timestamps = group["timestamp"].tolist()
        # Δt = 当前行为 - 上一行为
        delta_t = [0] + [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]

        user_sequences[user] = items
        user_delta_t[user] = delta_t

    return user_sequences, user_delta_t


# ------------------------------
# 4. 固定窗口切片
# ------------------------------
def slice_sequence_fixed_window(seq: List[int], delta_t_seq: List[int], window_size: int = 10):
    slices = []
    for start in range(0, len(seq), window_size):
        end = start + window_size
        slices.append((seq[start:end], delta_t_seq[start:end]))
    return slices


# ------------------------------
# 5. 行为过滤（可扩展）
# ------------------------------
def filter_behavior(seq: List[int], delta_t_seq: List[int], min_rating: int = None):
    """
    可按评分/行为类型过滤，当前仅返回原序列
    """
    return seq, delta_t_seq


# ------------------------------
# 6. Dataset 类
# ------------------------------
class Dataset:
    def __init__(self, user_sequences: Dict[int, List[int]], user_delta_t: Dict[int, List[int]], window_size: int = 10):
        self.user_ids = []
        self.seq_slices = []
        self.delta_t_slices = []

        for user in user_sequences:
            seq = user_sequences[user]
            delta_t_seq = user_delta_t[user]
            seq, delta_t_seq = filter_behavior(seq, delta_t_seq)
            slices = slice_sequence_fixed_window(seq, delta_t_seq, window_size)
            for s, dt in slices:
                self.user_ids.append(user)
                self.seq_slices.append(s)
                self.delta_t_slices.append(dt)

        print(f"[INFO] Dataset prepared: {len(self.seq_slices)} sequences")


# ------------------------------
# 7. 示例 pipeline
# ------------------------------
def main():
    # MovieLens 示例
    df_ml = load_dataset("ml-latest-small/ratings.csv", "MovieLens")
    df_ml_sorted = sort_by_user_and_time(df_ml)
    user_seq, user_dt = build_user_sequences(df_ml_sorted)
    dataset_ml = Dataset(user_seq, user_dt, window_size=10)

    print(f"User {dataset_ml.user_ids[0]} first window items: {dataset_ml.seq_slices[0]}")
    print(f"Delta_t: {dataset_ml.delta_t_slices[0]}")


if __name__ == "__main__":
    main()
