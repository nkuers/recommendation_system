import pandas as pd
from pathlib import Path
import kagglehub

# ------------------------------
# 1. 通用加载函数（MovieLens / Steam / Amazon）
# ------------------------------

def load_dataset(path: str):
    """
    通用数据集加载函数
    """
    df = pd.read_csv(path)
    print(f"[INFO] Loaded dataset: {path}")
    print(f"[INFO] Raw data shape: {df.shape}")
    return df


# ------------------------------
# 2. 时间排序
# ------------------------------

def sort_by_user_and_time(df: pd.DataFrame):
    """
    按 userId 和 timestamp 排序
    """
    return df.sort_values(by=["userId", "timestamp"], ascending=True)


# ------------------------------
# 3. Δt 时间差计算（第四周新增）
# ------------------------------

def add_delta_t(df: pd.DataFrame):
    """
    为每一条行为加入与上一条行为的时间差 Δt（秒）
    第一条行为的 Δt 记为 0
    """
    df["delta_t"] = df.groupby("userId")["timestamp"].diff().fillna(0)
    return df


# ------------------------------
# 4. 构建基本用户行为序列骨架（itemId + Δt）
# ------------------------------

def build_sequence_with_dt(df: pd.DataFrame):
    """
    返回格式：
        user_sequences[userId] = [
            (itemId1, Δt1),
            (itemId2, Δt2),
            ...
        ]
    """
    user_sequences = {}

    for uid, group in df.groupby("userId"):
        items = group["movieId"].tolist()
        dts = group["delta_t"].tolist()
        user_sequences[uid] = list(zip(items, dts))

    return user_sequences


# ------------------------------
# 5. 打印示例
# ------------------------------

def print_example(user_sequences: dict):
    first_user = list(user_sequences.keys())[0]
    print("[Example user sequence with Δt]")
    print(f"user {first_user}: {user_sequences[first_user][:10]} ...")


# ------------------------------
# 6. 主函数
# ------------------------------

def main():
    # 1. 加载数据集
    df = load_dataset("ml-latest-small/ratings.csv")

    # 2. 时间排序
    df = sort_by_user_and_time(df)

    # 3. Δt 计算（本周新增）
    df = add_delta_t(df)

    # 4. 构建序列（包含 itemId + Δt）
    seqs = build_sequence_with_dt(df)

    # 5. 打印示例
    print_example(seqs)


    # Download latest version
    path = kagglehub.dataset_download("nikdavis/steam-store-games")

    print("Path to dataset files:", path)


if __name__ == "__main__":
    main()
