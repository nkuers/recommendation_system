# preprocess/steam.py
import pandas as pd


def load_steam(path: str) -> pd.DataFrame:
    """
    读取 Steam-200k（CSV逗号分隔版本）
    原始列：
      userId, game, action, value, timestamp_raw(通常为0)
    注意：原数据基本没有真实 timestamp，这里用用户内行为顺序作为伪 timestamp。
    """
    df = pd.read_csv(
        path,
        header=None,
        names=["userId", "game", "action", "value", "timestamp_raw"],
        engine="python"   # 允许带引号的逗号分隔更稳
    )

    # 清理 action
    df["action"] = df["action"].astype(str).str.strip().str.lower()

    # 只保留 play 行为
    df = df[df["action"] == "play"].copy()

    # 连续编码
    df["userId"] = df["userId"].astype("category").cat.codes
    df["itemId"] = df["game"].astype("category").cat.codes

    # 伪 timestamp：同一用户内按出现顺序编号
    df["timestamp"] = df.groupby("userId").cumcount().astype(int)

    df = df[["userId", "itemId", "timestamp"]]

    print(f"[INFO] Steam loaded: {df.shape}, "
          f"num_users={df['userId'].nunique()}, "
          f"num_items={df['itemId'].nunique()}")
    return df
