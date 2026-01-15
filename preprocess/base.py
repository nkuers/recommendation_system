# preprocess/base.py
from __future__ import annotations

import random
from typing import List, Dict, Tuple, Optional
import pandas as pd


# ------------------------------
# 1) 时间排序
# ------------------------------
def sort_by_user_and_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    按 userId + timestamp 升序排序
    """
    return df.sort_values(by=["userId", "timestamp"], ascending=True)


# ------------------------------
# 2) 构建用户序列 + Δt
# ------------------------------
def build_user_sequences(df: pd.DataFrame) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    输入 df 需至少包含: userId, itemId, timestamp
    输出:
      user_sequences[user] = [itemId...]
      user_delta_t[user]   = [0, dt2, dt3, ...]
    """
    user_sequences: Dict[int, List[int]] = {}
    user_delta_t: Dict[int, List[int]] = {}

    for user, group in df.groupby("userId", sort=False):
        items = group["itemId"].tolist()
        timestamps = group["timestamp"].tolist()
        if len(timestamps) == 0:
            continue
        delta_t = [0] + [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]

        user_sequences[int(user)] = items
        user_delta_t[int(user)] = delta_t

    return user_sequences, user_delta_t


# ------------------------------
# 3) 固定窗口切片
# ------------------------------
def slice_sequence_fixed_window(
    seq: List[int],
    delta_t_seq: List[int],
    window_size: int = 10
) -> List[Tuple[List[int], List[int]]]:
    """
    将 seq 和 delta_t_seq 按固定窗口切片
    注意：最后一个窗口可能不足 window_size（保留）
    """
    assert len(seq) == len(delta_t_seq), "seq and delta_t_seq must have same length"

    slices: List[Tuple[List[int], List[int]]] = []
    for start in range(0, len(seq), window_size):
        end = start + window_size
        slices.append((seq[start:end], delta_t_seq[start:end]))
    return slices


# ------------------------------
# 4) 行为过滤接口（占位，后续可扩展）
# ------------------------------
def filter_behavior(
    seq: List[int],
    delta_t_seq: List[int],
    hours_seq: Optional[List[int]] = None,
    min_rating: Optional[float] = None,
    **kwargs
) -> Tuple[List[int], List[int], Optional[List[int]]]:
    """
    行为过滤占位：
    - 目前不做任何过滤，只原样返回
    - 未来你可以在这里加入：按评分阈值、按行为类型、去重、截断等
    """
    return seq, delta_t_seq, hours_seq


# ------------------------------
# 5) Dataset 类（统一输出 seq_slices / delta_t_slices / hour_slices）
# ------------------------------
class Dataset:
    """
    把 user->序列 映射成可训练的窗口样本集合。

    输出属性：
      - user_ids:       List[int]
      - seq_slices:     List[List[int]]
      - delta_t_slices: List[List[int]]
      - hour_slices:    List[List[int]]   (可选，只有传 user_hours 才会有内容)
    """
    def __init__(
        self,
        user_sequences: Dict[int, List[int]],
        user_delta_t: Dict[int, List[int]],
        window_size: int = 10,
        user_hours: Optional[Dict[int, List[int]]] = None,
        min_rating: Optional[float] = None,
        **kwargs
    ):
        self.user_ids: List[int] = []
        self.seq_slices: List[List[int]] = []
        self.delta_t_slices: List[List[int]] = []
        self.hour_slices: List[List[int]] = []

        self.num_items = 0  # Initialize num_items

        # 为了稳定可复现：强制按 userId 排序遍历
        for user in sorted(user_sequences.keys()):
            seq = user_sequences[user]
            dt_seq = user_delta_t[user]

            hrs_seq = None
            if user_hours is not None:
                if user not in user_hours:
                    raise KeyError(f"user_hours missing userId={user}")
                hrs_seq = user_hours[user]
                if len(hrs_seq) != len(seq):
                    raise ValueError(f"Length mismatch for user {user}: "
                                     f"len(hours)={len(hrs_seq)} vs len(seq)={len(seq)}")

            # 可扩展过滤（目前不做）
            seq, dt_seq, hrs_seq = filter_behavior(seq, dt_seq, hrs_seq, min_rating=min_rating, **kwargs)

            # 切片（items & dt）
            slices = slice_sequence_fixed_window(seq, dt_seq, window_size)

            # 切片（hours）——关键：与 slices 同步切！
            hour_slices_for_user: Optional[List[List[int]]] = None
            if hrs_seq is not None:
                hour_slices_for_user = [hrs_seq[i:i + window_size] for i in range(0, len(hrs_seq), window_size)]
                if len(hour_slices_for_user) != len(slices):
                    raise ValueError(f"Hour slice count mismatch for user {user}: "
                                     f"{len(hour_slices_for_user)} vs {len(slices)}")

            # 汇总到全局列表
            for idx, (s, dt) in enumerate(slices):
                self.user_ids.append(user)
                self.seq_slices.append(s)
                self.delta_t_slices.append(dt)

                if hour_slices_for_user is not None:
                    self.hour_slices.append(hour_slices_for_user[idx])

        if len(self.seq_slices) == 0:
            self.num_items = 0
        else:
            self.num_items = max(max(seq) for seq in self.seq_slices) + 1  # 计算 num_items

        print(f"[INFO] Dataset prepared: {len(self.seq_slices)} sequences")

    def __len__(self):
        # print(f"[INFO] Dataset size: {len(self.seq_slices)}")  # 打印数据集大小
        return len(self.seq_slices)

    def __getitem__(self, idx):
        seq = self.seq_slices[idx]
        delta_t_seq = self.delta_t_slices[idx]
        hour_seq = self.hour_slices[idx] if len(self.hour_slices) > 0 else [0] * len(seq)  # 确保 hour_seq 始终存在

        if self.num_items <= 1:
            neg = 0
        else:
            neg = random.randint(1, self.num_items - 1)
            tries = 0
            while neg in seq:
                neg = random.randint(1, self.num_items - 1)  # 确保负样本不在序列中
                tries += 1
                if tries > 1000:
                    break

        # 正样本是序列中的最后一个元素
        y_pos = seq[-1]

        return seq, hour_seq, y_pos, neg






