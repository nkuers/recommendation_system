# experiments/train_hour_baseline.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from preprocess.movielens import load_movielens
from preprocess.base import sort_by_user_and_time, build_user_sequences, Dataset
from models.hour_seq import (
    SeqTrainDatasetWithHour,
    collate_pad_items_hours,
    MeanSeqHourModel,
    bpr_loss
)


def get_num_items(user_seq: dict) -> int:
    mx = 0
    for _, items in user_seq.items():
        if items:
            mx = max(mx, max(items))
    return mx + 1


def extract_hours_from_df(df):
    """
    df 已经按 userId,timestamp 排序
    返回 user -> hour 序列（长度与 user_seq 对齐）
    hour = (timestamp // 3600) % 24  (Unix秒)
    """
    df = df.copy()
    df["hour"] = ((df["timestamp"] // 3600) % 24).astype(int)
    user_hours = df.groupby("userId")["hour"].apply(list).to_dict()
    return user_hours


def main():
    # 1) 加载 & 排序
    df = load_movielens("../data/movielens/ratings.csv")
    df = sort_by_user_and_time(df)

    # 2) 用户 item 序列 + dt
    user_seq, user_dt = build_user_sequences(df)

    # 3) 提取 hour 序列（user -> list[hour]）
    user_hours = extract_hours_from_df(df)

    # 4) Dataset：在内部同步切 seq/dt/hour（彻底避免错配）
    window_size = 10
    ds = Dataset(user_seq, user_dt, window_size=window_size, user_hours=user_hours)

    # Dataset 内已经生成好 hour_slices
    hour_slices = ds.hour_slices
    assert len(hour_slices) == len(ds.seq_slices), "hour_slices 与 seq_slices 数量不一致（不应发生）"

    # 5) num_items
    num_items = get_num_items(user_seq)
    print("[INFO] num_items =", num_items)

    # 6) 训练数据
    train_data = SeqTrainDatasetWithHour(ds.seq_slices, hour_slices, num_items=num_items)
    loader = DataLoader(
        train_data,
        batch_size=256,
        shuffle=True,
        collate_fn=collate_pad_items_hours
    )

    # 7) 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MeanSeqHourModel(num_items=num_items, emb_dim=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 8) 训练
    model.train()
    for epoch in range(3):
        total = 0.0
        for items_pad, hours_pad, mask, pos, neg in loader:
            items_pad = items_pad.to(device)
            hours_pad = hours_pad.to(device)
            mask = mask.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            pos_s, neg_s = model(items_pad, hours_pad, mask, pos, neg)
            loss = bpr_loss(pos_s, neg_s)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        # print(f"[Epoch {epoch}] loss={total/len(loader):.4f}")
        alpha = F.softplus(model.raw_alpha).item()
        print(f"[Epoch {epoch}] loss={total / len(loader):.4f}, alpha={alpha:.4f}")


if __name__ == "__main__":
    main()
