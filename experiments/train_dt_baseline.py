# experiments/train_dt_baseline.py
import math
import torch
from torch.utils.data import DataLoader

from preprocess.movielens import load_movielens
from preprocess.base import sort_by_user_and_time, build_user_sequences, Dataset
from models.dt_seq import (
    SeqTrainDatasetWithDT,
    collate_pad_items_dt,
    MeanSeqDTModel,
    bpr_loss
)


def get_num_items(user_seq: dict) -> int:
    mx = 0
    for _, items in user_seq.items():
        if items:
            mx = max(mx, max(items))
    return mx + 1


def dt_to_bucket(dt_seconds: int) -> int:
    """
    简化版 s(Δt)：log1p 压缩 + 分桶
    返回 bucket id（从 1 开始；0 留给 padding）
    """
    # 负数/异常保护
    if dt_seconds is None or dt_seconds < 0:
        dt_seconds = 0

    # log 压缩
    x = math.log1p(dt_seconds)

    # 手工阈值分桶（可调，后续做参数实验）
    # bucket 0: padding
    # bucket 1..N: 不同时间尺度
    # 这些阈值大致对应：秒级->分钟->小时->天->周->更久
    bins = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]  # 可改
    for i in range(1, len(bins)):
        if x <= bins[i]:
            return i
    return len(bins)  # 最后一个桶


def build_dt_bucket_slices(delta_t_slices, pad_bucket_id=0):
    """
    把 Dataset.delta_t_slices 转成 bucket 序列
    """
    out = []
    for dt_seq in delta_t_slices:
        buckets = []
        for dt in dt_seq:
            b = dt_to_bucket(int(dt))
            buckets.append(b)
        out.append(buckets)
    return out


def main():
    # 1) 加载 & 排序
    df = load_movielens("../data/movielens/ratings.csv")
    df = sort_by_user_and_time(df)

    # 2) 用户 item 序列 + dt
    user_seq, user_dt = build_user_sequences(df)

    # 3) Dataset 切片（得到 seq_slices / delta_t_slices）
    window_size = 10
    ds = Dataset(user_seq, user_dt, window_size=window_size)

    # 4) Δt -> bucket_slices
    dt_bucket_slices = build_dt_bucket_slices(ds.delta_t_slices)

    assert len(dt_bucket_slices) == len(ds.seq_slices)

    # 5) num_items / num_dt_buckets
    num_items = get_num_items(user_seq)
    num_dt_buckets = max(max(x) for x in dt_bucket_slices) + 1  # +1 because bucket id includes max
    print("[INFO] num_items =", num_items)
    print("[INFO] num_dt_buckets =", num_dt_buckets)

    # 6) 训练数据
    train_data = SeqTrainDatasetWithDT(ds.seq_slices, dt_bucket_slices, num_items=num_items)
    loader = DataLoader(
        train_data,
        batch_size=256,
        shuffle=True,
        collate_fn=collate_pad_items_dt
    )

    # 7) 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MeanSeqDTModel(num_items=num_items, num_dt_buckets=num_dt_buckets, emb_dim=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 8) 训练
    model.train()
    for epoch in range(3):
        total = 0.0
        for items_pad, dts_pad, mask, pos, neg in loader:
            items_pad = items_pad.to(device)
            dts_pad = dts_pad.to(device)
            mask = mask.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            pos_s, neg_s = model(items_pad, dts_pad, mask, pos, neg)
            loss = bpr_loss(pos_s, neg_s)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        beta = torch.nn.functional.softplus(model.raw_beta).item()
        print(f"[Epoch {epoch}] loss={total/len(loader):.4f}, beta={beta:.4f}")


if __name__ == "__main__":
    main()
