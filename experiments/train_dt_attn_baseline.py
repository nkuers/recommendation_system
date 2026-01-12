# experiments/train_dt_attn_baseline.py
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from preprocess.movielens import load_movielens
from preprocess.base import sort_by_user_and_time, build_user_sequences, Dataset
from models.dt_attn import (
    SeqTrainDatasetWithDT,
    collate_pad_items_dt,
    DTAttentionModel,
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
    简化 s(Δt): log1p 压缩 + 分桶
    bucket 0: padding
    bucket 1..N: 时间尺度桶
    """
    if dt_seconds is None or dt_seconds < 0:
        dt_seconds = 0

    x = math.log1p(dt_seconds)

    # 和你 dt-embedding baseline 一致（可调）
    bins = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    for i in range(1, len(bins)):
        if x <= bins[i]:
            return i
    return len(bins)


def build_dt_bucket_slices(delta_t_slices):
    out = []
    for dt_seq in delta_t_slices:
        out.append([dt_to_bucket(int(dt)) for dt in dt_seq])
    return out


def main():
    # 1) load & sort
    df = load_movielens("../data/movielens/ratings.csv")
    df = sort_by_user_and_time(df)

    # 2) sequences + delta_t
    user_seq, user_dt = build_user_sequences(df)

    # 3) dataset windows
    window_size = 10
    ds = Dataset(user_seq, user_dt, window_size=window_size)

    # 4) dt -> bucket
    dt_bucket_slices = build_dt_bucket_slices(ds.delta_t_slices)
    assert len(dt_bucket_slices) == len(ds.seq_slices)

    # 5) num_items / num_dt_buckets
    num_items = get_num_items(user_seq)
    num_dt_buckets = max(max(x) for x in dt_bucket_slices) + 1
    print("[INFO] num_items =", num_items)
    print("[INFO] num_dt_buckets =", num_dt_buckets)

    # 6) dataloader
    train_data = SeqTrainDatasetWithDT(ds.seq_slices, dt_bucket_slices, num_items=num_items)
    loader = DataLoader(
        train_data,
        batch_size=256,
        shuffle=True,
        collate_fn=collate_pad_items_dt
    )

    # 7) model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DTAttentionModel(num_items=num_items, num_dt_buckets=num_dt_buckets, emb_dim=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 8) train
    model.train()
    for epoch in range(20):
        total = 0.0

        for items_pad, dts_pad, mask, pos, neg in loader:
            items_pad = items_pad.to(device)
            dts_pad = dts_pad.to(device)
            mask = mask.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            pos_s, neg_s, attn = model(items_pad, dts_pad, mask, pos, neg)
            loss = bpr_loss(pos_s, neg_s)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        tau = (F.softplus(model.raw_tau) + 1e-6).item()
        print(f"[Epoch {epoch}] loss={total/len(loader):.4f}, tau={tau:.4f}")


if __name__ == "__main__":
    main()
