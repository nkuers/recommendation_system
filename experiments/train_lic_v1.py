# experiments/train_lic_v1.py
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from preprocess.movielens import load_movielens
from preprocess.base import sort_by_user_and_time, build_user_sequences, Dataset
from models.lic_v1 import (
    SeqTrainDatasetLIC,
    collate_lic,
    LICv1,
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
    简化 s(Δt)：log1p + 分桶（0 留给 padding）
    """
    if dt_seconds is None or dt_seconds < 0:
        dt_seconds = 0
    x = math.log1p(dt_seconds)

    bins = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    for i in range(1, len(bins)):
        if x <= bins[i]:
            return i
    return len(bins)


def build_dt_bucket_slices(delta_t_slices):
    return [[dt_to_bucket(int(dt)) for dt in dt_seq] for dt_seq in delta_t_slices]


def main():
    # 1) load & sort
    df = load_movielens("../data/movielens/ratings.csv")
    df = sort_by_user_and_time(df)

    # 2) sequences + delta_t
    user_seq, user_dt = build_user_sequences(df)

    # 3) long window dataset（GSU 用长窗口）
    long_window = 50
    short_len = 10

    ds_long = Dataset(user_seq, user_dt, window_size=long_window)

    # 4) dt -> bucket（按 long window 对齐）
    dt_bucket_slices_long = build_dt_bucket_slices(ds_long.delta_t_slices)

    # 5) num_items / num_dt_buckets
    num_items = get_num_items(user_seq)
    num_dt_buckets = max(max(x) for x in dt_bucket_slices_long) + 1
    print("[INFO] num_items =", num_items)
    print("[INFO] num_dt_buckets =", num_dt_buckets)
    print("[INFO] long_window =", long_window, "short_len =", short_len)

    # 6) train dataset
    train_data = SeqTrainDatasetLIC(
        seq_slices_long=ds_long.seq_slices,
        dt_bucket_slices_long=dt_bucket_slices_long,
        num_items=num_items,
        short_len=short_len
    )

    loader = DataLoader(
        train_data,
        batch_size=256,
        shuffle=True,
        collate_fn=collate_lic
    )

    # 7) model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LICv1(
        num_items=num_items,
        num_dt_buckets=num_dt_buckets,
        emb_dim=64
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 8) train
    model.train()
    for epoch in range(3):
        total = 0.0
        for (items_long_pad, dts_long_pad, mask_long,
             items_short_pad, dts_short_pad, mask_short,
             pos, neg) in loader:

            items_long_pad = items_long_pad.to(device)
            dts_long_pad = dts_long_pad.to(device)
            mask_long = mask_long.to(device)

            items_short_pad = items_short_pad.to(device)
            dts_short_pad = dts_short_pad.to(device)
            mask_short = mask_short.to(device)

            pos = pos.to(device)
            neg = neg.to(device)

            pos_s, neg_s, lam, tau_gsu, tau_esu = model(
                items_long_pad, dts_long_pad, mask_long,
                items_short_pad, dts_short_pad, mask_short,
                pos, neg
            )

            loss = bpr_loss(pos_s, neg_s)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        lam_val = lam.item() if hasattr(lam, "item") else float(lam)
        print(f"[Epoch {epoch}] loss={total/len(loader):.4f}, "
              f"lambda={lam_val:.3f}, tau_gsu={float(tau_gsu):.3f}, tau_esu={float(tau_esu):.3f}")


if __name__ == "__main__":
    main()
