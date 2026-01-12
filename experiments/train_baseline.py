import torch
from torch.utils.data import DataLoader

from preprocess.movielens import load_movielens
from preprocess.base import sort_by_user_and_time, build_user_sequences, Dataset
from models.baseline_mfseq import SeqTrainDataset, collate_pad, MeanSeqModel, bpr_loss


def get_num_items(user_seq: dict) -> int:
    mx = 0
    for _, items in user_seq.items():
        if items:
            mx = max(mx, max(items))
    return mx + 1


def main():
    # 1) 读取 & 预处理 MovieLens
    df = load_movielens("../data/movielens/ratings.csv")
    df = sort_by_user_and_time(df)
    user_seq, user_dt = build_user_sequences(df)

    # 2) 固定窗口切片
    ds = Dataset(user_seq, user_dt, window_size=10)

    # 3) 统计 item 数
    num_items = get_num_items(user_seq)
    print("[INFO] num_items =", num_items)

    # 4) 构造训练数据（只用 seq_slices）
    train_data = SeqTrainDataset(ds.seq_slices, num_items=num_items)
    loader = DataLoader(train_data, batch_size=256, shuffle=True, collate_fn=collate_pad)

    # 5) 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MeanSeqModel(num_items=num_items, emb_dim=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 6) 训练（先跑 1~3 个 epoch 验证链路）
    model.train()
    for epoch in range(3):
        total = 0.0
        for x_pad, mask, pos, neg in loader:
            x_pad = x_pad.to(device)
            mask = mask.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            pos_s, neg_s = model(x_pad, mask, pos, neg)
            loss = bpr_loss(pos_s, neg_s)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"[Epoch {epoch}] loss={total/len(loader):.4f}")


if __name__ == "__main__":
    main()
