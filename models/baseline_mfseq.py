import random
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset, DataLoader


class SeqTrainDataset(TorchDataset):
    """
    从你预处理的 window 序列中构造训练样本：
    输入: 前 L-1 个 item
    正样本: 第 L 个 item (next-item)
    负样本: 随机采一个用户未交互 item
    """
    def __init__(self, seq_slices: List[List[int]], num_items: int):
        self.samples = []
        self.num_items = num_items

        for seq in seq_slices:
            if len(seq) < 2:
                continue
            x = seq[:-1]
            y = seq[-1]
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        # 负采样
        neg = random.randint(0, self.num_items - 1)
        while neg == y:
            neg = random.randint(0, self.num_items - 1)
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long), torch.tensor(neg, dtype=torch.long)


def collate_pad(batch, pad_id=0):
    """
    把不同长度的序列 pad 成同长度，并返回 mask
    """
    xs, ys, negs = zip(*batch)
    lengths = [len(x) for x in xs]
    max_len = max(lengths)

    x_pad = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(xs), max_len), dtype=torch.bool)

    for i, x in enumerate(xs):
        x_pad[i, :len(x)] = x
        mask[i, :len(x)] = 1

    return x_pad, mask, torch.stack(ys), torch.stack(negs)


class MeanSeqModel(nn.Module):
    """
    baseline: item embedding -> 对序列做 masked mean pooling -> 与候选 item 做打分
    """
    def __init__(self, num_items: int, emb_dim: int = 64, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.item_emb = nn.Embedding(num_items, emb_dim, padding_idx=pad_id)

    def encode(self, x_pad, mask):
        # x_pad: [B, L]
        e = self.item_emb(x_pad)  # [B, L, D]
        mask_f = mask.unsqueeze(-1).float()  # [B, L, 1]
        summed = (e * mask_f).sum(dim=1)     # [B, D]
        denom = mask_f.sum(dim=1).clamp(min=1.0)  # [B, 1]
        return summed / denom

    def forward(self, x_pad, mask, pos_items, neg_items):
        h = self.encode(x_pad, mask)               # [B, D]
        pos_e = self.item_emb(pos_items)           # [B, D]
        neg_e = self.item_emb(neg_items)           # [B, D]
        pos_score = (h * pos_e).sum(dim=-1)        # [B]
        neg_score = (h * neg_e).sum(dim=-1)        # [B]
        return pos_score, neg_score


def bpr_loss(pos_score, neg_score):
    return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
