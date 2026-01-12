# models/hour_seq.py
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset


# ======================================================
# 1) 训练数据集（items + hours）
# ======================================================
class SeqTrainDatasetWithHour(TorchDataset):
    """
    训练样本：
      x_items: 前 L-1 个 item
      x_hours: 前 L-1 个 hour
      y_pos:   第 L 个 item
      y_neg:   负采样 item
    """
    def __init__(self, seq_slices: List[List[int]],
                 hour_slices: List[List[int]],
                 num_items: int):
        assert len(seq_slices) == len(hour_slices)

        self.samples: List[Tuple[List[int], List[int], int]] = []
        self.num_items = num_items

        for items, hours in zip(seq_slices, hour_slices):
            if len(items) < 2:
                continue
            self.samples.append((
                items[:-1],
                hours[:-1],
                items[-1]
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_items, x_hours, y_pos = self.samples[idx]

        neg = random.randint(0, self.num_items - 1)
        while neg == y_pos:
            neg = random.randint(0, self.num_items - 1)

        return (
            torch.tensor(x_items, dtype=torch.long),
            torch.tensor(x_hours, dtype=torch.long),
            torch.tensor(y_pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )


# ======================================================
# 2) collate：padding + mask
# ======================================================
def collate_pad_items_hours(batch, pad_item_id=0, pad_hour_id=24):
    # 解包
    x_items, x_hours, y_pos, y_neg = zip(*batch)
    lengths = [len(x) for x in x_items]
    max_len = max(lengths)

    # 创建填充张量
    items_pad = torch.full((len(batch), max_len), pad_item_id, dtype=torch.long)
    hours_pad = torch.full((len(batch), max_len), pad_hour_id, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)

    for i, L in enumerate(lengths):
        items_pad[i, :L] = x_items[i].clone().detach()
        hours_pad[i, :L] = x_hours[i].clone().detach()
        mask[i, :L] = 1

    # 将 y_pos 和 y_neg 转换为张量
    y_pos_tensor = torch.tensor(y_pos, dtype=torch.long)
    y_neg_tensor = torch.tensor(y_neg, dtype=torch.long)

    # 返回填充后的张量
    return items_pad, hours_pad, mask, y_pos_tensor, y_neg_tensor




# ======================================================
# 3) Hour Embedding baseline（Softplus α 门控）
# ======================================================
class MeanSeqHourModel(nn.Module):
    """
    item embedding + alpha * hour embedding
    alpha = softplus(raw_alpha) >= 0
    """
    def __init__(self,
                 num_items: int,
                 emb_dim: int = 64,
                 num_hours: int = 25,
                 pad_item_id: int = 0,
                 pad_hour_id: int = 24,
                 raw_alpha_init: float = -3.0):
        super().__init__()

        self.item_emb = nn.Embedding(
            num_items, emb_dim, padding_idx=pad_item_id
        )
        self.hour_emb = nn.Embedding(
            num_hours, emb_dim, padding_idx=pad_hour_id
        )

        # 可学习的 raw_alpha，通过 softplus 约束 alpha >= 0
        # softplus(-3) ≈ 0.048，初始 hour 影响极小
        self.raw_alpha = nn.Parameter(
            torch.tensor(raw_alpha_init, dtype=torch.float)
        )

    def encode(self, items_pad, hours_pad, mask):
        e_item = self.item_emb(items_pad)      # [B, L, D]
        e_hour = self.hour_emb(hours_pad)      # [B, L, D]

        alpha = F.softplus(self.raw_alpha)     # >= 0
        e = e_item + alpha * e_hour

        mask_f = mask.unsqueeze(-1).float()
        summed = (e * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def forward(self, items_pad, hours_pad, mask, pos_items, neg_items):
        h = self.encode(items_pad, hours_pad, mask)
        pos_e = self.item_emb(pos_items)
        neg_e = self.item_emb(neg_items)
        pos_score = (h * pos_e).sum(dim=-1)
        neg_score = (h * neg_e).sum(dim=-1)
        return pos_score, neg_score


# ======================================================
# 4) BPR loss
# ======================================================
def bpr_loss(pos_score, neg_score):
    return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
