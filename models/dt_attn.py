# models/dt_attn.py
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset


class SeqTrainDatasetWithDT(TorchDataset):
    """
    训练样本：
      x_items: 前 L-1 个 item
      x_dt:    前 L-1 个 dt_bucket
      y_pos:   第 L 个 item
      y_neg:   负采样 item
    """
    def __init__(self, seq_slices: List[List[int]],
                 dt_bucket_slices: List[List[int]],
                 num_items: int):
        assert len(seq_slices) == len(dt_bucket_slices)
        self.samples: List[Tuple[List[int], List[int], int]] = []
        self.num_items = num_items

        for items, dts in zip(seq_slices, dt_bucket_slices):
            if len(items) < 2:
                continue
            self.samples.append((items[:-1], dts[:-1], items[-1]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_items, x_dts, y_pos = self.samples[idx]

        neg = random.randint(0, self.num_items - 1)
        while neg == y_pos:
            neg = random.randint(0, self.num_items - 1)

        return (
            torch.tensor(x_items, dtype=torch.long),
            torch.tensor(x_dts, dtype=torch.long),
            torch.tensor(y_pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )


def collate_pad_items_dt(batch, pad_item_id=0, pad_dt_id=0):
    """
    pad items 和 dt_bucket 到同长度，并返回 mask
    """
    x_items, x_dts, y_pos, y_neg = zip(*batch)
    lengths = [len(x) for x in x_items]
    max_len = max(lengths)

    items_pad = torch.full((len(batch), max_len), pad_item_id, dtype=torch.long)
    dts_pad = torch.full((len(batch), max_len), pad_dt_id, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)

    for i, L in enumerate(lengths):
        items_pad[i, :L] = x_items[i]
        dts_pad[i, :L] = x_dts[i]
        mask[i, :L] = 1

    return items_pad, dts_pad, mask, torch.stack(y_pos), torch.stack(y_neg)


class DTAttentionModel(nn.Module):
    """
    dt-attention baseline:
      attn_logits_i = w_bucket[dt_bucket_i]
      attn = softmax(attn_logits + mask)
      h = sum(attn_i * item_emb_i)
    """
    def __init__(self,
                 num_items: int,
                 num_dt_buckets: int,
                 emb_dim: int = 64,
                 pad_item_id: int = 0,
                 pad_dt_id: int = 0):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, emb_dim, padding_idx=pad_item_id)

        # 每个 dt bucket 一个可学习标量权重
        # padding bucket(0) 也有权重，但会被 mask 掉
        self.dt_bucket_w = nn.Embedding(num_dt_buckets, 1, padding_idx=pad_dt_id)

        # 可选：温度参数，让 attention 更平滑/更尖锐（>=0）
        self.raw_tau = nn.Parameter(torch.tensor(0.0))  # softplus(0)=0.693

    def encode(self, items_pad, dts_pad, mask):
        e_item = self.item_emb(items_pad)  # [B,L,D]

        logits = self.dt_bucket_w(dts_pad).squeeze(-1)  # [B,L]
        tau = F.softplus(self.raw_tau) + 1e-6           # >=0, avoid /0
        logits = logits / tau

        # mask: False 的位置给 -inf
        logits = logits.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(logits, dim=1)             # [B,L]
        h = torch.bmm(attn.unsqueeze(1), e_item).squeeze(1)  # [B,D]
        return h, attn

    def forward(self, items_pad, dts_pad, mask, pos_items, neg_items):
        h, attn = self.encode(items_pad, dts_pad, mask)
        pos_e = self.item_emb(pos_items)
        neg_e = self.item_emb(neg_items)
        pos_score = (h * pos_e).sum(dim=-1)
        neg_score = (h * neg_e).sum(dim=-1)
        return pos_score, neg_score, attn


def bpr_loss(pos_score, neg_score):
    return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
