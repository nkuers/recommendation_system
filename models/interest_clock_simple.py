# models/interest_clock_simple.py
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


class SimpleInterestClock(nn.Module):
    """
    简化版 Interest Clock（候选 item 相关 + 时间差调制）

    对每个位置 i：
      sim_i = <q, k_i>   (候选 item embedding 与历史 item embedding 的相似度)
      g_i   = g(dt_bucket_i)  (时间差函数，输出标量)
      logits_i = sim_i * g_i / tau
      attn = softmax(logits + mask)
      u(q) = Σ attn_i * k_i

    score(q) = <u(q), q>
    BPR: score(pos) vs score(neg)
    """
    def __init__(self,
                 num_items: int,
                 num_dt_buckets: int,
                 emb_dim: int = 64,
                 pad_item_id: int = 0,
                 pad_dt_id: int = 0):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, emb_dim, padding_idx=pad_item_id)

        # g(dt): 每个 bucket 一个可学习标量（可视为简化 s(Δt)）
        self.dt_gate = nn.Embedding(num_dt_buckets, 1, padding_idx=pad_dt_id)

        # 温度 tau >= 0，控制 attention 尖锐程度
        self.raw_tau = nn.Parameter(torch.tensor(0.0))  # softplus(0)≈0.693

    def _candidate_aware_uservec(self, items_pad, dts_pad, mask, cand_items):
        """
        items_pad: [B,L]
        dts_pad:   [B,L]
        mask:      [B,L] bool
        cand_items:[B]
        return:
          u:    [B,D]
          attn: [B,L]
        """
        k = self.item_emb(items_pad)          # [B,L,D]
        q = self.item_emb(cand_items)         # [B,D]

        # sim_i = <q, k_i>
        sim = (k * q.unsqueeze(1)).sum(dim=-1)  # [B,L]

        # g(dt_i)
        gate = self.dt_gate(dts_pad).squeeze(-1)  # [B,L]

        tau = F.softplus(self.raw_tau) + 1e-6
        logits = (sim * gate) / tau               # [B,L]
        logits = logits.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(logits, dim=1)       # [B,L]
        u = torch.bmm(attn.unsqueeze(1), k).squeeze(1)  # [B,D]
        return u, attn

    def forward(self, items_pad, dts_pad, mask, pos_items, neg_items):
        u_pos, attn_pos = self._candidate_aware_uservec(items_pad, dts_pad, mask, pos_items)
        u_neg, attn_neg = self._candidate_aware_uservec(items_pad, dts_pad, mask, neg_items)

        q_pos = self.item_emb(pos_items)
        q_neg = self.item_emb(neg_items)

        pos_score = (u_pos * q_pos).sum(dim=-1)
        neg_score = (u_neg * q_neg).sum(dim=-1)

        return pos_score, neg_score, attn_pos


def bpr_loss(pos_score, neg_score):
    return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
