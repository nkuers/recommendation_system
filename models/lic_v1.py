# models/lic_v1.py
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset


# ======================================================
# Dataset：同一条序列同时产出 long/short 两个窗口
# ======================================================
class SeqTrainDatasetLIC(TorchDataset):
    """
    输入是按固定窗口切片后的序列（例如 window=50），每条样本再切出：
      - long:  全部 x_items_long（L-1）
      - short: 最近 short_len 个 x_items_short（<=short_len）
    """
    def __init__(self,
                 seq_slices_long: List[List[int]],
                 dt_bucket_slices_long: List[List[int]],
                 num_items: int,
                 short_len: int = 10):
        assert len(seq_slices_long) == len(dt_bucket_slices_long)
        self.samples: List[Tuple[List[int], List[int], List[int], List[int], int]] = []
        self.num_items = num_items
        self.short_len = short_len

        for items, dts in zip(seq_slices_long, dt_bucket_slices_long):
            if len(items) < 2:
                continue

            x_items = items[:-1]
            x_dts = dts[:-1]
            y_pos = items[-1]

            # short window（从 long 的末尾截取）
            x_items_short = x_items[-short_len:]
            x_dts_short = x_dts[-short_len:]

            self.samples.append((x_items, x_dts, x_items_short, x_dts_short, y_pos))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_items_long, x_dts_long, x_items_short, x_dts_short, y_pos = self.samples[idx]

        neg = random.randint(0, self.num_items - 1)
        while neg == y_pos:
            neg = random.randint(0, self.num_items - 1)

        return (
            torch.tensor(x_items_long, dtype=torch.long),
            torch.tensor(x_dts_long, dtype=torch.long),
            torch.tensor(x_items_short, dtype=torch.long),
            torch.tensor(x_dts_short, dtype=torch.long),
            torch.tensor(y_pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )


def _pad_2d(seqs: List[torch.Tensor], pad_id: int):
    lengths = [len(x) for x in seqs]
    max_len = max(lengths) if lengths else 0
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    for i, x in enumerate(seqs):
        L = len(x)
        if L > 0:
            out[i, :L] = x
            mask[i, :L] = 1
    return out, mask


def collate_lic(batch, pad_item_id=0, pad_dt_id=0):
    """
    batch:
      x_items_long, x_dts_long, x_items_short, x_dts_short, y_pos, y_neg
    返回：
      items_long_pad, dts_long_pad, mask_long
      items_short_pad, dts_short_pad, mask_short
      y_pos, y_neg
    """
    x_items_long, x_dts_long, x_items_short, x_dts_short, y_pos, y_neg = zip(*batch)

    items_long_pad, mask_long = _pad_2d(list(x_items_long), pad_item_id)
    dts_long_pad, _ = _pad_2d(list(x_dts_long), pad_dt_id)

    items_short_pad, mask_short = _pad_2d(list(x_items_short), pad_item_id)
    dts_short_pad, _ = _pad_2d(list(x_dts_short), pad_dt_id)

    return (
        items_long_pad, dts_long_pad, mask_long,
        items_short_pad, dts_short_pad, mask_short,
        torch.stack(y_pos), torch.stack(y_neg)
    )


# ======================================================
# LIC-v1 模型：GSU + ESU + fusion
# ======================================================
class CandidateAwareClock(nn.Module):
    """
    候选感知时间注意力模块（复用你 SimpleInterestClock 的核心）：
      logits_i = <q, k_i> * g(dt_i) / tau
      attn = softmax(logits + mask)
      u(q) = Σ attn_i * k_i
    """
    def __init__(self,
                 item_emb: nn.Embedding,
                 num_dt_buckets: int,
                 pad_dt_id: int = 0):
        super().__init__()
        self.item_emb = item_emb  # 共享 item embedding（重要）
        self.dt_gate = nn.Embedding(num_dt_buckets, 1, padding_idx=pad_dt_id)
        self.raw_tau = nn.Parameter(torch.tensor(0.0))  # softplus -> tau

    def forward(self, items_pad, dts_pad, mask, cand_items):
        k = self.item_emb(items_pad)              # [B,L,D]
        q = self.item_emb(cand_items)             # [B,D]
        sim = (k * q.unsqueeze(1)).sum(dim=-1)    # [B,L]
        gate = self.dt_gate(dts_pad).squeeze(-1)  # [B,L]

        tau = F.softplus(self.raw_tau) + 1e-6
        logits = (sim * gate) / tau
        logits = logits.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(logits, dim=1)       # [B,L]
        u = torch.bmm(attn.unsqueeze(1), k).squeeze(1)  # [B,D]
        return u, attn, tau


class LICv1(nn.Module):
    """
    LIC-v1:
      u_gsu(q) = Clock-GSU(long history)
      u_esu(q) = Clock-ESU(short history)
      u(q) = λ*u_gsu(q) + (1-λ)*u_esu(q), λ=softmax([w1,w2])[0]
      score(q) = <u(q), q>
    """
    def __init__(self,
                 num_items: int,
                 num_dt_buckets: int,
                 emb_dim: int = 64,
                 pad_item_id: int = 0,
                 pad_dt_id: int = 0):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, emb_dim, padding_idx=pad_item_id)

        self.gsu = CandidateAwareClock(self.item_emb, num_dt_buckets, pad_dt_id=pad_dt_id)
        self.esu = CandidateAwareClock(self.item_emb, num_dt_buckets, pad_dt_id=pad_dt_id)

        # 融合权重（可学习，两项 softmax，保证在 [0,1] 且和为1）
        self.fuse_logits = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float))

    def _fuse(self, u_gsu, u_esu):
        w = torch.softmax(self.fuse_logits, dim=0)  # [2]
        lam = w[0]
        return lam * u_gsu + (1.0 - lam) * u_esu, lam

    def score(self,
              items_long_pad, dts_long_pad, mask_long,
              items_short_pad, dts_short_pad, mask_short,
              cand_items):
        u_gsu, attn_gsu, tau_gsu = self.gsu(items_long_pad, dts_long_pad, mask_long, cand_items)
        u_esu, attn_esu, tau_esu = self.esu(items_short_pad, dts_short_pad, mask_short, cand_items)

        u, lam = self._fuse(u_gsu, u_esu)
        q = self.item_emb(cand_items)
        s = (u * q).sum(dim=-1)

        return s, lam, tau_gsu, tau_esu

    def forward(self,
                items_long_pad, dts_long_pad, mask_long,
                items_short_pad, dts_short_pad, mask_short,
                pos_items, neg_items):
        pos_s, lam, tau_gsu, tau_esu = self.score(
            items_long_pad, dts_long_pad, mask_long,
            items_short_pad, dts_short_pad, mask_short,
            pos_items
        )
        neg_s, _, _, _ = self.score(
            items_long_pad, dts_long_pad, mask_long,
            items_short_pad, dts_short_pad, mask_short,
            neg_items
        )
        return pos_s, neg_s, lam, tau_gsu, tau_esu


def bpr_loss(pos_score, neg_score):
    return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
