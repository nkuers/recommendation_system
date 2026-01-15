# models/interest_clock_simple.py
import math
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

    def _sample_neg(self, forbidden):
        if self.num_items <= 1:
            return 0
        max_id = self.num_items - 1
        if len(forbidden) >= max_id:
            return random.randint(1, max_id)
        neg = random.randint(1, max_id)
        tries = 0
        while neg in forbidden:
            neg = random.randint(1, max_id)
            tries += 1
            if tries > 1000:
                break
        return neg

    def __getitem__(self, idx):
        x_items, x_dts, y_pos = self.samples[idx]

        forbidden = set(x_items)
        forbidden.add(y_pos)
        neg = self._sample_neg(forbidden)

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
    简化版 Interest Clock v2（更接近论文思想）：

    核心思想：
      - 历史每个位置 i 的表示融合：
            item_emb_i + dt_bucket_emb_i + clock_enc_i
      - 候选 item 作为 query，对历史序列做时间感知注意力：
            logit_i = <q, K_i> / tau
        其中 K_i 来源于上面的 time-aware 表示；
      - 'clock_enc' 是基于 dt_bucket 的 [sin θ, cos θ] 编码，
        模拟“时钟/周期”感知；
      - 最后用 BPR 对正负样本打分。

    接口保持不变：
        forward(items_pad, dts_pad, mask, pos_items, neg_items)
    并返回：
        pos_score, neg_score, attn_pos
    这样无需改动现有的 run_comparative_experiment.py。
    """

    def __init__(self,
                 num_items: int,
                 num_dt_buckets: int,
                 emb_dim: int = 64,
                 pad_item_id: int = 0,
                 pad_dt_id: int = 0,
                 dropout_p: float = 0.0):
        super().__init__()

        # item embedding
        self.item_emb = nn.Embedding(num_items, emb_dim, padding_idx=pad_item_id)
        # Δt bucket embedding
        self.dt_emb = nn.Embedding(num_dt_buckets, emb_dim, padding_idx=pad_dt_id)

        # 简单“clock encoding”：由 [sinθ, cosθ] -> emb_dim
        self.clock_proj = nn.Linear(2, emb_dim)

        # 历史表示 -> key / value
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(dropout_p)
        self.kv_dropout = nn.Dropout(dropout_p)

        # 注意力温度参数 tau >= 0
        self.raw_tau = nn.Parameter(torch.tensor(0.0))  # softplus(0)≈0.693

    # --------- 时钟编码 ---------
    def build_clock_encoding(self, dt_buckets: torch.Tensor) -> torch.Tensor:
        """
        利用 dt_bucket_id 构造一个简易“时钟”编码：
            angle = 2π * (bucket_id / B)
            clock = [sin(angle), cos(angle)] -> 线性映射到 emb_dim

        dt_buckets: [B, L], int64
        return: [B, L, D]
        """
        # num_embeddings = B(+1,含padding)，可用 bucket 范围近似 [0, B-1]
        B = max(int(self.dt_emb.num_embeddings - 1), 1)

        x = dt_buckets.float() / max(B, 1)          # 归一化到 [0,1]
        angle = 2.0 * math.pi * x                   # [B, L]

        sin_part = torch.sin(angle)
        cos_part = torch.cos(angle)
        feat = torch.stack([sin_part, cos_part], dim=-1)  # [B, L, 2]

        clock_emb = self.clock_proj(feat)           # [B, L, D]
        return clock_emb

    # --------- 候选 item 感知用户表示 ---------
    def _candidate_aware_uservec(self,
                                 items_pad: torch.Tensor,
                                 dts_pad: torch.Tensor,
                                 mask: torch.Tensor,
                                 cand_items: torch.Tensor):
        """
        items_pad: [B, L]
        dts_pad:   [B, L]
        mask:      [B, L]  True=有效
        cand_items:[B]
        return:
          u:    [B, D]   （与候选 item 相关的用户表示）
          attn: [B, L]   （注意力权重）
        """
        # 历史 item + Δt + clock
        e_item = self.item_emb(items_pad)          # [B, L, D]
        e_dt = self.dt_emb(dts_pad)               # [B, L, D]
        e_clock = self.build_clock_encoding(dts_pad)  # [B, L, D]

        h_hist = e_item + e_dt + e_clock          # [B, L, D]
        h_hist = self.dropout(h_hist)

        K = self.k_proj(h_hist)                   # [B, L, D]
        V = self.v_proj(h_hist)                   # [B, L, D]
        K = self.kv_dropout(K)
        V = self.kv_dropout(V)

        # 候选 item 作为 query
        q = self.item_emb(cand_items)             # [B, D]
        q = q.unsqueeze(1)                        # [B, 1, D]

        tau = F.softplus(self.raw_tau) + 1e-6     # 标量 >= 0
        # 点积注意力：logits = <q, K_i> / tau
        logits = torch.sum(q * K, dim=-1) / tau   # [B, L]

        # mask 无效位置
        logits = logits.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(logits, dim=-1)      # [B, L]
        # 加权求和得到用户表示
        u = torch.bmm(attn.unsqueeze(1), V).squeeze(1)  # [B, D]

        return u, attn

    def forward(self,
                items_pad: torch.Tensor,
                dts_pad: torch.Tensor,
                mask: torch.Tensor,
                pos_items: torch.Tensor,
                neg_items: torch.Tensor):
        """
        与原版本接口保持一致：
          返回 (pos_score, neg_score, attn_pos)
        """
        # 正样本作为 query
        u_pos, attn_pos = self._candidate_aware_uservec(
            items_pad, dts_pad, mask, pos_items
        )
        # 负样本也可以视为一个候选 query（保持 candidate-aware 设定）
        u_neg, _ = self._candidate_aware_uservec(
            items_pad, dts_pad, mask, neg_items
        )

        q_pos = self.item_emb(pos_items)          # [B, D]
        q_neg = self.item_emb(neg_items)          # [B, D]

        pos_score = (u_pos * q_pos).sum(dim=-1)   # [B]
        neg_score = (u_neg * q_neg).sum(dim=-1)   # [B]

        return pos_score, neg_score, attn_pos


def bpr_loss(pos_score, neg_score):
    return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
