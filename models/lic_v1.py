# models/lic_v1.py
import math
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
                 short_len: int = 10,
                 max_steps: int = 10):
        assert len(seq_slices_long) == len(dt_bucket_slices_long)
        self.samples: List[Tuple[List[int], List[int], List[int], List[int], int]] = []
        self.num_items = num_items
        self.short_len = short_len
        self.max_steps = max_steps

        for items, dts in zip(seq_slices_long, dt_bucket_slices_long):
            if len(items) < 2:
                continue

            L = len(items)
            start_idx = max(1, L - max_steps)
            for t in range(start_idx, L):
                x_items = items[:t]
                x_dts = dts[:t]
                y_pos = items[t]

                # short window（从 long 的末尾截取）
                x_items_short = x_items[-short_len:]
                x_dts_short = x_dts[-short_len:]

                self.samples.append((x_items, x_dts, x_items_short, x_dts_short, y_pos))

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
        x_items_long, x_dts_long, x_items_short, x_dts_short, y_pos = self.samples[idx]

        forbidden = set(x_items_long)
        forbidden.add(y_pos)
        neg = self._sample_neg(forbidden)

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
# LIC-v2 核心：更贴近论文的 CandidateAwareClock + query-aware fusion
# ======================================================
class CandidateAwareClock(nn.Module):
    """
    更贴近论文思想的候选感知时间注意力模块：
    - token 表示： item_emb + dt_bucket_emb + clock_enc(sin/cos)
    - K/V 由线性层投影得到
    - attention logits = <q, K_i> / tau
    """
    def __init__(self,
                 item_emb: nn.Embedding,
                 num_dt_buckets: int,
                 emb_dim: int,
                 pad_dt_id: int = 0,
                 dropout_p: float = 0.0):
        super().__init__()
        self.item_emb = item_emb

        self.dt_emb = nn.Embedding(num_dt_buckets, emb_dim, padding_idx=pad_dt_id)
        self.clock_proj = nn.Linear(2, emb_dim)

        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(dropout_p)
        self.kv_dropout = nn.Dropout(dropout_p)

        self.raw_tau = nn.Parameter(torch.tensor(0.0))  # softplus -> tau

    def _clock_encoding(self, dts_pad: torch.Tensor) -> torch.Tensor:
        """
        dt bucket -> angle -> sin/cos -> proj
        dts_pad: [B, L]
        """
        B = max(int(self.dt_emb.num_embeddings - 1), 1)
        x = dts_pad.float() / max(B, 1)
        angle = 2.0 * math.pi * x
        feat = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)  # [B,L,2]
        return self.clock_proj(feat)  # [B,L,D]

    def forward(self, items_pad, dts_pad, mask, cand_items):
        """
        items_pad: [B,L]
        dts_pad:   [B,L]
        mask:      [B,L]
        cand_items:[B]
        """
        e_item = self.item_emb(items_pad)         # [B,L,D]
        e_dt = self.dt_emb(dts_pad)               # [B,L,D]
        e_clock = self._clock_encoding(dts_pad)   # [B,L,D]
        token = e_item + e_dt + e_clock           # [B,L,D]
        token = self.dropout(token)

        K = self.k_proj(token)                    # [B,L,D]
        V = self.v_proj(token)                    # [B,L,D]
        K = self.kv_dropout(K)
        V = self.kv_dropout(V)

        q = self.item_emb(cand_items)             # [B,D]
        q = q.unsqueeze(1)                        # [B,1,D]

        tau = F.softplus(self.raw_tau) + 1e-6
        logits = torch.sum(q * K, dim=-1) / tau   # [B,L]
        logits = logits.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(logits, dim=1)       # [B,L]
        u = torch.bmm(attn.unsqueeze(1), V).squeeze(1)  # [B,D]
        return u, attn, tau


class LICv1(nn.Module):
    """
    仍保留类名 LICv1，保证你的 runner 不用改。
    但内部已升级为更贴近论文的 v2 版本：

    - GSU: CandidateAwareClock(long history)
    - ESU: CandidateAwareClock(short history)
    - 融合权重 λ：改为 query-aware（随候选 item 自适应）
      λ = sigmoid(MLP([q, u_gsu, u_esu]))
    """
    def __init__(self,
                 num_items: int,
                 num_dt_buckets: int,
                 emb_dim: int = 64,
                 pad_item_id: int = 0,
                 pad_dt_id: int = 0,
                 dropout_p: float = 0.0):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, emb_dim, padding_idx=pad_item_id)

        self.gsu = CandidateAwareClock(
            self.item_emb, num_dt_buckets, emb_dim, pad_dt_id=pad_dt_id, dropout_p=dropout_p
        )
        self.esu = CandidateAwareClock(
            self.item_emb, num_dt_buckets, emb_dim, pad_dt_id=pad_dt_id, dropout_p=dropout_p
        )

        # query-aware 融合：输入 [q, u_gsu, u_esu] -> λ
        self.fuse_mlp = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(emb_dim, 1)
        )

    def _fuse(self, q_vec, u_gsu, u_esu):
        x = torch.cat([q_vec, u_gsu, u_esu], dim=-1)   # [B,3D]
        lam = torch.sigmoid(self.fuse_mlp(x)).squeeze(-1)  # [B]
        u = lam.unsqueeze(-1) * u_gsu + (1.0 - lam).unsqueeze(-1) * u_esu
        return u, lam.mean()  # 返回 mean λ 便于日志输出

    def score(self,
              items_long_pad, dts_long_pad, mask_long,
              items_short_pad, dts_short_pad, mask_short,
              cand_items):
        u_gsu, _, tau_gsu = self.gsu(items_long_pad, dts_long_pad, mask_long, cand_items)
        u_esu, _, tau_esu = self.esu(items_short_pad, dts_short_pad, mask_short, cand_items)

        q = self.item_emb(cand_items)
        u, lam = self._fuse(q, u_gsu, u_esu)

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
