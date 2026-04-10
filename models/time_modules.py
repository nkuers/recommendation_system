import torch
from torch import nn


class Time2Vec(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        if out_dim < 1:
            raise ValueError("out_dim must be >= 1")
        self.out_dim = out_dim
        self.periodic_dim = out_dim - 1
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.zeros(1))
        if self.periodic_dim > 0:
            self.w = nn.Parameter(torch.randn(self.periodic_dim))
            self.b = nn.Parameter(torch.zeros(self.periodic_dim))
        else:
            self.register_parameter("w", None)
            self.register_parameter("b", None)

    def forward(self, t):
        # t: [B, L]
        t = t.unsqueeze(-1)
        linear = self.w0 * t + self.b0
        if self.periodic_dim <= 0:
            return linear
        periodic = torch.sin(t * self.w + self.b)
        return torch.cat([linear, periodic], dim=-1)


class ContinuousTimeEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.time2vec = Time2Vec(hidden_size)

    def forward(self, time_seq):
        # time_seq: [B, L]
        time_seq = time_seq.float()
        delta = time_seq.clone()
        if time_seq.size(1) > 1:
            delta[:, 1:] = time_seq[:, 1:] - time_seq[:, :-1]
        delta = torch.clamp(delta, min=0.0)
        delta = torch.log1p(delta)
        return self.time2vec(delta)


class TimeGate(nn.Module):
    def __init__(self, hidden_size, gate_hidden, use_item_context=True):
        super().__init__()
        self.use_item_context = use_item_context
        in_dim = hidden_size * 2 if use_item_context else hidden_size
        self.net = nn.Sequential(
            nn.Linear(in_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, hidden_size),
            nn.Sigmoid(),
        )

    def forward(self, time_emb, item_emb=None):
        if self.use_item_context and item_emb is not None:
            x = torch.cat([time_emb, item_emb], dim=-1)
        else:
            x = time_emb
        return self.net(x)


def augment_time_sequence(time_seq, mode="mix", strength=0.3, prob=1.0):
    # time_seq: [B, L]
    if strength <= 0:
        return time_seq
    time_seq_f = time_seq.float()
    delta = time_seq_f.clone()
    if time_seq_f.size(1) > 1:
        delta[:, 1:] = time_seq_f[:, 1:] - time_seq_f[:, :-1]
    delta = torch.clamp(delta, min=0.0)

    delta_new = delta
    if mode in {"mix", "median"}:
        median = delta.median(dim=1, keepdim=True).values
        delta_new = (1.0 - strength) * delta_new + strength * median
    if mode in {"mix", "jitter"}:
        log_delta = torch.log1p(delta_new)
        noise = torch.randn_like(log_delta) * (0.1 * strength)
        delta_new = torch.expm1(torch.clamp(log_delta + noise, min=0.0))

    if prob < 1.0:
        mask = torch.rand_like(delta_new) < prob
        delta_new = torch.where(mask, delta_new, delta)

    delta_new = delta_new.clone()
    delta_new[:, 0] = 0.0
    base = time_seq_f[:, :1]
    time_aug = base + torch.cumsum(delta_new, dim=1)

    pad_mask = time_seq_f <= 0
    if pad_mask.any():
        time_aug = time_aug.masked_fill(pad_mask, 0.0)
    return time_aug.to(time_seq.dtype)


def scheduled_time_augmentation(
    time_seq,
    item_seq_len,
    step,
    warmup_steps=1000,
    ramp_steps=2000,
    min_seq_len=8,
    base_prob=0.15,
    base_strength=0.1,
    mode="mix",
):
    """Apply scheduled time augmentation on a subset of sequences.

    Returns:
        time_seq_aug: augmented tensor or None when augmentation is skipped
        apply_mask: boolean mask of augmented samples or None
    """
    if step <= warmup_steps:
        return None, None

    warm = float(step - warmup_steps) / float(max(ramp_steps, 1))
    warm = max(0.0, min(1.0, warm))
    effective_prob = float(base_prob) * warm
    effective_strength = float(base_strength) * warm
    if effective_prob <= 0.0 or effective_strength <= 0.0:
        return None, None

    seq_mask = item_seq_len >= int(min_seq_len)
    rnd_mask = torch.rand_like(item_seq_len.float()) < effective_prob
    apply_mask = seq_mask & rnd_mask
    if int(apply_mask.sum().item()) < 2:
        return None, None

    time_seq_aug = time_seq.clone()
    time_seq_aug_part = augment_time_sequence(
        time_seq[apply_mask],
        mode=mode,
        strength=effective_strength,
        prob=1.0,
    )
    time_seq_aug[apply_mask] = time_seq_aug_part
    return time_seq_aug, apply_mask
