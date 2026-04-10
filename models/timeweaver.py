import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss

try:
    from models.time_modules import (
        ContinuousTimeEmbedding,
        TimeGate,
        scheduled_time_augmentation,
    )
except ModuleNotFoundError:
    from .time_modules import (
        ContinuousTimeEmbedding,
        TimeGate,
        scheduled_time_augmentation,
    )


class TimeAwareAugmentation(nn.Module):
    def __init__(self, max_seq_length, hidden_size, time_bucket_size, dropout,
                 use_continuous_time=False, use_time_gate=False, time_gate_hidden=16):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        self.time_bucket_size = time_bucket_size
        self.use_continuous_time = use_continuous_time
        self.use_time_gate = use_time_gate

        self.pos_embedding = nn.Embedding(max_seq_length, hidden_size)
        if self.use_continuous_time:
            self.time_embedder = ContinuousTimeEmbedding(hidden_size)
        else:
            self.time_embedding = nn.Embedding(time_bucket_size, hidden_size)
        self.time_scale = nn.Parameter(torch.tensor(0.1))
        if self.use_time_gate:
            self.time_gate = TimeGate(hidden_size, time_gate_hidden, use_item_context=True)
            self.gate_scale = nn.Parameter(torch.tensor(0.5))
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)

        pos_ids = torch.arange(max_seq_length, dtype=torch.long)
        self.register_buffer("pos_ids", pos_ids, persistent=False)

    def forward(self, item_emb, time_seq):
        # item_emb: [B, L, D], time_seq: [B, L]
        batch_size, seq_len, _ = item_emb.size()
        pos_emb = self.pos_embedding(self.pos_ids[:seq_len]).unsqueeze(0).expand(batch_size, -1, -1)

        if self.use_continuous_time:
            time_emb = self.time_embedder(time_seq)
        else:
            time_seq = time_seq.float()
            delta = time_seq.clone()
            if seq_len > 1:
                delta[:, 1:] = time_seq[:, 1:] - time_seq[:, :-1]
            delta = torch.clamp(delta, min=0.0)
            time_idx = torch.remainder(delta.long(), self.time_bucket_size)
            time_emb = self.time_embedding(time_idx)

        # Global position offset: mean time embedding at each position in batch
        pos_offset = time_emb.mean(dim=0, keepdim=True)
        pos_calib = pos_emb + self.time_scale * pos_offset

        if self.use_time_gate:
            gate = self.time_gate(time_emb, item_emb=item_emb)
            valid_mask = (time_seq > 0).unsqueeze(-1).to(item_emb.dtype)
            gated_time = gate * time_emb * valid_mask
            x = item_emb + pos_calib + self.gate_scale * gated_time
        else:
            x = item_emb + pos_calib
        x = self.layer_norm(x)
        x = self.drop(x)
        return x


class DepthwiseConv1d(nn.Module):
    def __init__(self, hidden_size, kernel_size, dropout):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=pad, groups=hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, D]
        x_t = x.transpose(1, 2)
        y = self.conv(x_t)
        y = self.bn(y)
        y = self.drop(y)
        return y.transpose(1, 2)


class ParallelConvInter(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.proj = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, D]
        x_t = x.transpose(1, 2)
        z1 = self.conv1(x_t)
        z2 = self.conv3(x_t)
        g = (z1 * F.gelu(z2)) + (z2 * F.gelu(z1))
        g = self.proj(g)
        g = self.drop(g)
        return g.transpose(1, 2)


class TCNNextBlock(nn.Module):
    def __init__(self, hidden_size, kernel_size, ffn_hidden, dropout):
        super().__init__()
        self.dw = DepthwiseConv1d(hidden_size, kernel_size, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)

        self.conv_inter = ParallelConvInter(hidden_size, dropout)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        y = self.dw(x)
        x = self.norm1(x + y)

        y = self.conv_inter(x)
        x = self.norm2(x + y)

        y = self.ffn(x)
        x = self.norm3(x + y)
        return x


class EMAStream(nn.Module):
    def __init__(self, hidden_size, ema_decay):
        super().__init__()
        self.ema_decay = ema_decay
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # x: [B, L, D]
        alpha = self.ema_decay
        _, seq_len, _ = x.size()
        device = x.device
        dtype = x.dtype

        p = torch.arange(seq_len - 1, -1, -1, device=device, dtype=dtype)
        w_prime = (1.0 - alpha) ** p
        w = w_prime.clone()
        if seq_len > 1:
            w[1:] = w[1:] * alpha

        w = w.view(1, seq_len, 1)
        w_prime = w_prime.view(1, seq_len, 1)

        c = torch.cumsum(x * w, dim=1)
        s = c / w_prime
        return self.proj(s)


class StreamWeaver(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size * 2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, h_context, h_dynamic, h_res):
        x = torch.cat([h_context, h_dynamic], dim=-1)
        x = self.proj(x)
        x = self.norm(x + h_res)
        return x


class TimeWeaver(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        def _cfg(name, default):
            return config[name] if name in config else default

        self.hidden_size = config["hidden_size"]
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
        self.num_layers = _cfg("num_layers", 2)
        self.hidden_dropout_prob = _cfg("hidden_dropout_prob", 0.5)
        self.initializer_range = _cfg("initializer_range", 0.02)
        self.loss_type = _cfg("loss_type", "CE")

        self.time_bucket_size = _cfg("time_bucket_size", 1000)
        self.ema_decay = _cfg("ema_decay", 0.1)
        self.tcn_kernel_size = _cfg("tcn_kernel_size", -1)
        self.ffn_hidden_size = _cfg("ffn_hidden_size", self.hidden_size * 4)
        self.use_continuous_time = _cfg("use_continuous_time", False)
        self.use_time_gate = _cfg("use_time_gate", False)
        self.time_gate_hidden = _cfg("time_gate_hidden", max(8, self.hidden_size // 4))
        self.use_time_augmentation = _cfg("use_time_augmentation", False)
        self.time_aug_strength = _cfg("time_aug_strength", 0.1)
        self.time_aug_prob = _cfg("time_aug_prob", 0.15)
        self.time_aug_mode = _cfg("time_aug_mode", "mix")
        self.time_aug_min_seq_len = int(_cfg("time_aug_min_seq_len", 8))
        self.time_aug_warmup_steps = int(_cfg("time_aug_warmup_steps", 0))
        self.time_aug_ramp_steps = int(_cfg("time_aug_ramp_steps", 2000))
        self.contrastive_weight = _cfg("contrastive_weight", 0.03)
        self.contrastive_temp = _cfg("contrastive_temp", 0.2)
        self._aug_step = 0

        self.TIME_FIELD = config["TIME_FIELD"]
        self.TIME_SEQ = self.TIME_FIELD + config["LIST_SUFFIX"]

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.time_aug = TimeAwareAugmentation(
            self.max_seq_length,
            self.hidden_size,
            self.time_bucket_size,
            self.hidden_dropout_prob,
            use_continuous_time=self.use_continuous_time,
            use_time_gate=self.use_time_gate,
            time_gate_hidden=self.time_gate_hidden,
        )

        if self.tcn_kernel_size <= 0:
            base = max(1, self.max_seq_length // 3)
            if base % 2 == 0:
                base += 1
            self.tcn_kernel_size = min(15, base)

        self.context_layers = nn.ModuleList(
            [
                TCNNextBlock(
                    hidden_size=self.hidden_size,
                    kernel_size=self.tcn_kernel_size,
                    ffn_hidden=self.ffn_hidden_size,
                    dropout=self.hidden_dropout_prob,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.dynamic_layers = nn.ModuleList(
            [
                EMAStream(hidden_size=self.hidden_size, ema_decay=self.ema_decay)
                for _ in range(self.num_layers)
            ]
        )
        self.weavers = nn.ModuleList(
            [StreamWeaver(self.hidden_size) for _ in range(self.num_layers)]
        )

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("loss_type must be one of ['BPR', 'CE']")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, time_seq):
        item_emb = self.item_embedding(item_seq)
        h = self.time_aug(item_emb, time_seq)

        for ctx_layer, dyn_layer, weaver in zip(self.context_layers, self.dynamic_layers, self.weavers):
            h_ctx = ctx_layer(h)
            h_dyn = dyn_layer(h)
            h = weaver(h_ctx, h_dyn, h)
        return h

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        time_seq = interaction[self.TIME_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, time_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
        else:
            logits = torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

        if self.use_time_augmentation and self.training:
            self._aug_step += 1
            time_seq_aug, apply_mask = scheduled_time_augmentation(
                time_seq=time_seq,
                item_seq_len=item_seq_len,
                step=self._aug_step,
                warmup_steps=self.time_aug_warmup_steps,
                ramp_steps=self.time_aug_ramp_steps,
                min_seq_len=self.time_aug_min_seq_len,
                base_prob=self.time_aug_prob,
                base_strength=self.time_aug_strength,
                mode=self.time_aug_mode,
            )
            if time_seq_aug is not None:
                seq_output_aug = self.forward(item_seq, time_seq_aug)
                seq_output_aug = self.gather_indexes(seq_output_aug, item_seq_len - 1)

                z = F.normalize(seq_output[apply_mask], dim=-1)
                z_aug = F.normalize(seq_output_aug[apply_mask], dim=-1)
                logits = torch.matmul(z, z_aug.transpose(0, 1)) / max(self.contrastive_temp, 1e-6)
                labels = torch.arange(logits.size(0), device=logits.device)
                cl_loss = F.cross_entropy(logits, labels)
                loss = loss + self.contrastive_weight * cl_loss

        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        time_seq = interaction[self.TIME_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, time_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        if self.loss_type == "CE":
            logits = torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
            scores = logits.gather(1, test_item.view(-1, 1)).squeeze(1)
        else:
            test_item_emb = self.item_embedding(test_item)
            scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        time_seq = interaction[self.TIME_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, time_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        scores = torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
        return scores
