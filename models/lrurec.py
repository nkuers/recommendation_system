import math

import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class LRUBlock(nn.Module):
    def __init__(self, hidden_size, ffn_hidden, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        # Complex parameters represented with real/imag parts
        self.B_re = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.B_im = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.C_re = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.C_im = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.D = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.nu_log = nn.Parameter(torch.empty(hidden_size))
        self.theta_log = nn.Parameter(torch.empty(hidden_size))
        self.gamma_log = nn.Parameter(torch.empty(hidden_size))

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_size),
            nn.Dropout(dropout),
        )

        self.reset_parameters()

    def reset_parameters(self):
        # Ring initialization for lambda magnitude
        with torch.no_grad():
            r = torch.empty(self.hidden_size).uniform_(0.8, 0.99)
            theta = torch.empty(self.hidden_size).uniform_(1e-3, 2 * math.pi)
            nu = -torch.log(r)
            self.nu_log.copy_(torch.log(nu))
            self.theta_log.copy_(torch.log(theta))
            self.gamma_log.copy_(torch.log(torch.sqrt(1.0 - r * r)))

        for p in [self.B_re, self.B_im, self.C_re, self.C_im, self.D]:
            nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, x):
        # x: [B, L, D]
        bsz, seq_len, _ = x.size()
        device = x.device
        dtype = x.dtype

        # Lambda = exp(-exp(nu_log) + i * exp(theta_log))
        nu = torch.exp(self.nu_log)
        theta = torch.exp(self.theta_log)
        lam_re = torch.exp(-nu) * torch.cos(theta)
        lam_im = torch.exp(-nu) * torch.sin(theta)

        gamma = torch.exp(self.gamma_log)

        h_re = torch.zeros((bsz, self.hidden_size), device=device, dtype=dtype)
        h_im = torch.zeros((bsz, self.hidden_size), device=device, dtype=dtype)

        out = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            bx_re = F.linear(x_t, self.B_re)
            bx_im = F.linear(x_t, self.B_im)

            h_re = lam_re * h_re - lam_im * h_im + gamma * bx_re
            h_im = lam_re * h_im + lam_im * h_re + gamma * bx_im

            ch_re = F.linear(h_re, self.C_re) - F.linear(h_im, self.C_im)
            dx = F.linear(x_t, self.D)
            y_t = ch_re + dx
            out.append(y_t)

        y = torch.stack(out, dim=1)
        y = self.dropout(y)
        y = self.norm1(x + y)

        z = self.ffn(y)
        z = self.norm2(y + z)
        return z


class LRURec(SequentialRecommender):
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

        self.ffn_hidden_size = _cfg("ffn_hidden_size", self.hidden_size * 4)

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.pos_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(self.hidden_size)

        self.layers = nn.ModuleList(
            [
                LRUBlock(
                    hidden_size=self.hidden_size,
                    ffn_hidden=self.ffn_hidden_size,
                    dropout=self.hidden_dropout_prob,
                )
                for _ in range(self.num_layers)
            ]
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

    def forward(self, item_seq):
        # item_seq: [B, L]
        bsz, seq_len = item_seq.size()
        pos_ids = torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand(bsz, -1)

        x = self.item_embedding(item_seq) + self.pos_embedding(pos_ids)
        x = self.layer_norm(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)
        return x

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            return self.loss_fct(pos_score, neg_score)
        logits = torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
        return self.loss_fct(logits, pos_items)

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq)
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
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        scores = torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
        return scores
