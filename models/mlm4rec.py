import math

import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SequenceMixer(nn.Module):
    def __init__(self, seq_len, hidden_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(seq_len, seq_len)
        self.fc2 = nn.Linear(seq_len, seq_len)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, D]
        x_norm = self.norm(x)
        x_t = x_norm.permute(0, 2, 1)  # [B, D, N]
        x_t = self.fc2(self.act(self.fc1(x_t)))
        x_t = self.drop(x_t)
        return x + x_t.permute(0, 2, 1)


class ChannelMixer(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x_norm = self.norm(x)
        x_out = self.fc2(self.act(self.fc1(x_norm)))
        x_out = self.drop(x_out)
        return x + x_out


class GlobalEncoder(nn.Module):
    def __init__(self, seq_len, hidden_dim, n_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.ModuleList([SequenceMixer(seq_len, hidden_dim, dropout),
                            ChannelMixer(hidden_dim, dropout)])
             for _ in range(n_layers)]
        )

    def forward(self, x):
        for seq_mixer, chan_mixer in self.layers:
            x = seq_mixer(x)
            x = chan_mixer(x)
        return x


class MultiGranularityCLE(nn.Module):
    def __init__(self, hidden_dim_half, kernel_sizes):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim_half)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    hidden_dim_half,
                    hidden_dim_half,
                    k,
                    stride=1,
                    padding=k // 2,
                    groups=hidden_dim_half,
                )
                for k in kernel_sizes
            ]
        )
        self.act = nn.SiLU()

    def forward(self, h2):
        # h2: [B, N, D/2]
        h2n = self.norm(h2)
        h2n = h2n.permute(0, 2, 1)  # [B, D/2, N]
        outs = []
        for conv in self.convs:
            o = self.act(conv(h2n))
            outs.append(o)
        fused = outs[0]
        for o in outs[1:]:
            fused = fused * o
        fused = fused.permute(0, 2, 1)  # [B, N, D/2]
        return fused


class LocalEncoderBlock(nn.Module):
    def __init__(self, hidden_dim, local_hidden, kernel_sizes, dropout):
        super().__init__()
        if local_hidden % 2 != 0:
            raise ValueError("local_hidden must be even for CLE split")
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj_in = nn.Linear(hidden_dim, local_hidden)
        self.proj_out = nn.Linear(local_hidden // 2, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.cle = MultiGranularityCLE(local_hidden // 2, kernel_sizes)

    def forward(self, x):
        # x: [B, N, D]
        h = self.act(self.proj_in(self.norm(x)))  # [B, N, Hl]
        h1, h2 = torch.chunk(h, 2, dim=-1)
        h2g = self.cle(h2)
        cle_out = h1 * h2g
        out = self.proj_out(cle_out)
        out = self.drop(out)
        return x + out


class LocalEncoder(nn.Module):
    def __init__(self, hidden_dim, local_hidden, kernel_sizes, n_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                LocalEncoderBlock(hidden_dim, local_hidden, kernel_sizes, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class AdaptiveFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.att_fc = nn.Linear(hidden_dim, 1)
        self.wg = nn.Linear(hidden_dim, 1)
        self.wl = nn.Linear(hidden_dim, 1)

    def _att_pool(self, x):
        # x: [B, N, D]
        att = self.att_fc(x)  # [B, N, 1]
        att = torch.softmax(att, dim=1)
        pooled = torch.sum(att * x, dim=1)  # [B, D]
        return pooled

    def forward(self, yg, yl):
        y_g = self._att_pool(yg)
        y_l = self._att_pool(yl)
        wg = self.wg(y_g)
        wl = self.wl(y_l)
        w = torch.softmax(torch.cat([wg, wl], dim=-1), dim=-1)
        w_g = w[:, 0].view(-1, 1, 1)
        w_l = w[:, 1].view(-1, 1, 1)
        return w_g * yg + w_l * yl


class MLM4Rec(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        def _cfg(name, default):
            return config[name] if name in config else default

        self.hidden_size = config["hidden_size"]
        self.num_layers = _cfg("num_hidden_layers", 2)
        self.hidden_dropout_prob = _cfg("hidden_dropout_prob", 0.5)
        self.initializer_range = _cfg("initializer_range", 0.02)
        self.max_seq_length = _cfg("max_seq_length", config["MAX_ITEM_LIST_LENGTH"])
        self.loss_type = _cfg("loss_type", "BPR")

        self.local_hidden = _cfg("local_hidden_size", self.hidden_size * 2)
        self.kernel_sizes = _cfg("kernel_sizes", [3, 5, 7, 9])

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.global_encoder = GlobalEncoder(
            seq_len=self.max_seq_length,
            hidden_dim=self.hidden_size,
            n_layers=self.num_layers,
            dropout=self.hidden_dropout_prob,
        )
        self.local_encoder = LocalEncoder(
            hidden_dim=self.hidden_size,
            local_hidden=self.local_hidden,
            kernel_sizes=self.kernel_sizes,
            n_layers=self.num_layers,
            dropout=self.hidden_dropout_prob,
        )
        self.afl = AdaptiveFusion(self.hidden_size)
        self.pred_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.pred_fc2 = nn.Linear(self.hidden_size, self.n_items)
        self.pred_act = nn.GELU()

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
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq):
        x = self.item_embedding(item_seq)
        x = self.dropout(x)
        yg = self.global_encoder(x)
        yl = self.local_encoder(x)
        y = self.afl(yg, yl)
        return y

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
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        logits = self.pred_fc2(self.pred_act(self.pred_fc1(seq_output)))
        loss = self.loss_fct(logits, pos_items)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        if self.loss_type == "CE":
            logits = self.pred_fc2(self.pred_act(self.pred_fc1(seq_output)))
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
        if self.loss_type == "CE":
            scores = self.pred_fc2(self.pred_act(self.pred_fc1(seq_output)))
        else:
            test_items_emb = self.item_embedding.weight
            scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores
