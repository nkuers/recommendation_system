import torch
from torch import nn
import torch.nn.functional as F
from collections import defaultdict

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder


class BASRec(SequentialRecommender):
    """RecBole adaptation of BASRec-style balanced augmentation over a SASRec backbone."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]

        self.rec_weight = float(config["rec_weight"])
        self.aml_weight = float(config["aml_weight"])
        self.wml_weight = float(config["wml_weight"])
        self.n_pairs = int(config["n_pairs"])
        self.n_whole_level = int(config["n_whole_level"])
        self.substitute_rate = float(config["substitute_rate"])
        self.reorder_rate = float(config["reorder_rate"])
        self.rate_min = float(config["rate_min"])
        self.rate_max = float(config["rate_max"])
        self.substitute_topk = int(config["substitute_topk"])
        self.beta = float(config["beta"])
        self.base_augment_type = str(config["base_augment_type"])

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.beta_dist = torch.distributions.Beta(self.beta, self.beta)
        self.sim_neighbors, self.sim_scores = self._build_item_similarity(dataset)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _get_attention_mask(self, item_seq):
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        max_len = attention_mask.size(-1)
        subsequent_mask = torch.triu(
            torch.ones((1, max_len, max_len), device=item_seq.device), diagonal=1
        )
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1).long()
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        return (1.0 - extended_attention_mask) * -10000.0

    def forward(self, item_seq):
        pos_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        pos_ids = pos_ids.unsqueeze(0).expand_as(item_seq)
        input_emb = self.item_embedding(item_seq) + self.position_embedding(pos_ids)
        input_emb = self.layer_norm(input_emb)
        input_emb = self.dropout(input_emb)
        trm_output = self.trm_encoder(
            input_emb,
            self._get_attention_mask(item_seq),
            output_all_encoded_layers=True,
        )
        return trm_output[-1]

    def _build_pos_seq(self, item_seq, item_seq_len, pos_items):
        pos_seq = torch.zeros_like(item_seq)
        bsz = item_seq.size(0)
        for i in range(bsz):
            l = int(item_seq_len[i].item())
            if l <= 0:
                continue
            if l > 1:
                pos_seq[i, : l - 1] = item_seq[i, 1:l]
            pos_seq[i, l - 1] = pos_items[i]
        return pos_seq

    def _sample_neg_seq(self, item_seq, item_seq_len):
        neg_seq = torch.zeros_like(item_seq)
        bsz, seqlen = item_seq.size()
        for i in range(bsz):
            l = int(item_seq_len[i].item())
            if l <= 0:
                continue
            seen = set(item_seq[i, :l].tolist())
            for t in range(l):
                n = int(torch.randint(1, self.n_items, (1,), device=item_seq.device).item())
                retry = 0
                while n in seen and retry < 20:
                    n = int(torch.randint(1, self.n_items, (1,), device=item_seq.device).item())
                    retry += 1
                neg_seq[i, t] = n
        return neg_seq

    def _bce_from_seq_output(self, seq_output, pos_seq, neg_seq, sample_weight=None):
        pos_emb = self.item_embedding(pos_seq)
        neg_emb = self.item_embedding(neg_seq)

        pos_logits = (seq_output * pos_emb).sum(dim=-1)
        neg_logits = (seq_output * neg_emb).sum(dim=-1)
        mask = (pos_seq > 0).float()

        pos_loss = -torch.log(torch.sigmoid(pos_logits) + 1e-24) * mask
        neg_loss = -torch.log(1.0 - torch.sigmoid(neg_logits) + 1e-24) * mask
        loss = pos_loss + neg_loss
        if sample_weight is not None:
            sw = sample_weight.clamp_min(1e-6).unsqueeze(1)
            loss = loss * sw

        denom = mask.sum().clamp_min(1.0)
        return loss.sum() / denom

    def _mix_bce_loss(self, mixed_seq_output, pos_a, neg_a, pos_b, neg_b, alpha_weight):
        alpha = alpha_weight.clamp(0.0, 1.0).unsqueeze(1)
        loss_a = self._bce_from_seq_output(mixed_seq_output, pos_a, neg_a, sample_weight=alpha_weight)
        loss_b = self._bce_from_seq_output(mixed_seq_output, pos_b, neg_b, sample_weight=(1.0 - alpha_weight))
        return alpha.mean() * loss_a + (1.0 - alpha.mean()) * loss_b

    def _build_item_similarity(self, dataset):
        inter = dataset.inter_feat
        user_ids = inter[self.USER_ID].cpu().numpy().tolist()
        item_ids = inter[self.ITEM_ID].cpu().numpy().tolist()
        time_field = None
        if hasattr(dataset, "time_field") and dataset.time_field in inter:
            time_field = dataset.time_field
        elif "timestamp" in inter:
            time_field = "timestamp"
        elif "timestamp:float" in inter:
            time_field = "timestamp:float"

        if time_field is not None:
            times = inter[time_field].cpu().numpy().tolist()
            order = sorted(range(len(user_ids)), key=lambda i: (user_ids[i], times[i], i))
        else:
            order = sorted(range(len(user_ids)), key=lambda i: (user_ids[i], i))

        co = defaultdict(lambda: defaultdict(int))
        prev_u = None
        prev_i = None
        for idx in order:
            u = int(user_ids[idx])
            it = int(item_ids[idx])
            if it <= 0:
                continue
            if prev_u == u and prev_i is not None and prev_i > 0 and prev_i != it:
                co[prev_i][it] += 1
                co[it][prev_i] += 1
            prev_u = u
            prev_i = it

        neighbors = {}
        scores = {}
        for it, rel in co.items():
            top = sorted(rel.items(), key=lambda x: x[1], reverse=True)[: self.substitute_topk]
            if not top:
                continue
            mx = float(max(v for _, v in top))
            neighbors[it] = [int(j) for j, _ in top]
            scores[it] = [float(v) / mx if mx > 0 else 0.0 for _, v in top]
        return neighbors, scores

    def _sample_rate(self):
        if self.rate_max > self.rate_min:
            return float(torch.empty(1).uniform_(self.rate_min, self.rate_max).item())
        # backward compatible fallback
        return float(max(self.rate_min, self.substitute_rate if self.base_augment_type == "substitute" else self.reorder_rate))

    def _reorder_aug(self, item_seq, item_seq_len):
        aug = item_seq.clone()
        aug_weight = torch.zeros(item_seq.size(0), device=item_seq.device)
        for i in range(item_seq.size(0)):
            l = int(item_seq_len[i].item())
            if l <= 2:
                aug_weight[i] = 1.0
                continue
            rate = self._sample_rate()
            seg = max(2, int(l * rate))
            seg = min(seg, l)
            start = torch.randint(0, l - seg + 1, (1,), device=item_seq.device).item()
            part = aug[i, start:start + seg].clone()
            perm = torch.randperm(seg, device=item_seq.device)
            aug[i, start:start + seg] = part[perm]
            aug_weight[i] = max(rate, 1e-6)
        return aug, aug_weight

    def _substitute_aug(self, item_seq, item_seq_len):
        aug = item_seq.clone()
        aug_weight = torch.zeros(item_seq.size(0), device=item_seq.device)
        for i in range(item_seq.size(0)):
            l = int(item_seq_len[i].item())
            if l <= 1:
                aug_weight[i] = 1.0
                continue
            rate = self._sample_rate()
            n_sub = max(1, int(l * rate))
            pos = torch.randperm(l, device=item_seq.device)[:n_sub]
            sim_score_sum = 0.0
            sim_used = 0
            for p in pos.tolist():
                src = int(aug[i, p].item())
                cand = self.sim_neighbors.get(src)
                cand_scores = self.sim_scores.get(src)
                if cand:
                    ridx = int(torch.randint(0, len(cand), (1,), device=item_seq.device).item())
                    aug[i, p] = int(cand[ridx])
                    if cand_scores and ridx < len(cand_scores):
                        sim_score_sum += float(cand_scores[ridx])
                        sim_used += 1
                else:
                    aug[i, p] = int(torch.randint(1, self.n_items, (1,), device=item_seq.device).item())
            quality = (sim_score_sum / sim_used) if sim_used > 0 else 0.5
            aug_weight[i] = max(rate * quality, 1e-6)
        return aug, aug_weight

    def _augment(self, item_seq, item_seq_len):
        if self.base_augment_type == "reorder":
            return self._reorder_aug(item_seq, item_seq_len)
        if self.base_augment_type == "substitute":
            return self._substitute_aug(item_seq, item_seq_len)
        if torch.rand(1, device=item_seq.device).item() < 0.5:
            return self._reorder_aug(item_seq, item_seq_len)
        return self._substitute_aug(item_seq, item_seq_len)

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        pos_seq = self._build_pos_seq(item_seq, item_seq_len, pos_items)
        neg_seq = self._sample_neg_seq(item_seq, item_seq_len)

        seq_output = self.forward(item_seq)
        rec_loss = self._bce_from_seq_output(seq_output, pos_seq, neg_seq)
        total_loss = self.rec_weight * rec_loss

        if self.training:
            aug_loss = 0.0
            if self.aml_weight > 0:
                for _ in range(self.n_pairs):
                    for _ in range(2):
                        aug_seq, aug_weight = self._augment(item_seq, item_seq_len)
                        aug_output = self.forward(aug_seq)
                        aug_pos_seq = self._build_pos_seq(aug_seq, item_seq_len, pos_items)
                        aug_neg_seq = self._sample_neg_seq(aug_seq, item_seq_len)
                        mix_weight = self.beta_dist.sample((item_seq.size(0),)).to(item_seq.device).unsqueeze(1)
                        mixed_output = mix_weight.unsqueeze(-1) * seq_output + (1.0 - mix_weight.unsqueeze(-1)) * aug_output

                        loss_weight = 1.0 / (mix_weight.squeeze(1).clamp_min(1e-6) * aug_weight.clamp_min(1e-6))
                        lw_min = loss_weight.min()
                        lw_max = loss_weight.max()
                        if (lw_max - lw_min) > 1e-8:
                            loss_weight = (loss_weight - lw_min) / (lw_max - lw_min)
                        else:
                            loss_weight = torch.ones_like(loss_weight)

                        aug_loss = aug_loss + self._bce_from_seq_output(
                            mixed_output, aug_pos_seq, aug_neg_seq, sample_weight=loss_weight
                        )
                aug_loss = aug_loss / max(self.n_pairs * 2, 1)
                total_loss = total_loss + self.aml_weight * aug_loss

            whole_mix_loss = 0.0
            if self.wml_weight > 0:
                for _ in range(self.n_whole_level):
                    perm = torch.randperm(item_seq.size(0), device=item_seq.device)
                    pos_seq_perm = pos_seq[perm]
                    neg_seq_perm = neg_seq[perm]

                    # Item-wise mix: one alpha per sample.
                    alpha_item = self.beta_dist.sample((item_seq.size(0),)).to(item_seq.device).unsqueeze(1)
                    mixed_item = alpha_item.unsqueeze(-1) * seq_output + (1.0 - alpha_item.unsqueeze(-1)) * seq_output[perm]
                    whole_mix_loss = whole_mix_loss + self._mix_bce_loss(
                        mixed_item,
                        pos_seq,
                        neg_seq,
                        pos_seq_perm,
                        neg_seq_perm,
                        alpha_item.squeeze(1),
                    )

                    # Feature-wise mix: one alpha per hidden feature.
                    alpha_feat = self.beta_dist.sample((item_seq.size(0), self.hidden_size)).to(item_seq.device)
                    mixed_feat = alpha_feat.unsqueeze(1) * seq_output + (1.0 - alpha_feat.unsqueeze(1)) * seq_output[perm]
                    whole_mix_loss = whole_mix_loss + self._mix_bce_loss(
                        mixed_feat,
                        pos_seq,
                        neg_seq,
                        pos_seq_perm,
                        neg_seq_perm,
                        alpha_feat.mean(dim=1),
                    )
                whole_mix_loss = whole_mix_loss / max(self.n_whole_level * 2, 1)
                total_loss = total_loss + self.wml_weight * whole_mix_loss

        return total_loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_item_emb = self.item_embedding(test_item)
        return torch.mul(seq_output, test_item_emb).sum(dim=1)

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        return torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
