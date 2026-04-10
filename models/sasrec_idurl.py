import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss


class SASRec_IDURL(SequentialRecommender):
    """IDURL-style SASRec backbone with offline newc_degree/sem_aug fields."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']
        self.inner_size = config['inner_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

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

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=0)
        else:
            raise NotImplementedError("loss_type must be one of ['BPR', 'CE']")

        self.k = config['n_facet_all']
        self.project_arr = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.k)])
        self.after_proj_ln_arr = nn.ModuleList([nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps) for _ in range(self.k)])
        self.repr_dp = config['new_repr_dp']
        self.repr_dropout = nn.Dropout(self.hidden_dropout_prob) if self.repr_dp else nn.Identity()
        self.new_repr_ln_arr = nn.ModuleList([nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps) for _ in range(self.k)])
        self.idra_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.disen_lambda = config['disen_lambda']
        self.disen_loss_fct = nn.CrossEntropyLoss()

        self.idra = config['idra']
        self.batch_size = config['train_batch_size']
        self.align_lambda = config['align_lambda']
        self.tau = 1.0
        self.mask_default = self.mask_correlated_samples(self.batch_size)
        self.cl_loss_fct = nn.CrossEntropyLoss()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        max_len = attention_mask.size(-1)
        subsequent_mask = torch.triu(torch.ones((1, max_len, max_len), device=item_seq.device), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1).long()
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        return (1.0 - extended_attention_mask) * -10000.0

    def mask_correlated_samples(self, batch_size):
        n = 2 * batch_size
        mask = torch.ones((n, n), dtype=torch.bool)
        mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def cl_task(self, z_i, z_j, temp, batch_size):
        n = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        sim = torch.mm(z, z.T) / temp
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(n, 1)
        mask = self.mask_correlated_samples(batch_size) if batch_size != self.batch_size else self.mask_default
        negative_samples = sim[mask].reshape(n, -1)
        labels = torch.zeros(n, device=positive_samples.device, dtype=torch.long)
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def get_repr_list(self, ori_u_repr):
        repr_list = []
        for i in range(self.k):
            projected_repr = self.project_arr[i](ori_u_repr)
            drift_repr = self.after_proj_ln_arr[i](self.repr_dropout(projected_repr) + ori_u_repr)
            repr_list.append(drift_repr)

        drift_repr_list = []
        for i, new_repr in enumerate(repr_list):
            drift_repr_list.append(self.new_repr_ln_arr[i](self.repr_dropout(new_repr)).unsqueeze(-1))
        return drift_repr_list

    def forward(self, item_seq):
        pos_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        pos_ids = pos_ids.unsqueeze(0).expand_as(item_seq)
        input_emb = self.item_embedding(item_seq) + self.position_embedding(pos_ids)
        input_emb = self.layer_norm(input_emb)
        input_emb = self.dropout(input_emb)
        trm_output = self.trm_encoder(input_emb, self.get_attention_mask(item_seq), output_all_encoded_layers=True)
        return trm_output[-1]

    def calculate_loss_prob(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        seq_output = self.forward(item_seq)
        ori_u_repr = self.gather_indexes(seq_output, item_seq_len - 1).unsqueeze(1)
        drift_repr_list = self.get_repr_list(ori_u_repr)

        if 'newc_degree' not in interaction:
            raise ValueError(
                "IDURL requires offline field `newc_degree` in dataset. "
                "Please prepare IDURL dataset files first."
            )
        target_item_drift_degree = interaction['newc_degree'].long()

        loss = 0.0

        if self.idra == 1 and self.training:
            repr_sum = torch.sum(torch.cat(drift_repr_list, dim=1), dim=1).squeeze(-1)
            un_aug_repr = self.idra_ln(self.dropout(repr_sum))

            if 'sem_aug' not in interaction or 'sem_aug_lengths' not in interaction:
                raise ValueError(
                    "IDURL with IDRA requires offline fields `sem_aug` and `sem_aug_lengths` in dataset."
                )
            su_aug_seq = interaction['sem_aug']
            su_aug_lengths = interaction['sem_aug_lengths'].long().clamp_min(1).clamp_max(item_seq.size(1))

            su_seq_output = self.forward(su_aug_seq)
            su_u_repr = self.gather_indexes(su_seq_output, su_aug_lengths - 1).unsqueeze(1)
            su_drift_repr_list = self.get_repr_list(su_u_repr)
            su_repr_sum = torch.sum(torch.cat(su_drift_repr_list, dim=1), dim=1).squeeze(-1)
            su_aug_repr = self.idra_ln(su_repr_sum)

            cl_logits, cl_labels = self.cl_task(un_aug_repr, su_aug_repr, temp=self.tau, batch_size=item_seq_len.shape[0])
            loss = loss + self.align_lambda * self.cl_loss_fct(cl_logits, cl_labels)

        if self.disen_lambda > 0 and self.training:
            labels = target_item_drift_degree
            if labels.min() >= 1 and labels.max() <= self.k:
                labels = labels - 1
            labels = labels.clamp(0, self.k - 1)
            target_item_emb = self.item_embedding(pos_items).unsqueeze(1)
            drift_reprs = torch.cat(drift_repr_list, dim=1).squeeze(-1)
            disen_logits = (target_item_emb * drift_reprs).sum(-1)
            loss = loss + self.disen_lambda * self.disen_loss_fct(disen_logits.view(-1, self.k), labels)

        logits_list = []
        for drift_repr in drift_repr_list:
            logits = F.linear(drift_repr.squeeze(-1), self.item_embedding.weight, None)
            logits_list.append(logits.unsqueeze(-1))

        candi_query_logits = torch.cat(logits_list, dim=-1)
        candi_id_distr = candi_query_logits.softmax(dim=-1)
        final_logits = (candi_id_distr * candi_query_logits).sum(dim=-1)
        prediction_prob = final_logits.softmax(dim=-1)

        inp = torch.log(prediction_prob.view(-1, self.n_items) + 1e-8)
        ce_loss = self.loss_fct(inp, pos_items.view(-1)).mean()
        return loss + ce_loss, prediction_prob.squeeze(1)

    def calculate_loss(self, interaction):
        loss, _ = self.calculate_loss_prob(interaction)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_item_emb = self.item_embedding(test_item)
        return torch.mul(seq_output, test_item_emb).sum(dim=1)

    def full_sort_predict(self, interaction):
        _, prediction_prob = self.calculate_loss_prob(interaction)
        return prediction_prob
