import random

import torch
from torch import nn
from recbole.model.sequential_recommender import GRU4Rec


class EnvGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config['device']
        self.num_subseq = config['num_subseq']
        self.len_seq = config['MAX_ITEM_LIST_LENGTH']
        self.temp = config['temp']
        self.latent_dim = config['embedding_size']
        self.linear_in = nn.ModuleList([nn.Linear(self.len_seq, 128) for _ in range(self.num_subseq)])
        self.linear_out = nn.ModuleList([nn.Linear(128, self.len_seq) for _ in range(self.num_subseq)])
        self.activate = nn.ReLU()

    def get_env_mask(self, seq_item_emb):
        seq_emb = torch.mean(seq_item_emb, dim=-1)
        all_masked_seq_emb = []
        for k in range(self.num_subseq):
            tmp_logit = self.activate(self.linear_in[k](seq_emb))
            logit = self.linear_out[k](tmp_logit)
            eps = torch.rand(logit.shape, device=self.device)
            eps = torch.log(eps) - torch.log(1 - eps)
            mask = torch.sigmoid((logit + eps) / self.temp)
            mask_expand = mask.unsqueeze(-1).expand(-1, -1, self.latent_dim)
            all_masked_seq_emb.append(mask_expand * seq_item_emb)
        return all_masked_seq_emb


class IDEAGRU(GRU4Rec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.fake_ratio = config['fake_ratio']
        self.num_env = config['num_env']
        self.alpha = config['alpha']
        self.latent_dim = self.embedding_size
        self.seq_len = config['MAX_ITEM_LIST_LENGTH']
        self.var_hyper = config['var_hyper']

        self.num_fake_items = 1000 if self.fake_ratio > 0 else 0
        self.item_embedding = nn.Embedding(self.n_items + self.num_fake_items, self.embedding_size, padding_idx=0)
        self.env = EnvGenerator(config).to(self.device)

    def train_forward(self, item_seq_emb, item_seq_len):
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        return self.gather_indexes(gru_output, item_seq_len - 1)

    def env_mixup(self, all_masked_seq_emb):
        emb1, emb2 = random.sample(all_masked_seq_emb, 2)
        batch_size = emb1.shape[0]
        lambda_val = torch.distributions.Beta(self.alpha, self.alpha).sample([batch_size, 1, 1]).to(self.device)
        lambda_val = lambda_val.expand(-1, self.seq_len, self.latent_dim)
        return lambda_val * emb1 + (1 - lambda_val) * emb2

    def get_fake_seq(self, item_seq):
        if self.fake_ratio == 0 or self.num_fake_items == 0:
            return item_seq
        batch_size, _ = item_seq.size()
        fake_seq = item_seq.clone()
        for i in range(batch_size):
            seq_len = (item_seq[i] > 0).sum().item()
            num_fake = max(1, int(self.fake_ratio * seq_len))
            fake_idxs = random.sample(range(seq_len), num_fake)
            fake_items = torch.randint(self.n_items, self.n_items + self.num_fake_items, (num_fake,), device=self.device)
            fake_seq[i, fake_idxs] = fake_items
        return fake_seq

    def get_env_loss(self, seq_output, pos_items):
        logits = torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
        return self.loss_fct(logits, pos_items)

    def calculate_all_loss(self, interaction):
        seq_len = interaction[self.ITEM_SEQ_LEN]
        item_seq = interaction[self.ITEM_SEQ]
        pos_items = interaction[self.POS_ITEM_ID]

        seq_item_emb = self.item_embedding(self.get_fake_seq(item_seq))
        all_masked_seq_emb = self.env.get_env_mask(seq_item_emb)

        all_erm_loss = []
        for _ in range(self.num_env):
            mixed_seq_emb = self.env_mixup(all_masked_seq_emb)
            mixed_seq_output = self.train_forward(mixed_seq_emb, seq_len)
            all_erm_loss.append(self.get_env_loss(mixed_seq_output, pos_items))

        var_erm = torch.var(torch.stack(all_erm_loss))
        mean_erm = torch.mean(torch.stack(all_erm_loss))
        return mean_erm, var_erm

    def calculate_loss_env(self, interaction):
        _, var_erm = self.calculate_all_loss(interaction)
        return -var_erm

    def calculate_loss_il(self, interaction):
        mean_erm, var_erm = self.calculate_all_loss(interaction)
        return mean_erm, self.var_hyper * var_erm

    def calculate_loss(self, interaction):
        mean_erm, var_erm = self.calculate_all_loss(interaction)
        return mean_erm + self.var_hyper * var_erm

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        return torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
