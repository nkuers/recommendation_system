import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.utils import InputType

try:
    from models.ti_transformer import TransformerEncoder
except ModuleNotFoundError:
    from .ti_transformer import TransformerEncoder


class TiSASRec(SequentialRecommender):
    input_type = InputType.PAIRWISE
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
        loss_type = config["loss_type"] if "loss_type" in config else "BCE"
        if not isinstance(loss_type, str):
            loss_type = str(loss_type)
        loss_type = loss_type.upper()
        if loss_type not in {"BPR", "CE", "BCE"}:
            loss_type = "BCE"
        self.loss_type = loss_type
        self.time_span = config["time_span"]

        self.TIME_FIELD = config["TIME_FIELD"]
        self.TIME_SEQ = self.TIME_FIELD + config["LIST_SUFFIX"]

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.time_matrix_K_emb = nn.Embedding(self.time_span + 1, self.hidden_size)
        self.time_matrix_V_emb = nn.Embedding(self.time_span + 1, self.hidden_size)

        self.position_key_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.position_value_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
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

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            self.loss_fct = None

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def compute_time_matrix(self, time_stamp, item_seq):
        seq_len = time_stamp.shape[1]
        time_stamp = time_stamp.float()
        valid_mask = item_seq > 0

        masked_time = time_stamp.clone()
        masked_time[~valid_mask] = float("inf")
        first_time = masked_time.min(dim=1).values
        first_time = torch.where(torch.isfinite(first_time), first_time, torch.zeros_like(first_time))
        time_stamp = time_stamp.clone()
        time_stamp[~valid_mask] = first_time.unsqueeze(1).expand(-1, seq_len)[~valid_mask]

        timestamp_matrix = time_stamp.unsqueeze(1)
        timestamp_matrix_repeat = timestamp_matrix.repeat(1, seq_len, 1)
        diff_matrix = torch.abs(timestamp_matrix_repeat.transpose(-1, -2) - timestamp_matrix_repeat)

        valid_pair = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
        diff_pos = diff_matrix.masked_fill(~valid_pair, float("inf"))
        diff_pos = diff_pos.masked_fill(diff_pos <= 0, float("inf"))
        min_interval = diff_pos.min(dim=-1).values.min(dim=-1).values
        min_interval = torch.where(torch.isfinite(min_interval), min_interval, torch.ones_like(min_interval))
        min_interval = torch.clamp(min_interval, min=1.0)

        scaled = torch.floor(diff_matrix / min_interval.view(-1, 1, 1)).to(torch.long)
        return torch.clamp(scaled, max=self.time_span)

    def forward(self, item_seq, item_seq_len, time_seq):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_key = self.position_key_embedding(position_ids)
        position_value = self.position_value_embedding(position_ids)

        time_matrix = self.compute_time_matrix(time_seq, item_seq)
        time_key = self.dropout(self.time_matrix_K_emb(time_matrix))
        time_value = self.dropout(self.time_matrix_V_emb(time_matrix))

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb
        input_emb = self.layer_norm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb,
            extended_attention_mask,
            time_key,
            time_value,
            position_key=position_key,
            position_value=position_value,
            output_all_encoded_layers=True,
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_seq = interaction[self.TIME_SEQ]
        seq_output = self.forward(item_seq, item_seq_len, time_seq)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
        elif self.loss_type == "BCE":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-8)
            neg_loss = -torch.log(1.0 - torch.sigmoid(neg_score) + 1e-8)
            loss = (pos_loss + neg_loss).mean()
        else:
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        time_seq = interaction[self.TIME_SEQ]
        seq_output = self.forward(item_seq, item_seq_len, time_seq)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_seq = interaction[self.TIME_SEQ]
        seq_output = self.forward(item_seq, item_seq_len, time_seq)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores
