import numpy as np
import torch
import torch.nn as nn
from .embedding import SinusoidalPositionEmbedding
from .utils import CustomTransformerEncoderLayer


class GEAR(torch.nn.Module):
    def __init__(self, 
        max_len: int = None,
        num_items: int = None,
        n_layer: int = None,
        n_head: int = None,
        num_users: int = None,
        n_b: int = None,
        d_model: int = None,
        dropout: float = .0,
    ):
        super(GEAR, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.d_model = d_model
        self.max_len = max_len
        self.low_n_layer = 1
        self.n_head = n_head
        # self.device = device
        self.slopes = torch.pow(2, -8 / self.n_head * torch.arange(self.n_head, dtype=torch.float32))  

        # Embeddings
        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)  # +1 for padding index
        self.pos_emb = SinusoidalPositionEmbedding(2 * max_len, d_model)
        self.behavior_emb = nn.Embedding(n_b + 1, d_model, padding_idx=0)

        # Transformer layers
        self.item_attention_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
            for _ in range(self.low_n_layer)
        ])
        self.behavior_attention_blocks = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
            for _ in range(self.low_n_layer)
        ])
        self.upper_transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
            for _ in range(n_layer)
        ])

        # Dropout and activation
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-6)

        # Initialization
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.item_emb.weight)

    def log2feats(self, item_seqs, behavior_seqs, time_bias):
        device = item_seqs.device

        slopes = self.slopes.view(1, self.n_head, 1, 1).to(device)
        # time_bias = torch.where(time_bias > 0, torch.log(time_bias), torch.tensor(0.0))
        time_bias = torch.log(1 + torch.clamp(time_bias, min=0))
        time_bias = time_bias.unsqueeze(1).repeat(1, self.n_head, 1, 1)
        time_bias = -slopes * time_bias

        # Input embeddings
        seqs_i = self.item_emb(item_seqs)  # [batch_size, seq_len, d_model]
        # poss = self.pos_emb(item_seqs).to(device)  # position ids
        # seqs_i = seqs_i + poss

        seqs_b = self.behavior_emb(behavior_seqs) 
        # + self.pos_emb(behavior_seqs).to(device)

        # Transformer
        seq_len_i = item_seqs.size(1)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(seq_len_i).to(device)
        seqs_i = self.dropout(seqs_i)
        seqs_i = self.LayerNorm(seqs_i)
        for block in self.item_attention_blocks:
            seqs_i = block(seqs_i, src_mask=attn_mask)
        
        seq_len_b = behavior_seqs.size(1)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(seq_len_b).to(device)
        seqs_b = self.dropout(seqs_b)
        seqs_b = self.LayerNorm(seqs_b)
        for block in self.behavior_attention_blocks:
            seqs_b = block(seqs_b, src_mask=attn_mask, time_bias=time_bias)

        # Dropout
        seqs_i = self.dropout(seqs_i)
        seqs_b = self.dropout(seqs_b)

        seqs_alt = torch.empty(item_seqs.size(0), seq_len_b + seq_len_i, self.d_model, dtype=seqs_i.dtype, device=device)
        seqs_alt[:, 1::2, :] = seqs_i
        seqs_alt[:, 0::2, :] = seqs_b

        # Transformer encoder
        # seqs = self.LayerNorm(seqs)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(seq_len_b + seq_len_i).to(device)
        for block in self.upper_transformer:
            seqs_alt = block(seqs_alt, src_mask=attn_mask)

        return seqs_alt


    def forward(self, item_seqs, behavior_seqs, time_bias):
        time_bias = time_bias.to(torch.float32)
        seq_feats = self.log2feats(item_seqs, behavior_seqs, time_bias)  # [batch_size, seq_len * 2, d_model]

        seq_feats_item = seq_feats[:, 1::2, :]  # [batch_size, seq_len_item, d_model]
        seq_feats_behavior = seq_feats[:, 0::2, :]  # [batch_size, seq_len_behavior, d_model]

        logits_item = torch.matmul(seq_feats_behavior, self.item_emb.weight.T)  # [batch_size, seq_len_item, num_items + 1]
        logits_behavior = torch.matmul(seq_feats_item, self.behavior_emb.weight.T)  # [batch_size, seq_len_behavior, num_behaviors + 1]

        return logits_item, logits_behavior

    
    def predict(self, item_seqs, behavior_seqs, time_bias, candidates):
        # Sequence embeddings
        seq_feats = self.log2feats(item_seqs, behavior_seqs, time_bias)  # [batch_size, seq_len, hidden_dim]

        # Get the last relevant sequence embedding for prediction
        seq_feats = seq_feats[:, -1, :]  # [batch_size, hidden_dim]

        # Candidate embeddings
        candidate_embs = self.item_emb(candidates)  # [batch_size, num_candidates, hidden_dim]

        # Dot product for logits
        logits = torch.bmm(candidate_embs, seq_feats.unsqueeze(-1)).squeeze(-1)  # [batch_size, num_candidates]

        return logits
    