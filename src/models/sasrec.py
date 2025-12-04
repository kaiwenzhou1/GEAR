import numpy as np
import torch
import torch.nn as nn


class SASRec(torch.nn.Module):
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
        super(SASRec, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.d_model = d_model
        self.max_len = max_len
        # self.device = device

        # Embeddings
        self.user_emb = nn.Embedding(num_users, d_model)
        self.item_emb = nn.Embedding(num_items + 5, d_model, padding_idx=0)  # +1 for padding index
        self.pos_emb = nn.Embedding(2 * max_len + 1, d_model)

        # Transformer layers
        self.attention_blocks = nn.ModuleList([
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
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.pos_emb.weight)

    def log2feats(self, log_seqs):
        # Input embeddings
        seqs = self.item_emb(log_seqs)  # [batch_size, seq_len, d_model]
        poss = torch.arange(seqs.size(1), device=log_seqs.device).unsqueeze(0).expand(seqs.size(0), -1)  # position ids
        seqs += self.pos_emb(poss)

        seq_len = log_seqs.size(1)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(log_seqs.device)

        # Dropout
        seqs = self.dropout(seqs)

        # Transformer encoder
        seqs = self.LayerNorm(seqs)
        for block in self.attention_blocks:
            seqs = block(seqs, src_mask=attn_mask)

        return seqs

    def forward(self, log_seqs):
        # Sequence embeddings
        seq_feats = self.log2feats(log_seqs)  # [batch_size, seq_len, d_model]

        # Predict next item for each position
        logits = torch.matmul(seq_feats, self.item_emb.weight.T)  # [batch_size, seq_len, num_items + 1]

        # Select logits for even positions (0, 2, 4, ...)
        # logits = logits[:, ::2, :]   # [batch_size, seq_len // 2, num_items + 1]

        return logits
    
    def predict(self, log_seqs, candidates):
        # Sequence embeddings
        seq_feats = self.log2feats(log_seqs)  # [batch_size, seq_len, hidden_dim]

        # Get the last relevant sequence embedding for prediction
        seq_feats = seq_feats[:, -1, :]  # [batch_size, hidden_dim]

        # Candidate embeddings
        candidate_embs = self.item_emb(candidates)  # [batch_size, num_candidates, hidden_dim]

        # Dot product for logits
        logits = torch.bmm(candidate_embs, seq_feats.unsqueeze(-1)).squeeze(-1)  # [batch_size, num_candidates]

        return logits
    