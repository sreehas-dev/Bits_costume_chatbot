import torch
import torch.nn as nn

class BiEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            emb_dim, hidden_dim, batch_first=True, bidirectional=True
        )

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.encoder(emb)
        return out.mean(dim=1)
