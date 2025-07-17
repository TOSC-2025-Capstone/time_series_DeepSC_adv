import torch
import torch.nn as nn
import torch.nn.functional as F
from models.lstm import *

class LSTMAttentionDeepSC(nn.Module):
    """LSTM + Self-Attention 기반 DeepSC 모델"""
    def __init__(self, input_dim=6, seq_len=128, hidden_dim=128, compressed_len=64, compressed_features=2, num_layers=2, dropout=0.1, num_heads=4, params=None, **kwargs):
        super().__init__()
        p = params if params is not None else {}
        self.input_dim = p.get("input_dim", input_dim)
        self.seq_len = p.get("seq_len", seq_len)
        self.hidden_dim = p.get("hidden_dim", hidden_dim)
        self.compressed_len = p.get("compressed_len", compressed_len)
        self.compressed_features = p.get("compressed_features", compressed_features)
        self.num_layers = p.get("num_layers", num_layers)
        self.dropout = p.get("dropout", dropout)
        self.num_heads = p.get("num_heads", num_heads)
        self.encoder = LSTMCompressor_Both(
            self.input_dim, self.hidden_dim, self.compressed_len, self.compressed_features, self.num_layers, self.dropout
        )
        # attention의 embed_dim은 compressed_features로 맞춤
        self.attn = nn.MultiheadAttention(embed_dim=self.compressed_features, num_heads=self.num_heads, batch_first=True)
        self.decoder = LSTMDecompressor_Both(
            self.compressed_features, self.hidden_dim, self.seq_len, self.input_dim, self.num_layers, self.dropout
        )
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        compressed = self.encoder(x)  # [batch, compressed_len, compressed_features]
        # Self-attention 적용
        attn_out, _ = self.attn(compressed, compressed, compressed)  # [batch, compressed_len, compressed_features]
        reconstructed = self.decoder(attn_out)  # [batch, seq_len, input_dim-2]
        return reconstructed
    def get_compression_ratio(self):
        original_size = self.input_dim * self.seq_len
        compressed_size = self.compressed_len * self.compressed_features
        return compressed_size / original_size