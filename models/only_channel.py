import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Channels

# 모델
class OnlyChannel(nn.Module):
    def __init__(self, input_dim=6, seq_len=128, hidden_dim=128, compressed_len=64, compressed_features=3, num_layers=2, dropout=0.1, params=None, **kwargs):
        super().__init__()
        self.channels = Channels()

    def forward(self, x):
        noised = self.channels.Rayleigh(x, 0.1)

        return noised

    def get_compression_ratio(self):
        """압축률 계산"""
        original_size = self.input_dim * self.seq_len
        compressed_size = self.compressed_len * self.compressed_features
        return compressed_size / original_size
