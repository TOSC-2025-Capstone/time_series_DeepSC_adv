import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Channels
import pdb
# from parameters.parameters import noise_std

# 시퀀스, 피쳐 압축
class GRUCompressor_Both(nn.Module):
    def __init__(self, input_dim, seq_len, compressed_len=64, compressed_features=3, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, compressed_features, num_layers, dropout=dropout, batch_first=True)
        self.sequence_compress = nn.Linear(seq_len, compressed_len)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        # 피쳐 차원 압축 (input_dim -> compressed_features)
        feature_compressed, _ = self.gru(x)  # [batch, hidden_dim, 64]
        # 시계열 길이 압축 (seq_len -> compressed_len)
        feature_compressed = feature_compressed.permute(0, 2, 1)  # [batch, compressed_features, seq_len]
        compressed = self.sequence_compress(feature_compressed)  # [batch, 64, 3]
        compressed = compressed.permute(0, 2, 1)  # [batch, compressed_len, compressed_features]

        return compressed

# 시퀀스, 피쳐 복원
class GRUDecompressor_Both(nn.Module):
    def __init__(self, compressed_len, compressed_features, reconstruct_len=128, reconstruct_features=6, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(compressed_features, reconstruct_features, num_layers, dropout=dropout, batch_first=True)

        self.time_expand = nn.Linear(compressed_len, reconstruct_len) # 251010

    def forward(self, x):
        # x: [batch, compressed_len, compressed_features]
        # 시계열 길이 복원 (compressed_len -> reconstruct_len)
        x = x.permute(0, 2, 1)  # [batch, compressed_features, compressed_len]
        time_expanded = self.time_expand(x)  # [batch, hidden_dim(seq_len), reconstruct_features]
        time_expanded = time_expanded.permute(0, 2, 1)  # [batch, reconstruct_len, compressed_features]
        # 피쳐 차원 복원
        output, _ = self.gru(time_expanded)       # [batch, compressed_features, hidden_dim(seq_len)]
        return output

# 모델
class GRUDeepSC(nn.Module):
    """GRU 기반 DeepSC 모델, hidden_dim->compressed_len 시퀀스 압축, input_dim->compressed_features 피쳐 압축"""
    def __init__(self, input_dim=6, seq_len=128, hidden_dim=128, compressed_len=64, compressed_features=3, num_layers=2, dropout=0.1, params=None, **kwargs):
        super().__init__()
        # params 딕셔너리가 있으면 거기서 값을 꺼내고, 없으면 인자로 받은 값을 사용
        p = params if params is not None else {}
        self.input_dim = p.get("input_dim", input_dim)
        self.seq_len = p.get("seq_len", seq_len)
        self.hidden_dim = p.get("hidden_dim", hidden_dim)  # 251010 안씀
        self.compressed_len = p.get("compressed_len", compressed_len)
        self.compressed_features = p.get("compressed_features", compressed_features)
        self.num_layers = p.get("num_layers", num_layers)
        self.dropout = p.get("dropout", dropout)
        self.channels = Channels()

        # 올바른 파라미터 전달
        self.encoder = GRUCompressor_Both( # hidden_dim = compressed_len 로 시퀀스 압축
            self.input_dim, self.seq_len,
            self.compressed_len, self.compressed_features,
            self.num_layers, self.dropout
        )
        self.decoder = GRUDecompressor_Both( # hidden_dim = seq_len 로 시퀀스 복원
            self.compressed_len, self.compressed_features,
            self.seq_len, self.input_dim,
            self.num_layers, self.dropout
        )

    def forward(self, x):
        compressed = self.encoder(x)  # [batch, compressed_len, compressed_features]
        snr_db = 10
        # compressed_on_channel = self.channels.AWGN(compressed, snr_db)
        compressed_on_channel = compressed
        reconstructed = self.decoder(compressed_on_channel)  # [batch, seq_len, input_dim]
        return reconstructed

    def get_compression_ratio(self):
        """압축률 계산"""
        original_size = self.input_dim * self.seq_len
        compressed_size = self.compressed_len * self.compressed_features
        return compressed_size / original_size
