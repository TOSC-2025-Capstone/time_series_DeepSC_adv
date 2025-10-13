import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Channels, power_normalize
import pdb
from models.compressor import LearnableTimeCompressor, LearnableTimeDecompressor, ResidualChannelDecoder
# from parameters.parameters import noise_std

# 시퀀스, 피쳐 압축
class RNNBasedCompressor_Both(nn.Module):
    def __init__(self, input_dim, hidden_dim, compressed_len=64, compressed_features=3, num_layers=2, dropout=0.1, model_type="gru"):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        if model_type not in ["gru", "lstm"]:
            raise ValueError("model_type은 'gru' 또는 'lstm'이어야 합니다.")
        if model_type == "lstm":
            self.seq_model = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        elif model_type == "gru":  # "gru"
            self.seq_model = nn.GRU(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.time_compressor = LearnableTimeCompressor(d_model=hidden_dim, target_len=compressed_len)
        self.feature_compress = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),  # 128 → 64
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, compressed_features),  # 32 → 3
        )

    def forward(self, x):
        # x: [batch, 128, 6]
        x = self.input_projection(x)  # [batch, 128, hidden_dim]
        gru_out, _ = self.seq_model(x)  # [batch, 128, hidden_dim]
        # 시계열 길이 압축
        time_compressed = self.time_compressor(gru_out)  # [batch, 64, hidden_dim]
        # 피쳐 차원 압축
        compressed = self.feature_compress(time_compressed)  # [batch, 64, 3]
        return compressed

# 시퀀스, 피쳐 복원
class RNNBasedDecompressor_Both(nn.Module):
    def __init__(self, compressed_features, hidden_dim, reconstruct_len=128, reconstruct_features=6, num_layers=2, dropout=0.1, model_type="gru"):
        super().__init__()
        if model_type not in ["gru", "lstm"]:
            raise ValueError("model_type은 'gru' 또는 'lstm'이어야 합니다.")
        if model_type == "lstm":
            self.seq_model = nn.LSTM(hidden_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        elif model_type == "gru":  # "gru"
            self.seq_model = nn.GRU(hidden_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.feature_expand = ResidualChannelDecoder(d_comp = compressed_features, d_model = hidden_dim)
        self.decompressor = LearnableTimeDecompressor(d_model=hidden_dim, target_len=reconstruct_len)
        self.output_layer = nn.Linear(hidden_dim, reconstruct_features)

    def forward(self, x):
        # x: [batch, 64, 3]
        # 피쳐 차원 늘리기
        feature_expanded = self.feature_expand(x)  # [batch, 64, hidden_dim]
        # 시계열 길이 복원
        time_expanded = self.decompressor(feature_expanded)  # [batch, 128, hidden_dim]
        # GRU 처리
        # gru_out, _ = self.seq_model(time_expanded)  # [batch, 128, hidden_dim]
        # 피쳐 차원 복원 (원래 피쳐 수로 줄이기)
        output = self.output_layer(time_expanded)  # [batch, 128, 6]
        return output


# 모델
class RNNBasedSC(nn.Module):
    """LSTM/GRU 기반 DeepSC 모델, hidden_dim->compressed_len 시퀀스 압축, input_dim->compressed_features 피쳐 압축"""
    def __init__(self, input_dim=6, seq_len=128, hidden_dim=128, compressed_len=64, compressed_features=3, num_layers=2, dropout=0.1, params=None, **kwargs):
        super().__init__()
        # params 딕셔너리가 있으면 거기서 값을 꺼내고, 없으면 인자로 받은 값을 사용
        p = params if params is not None else {}
        self.input_dim = p.get("input_dim", input_dim)
        self.seq_len = p.get("seq_len", seq_len)
        self.hidden_dim = p.get("hidden_dim", hidden_dim)
        self.compressed_len = p.get("compressed_len", compressed_len)
        self.compressed_features = p.get("compressed_features", compressed_features)
        self.num_layers = p.get("num_layers", num_layers)
        self.dropout = p.get("dropout", dropout)
        self.channels = Channels()
        self.snr_db = p.get("snr_db", 5)  # AWGN 채널 SNR 값
        self.model_type = kwargs.get("model_type", "gru")  # 모델 타입 (gru/lstm)
        assert self.model_type in ["gru", "lstm"], "model_type은 'gru' 또는 'lstm'이어야 합니다."

        # 올바른 파라미터 전달
        self.encoder = RNNBasedCompressor_Both(
            self.input_dim, self.hidden_dim,
            self.compressed_len, self.compressed_features,
            self.num_layers, self.dropout, self.model_type
        )
        self.decoder = RNNBasedDecompressor_Both(
            self.compressed_features, self.hidden_dim,
            self.seq_len, self.input_dim,
            self.num_layers, self.dropout, self.model_type
        )

    def forward(self, x):
        compressed = self.encoder(x)  # [batch, compressed_len, compressed_features]
        tx_sig = power_normalize(compressed)
        # rx_sig = self.channels.AWGN(tx_sig, self.snr_db)
        rx_sig = tx_sig
        reconstructed = self.decoder(rx_sig)  # [batch, seq_len, input_dim]
        return reconstructed

    def get_compression_ratio(self):
        """압축률 계산"""
        original_size = self.input_dim * self.seq_len
        compressed_size = self.compressed_len * self.compressed_features
        return compressed_size / original_size
