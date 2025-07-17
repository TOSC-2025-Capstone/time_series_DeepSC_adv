import torch
import torch.nn as nn
import torch.nn.functional as F

# 모델 파라미터에서 파라미터 딕셔너리 불러오기 -> 그대로 아래에서 클래스 인스턴스 생성

# 시퀀스, 피쳐 압축
class LSTMCompressor_Both(nn.Module):
    def __init__(self, input_dim, hidden_dim, compressed_len=64, compressed_features=3, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.pool = nn.AdaptiveAvgPool1d(compressed_len)
        self.feature_compress = nn.Linear(hidden_dim, compressed_features)
        
    def forward(self, x):
        # x: [batch, 128, 6]
        lstm_out, _ = self.lstm(x)  # [batch, 128, hidden_dim]
        # 시계열 길이 압축
        time_compressed = lstm_out.permute(0, 2, 1)  # [batch, hidden_dim, 128]
        time_compressed = self.pool(time_compressed)  # [batch, hidden_dim, 64]
        time_compressed = time_compressed.permute(0, 2, 1)  # [batch, 64, hidden_dim]
        # 피쳐 차원 압축
        compressed = self.feature_compress(time_compressed)  # [batch, 64, 3]
        return compressed

# 위 둘 다 복원
class LSTMDecompressor_Both(nn.Module):
    def __init__(self, compressed_features, hidden_dim, reconstruct_len=128, reconstruct_features=6, num_layers=2, dropout=0.1):
        super().__init__()
        self.feature_expand = nn.Linear(compressed_features, hidden_dim)
        self.upsample = nn.Upsample(size=reconstruct_len, mode='linear', align_corners=False)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, reconstruct_features)
        
    def forward(self, x):
        # x: [batch, 64, 3]
        feature_expanded = self.feature_expand(x)  # [batch, 64, hidden_dim]
        # 시계열 길이 복원
        time_expanded = feature_expanded.permute(0, 2, 1)  # [batch, hidden_dim, 64]
        time_expanded = self.upsample(time_expanded)       # [batch, hidden_dim, 128]
        time_expanded = time_expanded.permute(0, 2, 1)    # [batch, 128, hidden_dim]
        # LSTM 처리
        lstm_out, _ = self.lstm(time_expanded)  # [batch, 128, hidden_dim]
        # 피쳐 차원 복원
        output = self.output_layer(lstm_out)  # [batch, 128, 6]
        return output

# 모델
class LSTMDeepSC(nn.Module):
    """LSTM 기반 DeepSC 모델"""
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
        
        # 올바른 파라미터 전달
        self.encoder = LSTMCompressor_Both(
            self.input_dim, self.hidden_dim, self.compressed_len, self.compressed_features, self.num_layers, self.dropout
        )
        self.decoder = LSTMDecompressor_Both(
            self.compressed_features, self.hidden_dim, self.seq_len, self.input_dim, self.num_layers, self.dropout
        )
        
    def forward(self, x):
        compressed = self.encoder(x)  # [batch, compressed_len, compressed_features]
        reconstructed = self.decoder(compressed)  # [batch, seq_len, input_dim]
        return reconstructed
    
    def get_compression_ratio(self):
        """압축률 계산"""
        original_size = self.input_dim * self.seq_len
        compressed_size = self.compressed_len * self.compressed_features
        return compressed_size / original_size
