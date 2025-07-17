import torch
import torch.nn as nn
import torch.nn.functional as F

# 시퀀스 길이 압축
class LSTMCompressor_TimeCompress(nn.Module):
    def __init__(self, input_dim, hidden_dim, target_len=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        # 시계열 길이 압축을 위한 pooling
        self.pool = nn.AdaptiveAvgPool1d(target_len)
        
    def forward(self, x):
        # x: [batch, 128, 6]
        lstm_out, _ = self.lstm(x)  # [batch, 128, hidden_dim]
        # 시계열 길이 압축
        compressed = lstm_out.permute(0, 2, 1)  # [batch, hidden_dim, 128]
        compressed = self.pool(compressed)       # [batch, hidden_dim, 64]
        compressed = compressed.permute(0, 2, 1) # [batch, 64, hidden_dim]
        return compressed

# 피쳐 수 압축
class LSTMCompressor_FeatureCompress(nn.Module):
    def __init__(self, input_dim, hidden_dim, target_features=3, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        # 피쳐 차원 압축
        self.feature_compress = nn.Linear(hidden_dim, target_features)
        
    def forward(self, x):
        # x: [batch, 128, 6]
        lstm_out, _ = self.lstm(x)  # [batch, 128, hidden_dim]
        # 피쳐 차원 압축
        compressed = self.feature_compress(lstm_out)  # [batch, 128, 3]
        return compressed

# 위 둘다 압축
class LSTMCompressor_Both(nn.Module):
    def __init__(self, input_dim, hidden_dim, target_len=64, target_features=3, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.pool = nn.AdaptiveAvgPool1d(target_len)
        self.feature_compress = nn.Linear(hidden_dim, target_features)
        
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
    def __init__(self, compressed_features, hidden_dim, target_len=128, target_features=6, num_layers=2, dropout=0.1):
        super().__init__()
        self.feature_expand = nn.Linear(compressed_features, hidden_dim)
        self.upsample = nn.Upsample(size=target_len, mode='linear', align_corners=False)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, target_features)
        
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

# 시퀀스 길이 압축
class GRUCompressor_TimeCompress(nn.Module):
    def __init__(self, input_dim, hidden_dim, target_len=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        # 시계열 길이 압축을 위한 pooling
        self.pool = nn.AdaptiveAvgPool1d(target_len)
        
    def forward(self, x):
        # x: [batch, 128, 6]
        gru_out, _ = self.gru(x)  # [batch, 128, hidden_dim]
        # 시계열 길이 압축
        compressed = gru_out.permute(0, 2, 1)  # [batch, hidden_dim, 128]
        compressed = self.pool(compressed)       # [batch, hidden_dim, 64]
        compressed = compressed.permute(0, 2, 1) # [batch, 64, hidden_dim]
        return compressed

# 피쳐 수 압축
class GRUCompressor_FeatureCompress(nn.Module):
    def __init__(self, input_dim, hidden_dim, target_features=3, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        # 피쳐 차원 압축
        self.feature_compress = nn.Linear(hidden_dim, target_features)
        
    def forward(self, x):
        # x: [batch, 128, 6]
        gru_out, _ = self.gru(x)  # [batch, 128, hidden_dim]
        # 피쳐 차원 압축
        compressed = self.feature_compress(gru_out)  # [batch, 128, 3]
        return compressed

# 위 둘다 압축
class GRUCompressor_Both(nn.Module):
    def __init__(self, input_dim, hidden_dim, target_len=64, target_features=3, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.pool = nn.AdaptiveAvgPool1d(target_len)
        self.feature_compress = nn.Linear(hidden_dim, target_features)
        
    def forward(self, x):
        # x: [batch, 128, 6]
        gru_out, _ = self.gru(x)  # [batch, 128, hidden_dim]
        # 시계열 길이 압축
        time_compressed = gru_out.permute(0, 2, 1)  # [batch, hidden_dim, 128]
        time_compressed = self.pool(time_compressed)  # [batch, hidden_dim, 64]
        time_compressed = time_compressed.permute(0, 2, 1)  # [batch, 64, hidden_dim]
        # 피쳐 차원 압축
        compressed = self.feature_compress(time_compressed)  # [batch, 64, 3]
        return compressed

# 위 둘 다 복원
class GRUDecompressor_Both(nn.Module):
    def __init__(self, compressed_features, hidden_dim, target_len=128, target_features=6, num_layers=2, dropout=0.1):
        super().__init__()
        self.feature_expand = nn.Linear(compressed_features, hidden_dim)
        self.upsample = nn.Upsample(size=target_len, mode='linear', align_corners=False)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, target_features)
        
    def forward(self, x):
        # x: [batch, 64, 3]
        feature_expanded = self.feature_expand(x)  # [batch, 64, hidden_dim]
        # 시계열 길이 복원
        time_expanded = feature_expanded.permute(0, 2, 1)  # [batch, hidden_dim, 64]
        time_expanded = self.upsample(time_expanded)       # [batch, hidden_dim, 128]
        time_expanded = time_expanded.permute(0, 2, 1)    # [batch, 128, hidden_dim]
        # GRU 처리
        gru_out, _ = self.gru(time_expanded)  # [batch, 128, hidden_dim]
        # 피쳐 차원 복원
        output = self.output_layer(gru_out)  # [batch, 128, 6]
        return output

## 모델 바로보기
class LSTMDeepSC(nn.Module):
    """LSTM 기반 DeepSC 모델"""
    def __init__(self, input_dim, seq_len, hidden_dim=128, target_len=64, target_features=3, num_layers=2, dropout=0.1):
        super(LSTMDeepSC, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.target_len = target_len
        self.target_features = target_features
        
        # 올바른 파라미터 전달
        # input_dim, hidden_dim, target_len=64, target_features=3, num_layers=2, dropout=0.1
        self.encoder = LSTMCompressor_Both(
            input_dim, hidden_dim, target_len, target_features, num_layers, dropout
        )
        # compressed_features, hidden_dim, target_len=128, target_features=6, num_layers=2, dropout=0.1
        self.decoder = LSTMDecompressor_Both(
            target_features, hidden_dim, seq_len, input_dim, num_layers, dropout
        )
        
    def forward(self, x):
        compressed = self.encoder(x)  # [batch, target_len, target_features]
        reconstructed = self.decoder(compressed)  # [batch, seq_len, input_dim]
        return reconstructed
    
    def get_compression_ratio(self):
        """압축률 계산"""
        original_size = self.input_dim * self.seq_len
        compressed_size = self.target_len * self.target_features
        return compressed_size / original_size
 
class GRUDeepSC(nn.Module):
    """GRU 기반 DeepSC 모델, hidden_dim->target_len으로 시퀀스 압축, input_dim->target_features로 피쳐 압축"""
    def __init__(self, input_dim, seq_len, hidden_dim=128, target_len=64, target_features=3, num_layers=2, dropout=0.1):
        super(GRUDeepSC, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.target_len = target_len
        self.target_features = target_features
        
        # 올바른 파라미터 전달
        self.encoder = GRUCompressor_Both(
            input_dim, hidden_dim, target_len, target_features, num_layers, dropout
        )
        self.decoder = GRUDecompressor_Both(
            target_features, hidden_dim, seq_len, input_dim, num_layers, dropout
        )
        
    def forward(self, x):
        compressed = self.encoder(x)  # [batch, target_len, target_features]
        reconstructed = self.decoder(compressed)  # [batch, seq_len, input_dim]
        return reconstructed
    
    def get_compression_ratio(self):
        """압축률 계산"""
        original_size = self.input_dim * self.seq_len
        compressed_size = self.target_len * self.target_features
        return compressed_size / original_size

class Seq2SeqAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, output_dim, num_layers=2, dropout=0.1):
        super(Seq2SeqAttention, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        # self.compressed_dim = hidden_dim//2 
        
        self.encoder = LSTMCompressor(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = LSTMDecompressor(hidden_dim, hidden_dim, seq_len, output_dim, num_layers, dropout)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
    def forward(self, x):
        encoder_outputs = self.encoder(x)  # (batch, seq_len, hidden_dim)
        
        batch_size = x.size(0)
        seq_len = self.seq_len
        embed_dim = encoder_outputs.size(-1)  # 보통 hidden_dim*2

        # 제로 벡터로 디코더 입력 생성
        decoder_input = torch.zeros(batch_size, seq_len, embed_dim, device=x.device)

        # Attention 적용
        attn_output = self.attn(decoder_input, encoder_outputs, encoder_outputs)
        # 디코더에 attn_output 입력
        output = self.decoder(attn_output)
        return output 

    def get_compression_ratio(self):
        """압축률 계산"""
        original_size = self.input_dim * self.seq_len
        compressed_size = self.hidden_dim
        return compressed_size / original_size

class BiLSTMCompressor_Both(nn.Module):
    def __init__(self, input_dim, hidden_dim, target_len=64, target_features=3, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.pool = nn.AdaptiveAvgPool1d(target_len)
        self.feature_compress = nn.Linear(hidden_dim*2, target_features)  # *2 for bidirectional
    def forward(self, x):
        # x: [batch, 128, 8]
        lstm_out, _ = self.lstm(x)  # [batch, 128, hidden_dim*2]
        time_compressed = lstm_out.permute(0, 2, 1)  # [batch, hidden_dim*2, 128]
        time_compressed = self.pool(time_compressed)  # [batch, hidden_dim*2, 64]
        time_compressed = time_compressed.permute(0, 2, 1)  # [batch, 64, hidden_dim*2]
        compressed = self.feature_compress(time_compressed)  # [batch, 64, target_features]
        return compressed

class BiLSTMDecompressor_Both(nn.Module):
    def __init__(self, compressed_features, hidden_dim, target_len=128, target_features=6, num_layers=2, dropout=0.1):
        super().__init__()
        self.feature_expand = nn.Linear(compressed_features, hidden_dim*2)
        self.upsample = nn.Upsample(size=target_len, mode='linear', align_corners=False)
        self.lstm = nn.LSTM(hidden_dim*2, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.output_layer = nn.Linear(hidden_dim*2, target_features)
    def forward(self, x):
        # x: [batch, 64, compressed_features]
        feature_expanded = self.feature_expand(x)  # [batch, 64, hidden_dim*2]
        time_expanded = feature_expanded.permute(0, 2, 1)  # [batch, hidden_dim*2, 64]
        time_expanded = self.upsample(time_expanded)       # [batch, hidden_dim*2, 128]
        time_expanded = time_expanded.permute(0, 2, 1)    # [batch, 128, hidden_dim*2]
        lstm_out, _ = self.lstm(time_expanded)  # [batch, 128, hidden_dim*2]
        output = self.output_layer(lstm_out)  # [batch, 128, target_features]
        return output

class BiLSTMDeepSC(nn.Module):
    """Bidirectional LSTM 기반 DeepSC 모델"""
    def __init__(self, input_dim, seq_len, hidden_dim=128, target_len=64, target_features=3, num_layers=2, dropout=0.1):
        super(BiLSTMDeepSC, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.target_len = target_len
        self.target_features = target_features
        self.encoder = BiLSTMCompressor_Both(
            input_dim, hidden_dim, target_len, target_features, num_layers, dropout
        )
        self.decoder = BiLSTMDecompressor_Both(
            target_features, hidden_dim, seq_len, input_dim-2, num_layers, dropout
        )
    def forward(self, x):
        compressed = self.encoder(x)  # [batch, target_len, target_features]
        reconstructed = self.decoder(compressed)  # [batch, seq_len, input_dim-2]
        return reconstructed
    def get_compression_ratio(self):
        original_size = self.input_dim * self.seq_len
        compressed_size = self.target_len * self.target_features
        return compressed_size / original_size

class LSTMAttentionDeepSC(nn.Module):
    """LSTM + Self-Attention 기반 DeepSC 모델"""
    def __init__(self, input_dim, seq_len, hidden_dim=128, target_len=64, target_features=2, num_layers=2, dropout=0.1, num_heads=4):
        super(LSTMAttentionDeepSC, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.target_len = target_len
        self.target_features = target_features
        self.encoder = LSTMCompressor_Both(
            input_dim, hidden_dim, target_len, target_features, num_layers, dropout
        )
        # attention의 embed_dim은 target_features로 맞춤
        self.attn = nn.MultiheadAttention(embed_dim=target_features, num_heads=num_heads, batch_first=True)
        self.decoder = LSTMDecompressor_Both(
            # target_features, hidden_dim, seq_len, input_dim-2, num_layers, dropout
            target_features, hidden_dim, seq_len, input_dim, num_layers, dropout
        )
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        compressed = self.encoder(x)  # [batch, target_len, target_features]
        # Self-attention 적용
        attn_out, _ = self.attn(compressed, compressed, compressed)  # [batch, target_len, target_features]
        reconstructed = self.decoder(attn_out)  # [batch, seq_len, input_dim-2]
        return reconstructed
    def get_compression_ratio(self):
        original_size = self.input_dim * self.seq_len
        compressed_size = self.target_len * self.target_features
        return compressed_size / original_size
