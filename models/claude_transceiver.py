import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Channels
import math

class InvertedMultiHeadAttention(nn.Module):
    """차원이 뒤바뀐 어텐션 - 변수(피쳐) 축에서 어텐션 수행"""
    def __init__(self, num_heads, seq_len, dropout=0.1):
        super().__init__()
        assert seq_len % num_heads == 0
        self.d_k = seq_len // num_heads
        self.num_heads = num_heads
        self.seq_len = seq_len

        self.wq = nn.Linear(seq_len, seq_len)
        self.wk = nn.Linear(seq_len, seq_len)
        self.wv = nn.Linear(seq_len, seq_len)
        self.dense = nn.Linear(seq_len, seq_len)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: [batch, seq_len, features] -> [batch, features, seq_len]
        x = x.transpose(1, 2)
        batch_size, num_features, seq_len = x.shape

        # Q, K, V 계산
        query = self.wq(x).view(batch_size, num_features, self.num_heads, self.d_k)
        query = query.transpose(1, 2)  # [batch, num_heads, num_features, d_k]

        key = self.wk(x).view(batch_size, num_features, self.num_heads, self.d_k)
        key = key.transpose(1, 2)

        value = self.wv(x).view(batch_size, num_features, self.num_heads, self.d_k)
        value = value.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, num_features, seq_len)

        # Final linear transformation
        output = self.dense(attn_output)
        output = self.dropout(output)

        # 다시 원래 차원으로 변환: [batch, features, seq_len] -> [batch, seq_len, features]
        output = output.transpose(1, 2)

        return output

class InvertedFeedForward(nn.Module):
    """각 변수별로 독립적인 FFN 적용"""
    def __init__(self, seq_len, d_ff, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len

        # 각 변수(feature)에 대해 독립적인 FFN
        self.w_1 = nn.Linear(seq_len, d_ff)
        self.w_2 = nn.Linear(d_ff, seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, features]
        x = x.transpose(1, 2)  # [batch, features, seq_len]

        # 각 feature별로 독립적으로 FFN 적용
        x = self.w_1(x)  # [batch, features, d_ff]
        x = F.relu(x)
        x = self.w_2(x)  # [batch, features, seq_len]
        x = self.dropout(x)

        # 다시 원래 차원으로
        x = x.transpose(1, 2)  # [batch, seq_len, features]

        return x

class iTransformerLayer(nn.Module):
    """iTransformer 레이어"""
    def __init__(self, seq_len, num_features, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.inverted_attention = InvertedMultiHeadAttention(num_heads, seq_len, dropout)
        self.inverted_ffn = InvertedFeedForward(seq_len, d_ff, dropout)

        # LayerNorm은 feature 차원에 적용 (각 변수의 시계열을 정규화)
        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(num_features)

    def forward(self, x):
        # Self-attention with residual connection
        attn_output = self.inverted_attention(x)
        x = self.norm1(x + attn_output)

        # FFN with residual connection
        ffn_output = self.inverted_ffn(x)
        x = self.norm2(x + ffn_output)

        return x

class iTransformerDeepSC(nn.Module):
    def __init__(
        self,
        num_layers=2,
        input_dim=6,
        max_len=128,
        num_heads=4,
        d_ff=512,
        dropout=0.1,
        compressed_len=64,
        d_comp=3,
        params=None,
        **kwargs
    ):
        super().__init__()
        p = params if params is not None else {}
        self.num_layers = p.get("num_layers", num_layers)
        self.input_dim = p.get("input_dim", input_dim)
        self.max_len = p.get("max_len", max_len)
        self.num_heads = p.get("num_heads", num_heads)
        self.d_ff = p.get("d_ff", d_ff)
        self.dropout = p.get("dropout", dropout)
        self.compressed_len = p.get("compressed_len", compressed_len)
        self.d_comp = p.get("d_comp", d_comp)

        # iTransformer 인코더 레이어들
        self.encoder_layers = nn.ModuleList([
            iTransformerLayer(self.max_len, self.input_dim, self.num_heads, self.d_ff, self.dropout)
            for _ in range(self.num_layers)
        ])

        # 시간 압축 (각 변수별로 독립적)
        self.time_compressor = nn.Conv1d(self.input_dim, self.input_dim,
                                       kernel_size=3, stride=2, padding=1,
                                       groups=self.input_dim)  # Grouped convolution
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.compressed_len)

        # 변수 차원 압축 (6 -> 3)
        self.feature_compressor = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.LayerNorm(self.input_dim // 2),
            nn.ReLU(),
            nn.Linear(self.input_dim // 2, self.d_comp),
        )

        self.channels = Channels()

        # 변수 차원 복원 (3 -> 6)
        self.feature_decompressor = nn.Sequential(
            nn.Linear(self.d_comp, self.input_dim // 2),
            nn.LayerNorm(self.input_dim // 2),
            nn.ReLU(),
            nn.Linear(self.input_dim // 2, self.input_dim),
        )

        # 시간 복원
        self.time_decompressor = nn.ConvTranspose1d(self.input_dim, self.input_dim,
                                                  kernel_size=3, stride=2, padding=1,
                                                  groups=self.input_dim)
        self.time_upsample = nn.AdaptiveAvgPool1d(self.max_len)

        # iTransformer 디코더 레이어들
        self.decoder_layers = nn.ModuleList([
            iTransformerLayer(self.max_len, self.input_dim, self.num_heads, self.d_ff, self.dropout)
            for _ in range(self.num_layers)
        ])

    def forward(self, x, src_mask=None):
        # x: [batch, seq_len, input_dim=6]

        # 1. iTransformer 인코딩 (변수 간 상관관계 학습)
        encoded = x
        for layer in self.encoder_layers:
            encoded = layer(encoded)

        # 2. 시간 압축 (변수별 독립적)
        encoded = encoded.transpose(1, 2)  # [batch, input_dim, seq_len]
        compressed = self.time_compressor(encoded)
        compressed = self.adaptive_pool(compressed)  # [batch, input_dim, compressed_len]
        compressed = compressed.transpose(1, 2)  # [batch, compressed_len, input_dim]

        # 3. 변수 차원 압축 (6 -> 3)
        feature_compressed = self.feature_compressor(compressed)  # [batch, compressed_len, 3]

        # 4. 채널 노이즈
        channel_syms = self.channels.Rayleigh(feature_compressed, 0.1)

        # 5. 변수 차원 복원 (3 -> 6)
        feature_restored = self.feature_decompressor(channel_syms)  # [batch, compressed_len, 6]

        # 6. 시간 복원
        feature_restored = feature_restored.transpose(1, 2)  # [batch, input_dim, compressed_len]
        time_restored = self.time_decompressor(feature_restored)
        time_restored = self.time_upsample(time_restored)  # [batch, input_dim, max_len]
        time_restored = time_restored.transpose(1, 2)  # [batch, max_len, input_dim]

        # 7. iTransformer 디코딩 (변수 간 상관관계 복원)
        output = time_restored
        for layer in self.decoder_layers:
            output = layer(output)

        return output
