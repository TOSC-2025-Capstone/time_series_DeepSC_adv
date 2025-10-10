# -*- coding: utf-8 -*-
"""
Created on Mon May 25 20:33:53 2020

@author: HQ Xie
这是一个Transformer的网络结构
"""
"""
Transformer includes:
    Encoder
        1. Positional coding
        2. Multihead-attention
        3. PositionwiseFeedForward
    Decoder
        1. Positional coding
        2. Multihead-attention
        3. Multihead-attention
        4. PositionwiseFeedForward
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math
import pdb

# from samba_mixer.model.input_projections.linear_projection_time_embedding_cycle_diff_embedding import LinearProjectionWithLocalTimeAndGlobalDiffEmbedding
from utils import Channels

class TimeSeriesPositionalEncoding(nn.Module):
    """시계열 특화 위치 인코딩"""
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 1. 기본 sinusoidal PE
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

        # 2. 학습 가능한 시계열 위치 임베딩 추가
        self.learnable_pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.1)

        # 3. 시간적 스케일 임베딩
        self.scale_embedding = nn.Linear(1, d_model)

    def forward(self, x):
        seq_len = x.size(1)

        # 기본 PE + 학습 가능한 PE
        pos_enc = self.pe[:, :seq_len] + self.learnable_pe[:, :seq_len]

        # 시간적 스케일 정보 추가 (0~1로 정규화된 시간 인덱스)
        time_scale = torch.linspace(0, 1, seq_len, device=x.device).unsqueeze(0).unsqueeze(-1)
        scale_enc = self.scale_embedding(time_scale)

        x = x + pos_enc + scale_enc
        return self.dropout(x)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )  # math.log(math.exp(1)) = 1
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

        # self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        x = self.dropout(x)
        return x

class NoiseAdaptivePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 기본 sinusoidal PE
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

        # 노이즈 감지 게이트
        self.noise_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        seq_len = x.size(1)

        # 노이즈 레벨 추정
        noise_level = self.noise_detector(x.mean(dim=1, keepdim=True))  # [batch, 1, 1]

        # 노이즈 많으면 PE 약하게 적용
        pe_strength = 1.0 - 0.5 * noise_level

        x = x + pe_strength * self.pe[:, :seq_len]
        return self.dropout(x)

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

        # self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)

        # 마스크 차원 올바르게 조정
        if mask is not None:
            # mask가 [seq_len, seq_len]이면 [1, 1, seq_len, seq_len]로 변환
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # [seq_len, seq_len] -> [1, 1, seq_len, seq_len]
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch, seq_len, seq_len] -> [batch, 1, seq_len, seq_len]

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k)
        query = query.transpose(1, 2)

        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k)
        key = key.transpose(1, 2)

        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k)
        value = value.transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)

        x = self.dense(x)
        x = self.dropout(x)

        return x

    def attention(self, query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # print(mask.shape)
        if mask is not None:
            # 根据mask，指定位置填充 -1e9
            scores += mask * -1e9
            # attention weights
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x


# class LayerNorm(nn.Module):
#     "Construct a layernorm module (See citation for details)."
#     # features = d_model
#     def __init__(self, features, eps=1e-6):
#         super(LayerNorm, self).__init__()
#         self.a_2 = nn.Parameter(torch.ones(features))
#         self.b_2 = nn.Parameter(torch.zeros(features))
#         self.eps = eps

#     def forward(self, x):
#         mean = x.mean(-1, keepdim=True)
#         std = x.std(-1, keepdim=True)
#         return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, window_size=None, dropout=0.1):
        """
        window_size=None: 일반 Attention
        window_size=int: Local Attention
        """
        super().__init__()

        # window_size에 따라 Attention 선택
        if window_size is not None:
            # self.mha = LocalMultiHeadedAttention(num_heads, d_model, window_size, dropout)
            # self.mha = DenoisingMultiHeadAttention(num_heads, d_model, noise_threshold=0.3, dropout=0.1)
            # self.mha = ResidualTemperatureAttention(num_heads, d_model, dropout=0.1)
            # self.mha = EMAAttention(num_heads, d_model, dropout=0.1)
            # self.mha = UnifiedAttention(num_heads, d_model, inverted=False, seq_len=None,
            #         denoising=True, noise_threshold=0.3,
            #         local_window=window_size, use_temperature=True,
            #         use_residual_attn=True, use_ema=True, ema_decay=0.9,
            #         dropout=0.1)
            self.mha = MultiHeadedAttention(num_heads, d_model, dropout)
        else:
            self.mha = MultiHeadedAttention(num_heads, d_model, dropout)

        self.ffn = PositionwiseFeedForward(d_model, dff, dropout)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask):
        attn_output = self.mha(x, x, x, mask)
        x = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.layernorm2(x + ffn_output)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1, window_size=None):
        super().__init__()

        # 기존 MultiHeadedAttention 사용
        self.self_attn = MultiHeadedAttention(num_heads, d_model, dropout)
        self.cross_attn = MultiHeadedAttention(num_heads, d_model, dropout)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, memory, self_mask=None, cross_mask=None):
        # 1. Self-attention (복원 위치 간 관계)
        attn_out = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + attn_out)

        # 2. Cross-attention (압축된 정보 활용)
        cross_out = self.cross_attn(x, memory, memory, cross_mask)
        x = self.norm2(x + cross_out)

        # 3. Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)

        return x

class Encoder(nn.Module):
    def __init__(
        self, num_layers, input_dim, max_len, d_model, num_heads, dff,
        window_size=None, dropout=0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        # self.pos_encoding = TimeSeriesPositionalEncoding(d_model, dropout, max_len)
        self.pos_encoding = NoiseAdaptivePositionalEncoding(d_model, dropout, max_len)

        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, dff, window_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, src_mask):
        x = self.input_projection(x)
        x = self.pos_encoding(x)

        for enc_layer in self.enc_layers:
            x = enc_layer(x, src_mask)

        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, max_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # 학습 가능한 위치별 쿼리 임베딩
        self.pos_queries = nn.Parameter(torch.randn(max_len, d_model) * 0.1)

        # 위치 인코딩 (기존 것 재사용 가능)
        # self.pos_encoding = TimeSeriesPositionalEncoding(d_model, dropout, max_len)
        self.pos_encoding = NoiseAdaptivePositionalEncoding(d_model, dropout, max_len)

        # 디코더 레이어들
        self.decoder_layers = nn.ModuleList([
            # DecoderLayer(d_model, num_heads, dff, dropout)
            DecoderLayer(d_model, num_heads, dff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, memory, target_len=None, use_mask=False):
        # memory: channel_decoder 출력 [batch, compressed_len, d_model]
        if target_len is None:
            target_len = self.max_len

        batch_size = memory.size(0)

        # 학습 가능한 쿼리 임베딩 생성
        query_embed = self.pos_queries[:target_len].unsqueeze(0).repeat(batch_size, 1, 1)

        # 위치 인코딩 추가
        x = self.pos_encoding(query_embed)

        # 마스크 생성 (필요한 경우)
        self_mask = None
        if use_mask:
            # Look-ahead mask (자기회귀적 복원을 위한 경우)
            self_mask = torch.triu(torch.ones(target_len, target_len), diagonal=1).bool()
            self_mask = self_mask.to(x.device)

        # 각 디코더 레이어 통과
        for layer in self.decoder_layers:
            x = layer(x, memory, self_mask, None)

        return x


class ChannelDecoder(nn.Module):
    def __init__(self, in_features, size1, size2):
        super(ChannelDecoder, self).__init__()

        self.linear1 = nn.Linear(in_features, size1)
        self.linear2 = nn.Linear(size1, size2)
        self.linear3 = nn.Linear(size2, size1)
        # self.linear4 = nn.Linear(size1, d_model)

        self.layernorm = nn.LayerNorm(size1, eps=1e-6)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = F.relu(x1)
        x3 = self.linear2(x2)
        x4 = F.relu(x3)
        x5 = self.linear3(x4)

        output = self.layernorm(x1 + x5)

        return output

class ResidualChannelDecoder(nn.Module):
    def __init__(self, d_comp, d_model):
        super().__init__()

        # 각 단계별 residual block
        self.expand1 = nn.Sequential(
            nn.Linear(d_comp, d_model // 4),
            # nn.LayerNorm(d_model // 4),
            nn.ReLU()
        )
        self.expand2 = nn.Sequential(
            nn.Linear(d_model // 4, d_model),
            # nn.LayerNorm(d_model // 2),
            # nn.Dropout(0.1),
            nn.ReLU()
        )

        # 각 단계별 skip projection
        self.skip1 = nn.Linear(d_comp, d_model // 4)
        self.skip2 = nn.Linear(d_model // 4, d_model)

    def forward(self, x):
        # 점진적 확장 + residual
        out1 = self.expand1(x) + self.skip1(x)
        out2 = self.expand2(out1) + self.skip2(out1)

        return out2

class LearnableTimeCompressor(nn.Module):
    """학습 가능한 시계열 압축기"""
    def __init__(self, d_model, target_len):
        super().__init__()
        self.target_len = target_len

        # 1D Conv로 학습 가능한 압축
        self.conv_compress = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # 정확한 길이 맞추기
        self.adaptive_pool = nn.AdaptiveAvgPool1d(target_len)

        # Skip connection을 위한 projection
        self.skip_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        skip = x  # Skip connection 저장

        x = x.permute(0, 2, 1)  # (batch, d_model, seq_len)
        x = self.conv_compress(x)
        x = self.adaptive_pool(x)  # (batch, d_model, target_len)
        x = x.permute(0, 2, 1)  # (batch, target_len, d_model)

        # Skip connection (원본을 target_len으로 압축)
        skip = skip.permute(0, 2, 1)
        skip = F.adaptive_avg_pool1d(skip, self.target_len)
        skip = skip.permute(0, 2, 1)
        skip = self.skip_proj(skip)

        return x + skip

class LearnableTimeDecompressor(nn.Module):
    """학습 가능한 시계열 복원기"""
    def __init__(self, d_model, target_len):
        super().__init__()
        self.target_len = target_len

        # 전치 합성곱으로 학습 가능한 복원
        self.deconv_decompress = nn.Sequential(
            nn.ConvTranspose1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # 정확한 길이 맞추기
        self.adaptive_pool = nn.AdaptiveAvgPool1d(target_len)

    def forward(self, x):
        # x: (batch, compressed_len, d_model)
        x = x.permute(0, 2, 1)  # (batch, d_model, compressed_len)
        x = self.deconv_decompress(x)
        x = self.adaptive_pool(x)  # (batch, d_model, target_len)
        x = x.permute(0, 2, 1)  # (batch, target_len, d_model)
        return x

# 250914 claude iTransformer
class InvertedMultiHeadAttention(nn.Module):
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

        # Feature importance gating 추가
        self.feature_gate = nn.Sequential(
            nn.Linear(seq_len, seq_len // 4),
            nn.ReLU(),
            nn.Linear(seq_len // 4, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x_transposed = x.transpose(1, 2)
        batch_size, num_features, seq_len = x_transposed.shape

        # Feature importance 계산
        feature_importance = self.feature_gate(x_transposed)  # [B, F, 1]

        # Q, K, V 계산
        query = self.wq(x_transposed).view(batch_size, num_features, self.num_heads, self.d_k)
        query = query.transpose(1, 2)  # [B, num_heads, F, d_k]

        key = self.wk(x_transposed).view(batch_size, num_features, self.num_heads, self.d_k)
        key = key.transpose(1, 2)  # [B, num_heads, F, d_k]

        value = self.wv(x_transposed).view(batch_size, num_features, self.num_heads, self.d_k)
        value = value.transpose(1, 2)  # [B, num_heads, F, d_k]

        # Scaled dot-product attention with feature gating
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores shape: [B, num_heads, F, F]

        # Feature importance를 attention에 반영 (차원 맞추기)
        # feature_importance: [B, F, 1] -> [B, 1, F, 1]로 확장
        feature_importance = feature_importance.unsqueeze(1)  # [B, 1, F, 1]

        # Query 측 feature importance 적용
        scores = scores * feature_importance  # 브로드캐스팅: [B, num_heads, F, F] * [B, 1, F, 1]

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, num_features, seq_len)

        output = self.dense(attn_output)
        output = self.dropout(output)
        output = output.transpose(1, 2)

        return output

class InvertedFeedForward(nn.Module):
    """각 변수별로 독립적인 FFN"""
    def __init__(self, seq_len, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(seq_len, d_ff)
        self.w_2 = nn.Linear(d_ff, seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, features]
        x = x.transpose(1, 2)  # [batch, features, seq_len]

        x = self.w_1(x)
        x = F.relu(x)
        x = self.w_2(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)  # [batch, seq_len, features]
        return x

class iTransformerEncoderLayer(nn.Module):
    def __init__(self, seq_len, num_features, num_heads, d_ff, window_size=None, dropout=0.1):
        """
        window_size=None: 일반 Inverted Attention
        window_size=int: Local Inverted Attention
        """
        super().__init__()

        # window_size에 따라 Attention 선택
        self.inverted_attention = InvertedMultiHeadAttention(num_heads, seq_len, dropout)

        self.inverted_ffn = InvertedFeedForward(seq_len, d_ff, dropout)
        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(num_features)

    def forward(self, x, mask=None):
        attn_output = self.inverted_attention(x)
        x = self.norm1(x + attn_output)

        ffn_output = self.inverted_ffn(x)
        x = self.norm2(x + ffn_output)

        return x

class iTransformerEncoder(nn.Module):
    def __init__(self, num_layers, input_dim, max_len, d_model, num_heads, dff,
                 window_size=None, dropout=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.max_len = max_len

        # Feature-wise positional encoding 추가
        self.feature_pos_embedding = nn.Parameter(
            torch.randn(1, 1, input_dim) * 0.1
        )

        # Time-wise positional encoding
        self.time_pos_embedding = nn.Parameter(
            torch.randn(1, max_len, 1) * 0.1
        )

        self.enc_layers = nn.ModuleList([
            iTransformerEncoderLayer(max_len, input_dim, num_heads, dff, window_size, dropout)
            for _ in range(num_layers)
        ])

        self.output_projection = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        # 양방향 positional encoding 추가
        x = x + self.time_pos_embedding + self.feature_pos_embedding
        x = self.dropout(x)

        for enc_layer in self.enc_layers:
            x = enc_layer(x, src_mask)

        x = self.output_projection(x)
        return x

class InvertedDecoderLayer(nn.Module):
    """
    iTransformer 기반 디코더 레이어:
      - Self-Attention: feature-axis
      - Cross-Attention: feature-axis 기준 encoder memory 참조
    """
    def __init__(self, seq_len, num_features, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = InvertedMultiHeadAttention(num_heads, seq_len, dropout)
        self.cross_attn = InvertedMultiHeadAttention(num_heads, seq_len, dropout)
        self.ffn = InvertedFeedForward(seq_len, d_ff, dropout)

        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(num_features)
        self.norm3 = nn.LayerNorm(num_features)

    def forward(self, x, memory, self_mask=None, cross_mask=None):
        """
        x: [batch, seq_len, num_features]
        memory: [batch, seq_len, num_features]
        """
        # Self-Attention (feature 축)
        attn_out = self.self_attn(x)
        x = self.norm1(x + attn_out)

        # Cross-Attention (encoder memory 참조)
        # memory를 같은 축으로 맞춰줌
        cross_out = self.cross_attn(memory)
        x = self.norm2(x + cross_out)

        # Feed Forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)

        return x


class iTransformerDecoder(nn.Module):
    """
    iTransformer 기반 디코더 (변수 축 복원)
      - feature 간 상관성에 기반한 복원 수행
    """
    def __init__(self, num_layers, compressed_len, seq_len, num_features, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.compressed_len = compressed_len
        self.num_features = num_features
        self.seq_len = seq_len

        # 학습 가능한 feature query embedding
        self.feature_queries = nn.Parameter(torch.randn(num_features, compressed_len) * 0.1)

        # ===== Positional Encoding 추가 (Encoder와 동일) =====
        # Feature-wise positional encoding
        self.feature_pos_embedding = nn.Parameter(
            torch.randn(1, 1, num_features) * 0.1
        )

        # Time-wise positional encoding (compressed_len에 맞춤)
        self.time_pos_embedding = nn.Parameter(
            torch.randn(1, compressed_len, 1) * 0.1
        )

        self.dropout = nn.Dropout(dropout)
        # =====================================================

        # 디코더 레이어
        self.decoder_layers = nn.ModuleList([
            InvertedDecoderLayer(compressed_len, num_features, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 출력 투영 (compressed_len -> seq_len 복원)
        self.output_projection = nn.Linear(compressed_len, seq_len)

    def forward(self, memory, target_len=None, use_mask=False):
        """
        memory: [batch, compressed_len, num_features]
        """
        batch_size = memory.size(0)

        # feature query 임베딩 준비
        query_embed = self.feature_queries.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, F, T]
        x = query_embed.transpose(1, 2)  # [B, T, F]

        # ===== Positional Encoding 적용 =====
        x = x + self.time_pos_embedding + self.feature_pos_embedding
        x = self.dropout(x)
        # ====================================

        # 각 디코더 레이어 통과
        for layer in self.decoder_layers:
            x = layer(x, memory)

        # 출력 투영 후 반환 (시간 축 복원)
        x = self.output_projection(x.transpose(1, 2)).transpose(1, 2)  # [B, seq_len, F]
        return x

class SmoothTimeDecompressor(nn.Module):
    def __init__(self, d_model, target_len):
        super().__init__()
        self.target_len = target_len

        # ConvTranspose 대신 Interpolation + Conv 사용
        self.decompress = nn.Sequential(
            nn.Linear(d_model, d_model * 2),  # 채널 확장
            nn.GELU(),
        )
        self.smooth_conv = nn.Conv1d(d_model * 2, d_model, kernel_size=5, padding=2)

    def forward(self, x):
        # x: [batch, compressed_len, d_model]
        batch, comp_len, d_model = x.shape

        # 1단계: 피처 확장
        x = self.decompress(x)  # [batch, comp_len, d_model*2]

        # 2단계: 시간 축 보간 (부드러운 업샘플링)
        x = x.permute(0, 2, 1)  # [batch, d_model*2, comp_len]
        x = F.interpolate(x, size=self.target_len, mode='linear', align_corners=False)

        # 3단계: 스무딩 컨볼루션
        x = self.smooth_conv(x)  # [batch, d_model, target_len]
        x = x.permute(0, 2, 1)  # [batch, target_len, d_model]

        return x

# ========== 수정된 TemporalFeatureCrossInteraction ==========
class TemporalFeatureCrossInteraction(nn.Module):
    """시간축과 변수축의 정보를 교차 교환 (차원 불일치 수정)"""
    def __init__(self, seq_len, num_features, dropout=0.1):
        super().__init__()

        # Temporal projection: [B, F, T] -> [B, F, T]
        self.time_projection = nn.Sequential(
            nn.Conv1d(num_features, num_features, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_features, num_features, kernel_size=1)
        )

        # Feature projection: [B, T, F] -> [B, T, F]
        self.feature_projection = nn.Sequential(
            nn.Linear(num_features, num_features * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_features * 2, num_features)
        )

        # Adaptive gating
        self.time_gate = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )

        self.feature_gate = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch, seq_len, num_features]
        batch_size, seq_len, num_features = x.shape

        # 1. Temporal processing (시간 축 정보)
        x_time = x.transpose(1, 2)  # [B, F, T]
        temporal_info = self.time_projection(x_time)  # [B, F, T]
        temporal_info = temporal_info.transpose(1, 2)  # [B, T, F]

        # 2. Feature processing (변수 축 정보)
        feature_info = self.feature_projection(x)  # [B, T, F]

        # 3. Gated fusion
        time_gate = self.time_gate(x)  # [B, T, 1]
        feature_gate = self.feature_gate(x)  # [B, T, 1]

        # 결합
        enhanced_x = x + time_gate * temporal_info + feature_gate * feature_info

        return enhanced_x


# ========== 수정된 MultiScaleFeatureExtractor ==========
class MultiScaleFeatureExtractor(nn.Module):
    """다중 스케일에서 feature 추출 (차원 불일치 수정)"""
    def __init__(self, seq_len, num_features, dropout=0.1):
        super().__init__()

        # 서로 다른 커널 사이즈로 다양한 시간 스케일 포착
        # [B, F, T] -> [B, F, T] 유지
        self.conv_short = nn.Conv1d(num_features, num_features, kernel_size=3, padding=1)
        self.conv_medium = nn.Conv1d(num_features, num_features, kernel_size=5, padding=2)
        self.conv_long = nn.Conv1d(num_features, num_features, kernel_size=7, padding=3)

        # Scale fusion: 3개의 scale을 합침
        self.fusion = nn.Sequential(
            nn.Linear(num_features * 3, num_features * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_features * 2, num_features)
        )

        self.norm = nn.LayerNorm(num_features)

    def forward(self, x):
        # x: [batch, seq_len, num_features]
        x_t = x.transpose(1, 2)  # [B, F, T]

        # 다양한 스케일 추출
        short_scale = F.gelu(self.conv_short(x_t))  # [B, F, T]
        medium_scale = F.gelu(self.conv_medium(x_t))  # [B, F, T]
        long_scale = F.gelu(self.conv_long(x_t))  # [B, F, T]

        # 다시 원래 shape으로
        short_scale = short_scale.transpose(1, 2)  # [B, T, F]
        medium_scale = medium_scale.transpose(1, 2)  # [B, T, F]
        long_scale = long_scale.transpose(1, 2)  # [B, T, F]

        # Concatenate and fuse
        multi_scale = torch.cat([short_scale, medium_scale, long_scale], dim=-1)  # [B, T, F*3]
        fused = self.fusion(multi_scale)  # [B, T, F]

        return self.norm(x + fused)


# ========== 수정된 UltimateInvertedAttention ==========
class UltimateInvertedAttention(nn.Module):
    """모든 개선사항을 통합한 최종 Attention (차원 불일치 수정)"""
    def __init__(self, num_heads, seq_len, num_features, dropout=0.1):
        super().__init__()

        # 1. Multi-scale feature extraction
        self.multi_scale = MultiScaleFeatureExtractor(seq_len, num_features, dropout)

        # 2. Cross-interaction
        self.cross_interaction = TemporalFeatureCrossInteraction(seq_len, num_features, dropout)

        # 3. Feature-axis attention (iTransformer)
        assert seq_len % num_heads == 0
        self.d_k = seq_len // num_heads
        self.num_heads = num_heads

        self.wq = nn.Linear(seq_len, seq_len)
        self.wk = nn.Linear(seq_len, seq_len)
        self.wv = nn.Linear(seq_len, seq_len)
        self.dense = nn.Linear(seq_len, seq_len)

        # 4. Time-axis attention (Transformer)
        assert num_features % num_heads == 0
        self.d_k_time = num_features // num_heads
        self.wq_time = nn.Linear(num_features, num_features)
        self.wk_time = nn.Linear(num_features, num_features)
        self.wv_time = nn.Linear(num_features, num_features)
        self.dense_time = nn.Linear(num_features, num_features)

        # 5. Dynamic routing weights
        self.routing = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # [B, F, T] -> [B, F, 1]
            nn.Flatten(),  # [B, F]
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # [multi_scale, feature_attn, time_attn]
            nn.Softmax(dim=-1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, num_features]
        batch_size, seq_len, num_features = x.shape

        # 1. Multi-scale features
        multi_scale_out = self.multi_scale(x)  # [B, T, F]

        # 2. Cross-interaction
        cross_out = self.cross_interaction(multi_scale_out)  # [B, T, F]

        # 3. Feature-axis attention
        x_feat = cross_out.transpose(1, 2)  # [B, F, T]

        q_feat = self.wq(x_feat).view(batch_size, num_features, self.num_heads, self.d_k)
        q_feat = q_feat.transpose(1, 2)

        k_feat = self.wk(x_feat).view(batch_size, num_features, self.num_heads, self.d_k)
        k_feat = k_feat.transpose(1, 2)

        v_feat = self.wv(x_feat).view(batch_size, num_features, self.num_heads, self.d_k)
        v_feat = v_feat.transpose(1, 2)

        scores_feat = torch.matmul(q_feat, k_feat.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_feat = F.softmax(scores_feat, dim=-1)
        feat_out = torch.matmul(attn_feat, v_feat)

        feat_out = feat_out.transpose(1, 2).contiguous().view(batch_size, num_features, seq_len)
        feat_out = self.dense(feat_out).transpose(1, 2)  # [B, T, F]

        # 4. Time-axis attention
        q_time = self.wq_time(cross_out).view(batch_size, seq_len, self.num_heads, self.d_k_time)
        q_time = q_time.transpose(1, 2)

        k_time = self.wk_time(cross_out).view(batch_size, seq_len, self.num_heads, self.d_k_time)
        k_time = k_time.transpose(1, 2)

        v_time = self.wv_time(cross_out).view(batch_size, seq_len, self.num_heads, self.d_k_time)
        v_time = v_time.transpose(1, 2)

        scores_time = torch.matmul(q_time, k_time.transpose(-2, -1)) / (self.d_k_time ** 0.5)
        attn_time = F.softmax(scores_time, dim=-1)
        time_out = torch.matmul(attn_time, v_time)

        time_out = time_out.transpose(1, 2).contiguous().view(batch_size, seq_len, num_features)
        time_out = self.dense_time(time_out)  # [B, T, F]

        # 5. Dynamic routing (학습 가능한 가중치 조합)
        # x를 [B, F, T]로 변환 후 global average pooling
        routing_input = cross_out.transpose(1, 2)  # [B, F, T]
        routing_weights = self.routing(routing_input)  # [B, 3]

        # Reshape for broadcasting: [B, 3] -> [B, 1, 1, 3]
        routing_weights = routing_weights.view(batch_size, 1, 1, 3)

        # Stack outputs: [B, T, F, 3]
        stacked = torch.stack([multi_scale_out, feat_out, time_out], dim=-1)

        # Weighted sum: [B, T, F, 3] * [B, 1, 1, 3] -> [B, T, F]
        output = (stacked * routing_weights).sum(dim=-1)

        return self.dropout(output)

class UltimateTransformerEncoderLayer(nn.Module):
    """최종 통합 인코더 레이어"""
    def __init__(self, seq_len, num_features, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.ultimate_attn = SimplifiedUltimateInvertedAttention(
            num_heads, seq_len, num_features, dropout
        )

        # Enhanced FFN with gating
        self.ffn = nn.Sequential(
            nn.Linear(seq_len, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, seq_len)
        )

        self.ffn_gate = nn.Sequential(
            nn.Linear(num_features, num_features // 4),
            nn.ReLU(),
            nn.Linear(num_features // 4, 1),
            nn.Sigmoid()
        )

        # Pre-norm
        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(num_features)

        # Learnable residual scaling
        self.res_scale = nn.Parameter(torch.ones(2) * 0.5)

    def forward(self, x, mask=None):
        # Attention block
        normed = self.norm1(x)
        attn_output = self.ultimate_attn(normed)
        x = x + self.res_scale[0] * attn_output

        # FFN block with gating
        normed = self.norm2(x)

        # Apply FFN on time axis
        ffn_out = normed.transpose(1, 2)  # [B, F, T]
        ffn_out = self.ffn(ffn_out)
        ffn_out = ffn_out.transpose(1, 2)  # [B, T, F]

        # Gated FFN
        gate = self.ffn_gate(normed)
        ffn_output = ffn_out * gate

        x = x + self.res_scale[1] * ffn_output

        return x


class UltimateITransformerEncoder(nn.Module):
    """모든 개선사항을 통합한 최종 인코더"""
    def __init__(self, num_layers, input_dim, max_len, d_model, num_heads, dff,
                 window_size=None, dropout=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.max_len = max_len

        # Input preprocessing
        self.input_norm = nn.LayerNorm(input_dim)

        # Enhanced positional encoding
        self.feature_pos_embedding = nn.Parameter(
            torch.randn(1, 1, input_dim) * 0.02
        )
        self.time_pos_embedding = nn.Parameter(
            torch.randn(1, max_len, 1) * 0.02
        )

        # Learnable position scales
        self.time_scale = nn.Parameter(torch.ones(1))
        self.feature_scale = nn.Parameter(torch.ones(1))

        # Ultimate encoder layers with layer-wise dropout
        self.enc_layers = nn.ModuleList([
            UltimateTransformerEncoderLayer(max_len, input_dim, num_heads, dff, dropout)
            for _ in range(num_layers)
        ])

        # Layer-wise dropout scheduling (깊은 레이어일수록 dropout 증가)
        self.layer_dropout_rates = [dropout * (1 + 0.1 * i) for i in range(num_layers)]

        # Output projection with residual
        self.output_projection = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

        self.output_skip = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        # Input normalization
        x = self.input_norm(x)

        # Scaled positional encoding
        pos_encoding = (self.time_scale * self.time_pos_embedding +
                       self.feature_scale * self.feature_pos_embedding)
        x = x + pos_encoding
        x = self.dropout(x)

        # Save for skip connection
        skip = x

        # Ultimate encoding with progressive dropout
        for i, enc_layer in enumerate(self.enc_layers):
            x = enc_layer(x, src_mask)

            # Apply layer-wise dropout
            if self.training:
                dropout_mask = torch.bernoulli(
                    torch.ones_like(x) * (1 - self.layer_dropout_rates[i])
                )
                x = x * dropout_mask

        # Output projection with skip connection
        output = self.output_projection(x) + self.output_skip(skip)

        return output

# ========== Decoder용 Cross-Attention ==========
class UltimateInvertedCrossAttention(nn.Module):
    """Decoder의 Cross-Attention (Encoder memory 참조)"""
    def __init__(self, num_heads, seq_len, num_features, dropout=0.1):
        super().__init__()

        assert seq_len % num_heads == 0
        self.d_k = seq_len // num_heads
        self.num_heads = num_heads

        # Query: decoder에서, Key/Value: encoder memory에서
        self.wq = nn.Linear(seq_len, seq_len)
        self.wk = nn.Linear(seq_len, seq_len)
        self.wv = nn.Linear(seq_len, seq_len)
        self.dense = nn.Linear(seq_len, seq_len)

        # Cross-attention에서도 multi-scale 정보 활용
        self.multi_scale_query = MultiScaleFeatureExtractor(seq_len, num_features, dropout)
        self.multi_scale_memory = MultiScaleFeatureExtractor(seq_len, num_features, dropout)

        # Adaptive fusion weight
        self.fusion_weight = nn.Sequential(
            nn.Linear(num_features, num_features // 4),
            nn.ReLU(),
            nn.Linear(num_features // 4, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, memory):
        # query: [B, T, F] (decoder)
        # memory: [B, T, F] (encoder)
        batch_size, seq_len, num_features = query.shape

        # Multi-scale enhancement
        query_enhanced = self.multi_scale_query(query)
        memory_enhanced = self.multi_scale_memory(memory)

        # Feature-axis cross-attention
        q = query_enhanced.transpose(1, 2)  # [B, F, T]
        k = memory_enhanced.transpose(1, 2)  # [B, F, T]
        v = memory_enhanced.transpose(1, 2)  # [B, F, T]

        q = self.wq(q).view(batch_size, num_features, self.num_heads, self.d_k)
        q = q.transpose(1, 2)  # [B, H, F, d_k]

        k = self.wk(k).view(batch_size, num_features, self.num_heads, self.d_k)
        k = k.transpose(1, 2)

        v = self.wv(v).view(batch_size, num_features, self.num_heads, self.d_k)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(batch_size, num_features, seq_len)
        attn_out = self.dense(attn_out).transpose(1, 2)  # [B, T, F]

        # Adaptive fusion
        fusion_gate = self.fusion_weight(query)
        output = fusion_gate * attn_out + (1 - fusion_gate) * query

        return self.dropout(output)


# ========== Ultimate Decoder Layer ==========
class UltimateInvertedDecoderLayer(nn.Module):
    """모든 개선사항을 통합한 Decoder Layer"""
    def __init__(self, seq_len, num_features, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # 1. Self-Attention (decoder 내부)
        self.self_attn = SimplifiedUltimateInvertedAttention(
            num_heads, seq_len, num_features, dropout
        )

        # 2. Cross-Attention (encoder memory 참조)
        self.cross_attn = UltimateInvertedCrossAttention(
            num_heads, seq_len, num_features, dropout
        )

        # 3. Enhanced FFN with gating
        self.ffn = nn.Sequential(
            nn.Linear(seq_len, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, seq_len)
        )

        self.ffn_gate = nn.Sequential(
            nn.Linear(num_features, num_features // 4),
            nn.ReLU(),
            nn.Linear(num_features // 4, 1),
            nn.Sigmoid()
        )

        # Pre-norm
        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(num_features)
        self.norm3 = nn.LayerNorm(num_features)

        # Learnable residual scaling
        self.res_scale = nn.Parameter(torch.ones(3) * 0.5)

    def forward(self, x, memory, self_mask=None, cross_mask=None):
        # 1. Self-Attention
        normed = self.norm1(x)
        self_attn_out = self.self_attn(normed)
        x = x + self.res_scale[0] * self_attn_out

        # 2. Cross-Attention
        normed = self.norm2(x)
        cross_attn_out = self.cross_attn(normed, memory)
        x = x + self.res_scale[1] * cross_attn_out

        # 3. FFN with gating
        normed = self.norm3(x)

        ffn_out = normed.transpose(1, 2)
        ffn_out = self.ffn(ffn_out)
        ffn_out = ffn_out.transpose(1, 2)

        gate = self.ffn_gate(normed)
        ffn_output = ffn_out * gate

        x = x + self.res_scale[2] * ffn_output

        return x


# ========== Ultimate iTransformer Decoder ==========
class UltimateITransformerDecoder(nn.Module):
    """모든 개선사항을 통합한 최종 Decoder"""
    def __init__(self, num_layers, compressed_len, seq_len, num_features,
                 num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.compressed_len = compressed_len
        self.num_features = num_features
        self.seq_len = seq_len

        # 학습 가능한 feature query embedding
        self.feature_queries = nn.Parameter(
            torch.randn(num_features, compressed_len) * 0.02
        )

        # Enhanced positional encoding
        self.feature_pos_embedding = nn.Parameter(
            torch.randn(1, 1, num_features) * 0.02
        )
        self.time_pos_embedding = nn.Parameter(
            torch.randn(1, compressed_len, 1) * 0.02
        )

        # Learnable position scales
        self.time_scale = nn.Parameter(torch.ones(1))
        self.feature_scale = nn.Parameter(torch.ones(1))

        # Input normalization
        self.input_norm = nn.LayerNorm(num_features)

        # Ultimate decoder layers
        self.decoder_layers = nn.ModuleList([
            UltimateInvertedDecoderLayer(
                compressed_len, num_features, num_heads, d_ff, dropout
            )
            for _ in range(num_layers)
        ])

        # Layer-wise dropout scheduling
        self.layer_dropout_rates = [dropout * (1 + 0.1 * i) for i in range(num_layers)]

        # Progressive upsampling for time dimension
        # compressed_len -> seq_len
        self.time_upsample = nn.Sequential(
            nn.Linear(compressed_len, compressed_len * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(compressed_len * 2, seq_len)
        )

        # Skip connection for upsampling
        self.upsample_skip = nn.Linear(compressed_len, seq_len)

        self.dropout = nn.Dropout(dropout)

    def forward(self, memory, target_len=None, use_mask=False):
        """
        memory: [batch, compressed_len, num_features] (encoder output)
        """
        batch_size = memory.size(0)

        # 1. Initialize decoder input with learnable queries
        query_embed = self.feature_queries.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, F, T]
        x = query_embed.transpose(1, 2)  # [B, T, F]

        # 2. Input normalization
        x = self.input_norm(x)

        # 3. Add positional encoding
        pos_encoding = (self.time_scale * self.time_pos_embedding +
                       self.feature_scale * self.feature_pos_embedding)
        x = x + pos_encoding
        x = self.dropout(x)

        # Save for skip connection
        skip = x

        # 4. Pass through decoder layers
        for i, decoder_layer in enumerate(self.decoder_layers):
            x = decoder_layer(x, memory)

            # Apply layer-wise dropout
            if self.training:
                dropout_mask = torch.bernoulli(
                    torch.ones_like(x) * (1 - self.layer_dropout_rates[i])
                )
                x = x * dropout_mask

        # 5. Time dimension upsampling (compressed_len -> seq_len)
        # [B, compressed_len, F] -> [B, seq_len, F]
        x_transposed = x.transpose(1, 2)  # [B, F, compressed_len]

        upsampled = self.time_upsample(x_transposed)  # [B, F, seq_len]
        skip_upsampled = self.upsample_skip(skip.transpose(1, 2))

        # Combine with skip connection
        output = upsampled + skip_upsampled
        output = output.transpose(1, 2)  # [B, seq_len, F]

        return output

class SimplifiedUltimateInvertedAttention(nn.Module):
    """
    num_heads 제약 없이 작동하는 간소화 버전
    Feature-axis attention만 사용 (iTransformer 스타일)
    """
    def __init__(self, num_heads, seq_len, num_features, dropout=0.1):
        super().__init__()

        # 1. Multi-scale feature extraction
        self.multi_scale = MultiScaleFeatureExtractor(seq_len, num_features, dropout)

        # 2. Cross-interaction
        self.cross_interaction = TemporalFeatureCrossInteraction(seq_len, num_features, dropout)

        # 3. Feature-axis attention (iTransformer) - num_heads 제약 없음
        # d_k를 고정값으로 설정
        self.d_k = max(seq_len // num_heads, 16)  # 최소 16 보장
        self.num_heads = num_heads

        self.wq = nn.Linear(seq_len, self.d_k * num_heads)
        self.wk = nn.Linear(seq_len, self.d_k * num_heads)
        self.wv = nn.Linear(seq_len, self.d_k * num_heads)
        self.dense = nn.Linear(self.d_k * num_heads, seq_len)

        # 4. Adaptive fusion
        self.fusion = nn.Sequential(
            nn.Linear(num_features * 2, num_features),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, num_features]
        batch_size, seq_len, num_features = x.shape

        # 1. Multi-scale features
        multi_scale_out = self.multi_scale(x)

        # 2. Cross-interaction
        cross_out = self.cross_interaction(multi_scale_out)

        # 3. Feature-axis attention
        x_feat = cross_out.transpose(1, 2)  # [B, F, T]

        q = self.wq(x_feat).view(batch_size, num_features, self.num_heads, self.d_k)
        q = q.transpose(1, 2)  # [B, H, F, d_k]

        k = self.wk(x_feat).view(batch_size, num_features, self.num_heads, self.d_k)
        k = k.transpose(1, 2)

        v = self.wv(x_feat).view(batch_size, num_features, self.num_heads, self.d_k)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, v)

        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(batch_size, num_features, self.d_k * self.num_heads)
        attn_out = self.dense(attn_out).transpose(1, 2)  # [B, T, F]

        # 4. Fusion with multi-scale and cross-interaction
        combined = torch.cat([multi_scale_out, attn_out], dim=-1)  # [B, T, F*2]
        output = self.fusion(combined)  # [B, T, F]

        return self.dropout(output + cross_out)

class LocalMultiHeadedAttention(nn.Module):
    """일반 Transformer용 Local Attention"""
    def __init__(self, num_heads, d_model, window_size=32, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.window_size = window_size

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def create_local_mask(self, seq_len, device):
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = False
        return mask

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        seq_len = query.size(1)

        # 로컬 마스크 생성
        local_mask = self.create_local_mask(seq_len, query.device)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            local_mask = local_mask.unsqueeze(0).unsqueeze(0) | mask
        else:
            local_mask = local_mask.unsqueeze(0).unsqueeze(0)

        # Q, K, V 계산
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        x, self.attn = self.attention(query, key, value, mask=local_mask)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)
        return self.dropout(self.dense(x))

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value), p_attn


class LocalInvertedMultiHeadAttention(nn.Module):
    """iTransformer용 Local Attention (변수 축)"""
    def __init__(self, num_heads, seq_len, window_size=32, dropout=0.1):
        super().__init__()
        assert seq_len % num_heads == 0
        self.d_k = seq_len // num_heads
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.window_size = window_size

        self.wq = nn.Linear(seq_len, seq_len)
        self.wk = nn.Linear(seq_len, seq_len)
        self.wv = nn.Linear(seq_len, seq_len)
        self.dense = nn.Linear(seq_len, seq_len)

        self.dropout = nn.Dropout(p=dropout)

    def create_local_mask(self, num_features, device):
        """변수 축에서의 로컬 마스크"""
        mask = torch.ones(num_features, num_features, dtype=torch.bool, device=device)
        for i in range(num_features):
            start = max(0, i - self.window_size // 2)
            end = min(num_features, i + self.window_size // 2 + 1)
            mask[i, start:end] = False
        return mask

    def forward(self, x):
        # x: [batch, seq_len, features] -> [batch, features, seq_len]
        x = x.transpose(1, 2)
        batch_size, num_features, seq_len = x.shape

        # 변수 축에서 로컬 마스크 생성
        local_mask = self.create_local_mask(num_features, x.device)
        local_mask = local_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, features, features]

        # Q, K, V 계산
        query = self.wq(x).view(batch_size, num_features, self.num_heads, self.d_k)
        query = query.transpose(1, 2)

        key = self.wk(x).view(batch_size, num_features, self.num_heads, self.d_k)
        key = key.transpose(1, 2)

        value = self.wv(x).view(batch_size, num_features, self.num_heads, self.d_k)
        value = value.transpose(1, 2)

        # Attention with local mask
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(local_mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, num_features, seq_len)

        output = self.dense(attn_output)
        output = self.dropout(output)

        # [batch, features, seq_len] -> [batch, seq_len, features]
        output = output.transpose(1, 2)
        return output

class DeepSC(nn.Module):
    def __init__(
        self,
        num_layers=2,
        input_dim=6,
        max_len=128,
        d_model=128,
        num_heads=4,
        dff=512,
        dropout=0.1,
        compressed_len=None,
        window_size=None,  # None=일반, int=Local
        use_itransformer=False,  # True=iTransformer, False=Transformer
        params=None,
        **kwargs
    ):
        super(DeepSC, self).__init__()
        p = params if params is not None else {}
        self.num_layers = p.get("num_layers", num_layers)
        self.input_dim = p.get("input_dim", input_dim)
        self.max_len = p.get("max_len", max_len)
        self.d_model = p.get("d_model", d_model)
        self.num_heads = p.get("num_heads", num_heads)
        self.dff = p.get("dff", dff)
        self.dropout = p.get("dropout", dropout)
        self.compressed_len = p.get("compressed_len", compressed_len)
        self.d_comp = p.get("d_comp", 3)
        self.window_size = p.get("window_size", window_size)
        self.use_itransformer = p.get("use_itransformer", use_itransformer)

        # 의미 인코더 = encoder + time_compressor
        if self.use_itransformer:
            self.encoder = UltimateITransformerEncoder(
            # self.encoder = iTransformerEncoder(
                self.num_layers, self.input_dim, self.max_len,
                self.d_model, self.num_heads, self.dff,
                window_size=self.window_size,
                dropout=self.dropout
            )
        else:
            self.encoder = Encoder(
                self.num_layers, self.input_dim, self.max_len,
                self.d_model, self.num_heads, self.dff,
                window_size=self.window_size,
                dropout=self.dropout
            )

        # 시계열 길이 압축 모듈 (선택)
        if self.compressed_len is not None:
            self.time_compressor = LearnableTimeCompressor(self.d_model, self.compressed_len)
        else:
            self.time_compressor = None

        # 점진적 압축으로 변경
        self.channel_encoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),  # 128 → 64
            nn.LayerNorm(self.d_model // 2),
            # nn.ReLU(),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model // 2, self.d_comp),  # 32 → 3
        )

        self.channels = Channels()

        if self.use_itransformer:
            self.decoder = UltimateITransformerDecoder(
            # self.decoder = iTransformerDecoder(
                num_layers=self.num_layers,
                compressed_len=self.compressed_len,
                seq_len=self.max_len,
                num_features=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.dff)
        else:
            self.decoder = Decoder(
                num_layers=self.num_layers,
                d_model=self.d_model,
                num_heads=self.num_heads,
                dff=self.dff,
                max_len=self.max_len,
                dropout=self.dropout
            )


        self.channel_decoder = ResidualChannelDecoder(
            self.d_comp, self.d_model
        )

        # 자연어 디코더 대신 시계열 출력 레이어 사용
        self.output_projection = nn.Linear(self.d_model, self.input_dim)

        # 시계열 길이 복원
        self.time_decompressor = LearnableTimeDecompressor(self.d_model, self.max_len)
        # self.time_decompressor = SmoothTimeDecompressor(self.d_model, self.max_len)

        # 업샘플링 레이어 추가 (compressed_len → max_len)
        self.upsample = nn.Upsample(
            size=self.max_len, mode="linear", align_corners=False
        )


    def forward(self, x, src_mask=None):
        # x: (batch_size, seq_len, input_dim) - 시계열 데이터

        # 1단계: 의미 인코더
        # (batch, max_len, input_dim->d_model)
        encoded = self.encoder(x, src_mask)

        # 2단계: sequence compress (downsampling) (시계열 압축)
        # (batch, max_len->compressed_len, d_model)
        compressed = self.time_compressor(encoded)

        # 3단계: 채널 인코더 (피쳐 압축)
        # (batch_size, compressed_len, d_model->d_comp)
        channel_encoded = self.channel_encoder(compressed)

        # 4단계 : 채널 상태 적용
        # (batch_size, compressed_len, d_comp)
        snr_db = 5
        channel_syms = self.channels.AWGN(channel_encoded, snr_db)
        # channel_syms = channel_encoded

        # 5단계 : 채널 디코더 (피쳐 복원 예측을 위한 linear 적용)
        # (batch_size, compressed_len, d_comp -> d_model)
        channel_decoded = self.channel_decoder(channel_syms)

        # 6단계 : sequence decompress (upsampling) (시계열 복원)
        # decompressed = self.time_decompressor(channel_decoded)

        # 6단계 : 의미 디코더
        # (batch_size, compressed_len->max_len, d_model)
        decompressed = self.decoder(channel_decoded, use_mask=True) # transformer
        # decompressed = self.decoder(channel_decoded) # itransformer

        # 7단계: 출력 투영
        # (batch_size, max_len, d_model->input_dim => 원래 피쳐 차원으로 복원)
        output = self.output_projection(decompressed)

        return output
