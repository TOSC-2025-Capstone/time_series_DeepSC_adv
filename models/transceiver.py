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
import numpy as np

def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)

    return noise_std

# from samba_mixer.model.input_projections.linear_projection_time_embedding_cycle_diff_embedding import LinearProjectionWithLocalTimeAndGlobalDiffEmbedding
from utils import Channels, power_normalize

import parameters.parameters as parameters
import parameters.model_parameters as mparams

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
    def __init__(self, num_heads, d_model, dropout=0.1, max_len=512):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        # # 수정: 학습 가능한 Temporal Bias 정의
        # # [num_heads, max_len, max_len] 크기로 정의하여 Attention Score에 더함
        # self.temporal_bias = nn.Parameter(
        #     torch.randn(1, num_heads, max_len, max_len) * 0.01
        # )

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

        # # 수정: Temporal Bias 추가
        # # bias를 현재 시퀀스 길이에 맞게 잘라서 scores에 더함
        # seq_len = scores.size(-1)
        # bias = self.temporal_bias[:, :, :seq_len, :seq_len] # [1, num_heads, seq_len, seq_len]

        # scores = scores + bias # scores에 시간적 위치 가중치 적용

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

class Conv1dFeedForward(nn.Module):
    """
    Conv1D 기반의 FFN 구현 (Temporal Block)
    d_model을 Channel로 간주하여 시계열 특징을 추출함
    """
    def __init__(self, d_model, d_ff, dropout=0.1, kernel_size=3):
        super().__init__()

        # 1. d_model -> d_ff로 확장하는 Conv1D (kernel_size=1은 Linear와 유사)
        # kernel_size=3을 사용하여 지역적인 시간 특징을 혼합
        self.conv_1 = nn.Conv1d(in_channels=d_model,
                                out_channels=d_ff,
                                kernel_size=kernel_size,
                                padding=kernel_size // 2) # 시퀀스 길이 유지를 위한 padding

        # 2. d_ff -> d_model로 축소하는 Conv1D (Projection 역할)
        self.conv_2 = nn.Conv1d(in_channels=d_ff,
                                out_channels=d_model,
                                kernel_size=kernel_size,
                                padding=kernel_size // 2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [Batch, Seq_Len, d_model]

        # 1. Conv1D 입력 형태 맞추기: [B, d_model, Seq_Len]
        x = x.transpose(1, 2)

        # 2. Conv 블록 통과
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv_2(x)

        # 3. Transformer 출력 형태 맞추기: [Batch, Seq_Len, d_model]
        x = x.transpose(1, 2)

        return x

class Conv1dInvertedFeedForward(nn.Module):
    """
    Conv1D 기반의 Inverted FFN 대체 (Temporal Block 역할)
    """
    def __init__(self, seq_len, d_ff, dropout=0.1, kernel_size=3):
        super().__init__()

        # Inverted Attention Layer의 출력 차원(Features)을 채널로 사용
        self.conv_1 = nn.Conv1d(in_channels=seq_len,      # 이전 Inverted FFN의 seq_len이 Channel 역할을 하도록
                                out_channels=d_ff,
                                kernel_size=kernel_size,
                                padding=kernel_size // 2)

        self.conv_2 = nn.Conv1d(in_channels=d_ff,
                                out_channels=seq_len,
                                kernel_size=kernel_size,
                                padding=kernel_size // 2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [Batch, Seq_Len, Features]
        # 1. Conv1D 입력 형태 맞추기: [B, Seq_Len, Features] -> [B, Features, Seq_Len]
        # d_model 차원 대신, Seq_Len 차원이 Channel 역할을 하도록 이미 구현되어 있음 (Inverted Logic)
        x = x.transpose(1, 2)

        # 2. Conv 블록 통과
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv_2(x)

        # 3. Transformer 출력 형태 맞추기: [Batch, Features, Seq_Len] -> [Batch, Seq_Len, Features]
        x = x.transpose(1, 2)

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
    def __init__(self, d_model, num_heads, dff, window_size=None, dropout=0.1, max_len=512):
        """
        window_size=None: 일반 Attention
        window_size=int: Local Attention
        """
        super().__init__()

        self.mha = MultiHeadedAttention(num_heads, d_model, dropout, max_len=max_len)

        self.ffn = PositionwiseFeedForward(d_model, dff, dropout)
        # self.ffn = Conv1dFeedForward(d_model, dff, dropout, kernel_size=5)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask):
        attn_output = self.mha(x, x, x, mask)
        x = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.layernorm2(x + ffn_output)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1, window_size=None, max_len=512):
        super().__init__()

        # 기존 MultiHeadedAttention 사용
        self.self_attn = MultiHeadedAttention(num_heads, d_model, dropout, max_len)
        self.cross_attn = MultiHeadedAttention(num_heads, d_model, dropout, max_len)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout)
        # self.ffn = Conv1dFeedForward(d_model, dff, dropout, kernel_size=5)

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
        self.pos_encoding = TimeSeriesPositionalEncoding(d_model, dropout, max_len)
        # self.pos_encoding = NoiseAdaptivePositionalEncoding(d_model, dropout, max_len)

        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, dff, window_size, dropout, max_len)
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
        self.pos_encoding = TimeSeriesPositionalEncoding(d_model, dropout, max_len)
        # self.pos_encoding = NoiseAdaptivePositionalEncoding(d_model, dropout, max_len)

        # 디코더 레이어들
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, dff, dropout, max_len=max_len)
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
        # skip = self.skip_proj(skip)

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
        self.input_dim = mparams.model_params.get("input_dim", None)

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

    def forward(self, query, key, value):
        query_transposed = query.transpose(1, 2)
        key_transposed = key.transpose(1, 2)
        value_transposed = value.transpose(1, 2)
        batch_size, num_features, seq_len = query_transposed.shape

        # Feature importance 계산
        # feature_importance = self.feature_gate(query_transposed)  # [B, F, 1]

        # Q, K, V 계산
        query = self.wq(query_transposed).view(batch_size, num_features, self.num_heads, self.d_k)
        query = query.transpose(1, 2)  # [B, num_heads, F, d_k]

        key = self.wk(key_transposed).view(batch_size, num_features, self.num_heads, self.d_k)
        key = key.transpose(1, 2)  # [B, num_heads, F, d_k]

        value = self.wv(value_transposed).view(batch_size, num_features, self.num_heads, self.d_k)
        value = value.transpose(1, 2)  # [B, num_heads, F, d_k]

        # Scaled dot-product attention with feature gating
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores shape: [B, num_heads, F, F]

        # Feature importance를 attention에 반영 (차원 맞추기)
        # feature_importance: [B, F, 1] -> [B, 1, F, 1]로 확장
        # feature_importance = feature_importance.unsqueeze(1)  # [B, 1, F, 1]

        # Query 측 feature importance 적용
        # scores = scores * feature_importance  # 브로드캐스팅: [B, num_heads, F, F] * [B, 1, F, 1]

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

        # self.inverted_ffn = InvertedFeedForward(seq_len, d_ff, dropout)
        self.inverted_ffn = Conv1dInvertedFeedForward(seq_len, d_ff, dropout, kernel_size=mparams.model_params.get("kernel_size"))
        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(num_features)

    def forward(self, x, mask=None):
        attn_output = self.inverted_attention(x,x,x)
        x = self.norm1(x + attn_output)

        ffn_output = self.inverted_ffn(x)
        x = self.norm2(x + ffn_output)

        return x

class iTransformerEncoder(nn.Module):
    def __init__(self, num_layers, input_dim, max_len, d_seq, d_model, num_heads, dff,
                 window_size=None, dropout=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.max_len = max_len

        self.input_projection = nn.Linear(input_dim, d_model) # case 45
        self.input_time_projection = nn.Linear(max_len, d_seq)

        # Feature-wise positional encoding 추가
        self.feature_pos_embedding = nn.Parameter(
            torch.randn(1, 1, d_model) * 0.1
        )

        # Time-wise positional encoding
        self.time_pos_embedding = nn.Parameter(
            torch.randn(1, max_len, 1) * 0.1
        )

        # self.layernorm_before_pe = nn.LayerNorm(input_dim, eps=1e-6)

        self.enc_layers = nn.ModuleList([
            # iTransformerEncoderLayer(max_len, input_dim, num_heads, dff, window_size, dropout)
            iTransformerEncoderLayer(d_seq, d_model, num_heads, dff, window_size, dropout) # case 45
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        # 양방향 positional encoding 추가
        x = self.input_projection(x)
        x = x.permute(0,2,1)
        x = self.input_time_projection(x)
        x = x.permute(0,2,1)

        # x = x + self.time_pos_embedding + self.feature_pos_embedding
        # x = x + self.time_pos_embedding

        x = self.dropout(x)

        for enc_layer in self.enc_layers:
            x = enc_layer(x, src_mask)

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
        # self.ffn = InvertedFeedForward(seq_len, d_ff, dropout)
        self.ffn = Conv1dInvertedFeedForward(seq_len, d_ff, dropout, kernel_size=mparams.model_params.get("kernel_size"))

        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(num_features)
        self.norm3 = nn.LayerNorm(num_features)

    def forward(self, x, memory, self_mask=None, cross_mask=None):
        """
        x: [batch, seq_len, num_features]
        memory: [batch, seq_len, num_features]
        """
        # Self-Attention (feature 축)
        attn_out = self.self_attn(x,x,x)
        x = self.norm1(x + attn_out)

        # Cross-Attention (encoder memory 참조)
        # memory를 같은 축으로 맞춰줌
        cross_out = self.cross_attn(x,memory,memory)
        x = self.norm2(x + cross_out)

        # Feed Forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)

        return x


class iTransformerDecoder(nn.Module):
    """
    iTransformer 기반 디코더 (수정 버전)
    - 입력으로 max_len 길이의 시퀀스를 받도록 변경
    """
    def __init__(self, num_layers, seq_len, num_features, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # compressed_len은 이제 사용되지 않지만, 호환성을 위해 파라미터는 유지
        self.num_features = num_features
        self.seq_len = seq_len # max_len에 해당

        self.feature_queries = nn.Parameter(torch.randn(num_features, seq_len) * 0.1)

        # Feature-wise positional encoding
        self.feature_pos_embedding = nn.Parameter(torch.randn(1, 1, num_features) * 0.1)
        self.time_pos_embedding = nn.Parameter(torch.randn(1, seq_len, 1) * 0.1)
        self.dropout = nn.Dropout(dropout)

        self.decoder_layers = nn.ModuleList([
            InvertedDecoderLayer(seq_len, num_features, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, memory, target_len=None, use_mask=False):
        """
        memory: [batch, seq_len, num_features] (이미 max_len으로 복원된 텐서)
        """
        batch_size = memory.size(0)

        # feature query 임베딩 준비
        query_embed = self.feature_queries.unsqueeze(0).repeat(batch_size, 1, 1) # [B, F, T(max_len)]
        x = query_embed.transpose(1, 2) # [B, T(max_len), F]

        # Positional Encoding 적용
        # x = x + self.time_pos_embedding + self.feature_pos_embedding
        # x = x + self.time_pos_embedding
        x = self.dropout(x)

        # 각 디코더 레이어 통과
        for layer in self.decoder_layers:
            # 이제 x와 memory 모두 시퀀스 길이가 max_len으로 동일함
            x = layer(x, memory)

        # 변경 5: self.output_projection 호출 삭제, x를 그대로 반환
        return x

class DeepSC(nn.Module):
    def __init__(
        self,
        num_layers=2,
        input_dim=6,
        output_dim=6,
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
        self.output_dim = p.get("output_dim", output_dim)
        self.max_len = p.get("max_len", max_len)
        self.seq_len = p.get("seq_len", 512)
        self.num_heads = p.get("num_heads", num_heads)
        self.dff = p.get("dff", dff)
        self.dropout = p.get("dropout", dropout)
        self.compressed_len = p.get("compressed_len", compressed_len)
        self.d_comp = p.get("d_comp", 3)
        self.d_model = p.get("d_model", d_model) # 모델 피쳐 확장차원
        self.d_seq = p.get("d_seq", 512) # 모델 시간 확장차원

        self.hidden_dim = p.get("hidden_dim", 512)
        self.compressed_features = p.get("compressed_features", 0)

        self.window_size = p.get("window_size", window_size)
        self.use_itransformer = p.get("use_itransformer", use_itransformer)
        self.snr_db = p.get("snr_db", 5)  # 기본 SNR 값
        self.model_type = kwargs.get("model_type", "deepsc")  # 모델 타입 (gru/lstm)
        assert self.model_type in ["deepsc","gru", "lstm"], "model_type은 'deepsc', 'gru' 또는 'lstm'이어야 합니다."

        # 의미 인코더 = encoder + time_compressor
        if self.model_type == 'deepsc':
            if self.use_itransformer:
                self.encoder = iTransformerEncoder(
                    # self.num_layers, self.input_dim, self.max_len,
                    self.num_layers, self.input_dim, self.max_len, self.d_seq,
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
        elif self.model_type == 'gru':
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.GRU(
                    input_size=self.hidden_dim,
                    hidden_size=self.hidden_dim,
                    num_layers=self.num_layers,
                    dropout=self.dropout,
                    batch_first=True,
                ),
            )
        elif self.model_type == 'lstm':
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.LSTM(
                    input_size=self.hidden_dim,
                    hidden_size=self.hidden_dim,
                    num_layers=self.num_layers,
                    dropout=self.dropout,
                    batch_first=True,
                ),
            )

        # 시계열 길이 압축 모듈 (선택)
        if self.compressed_len is not None:
            self.time_compressor = LearnableTimeCompressor(self.d_model, self.compressed_len)
            # self.time_compressor = LearnableTimeCompressor(self.input_dim, self.compressed_len)
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

        self.channel_decoder = ResidualChannelDecoder(
            self.d_comp, self.d_model
        )
        # self.channel_decoder = nn.Linear(self.d_comp, self.input_dim)

        # 시계열 길이 복원
        # self.time_decompressor = LearnableTimeDecompressor(self.input_dim, self.max_len)
        self.time_decompressor = LearnableTimeDecompressor(self.d_model, self.d_seq)

        if self.model_type == 'deepsc':
            if self.use_itransformer:
                self.decoder = iTransformerDecoder(
                    num_layers=self.num_layers,
                    # seq_len=self.max_len,
                    seq_len=self.d_seq,
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
                    # max_len=self.compressed_len, # time_decompressor에서 복원
                    dropout=self.dropout
                )
        elif self.model_type == 'gru':
            self.decoder = nn.GRU(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
                batch_first=True,
            )
        elif self.model_type == 'lstm':
            self.decoder = nn.LSTM(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
                batch_first=True,
            )

        # 자연어 디코더 대신 시계열 출력 레이어 사용
        self.output_projection = nn.Linear(self.d_model, self.input_dim)
        self.output_time_projection = nn.Linear(self.d_seq, self.seq_len)


    def forward(self, x, src_mask=None):
        # x: (batch_size, seq_len, input_dim) - 시계열 데이터
        # 1단계: 의미 인코더
        # (batch, max_len, input_dim->d_model)
        if self.model_type == 'deepsc' :
            encoded = self.encoder(x, src_mask)
        else:  # GRU/LSTM 인코더
            encoded, _ = self.encoder(x)
        # encoded = self.feat_expander(x)

        # 2단계: sequence compress (downsampling) (시계열 압축)
        # (batch, max_len->compressed_len, d_model)
        # encoded.permute(0, 2, 1)  # (batch, d_model, max_len)
        compressed = self.time_compressor(encoded)

        # 3단계: 채널 인코더 (피쳐 압축)
        # (batch_size, compressed_len, d_model->d_comp)
        channel_encoded = self.channel_encoder(compressed)

        tx_sig = power_normalize(channel_encoded)

        # 4단계 : 채널 상태 적용
        if parameters.is_train_phase == False:
            # (batch_size, compressed_len, d_comp)
            rx_sig = self.channels.Rayleigh(tx_sig, self.snr_db)
            # rx_sig = self.channels.fading(tx_sig, 0, n_std, detector="MMSE")
        else:
            rx_sig = tx_sig

        # 5단계 : 채널 디코더
        # (batch_size, compressed_len, d_comp -> d_model)
        channel_decoded = self.channel_decoder(rx_sig)

        # 6단계 : 시계열 복원
        # (batch_size, compressed_len->max_len, d_model)
        decompressed = self.time_decompressor(channel_decoded)

        # 7단계 : 의미 디코더
        if self.model_type == 'deepsc' :
            output = self.decoder(decompressed, use_mask=True)
        else:  # GRU/LSTM 인코더
            output, _ = self.decoder(decompressed)
        # output, _ = self.decoder(decompressed)
        # output = decompressed

        # 8단계: 출력 투영
        # (batch_size, max_len, d_model->input_dim => 원래 피쳐 차원으로 복원)
        final_output = self.output_projection(output)
        final_output = final_output.permute(0,2,1)
        final_output = self.output_time_projection(final_output)
        final_output = final_output.permute(0,2,1)

        # final_output = output

        return final_output
