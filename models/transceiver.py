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
            self.mha = EMAAttention(num_heads, d_model, dropout=0.1)
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



class TimeSeriesDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
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


class TimeSeriesDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, max_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # 학습 가능한 위치별 쿼리 임베딩
        self.pos_queries = nn.Parameter(torch.randn(max_len, d_model) * 0.1)

        # 위치 인코딩 (기존 것 재사용 가능)
        self.pos_encoding = TimeSeriesPositionalEncoding(d_model, dropout, max_len)

        # 디코더 레이어들
        self.decoder_layers = nn.ModuleList([
            TimeSeriesDecoderLayer(d_model, num_heads, dff, dropout)
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


class TimeSeriesCompressor(nn.Module):
    """
    시계열 길이(seq_len)를 원하는 길이로 압축하는 모듈
    """

    def __init__(self, target_len):
        super(TimeSeriesCompressor, self).__init__()
        self.target_len = target_len
        self.pool = nn.AdaptiveAvgPool1d(target_len)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x.permute(0, 2, 1)  # (batch, d_model, seq_len)
        x = self.pool(x)  # (batch, d_model, target_len)
        x = x.permute(0, 2, 1)  # (batch, target_len, d_model)
        return x

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
    """iTransformer의 핵심: 변수 축에서 어텐션 수행"""
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

        # Q, K, V 계산 (변수 축에서)
        query = self.wq(x).view(batch_size, num_features, self.num_heads, self.d_k)
        query = query.transpose(1, 2)

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

        # 다시 원래 차원으로: [batch, features, seq_len] -> [batch, seq_len, features]
        output = output.transpose(1, 2)

        return output

class DenoisingMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1, noise_threshold=0.3):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        self.noise_threshold = noise_threshold

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

        self.noise_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

        self.attn = None

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)

        # 1. 노이즈 레벨 추정
        noise_scores = self.noise_gate(key)  # [batch, seq_len, 1]

        # 2. Q, K, V 계산
        Q = self.wq(query).view(batch_size, seq_len, self.num_heads, self.d_k)
        Q = Q.transpose(1, 2)  # [batch, heads, seq_len, d_k]

        K = self.wk(key).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = K.transpose(1, 2)  # [batch, heads, seq_len, d_k]

        V = self.wv(value).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = V.transpose(1, 2)  # [batch, heads, seq_len, d_k]

        # 3. Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: [batch, heads, seq_len, seq_len]

        # 4. 기존 mask 처리
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask.bool(), -1e9)

        # 5. 노이즈 마스킹 수정
        noise_mask = (noise_scores >= self.noise_threshold).float()  # threshold 이상만 허용
        # noise_mask: [batch, seq_len, 1]

        # Key 위치에 대한 마스크 (마지막 차원)
        noise_mask = noise_mask.squeeze(-1)  # [batch, seq_len]
        noise_mask = noise_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]

        # 노이즈가 높은 key 위치 차단
        scores = scores.masked_fill((noise_mask < 0.5).bool(), -1e9)

        # 6. Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        # attn_weights: [batch, heads, seq_len, seq_len]
        self.attn = attn_weights

        # 7. Output 계산
        output = torch.matmul(attn_weights, V)
        # output: [batch, heads, seq_len, d_k]

        # 8. Reshape
        output = output.transpose(1, 2).contiguous()
        # output: [batch, seq_len, heads, d_k]

        output = output.view(batch_size, seq_len, self.d_model)
        # output: [batch, seq_len, d_model]

        # 9. Final transformation
        output = self.dense(output)
        output = self.dropout(output)

        return output

class LocalMultiHeadedAttention(nn.Module):
    """일반 Transformer용 Local Attention"""
    def __init__(self, num_heads, d_model, window_size=32, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.window_size = int(window_size)

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

class ResidualTemperatureAttention(nn.Module):
    """Residual Attention + Temperature Scaling 결합"""
    def __init__(self, num_heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

        self.temperature_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Softplus()
        )

        self.attn_alpha = nn.Parameter(torch.ones(1) * 0.5)

        self.attn = None
        self.prev_attn = None

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)

        # Temperature 예측
        temperature = self.temperature_predictor(query.mean(dim=1))
        temperature = temperature.unsqueeze(1).unsqueeze(2) + 1.0

        # Q, K, V 계산
        Q = self.wq(query).view(batch_size, seq_len, self.num_heads, self.d_k)
        Q = Q.transpose(1, 2)

        K = self.wk(key).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = K.transpose(1, 2)

        V = self.wv(value).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = V.transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Mask 처리
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask.bool(), -1e9)

        # Temperature Scaling
        scores = scores / temperature

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Residual Attention (배치 크기 체크 추가)
        if self.prev_attn is not None:
            # 배치 크기가 같을 때만 residual 적용
            if self.prev_attn.size(0) == batch_size:
                alpha = torch.sigmoid(self.attn_alpha)
                attn_weights = alpha * attn_weights + (1 - alpha) * self.prev_attn

        # 현재 attention 저장 (학습 모드일 때만)
        if self.training:
            self.prev_attn = attn_weights.detach()

        self.attn = attn_weights

        # Output 계산
        output = torch.matmul(attn_weights, V)

        # Reshape
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)

        # Final transformation
        output = self.dense(output)
        output = self.dropout(output)

        return output

class EMAAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1, ema_decay=0.9):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        self.ema_decay = ema_decay

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # EMA state
        self.register_buffer('ema_state', None)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)

        Q = self.wq(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.wk(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.wv(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask.bool(), -1e9)

        attn_weights = F.softmax(scores, dim=-1)

        # EMA 스무딩 (GRU의 hidden state 업데이트와 유사)
        if self.training:
            if self.ema_state is None or self.ema_state.size(0) != batch_size:
                self.ema_state = attn_weights.detach()
            else:
                self.ema_state = self.ema_decay * self.ema_state + (1 - self.ema_decay) * attn_weights.detach()

            # 스무딩된 attention 사용
            attn_weights = 0.7 * attn_weights + 0.3 * self.ema_state

        self.attn = attn_weights
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.dropout(self.dense(output))

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
        if window_size is not None:
            self.inverted_attention = LocalInvertedMultiHeadAttention(
                num_heads, seq_len, window_size, dropout
            )
        else:
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
    def __init__(
        self, num_layers, input_dim, max_len, d_model, num_heads, dff,
        window_size=None, dropout=0.1
    ):
        super().__init__()

        # 입력을 d_model 차원으로 변환하지 않고 그대로 사용
        # input_dim을 그대로 사용하여 변수 간 관계를 직접 학습
        self.input_dim = input_dim
        self.max_len = max_len

        # iTransformer 레이어들 (input_dim 변수에 대해 작동)
        self.enc_layers = nn.ModuleList([
            iTransformerEncoderLayer(max_len, input_dim, num_heads, dff, window_size, dropout)
            for _ in range(num_layers)
        ])

        # 기존 DeepSC와 호환성을 위해 d_model 차원으로 투영
        self.output_projection = nn.Linear(input_dim, d_model)

    def forward(self, x, src_mask):
        for enc_layer in self.enc_layers:
            x = enc_layer(x, src_mask)

        x = self.output_projection(x)
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
            self.encoder = iTransformerEncoder(
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
            # self.time_compressor = TimeSeriesCompressor(self.compressed_len)
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

        self.decoder = TimeSeriesDecoder(
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
        decompressed = self.time_decompressor(channel_decoded)

        # 6단계 : 의미 디코더
        # (batch_size, compressed_len->max_len, d_model)
        # decoded = self.decoder(channel_decoded, use_mask=True)

        # 7단계: 출력 투영
        # (batch_size, max_len, d_model->input_dim => 원래 피쳐 차원으로 복원)
        output = self.output_projection(decompressed)

        return output
