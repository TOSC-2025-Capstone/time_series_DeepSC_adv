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

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model)) #math.log(math.exp(1)) = 1
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #[1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x
  
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
        
        #self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k)
        query = query.transpose(1, 2)
        
        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k)
        key = key.transpose(1, 2)
        
        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k)
        value = value.transpose(1, 2)
        
        #        query, key, value = \
        #            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = self.attention(query, key, value, mask=mask)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.num_heads * self.d_k)
             
        x = self.dense(x)
        x = self.dropout(x)
        
        return x
    
    def attention(self, query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        #print(mask.shape)
        if mask is not None:
            # 根据mask，指定位置填充 -1e9  
            scores += (mask * -1e9)
            # attention weights
        p_attn = F.softmax(scores, dim = -1)
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

#class LayerNorm(nn.Module):
#    "Construct a layernorm module (See citation for details)."
#    # features = d_model
#    def __init__(self, features, eps=1e-6):
#        super(LayerNorm, self).__init__()
#        self.a_2 = nn.Parameter(torch.ones(features))
#        self.b_2 = nn.Parameter(torch.zeros(features))
#        self.eps = eps
#
#    def forward(self, x):
#        mean = x.mean(-1, keepdim=True)
#        std = x.std(-1, keepdim=True)
#        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
    
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, d_model, num_heads, dff, dropout = 0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadedAttention(num_heads, d_model, dropout = 0.1)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout = 0.1)
        
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        attn_output = self.mha(x, x, x, mask)
        x = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + ffn_output)
        
        return x
    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, d_model, num_heads, dff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadedAttention(num_heads, d_model, dropout = 0.1)
        self.src_mha = MultiHeadedAttention(num_heads, d_model, dropout = 0.1)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout = 0.1)
        
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
        
        #self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        "Follow Figure 1 (right) for connections."
        #m = memory
        
        attn_output = self.self_mha(x, x, x, look_ahead_mask)
        x = self.layernorm1(x + attn_output)
        
        src_output = self.src_mha(x, memory, memory, trg_padding_mask) # q, k, v
        x = self.layernorm2(x + src_output)
        
        fnn_output = self.ffn(x)
        x = self.layernorm3(x + fnn_output)
        return x

    
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, num_layers, input_dim, max_len, 
                 d_model, num_heads, dff, dropout = 0.1):
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        # nn.Embedding 대신 nn.Linear 사용 (시계열 데이터용)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        # input_proj + pos_encoding + cycle_diff
        # self.pos_encoding = LinearProjectionWithLocalTimeAndGlobalDiffEmbedding(d_model)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, dropout) 
                                            for _ in range(num_layers)])
        
    def forward(self, x, src_mask):
        
        "Pass the input (and mask) through each layer in turn."
        # x: (batch_size, seq_len, input_dim) - 시계열 데이터
        
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        x = self.pos_encoding(x)
        
        for enc_layer in self.enc_layers:
            x = enc_layer(x, src_mask)
        
        return x
        


class Decoder(nn.Module):
    def __init__(self, num_layers, trg_vocab_size, max_len, 
                 d_model, num_heads, dff, dropout = 0.1):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, dropout) 
                                            for _ in range(num_layers)])
    
    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for dec_layer in self.dec_layers:
            x = dec_layer(x, memory, look_ahead_mask, trg_padding_mask)
            
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
        x = self.pool(x)         # (batch, d_model, target_len)
        x = x.permute(0, 2, 1)  # (batch, target_len, d_model)
        return x

class DeepSC(nn.Module):
    def __init__(self, num_layers=2, input_dim=6, max_len=128, d_model=128, num_heads=4, dff=512, dropout=0.1, compressed_len=None, params=None, **kwargs):
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
        
        self.encoder = Encoder(self.num_layers, self.input_dim, self.max_len, 
                               self.d_model, self.num_heads, self.dff, self.dropout)

        # 시계열 길이 압축 모듈 (선택)
        if self.compressed_len is not None:
            self.time_compressor = TimeSeriesCompressor(self.compressed_len)
        else:
            self.time_compressor = None
        
        self.channel_encoder = nn.Sequential(nn.Linear(self.d_model, 256), 
                                             nn.ReLU(inplace=True),
                                             nn.Linear(256, self.d_comp))

        self.channels = Channels()

        self.channel_decoder = ChannelDecoder(self.d_comp, self.d_model, 512) # feature, output_dim, max_dim
        
        # 자연어 디코더 대신 시계열 출력 레이어 사용
        self.output_projection = nn.Linear(self.d_model, self.input_dim)

        # 업샘플링 레이어 추가 (compressed_len → max_len)
        self.upsample = nn.Upsample(size=self.max_len, mode='linear', align_corners=False)
        
    def forward(self, x, src_mask=None):
        # x: (batch_size, seq_len, input_dim) - 시계열 데이터
        # pdb.set_trace()
        
        # 1단계: 인코더
        encoded = self.encoder(x, src_mask) # (batch, max_len, d_model)
        
        # 2단계: sequence compress (downsampling)
        compressed = self.time_compressor(encoded)  # (batch, target_len, d_model)

        # 3단계: 채널 인코더 (압축)
        channel_encoded = self.channel_encoder(compressed)

        # 4단계 : 채널 상태 적용
        channel_syms = channel_encoded
        # channel_syms = self.channels.AWGN(channel_encoded, 0.1)
        # channel_syms = self.channels.Rayleigh(channel_encoded, 0.1)
        # channel_syms = self.channels.Rician(channel_encoded, 0.1)
        
        # 5단계: 채널 디코더 (복원)
        channel_decoded = self.channel_decoder(channel_syms)
        # channel_decoded = self.channel_decoder(channel_encoded)
        
        # 6단계: 출력 투영 (원래 차원으로 복원)
        output = self.output_projection(channel_decoded)
        
        # 7단계: upsampling (batch, compressed_len, input_dim) → (batch, max_len, input_dim)
        output = output.permute(0, 2, 1)  # (batch, input_dim, compressed_len), (PyTorch의 Upsample은 (batch, channels, length) 형태를 기대하므로 형태 수정
        output = self.upsample(output)     # (batch, input_dim, max_len)
        output = output.permute(0, 2, 1)  # (batch, max_len, input_dim)

        return output

