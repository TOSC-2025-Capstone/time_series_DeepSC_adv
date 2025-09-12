import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb
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

class SymmetricChannelProcessor(nn.Module):
    """대칭적 채널 인코더/디코더"""
    def __init__(self, d_model, d_comp, is_encoder=True):
        super().__init__()
        self.is_encoder = is_encoder

        if is_encoder:
            # 인코더: d_model → d_comp
            self.layers = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, d_comp),
                nn.Tanh()  # 출력 범위 제한
            )
        else:
            # 디코더: d_comp → d_model (인코더의 역순)
            self.layers = nn.Sequential(
                nn.Linear(d_comp, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, d_model)
            )

            # Residual connection을 위한 projection
            self.residual_proj = nn.Linear(d_comp, d_model)

    def forward(self, x):
        output = self.layers(x)

        if not self.is_encoder:
            # 디코더에서 residual connection 적용
            residual = self.residual_proj(x)
            output = output + residual

        return output

class MultiScaleLoss(nn.Module):
    """다중 스케일 손실 함수"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        # 시간 도메인 손실
        time_loss = self.mse(pred, target)

        # 주파수 도메인 손실 (FFT)
        pred_fft = torch.fft.fft(pred.transpose(1, 2), dim=-1)
        target_fft = torch.fft.fft(target.transpose(1, 2), dim=-1)
        freq_loss = self.mse(torch.abs(pred_fft), torch.abs(target_fft))

        # 기울기 손실 (시계열의 변화율 보존)
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        grad_loss = self.l1(pred_diff, target_diff)

        # 통계적 특성 손실
        pred_mean = pred.mean(dim=1)
        target_mean = target.mean(dim=1)
        stat_loss = self.mse(pred_mean, target_mean)

        return {
            'total': time_loss + 0.1 * freq_loss + 0.1 * grad_loss + 0.05 * stat_loss,
            'time': time_loss,
            'freq': freq_loss,
            'grad': grad_loss,
            'stat': stat_loss
        }

class ImprovedDeepSC(nn.Module):
    """시계열 최적화된 DeepSC"""
    def __init__(
        self,
        num_layers=2,
        input_dim=6,
        max_len=128,
        d_model=128,
        num_heads=4,
        dff=512,
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
        self.d_model = p.get("d_model", d_model)
        self.num_heads = p.get("num_heads", num_heads)
        self.dff = p.get("dff", dff)
        self.dropout = p.get("dropout", dropout)
        self.compressed_len = p.get("compressed_len", compressed_len)
        self.d_comp = p.get("d_comp", d_comp)

        # 입력 투영
        self.input_projection = nn.Linear(self.input_dim, self.d_model)

        # 시계열 특화 위치 인코딩
        self.pos_encoding = TimeSeriesPositionalEncoding(self.d_model, dropout, self.max_len)

        # # Transformer 인코더 레이어들
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads,
                dim_feedforward=self.dff,
                dropout=dropout,
                batch_first=True
            ) for _ in range(self.num_layers)
        ])

        # 학습 가능한 시간 압축기
        self.time_compressor = LearnableTimeCompressor(self.d_model, self.compressed_len)

        # 대칭적 채널 처리
        self.channel_encoder = SymmetricChannelProcessor(self.d_model, self.d_comp, is_encoder=True)
        self.channel_decoder = SymmetricChannelProcessor(self.d_model, self.d_comp, is_encoder=False)

        # 채널 (노이즈 시뮬레이션)
        self.channels = Channels()

        # 학습 가능한 시간 복원기
        self.time_decompressor = LearnableTimeDecompressor(self.d_model, self.max_len)

        # 출력 투영
        self.output_projection = nn.Linear(self.d_model, self.input_dim)

        # 다중 스케일 손실
        self.criterion = MultiScaleLoss()

    def forward(self, x, src_mask=None, return_intermediate=False):
        intermediate = {}

        # 1. 입력 투영 및 위치 인코딩
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x)

        if return_intermediate:
            intermediate['input_projected'] = x.clone()

        # 2. Transformer 인코딩
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x, src_key_padding_mask=src_mask)

        if return_intermediate:
            intermediate['encoded'] = x.clone()

        # 3. 시간 압축 (학습 가능)
        compressed = self.time_compressor(x)

        if return_intermediate:
            intermediate['time_compressed'] = compressed.clone()

        # 4. 채널 인코딩 (대칭적)
        channel_encoded = self.channel_encoder(compressed)

        if return_intermediate:
            intermediate['channel_encoded'] = channel_encoded.clone()

        # 5. 채널 노이즈 적용
        channel_syms = self.channels.Rayleigh(channel_encoded, 0.35)

        if return_intermediate:
            intermediate['channel_noisy'] = channel_syms.clone()

        # 6. 채널 디코딩 (대칭적)
        channel_decoded = self.channel_decoder(channel_syms)

        if return_intermediate:
            intermediate['channel_decoded'] = channel_decoded.clone()

        # 7. 시간 복원 (학습 가능)
        time_restored = self.time_decompressor(channel_decoded)

        if return_intermediate:
            intermediate['time_restored'] = time_restored.clone()

        # 8. 출력 투영
        output = self.output_projection(time_restored)

        if return_intermediate:
            return output, intermediate

        return output

    def compute_loss(self, pred, target):
        """다중 스케일 손실 계산"""
        return self.criterion(pred, target)

    def get_compression_ratio(self):
        """압축률 계산"""
        original_size = self.input_dim * self.max_len
        compressed_size = self.compressed_len * self.d_comp
        return compressed_size / original_size
