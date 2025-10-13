import torch
import torch.nn as nn
import torch.nn.functional as F

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
