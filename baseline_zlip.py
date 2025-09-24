import numpy as np
import pywt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import pdb

# =============================================================================
# 1️⃣ 예시 데이터: 배치 32x512x5
# =============================================================================
batch_size, seq_len, n_features = 32, 512, 5
data = np.random.rand(batch_size, seq_len, n_features).astype(np.float32)

# 목표 압축률
target_elements = 32 * 128 * 3  # 32x128x3
original_elements = data.size
compression_ratio = target_elements / original_elements
print(f"목표 압축률: {compression_ratio:.3f}")

# =============================================================================
# 2️⃣ Delta 기반 (차원 줄임으로 압축률 맞춤)
# =============================================================================
def delta_tensor_compress_reconstruct(batch, compressed_len):
    reconstructed = np.zeros_like(batch)
    for i in range(batch.shape[0]):
        # Delta 인코딩
        # pdb.set_trace()
        delta = np.diff(batch[i], axis=0, prepend=batch[i:i+1, :].squeeze())
        # 압축률 맞추기 위해 시퀀스 길이 샘플링
        idx = np.linspace(0, delta.shape[0]-1, compressed_len, dtype=int)
        compressed_delta = delta[idx]
        # 단순 linear interpolation으로 복원
        recon = np.zeros_like(batch[i])
        for f in range(batch.shape[1]):
            pdb.set_trace()
            recon[f, :] = np.interp(np.arange(batch.shape[1]), idx, compressed_delta[:, f])
        reconstructed[i] = recon
    return reconstructed

compressed_len = int(seq_len * (target_elements / original_elements * (seq_len/n_features)))
recon_delta = delta_tensor_compress_reconstruct(data, compressed_len)
mse_delta = mean_squared_error(data.flatten(), recon_delta.flatten())
print(f"Delta MSE: {mse_delta:.6f}")

# =============================================================================
# 3️⃣ PCA 압축
# =============================================================================
latent_dim = int(n_features * seq_len * compression_ratio / batch_size)  # 간략 추정
recon_pca = np.zeros_like(data)
for i in range(batch_size):
    pca = PCA(n_components=latent_dim)
    flat = data[i]
    compressed = pca.fit_transform(flat)
    recon_pca[i] = pca.inverse_transform(compressed)
mse_pca = mean_squared_error(data.flatten(), recon_pca.flatten())
print(f"PCA MSE: {mse_pca:.6f}")

# =============================================================================
# 4️⃣ DWT 압축
# =============================================================================
def dwt_compress_reconstruct(batch, wave='db1', target_elements=None):
    reconstructed = np.zeros_like(batch)
    for i in range(batch.shape[0]):
        for j in range(batch.shape[2]):
            coeffs = pywt.wavedec(batch[i,:,j], wave)
            # 목표 요소 개수에 맞춰 계수 선택
            all_coeffs = np.concatenate(coeffs)
            k = int(len(all_coeffs) * target_elements / batch[i,:,j].size)
            idx = np.argsort(np.abs(all_coeffs))[-k:]  # 가장 큰 k개 계수
            new_coeffs = np.zeros_like(all_coeffs)
            new_coeffs[idx] = all_coeffs[idx]
            # 계수 다시 분할
            split_sizes = [len(c) for c in coeffs]
            coeffs_trimmed = np.split(new_coeffs, np.cumsum(split_sizes)[:-1])
            reconstructed[i,:,j] = pywt.waverec(coeffs_trimmed, wave)[:batch.shape[1]]
    return reconstructed

recon_dwt = dwt_compress_reconstruct(data, target_elements=target_elements)
mse_dwt = mean_squared_error(data.flatten(), recon_dwt.flatten())
print(f"DWT MSE: {mse_dwt:.6f}")

# =============================================================================
# 5️⃣ 결과 요약
# =============================================================================
print("\n복원 성능 비교 (MSE, 텐서 원소 기준 동일 압축률)")
print(f"Delta: {mse_delta:.6f}")
print(f"PCA  : {mse_pca:.6f}")
print(f"DWT  : {mse_dwt:.6f}")
