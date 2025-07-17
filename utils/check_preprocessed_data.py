import torch
import matplotlib.pyplot as plt

# 데이터 로드
train_data = torch.load('model/preprocessed_data/train_data.pt')
test_data = torch.load('model/preprocessed_data/test_data.pt')

# TensorDataset 내부 텐서 추출
train_tensor = train_data.tensors[0]
test_tensor = test_data.tensors[0]

print("Train data shape:", train_tensor.shape)
print("Test data shape:", test_tensor.shape)

# 앞부분 일부 출력
print("\nTrain data sample (first window):")
print(train_tensor[0])

print("\nTest data sample (first window):")
print(test_tensor[0])

# 통계 정보
print("\nTrain data statistics:")
print("Mean:", train_tensor.mean().item())
print("Std:", train_tensor.std().item())
print("Min:", train_tensor.min().item())
print("Max:", train_tensor.max().item())

# 시각화: 첫 window의 feature별 시계열
plt.figure(figsize=(12, 6))
for i in range(train_tensor.shape[2]):
    plt.plot(train_tensor[0, :, i], label=f'Feature {i}')
plt.title('Train Data - First Window (Feature-wise)')
plt.xlabel('Time Step')
plt.ylabel('Scaled Value')
plt.legend()
plt.grid(True)
plt.show() 