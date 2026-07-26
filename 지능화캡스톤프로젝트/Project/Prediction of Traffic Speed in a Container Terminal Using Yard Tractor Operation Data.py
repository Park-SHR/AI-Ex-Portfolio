import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------
# Step 1: Mock Data Generation & Under-sampling
# ---------------------------------------------------------
def generate_mock_data(num_samples=10000):
    """
    논문 명세에 따른 38개 Feature 기반 가상 데이터 생성
    - Feature 1~3: 링크 유형 (One-hot)
    - Feature 4~13: 최근 10개 구간 평균 속도
    - Feature 14: 타겟 구간
    - Feature 15~38: 확정/추정 차량 밀도 (간섭, 상류, 하류 링크 포함)
    """
    X = np.random.rand(num_samples, 38).astype(np.float32)
    # 실제 환경을 모사하여 특정 Feature 조합이 정체(속도 0~0.1)를 유발하도록 Y값 임의 설계
    noise = np.random.normal(0, 0.1, num_samples)
    y = np.clip(np.mean(X[:, 14:], axis=1) + noise, 0.0, 0.999).astype(np.float32)
    return X, y

def apply_undersampling(X, y, num_bins=10):
    """
    속도 구간(1~10)별로 데이터 수를 균등하게 맞추는 언더샘플링 전처리
    """
    bins = np.linspace(0, 1, num_bins + 1)
    indices = np.digitize(y, bins) - 1
    
    # 가장 적은 데이터를 가진 구간의 샘플 수 파악
    min_count = min([np.sum(indices == i) for i in range(num_bins)])
    
    balanced_X, balanced_y = [], []
    for i in range(num_bins):
        bin_idx = np.where(indices == i)[0]
        if len(bin_idx) > 0:
            selected_idx = np.random.choice(bin_idx, min_count, replace=False)
            balanced_X.append(X[selected_idx])
            balanced_y.append(y[selected_idx])
            
    return np.vstack(balanced_X), np.concatenate(balanced_y)

# ---------------------------------------------------------
# Step 2: MLP 예측 모델 정의
# ---------------------------------------------------------
class TrafficSpeedPredictorMLP(nn.Module):
    """
    3개의 은닉층과 각 64개의 ReLU 노드로 구성된 다층 인공신경망
    """
    def __init__(self, input_dim=38):
        super(TrafficSpeedPredictorMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # 정규화된 속도 [0, 1] 회귀 출력
        )

    def forward(self, x):
        return self.network(x)

# ---------------------------------------------------------
# Step 3: 모델 훈련 파이프라인
# ---------------------------------------------------------
# 데이터 준비
X_raw, y_raw = generate_mock_data(50000)
X_balanced, y_balanced = apply_undersampling(X_raw, y_raw)

# Tensor 변환
X_tensor = torch.tensor(X_balanced)
y_tensor = torch.tensor(y_balanced).view(-1, 1)

dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 모델, 손실 함수, 최적화 기법 초기화
model = TrafficSpeedPredictorMLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting Model Training...")
epochs = 10
for epoch in range(epochs):
    epoch_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch+1) % 2 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_loader):.4f}")

# ---------------------------------------------------------
# Step 4: 랭킹 성능(Ranking Performance) 평가
# ---------------------------------------------------------
def evaluate_ranking_performance(model, X_test, y_test, num_pairs=10000):
    """
    무작위 추출된 데이터 쌍을 비교하여 더 빠른 차로를 정확히 판별하는지 평가
    """
    model.eval()
    correct_rankings = 0
    
    with torch.no_grad():
        predictions = model(torch.tensor(X_test)).numpy().flatten()
        
    for _ in range(num_pairs):
        idx1, idx2 = np.random.choice(len(y_test), 2, replace=False)
        
        actual_diff = y_test[idx1] - y_test[idx2]
        pred_diff = predictions[idx1] - predictions[idx2]
        
        # 실제 속도 차이와 예측 속도 차이의 부호가 같으면 랭킹 맞음
        if actual_diff * pred_diff > 0:
            correct_rankings += 1
            
    accuracy = (correct_rankings / num_pairs) * 100
    return accuracy

# 테스트 데이터 생성 및 평가
X_test_raw, y_test_raw = generate_mock_data(5000)
ranking_acc = evaluate_ranking_performance(model, X_test_raw, y_test_raw)
print(f"\\nEstimated Ranking Performance: {ranking_acc:.1f}%")