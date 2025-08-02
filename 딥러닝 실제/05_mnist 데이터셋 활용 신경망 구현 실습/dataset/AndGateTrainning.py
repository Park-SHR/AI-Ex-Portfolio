import numpy as np
import matplotlib.pyplot as plt

# Sigmoid 함수 및 그 미분 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

# AND 게이트 입력 및 정답 레이블
X = np.array([[0,0],[0,1],[1,0],[1,1]])  # 입력값
T = np.array([[0],[0],[0],[1]])          # AND의 타겟 출력

# 가중치, 편향 초기화
np.random.seed(42)
W = np.random.randn(2,1)  # 2 입력 -> 1 출력
b = np.random.randn(1)

# 하이퍼파라미터
lr = 0.1       # 학습률
epochs = 2000  # 반복 횟수

# 손실 값과 가중치 변화 저장
losses = []
w_history = []

# 학습 과정
for epoch in range(epochs):
    # 순전파 계산
    z = np.dot(X, W) + b
    y = sigmoid(z)

    # 손실 함수: 평균 제곱 오차(MSE)
    loss = np.mean(0.5 * (y - T)**2)
    losses.append(loss)
    w_history.append(W.copy())

    # 역전파 계산
    dz = (y - T) * sigmoid_grad(z)
    dW = np.dot(X.T, dz) / X.shape[0]
    db = np.sum(dz) / X.shape[0]

    # 가중치 및 편향 업데이트
    W -= lr * dW
    b -= lr * db

# 시각화를 위한 가중치 분리
w0 = [w[0][0] for w in w_history]
w1 = [w[1][0] for w in w_history]

# 그래프 그리기
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# 손실 함수 변화
axs[0].plot(losses)
axs[0].set_title('Loss Function (MSE) over Epochs')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')

# 가중치 변화
axs[1].plot(w0, label='w0')
axs[1].plot(w1, label='w1')
axs[1].set_title('Weight Changes over Epochs')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Weight Value')
axs[1].legend()

plt.tight_layout()
plt.show()
