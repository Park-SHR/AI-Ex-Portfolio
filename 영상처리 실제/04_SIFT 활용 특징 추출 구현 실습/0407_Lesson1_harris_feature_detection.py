import cv2 as cv
import numpy as np

# --- [1] 영상 입력 및 필터 정의 (PDF 19쪽) --- #
img = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,1,0,0,0,0,0,0,0],
    [0,0,0,1,1,1,0,0,0,0,0,0,0],
    [0,0,0,1,1,1,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0]
], dtype=np.float32)

# 수평, 수직 미분 필터 정의
ux = np.array([[-1, 0, 1]])
uy = np.array([-1, 0, 1]).reshape((3, 1))

# 가우시안 커널 정의
k = cv.getGaussianKernel(3, 1)
g = np.outer(k, k.transpose())

# --- [2] 해리스 응답 계산 (PDF 20쪽) --- #
# 미분 계산
dy = cv.filter2D(img, cv.CV_32F, uy)  # y 방향 미분
dx = cv.filter2D(img, cv.CV_32F, ux)  # x 방향 미분

# 기울기 제곱 및 곱
dyy = dy * dy
dxx = dx * dx
dyx = dy * dx

# 2차 모멘트 가우시안 스무딩
gdyy = cv.filter2D(dyy, cv.CV_32F, g)
gdxx = cv.filter2D(dxx, cv.CV_32F, g)
gdyx = cv.filter2D(dyx, cv.CV_32F, g)

# 해리스 응답 계산
C = gdyy * gdxx - gdyx * gdyx - 0.04 * (gdyy + gdxx) * (gdyy + gdxx)

# --- [3] 비최대 억제 및 특징점 표시 (PDF 21쪽 상단) --- #
for j in range(1, C.shape[0]-1):  # 경계 제외
    for i in range(1, C.shape[1]-1):
        if C[j, i] > 0.1 and np.sum(C[j, i] > C[j-1:j+2, i-1:i+2]) == 8:
            img[j, i] = 9  # 특징점에 9 표시

# --- [4] 결과 출력 (PDF 21쪽 하단) --- #
np.set_printoptions(precision=2)

# 중간 결과 출력 (디버깅용)
print("① dy\n", dy)
print("② dx\n", dx)
print("③ dyy\n", dyy)
print("④ dxx\n", dxx)
print("⑤ dyx\n", dyx)
print("⑥ gdyy\n", gdyy)
print("⑦ gdxx\n", gdxx)
print("⑧ gdyx\n", gdyx)
print("⑨ C (Harris Response)\n", C)
print("⑩ img (features marked with 9)\n", img)

# --- [5] 확대 시각화 (PDF 21쪽 하단) --- #
popping = np.zeros((160, 160), np.uint8)  # 16배 확대용 흑백 버퍼
for j in range(160):
    for i in range(160):
        # C 맵을 확대하고 대비 향상
        popping[j, i] = np.uint8((C[j // 16, i // 16] + 0.06) * 700)

# 이미지 표시
cv.imshow('Image Display2', popping)
cv.waitKey()
cv.destroyAllWindows()