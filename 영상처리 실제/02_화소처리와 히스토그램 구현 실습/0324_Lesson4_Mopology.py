import cv2 as cv                        # OpenCV 라이브러리 import
import numpy as np                     # NumPy 라이브러리 import
import matplotlib.pyplot as plt        # 그래프 시각화를 위한 matplotlib import

# 이미지 불러오기 (원본 채널 유지)
img = cv.imread('soccer.jpg', cv.IMREAD_UNCHANGED)  

# 이미지의 R 채널에 대해 Otsu 이진화 수행 (적절한 임계값 자동 결정)
t, bin_img = cv.threshold(img[:,:,2], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# 이진화된 결과 이미지 시각화
plt.imshow(bin_img, cmap='gray')       # 그레이스케일 컬러맵으로 출력
plt.xticks([]), plt.yticks([])         # x, y 축 눈금 제거
plt.show()

# 이미지의 아래쪽 절반, 왼쪽 절반 잘라내기 (ROI 추출)
b = bin_img[bin_img.shape[0]//2 : bin_img.shape[0],     # 세로 절반
            0 : bin_img.shape[0]//2 + 1]                # 가로 절반 (주의: shape[1]이 아닌 shape[0] 사용함)

# 잘라낸 영역 시각화
plt.imshow(b, cmap='gray')
plt.xticks([]), plt.yticks([])

# 구조화 요소(Structuring Element) 정의 (다이아몬드 모양)
se = np.uint8([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0]
])

# 팽창(Dilation) 적용 - 흰색 영역 확장
b_dilation = cv.dilate(b, se, iterations=1)

# 팽창 결과 시각화
plt.imshow(b_dilation, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()

# 침식(Erosion) 적용 - 흰색 영역 축소
b_erosion = cv.erode(b, se, iterations=1)

# 침식 결과 시각화
plt.imshow(b_erosion, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()

# 닫힘(Closing): 팽창 후 침식 → 작은 구멍 메우기
b_closing = cv.erode(cv.dilate(b, se, iterations=1), se, iterations=1)

# 닫힘 결과 시각화
plt.imshow(b_closing, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()
