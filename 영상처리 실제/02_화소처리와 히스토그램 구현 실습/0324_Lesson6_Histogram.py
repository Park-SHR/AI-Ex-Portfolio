import cv2 as cv                   # OpenCV 라이브러리 import
import matplotlib.pyplot as plt   # matplotlib (그래프 그리기) 라이브러리 import

# 이미지 불러오기
img = cv.imread('soccer_gray.jpg')  # 이미지를 BGR 컬러로 불러옴

# 그레이스케일로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # BGR 이미지를 그레이스케일로 변환

# 원본 그레이스케일 이미지 출력
plt.imshow(gray, cmap='gray')     # 회색조로 이미지 시각화
plt.xticks([]), plt.yticks([])    # x, y 축 눈금 제거
plt.show()                        # 이미지 보여주기

# 원본 이미지의 히스토그램 계산 (0번 채널 = 그레이스케일)
h = cv.calcHist([gray], [0], None, [256], [0, 256])  # 픽셀값 분포를 계산 (0~255 범위, 256개 구간)

# 원본 이미지의 히스토그램 그래프 출력
plt.plot(h, color='r', linewidth=1)  # 빨간색 실선으로 히스토그램 그림
plt.show()

# 히스토그램 평활화 적용 (equalizeHist 함수 사용)
equal = cv.equalizeHist(gray)  # 대비 향상 (픽셀 분포를 넓게 펴줌)

# 평활화된 이미지 시각화
plt.imshow(equal, cmap='gray')     # 결과 이미지 출력
plt.xticks([]), plt.yticks([])     # 축 눈금 제거
plt.show()

# 평활화된 이미지의 히스토그램 계산
h = cv.calcHist([equal], [0], None, [256], [0, 256])  # 평활화 후 픽셀 분포 확인

# 평활화된 이미지의 히스토그램 그래프 출력
plt.plot(h, color='r', linewidth=1)
plt.show()
