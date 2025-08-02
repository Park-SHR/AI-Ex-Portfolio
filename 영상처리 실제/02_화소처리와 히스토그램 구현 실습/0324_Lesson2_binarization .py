import cv2 as cv                          # OpenCV 라이브러리 import
import matplotlib.pyplot as plt           # 그래프를 그리기 위한 matplotlib import

# 이미지 불러오기 (컬러 이미지: BGR 형식)
img = cv.imread('soccer.jpg')

# R 채널(2번 채널)의 히스토그램 계산
h = cv.calcHist(
    [img],        # 입력 이미지 (리스트로)
    [2],          # 채널 인덱스 2 → R 채널 (B:0, G:1, R:2)
    None,         # 마스크 없음 (전체 이미지 사용)
    [256],        # 히스토그램 빈 개수 (0~255 → 총 256개)
    [0, 256]      # 픽셀 값의 범위
)

# R 채널 히스토그램 그래프 그리기
plt.plot(h, color='r', linewidth=1)       # 빨간색 실선으로 출력

# 그래프 보여주기
plt.show()
