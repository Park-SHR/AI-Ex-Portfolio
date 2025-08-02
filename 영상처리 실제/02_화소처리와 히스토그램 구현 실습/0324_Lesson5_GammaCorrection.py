import cv2 as cv                   # OpenCV 라이브러리 import
import numpy as np                # NumPy 라이브러리 import

# 이미지 불러오기
img = cv.imread('soccer.jpg')     # 'soccer.jpg' 이미지를 컬러로 불러옴

# 이미지 크기 줄이기 (원래 크기의 25%)
img = cv.resize(img, dsize=(0, 0), fx=0.25, fy=0.25)

# 감마 보정 함수 정의
def gamma(f, gamma=1.0):
    f1 = f / 255.0                # 이미지 픽셀 값을 0~1 사이로 정규화
    return np.uint8(255 * (f1 ** gamma))  # 감마 보정 후 다시 0~255로 변환하여 반환

# 감마 값 별로 보정한 이미지들을 좌우로 붙이기
gc = np.hstack((
    gamma(img, 0.5),              # 감마 < 1 → 이미지 밝아짐
    gamma(img, 0.75),             # 감마 0.75 → 약간 밝아짐
    gamma(img, 1.0),              # 감마 1 → 원본과 동일
    gamma(img, 2.0),              # 감마 > 1 → 이미지 어두워짐
    gamma(img, 3.0)               # 감마 3 → 더 어두워짐
))

# 감마 보정 결과 이미지 띄우기
cv.imshow('gamma', gc)

# 키 입력 대기 후 모든 창 닫기
cv.waitKey()
cv.destroyAllWindows()
