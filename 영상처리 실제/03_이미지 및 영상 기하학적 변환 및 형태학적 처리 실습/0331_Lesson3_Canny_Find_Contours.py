import cv2 as cv       # OpenCV 불러오기
import numpy as np     # NumPy 불러오기 (contour 처리에 사용)

# 이미지 파일 읽기
img = cv.imread('soccer.jpg')  # 컬러 이미지 읽기

# 그레이스케일로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Canny 엣지 검출 적용 (T_low=100, T_high=200)
canny = cv.Canny(gray, 100, 200)

# 윤곽선 검출 (cv.RETR_LIST: 모든 윤곽선 추출, 계층 구조 X / cv.CHAIN_APPROX_NONE: 모든 점 저장)
contour, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

# 필터링된 윤곽선 저장 리스트 초기화
lcontour = []

# 각 윤곽선 순회하며 점 개수 기준으로 필터링
for i in range(len(contour)):
    if contour[i].shape[0] > 100:         # 점 개수가 100개 넘는 윤곽선만 선택
        lcontour.append(contour[i])       # 리스트에 추가

# 선택된 윤곽선들을 이미지 위에 그림 (녹색, 두께 3)
cv.drawContours(img, lcontour, -1, (0, 255, 0), 3)

# 결과 이미지 출력 (윤곽선 포함)
cv.imshow('Original with contours', img)

# Canny 엣지 맵 출력
cv.imshow('Canny', canny)

# 키 입력 대기
cv.waitKey()

# 모든 창 닫기
cv.destroyAllWindows()
