import cv2 as cv 
# OpenCV 라이브러리 import (cv2를 cv로 줄여 사용)

img = cv.imread('mot_color70.jpg')
# 이미지 파일을 BGR 컬러 형식으로 읽음 (파일명은 사용 환경에 맞게 수정 필요)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 컬러 이미지를 그레이스케일로 변환 (SIFT는 그레이 이미지에서 특징 추출함)

sift = cv.SIFT_create()
# SIFT (Scale-Invariant Feature Transform) 객체 생성

kp, des = sift.detectAndCompute(gray, None)
# SIFT를 이용해 특징점(keypoints)과 기술자(descriptors) 계산

gray = cv.drawKeypoints(
    gray, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
# 계산된 특징점을 이미지 위에 시각화함
# DRAW_RICH_KEYPOINTS 옵션: 위치뿐 아니라 크기와 방향도 함께 그림

cv.imshow('sift', gray)
# 결과 이미지를 윈도우에 표시

k = cv.waitKey()
# 키 입력 대기 (아무 키나 누를 때까지 창 유지)

cv.destroyAllWindows()
# 모든 OpenCV 윈도우 닫기
