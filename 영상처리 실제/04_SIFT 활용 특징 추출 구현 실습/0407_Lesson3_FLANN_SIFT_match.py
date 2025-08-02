import cv2 as cv
import numpy as np
import time

# 1. 이미지 불러오기 및 모델 영역 크롭
img1 = cv.imread('mot_color70.jpg')[190:350, 440:560]  # 버스 부분을 잘라 모델 영상으로 사용
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)           # 모델 이미지 그레이 변환

# 2. 전체 장면 이미지 불러오기
img2 = cv.imread('mot_color83.jpg')                    # 전체 장면 이미지
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)           # 장면 이미지 그레이 변환

# 3. SIFT 특징점 추출
sift = cv.SIFT_create()                                # SIFT 객체 생성
kp1, des1 = sift.detectAndCompute(gray1, None)         # 모델 이미지 특징점 추출
kp2, des2 = sift.detectAndCompute(gray2, None)         # 장면 이미지 특징점 추출

# 4. 특징점 개수 출력
print('특징점 개수:', len(kp1), len(kp2))

# 5. FLANN 매칭 시작 (시간 측정 포함)
start = time.time()
flann_matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)  # FLANN 매처 객체 생성
knn_match = flann_matcher.knnMatch(des1, des2, 2)  # KNN 매칭 (각 특징점당 가장 가까운 2개 매칭)

# 6. 좋은 매칭 필터링 (거리 비율 기준)
T = 0.7
good_match = []
for nearest1, nearest2 in knn_match:
    if (nearest1.distance / nearest2.distance) < T:    # 최근접 이웃 거리 비율 조건
        good_match.append(nearest1)

# 7. 매칭 시간 출력
print('매칭에 걸린 시간:', time.time() - start)

# 8. 매칭 시각화를 위한 이미지 버퍼 생성
img_match = np.empty(
    (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3),
    dtype=np.uint8
)

# 9. 매칭 결과 그리기
cv.drawMatches(
    img1, kp1, img2, kp2, good_match, img_match,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 10. 결과 화면 출력
cv.imshow('Good Matches', img_match)
k = cv.waitKey()
cv.destroyAllWindows()
