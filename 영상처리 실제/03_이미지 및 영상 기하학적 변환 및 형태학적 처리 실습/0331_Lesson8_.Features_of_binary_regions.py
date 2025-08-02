import skimage                         # scikit-image: 예제 이미지 사용
import numpy as np                     # 수치 계산용
import cv2 as cv                       # OpenCV

# scikit-image에서 말(horse) 이미지 불러오기 (binary 형태)
orig = skimage.data.horse()

# 0/1 값을 0~255 범위로 반전 처리 (255 - 값) → 말이 흰색이 되게 만듦
img = 255 - np.uint8(orig) * 255

# 초기 이진 영상 출력
cv.imshow('Horse', img)

# 윤곽선 찾기: 외곽선만, 모든 좌표 저장
contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

# 이진 영상을 컬러 영상으로 변환 (윤곽선을 색으로 표시하기 위해)
img2 = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

# 모든 윤곽선을 보라색으로 그림
cv.drawContours(img2, contours, -1, (255, 0, 255), 2)

# 윤곽선이 포함된 이미지 출력
cv.imshow('Horse with contour', img2)

# 가장 바깥쪽 윤곽선 하나 선택
contour = contours[0]

# 윤곽선 모멘트 계산 (면적, 중심 등)
m = cv.moments(contour)

# 면적 계산
area = cv.contourArea(contour)

# 중심 좌표 계산: m10/m00, m01/m00
cx, cy = m['m10'] / m['m00'], m['m01'] / m['m00']

# 윤곽선 길이 (둘레) 계산
perimeter = cv.arcLength(contour, True)

# 원형도(Roundness) 계산: 4πA / P² → 원에 가까울수록 값이 1에 가까움
roundness = (4.0 * np.pi * area) / (perimeter * perimeter)

# 결과 출력
print('면적 =', area,
      '\n중심 = (', cx, ',', cy, ')',
      '\n둘레 =', perimeter,
      '\n둥근 정도 =', roundness)

# 다시 컬러로 변환 (윤곽선 근사와 convex hull 시각화용)
img3 = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

# 윤곽선을 직선으로 근사화 (허용 오차=8픽셀)
contour_approx = cv.approxPolyDP(contour, 8, True)

# 초록색으로 근사 윤곽선 시각화
cv.drawContours(img3, [contour_approx], -1, (0, 255, 0), 2)

# convex hull(볼록 껍질) 계산
hull = cv.convexHull(contour)

# hull을 (1, N, 2) 형태로 reshape해서 그리기 적합한 형태로 바꿈
# (오류)hull = hull.reshape(1, hull.shape[0], hull.shape[1])

# drawContours는 [hull] 형식으로 리스트 감싸서 전달
cv.drawContours(img3, [hull], -1, (0, 0, 255), 2)

# 파란색으로 convex hull 시각화
cv.drawContours(img3, hull, -1, (0, 0, 255), 2)

# 근사 윤곽선과 convex hull이 포함된 영상 출력
cv.imshow('Horse with line segments and convex hull', img3)

cv.waitKey()               # 키 입력 대기
cv.destroyAllWindows()     # 모든 창 닫기
