import cv2 as cv  # OpenCV 라이브러리 불러오기

# 이미지 불러오기 (컬러 → BGR)
img = cv.imread('soccer.jpg')

# 그레이스케일 이미지로 변환 (엣지 검출은 흑백 이미지에서 진행)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# X 방향으로 소벨 필터 적용 (float32 타입, ksize=3)
grad_x = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)

# Y 방향으로 소벨 필터 적용 (float32 타입, ksize=3)
grad_y = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)

# X 방향 그래디언트의 절댓값을 uint8로 변환 (음수 → 양수, 0~255 범위 클리핑)
sobel_x = cv.convertScaleAbs(grad_x)

# Y 방향 그래디언트의 절댓값을 uint8로 변환
sobel_y = cv.convertScaleAbs(grad_y)

# 가중치 합을 통해 최종 엣지 강도 계산
# 각 방향 결과를 0.5씩 가중 평균 → 0.5 * sobel_x + 0.5 * sobel_y
edge_strength = cv.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

# 원본 그레이스케일 이미지 출력
cv.imshow('Original', gray)

# X 방향 소벨 결과 출력
cv.imshow('sobel x', sobel_x)

# Y 방향 소벨 결과 출력
cv.imshow('sobel y', sobel_y)

# 엣지 강도 최종 결과 출력
cv.imshow('edge strength', edge_strength)

# 키 입력 대기
cv.waitKey()

# 모든 창 닫기
cv.destroyAllWindows()
