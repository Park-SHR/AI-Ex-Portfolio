import cv2 as cv  # OpenCV 라이브러리 불러오기

# 이미지 불러오기 ('apple.jpg')
img = cv.imread('apple.jpg')

# 그레이스케일 변환 (허프 서클 검출은 흑백 이미지 기반으로 동작)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 허프 원 검출 함수 적용 (HoughCircles)
apples = cv.HoughCircles(
    gray,                   # 입력 영상 (그레이스케일)
    cv.HOUGH_GRADIENT,      # 허프 변환 방식 (기본값)
    dp=1,                   # 누산기 해상도 비율 (1이면 동일 해상도)
    minDist=200,            # 원 중심 간 최소 거리 (겹치지 않도록)
    param1=150,             # Canny 엣지 검출 상위 임계값
    param2=20,              # 중심 검출에 필요한 누산기 임계값 (작을수록 민감하게 감지)
    minRadius=50,           # 검출할 원의 최소 반지름
    maxRadius=120           # 검출할 원의 최대 반지름
)

# 검출된 각 원에 대해 반복
for i in apples[0]:  # apples[0]에는 [[x, y, r], [x, y, r], ...] 형태의 원 정보가 들어있음
    cv.circle(       # 원을 이미지 위에 그림
        img,
        (int(i[0]), int(i[1])),  # 원의 중심 좌표 (x, y)
        int(i[2]),               # 반지름
        (255, 0, 0),             # 색상 (파란색 원)
        2                        # 선 두께
    )

# 결과 이미지 출력
cv.imshow('Apple detection', img)

# 키 입력 대기
cv.waitKey()

# 모든 창 닫기
cv.destroyAllWindows()
