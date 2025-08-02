import cv2 as cv  # OpenCV 라이브러리 불러오기

# 이미지 파일 읽기 (컬러 이미지로 불러옴)
img = cv.imread('soccer.jpg')  # 'soccer.jpg' 파일에서 이미지 읽기

# 그레이스케일 이미지로 변환 (엣지 검출은 흑백에서 진행)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Canny 엣지 검출 - 첫 번째 시도 (T_low=50, T_high=150)
canny1 = cv.Canny(gray, 50, 150)

# Canny 엣지 검출 - 두 번째 시도 (T_low=100, T_high=200)
canny2 = cv.Canny(gray, 100, 200)

# 원본 그레이스케일 이미지 출력
cv.imshow('Original', gray)

# 첫 번째 Canny 결과 출력
cv.imshow('Canny1', canny1)

# 두 번째 Canny 결과 출력
cv.imshow('Canny2', canny2)

# 키 입력 대기
cv.waitKey()

# 모든 창 닫기
cv.destroyAllWindows()
