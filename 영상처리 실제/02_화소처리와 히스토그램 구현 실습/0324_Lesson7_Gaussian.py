import cv2 as cv                   # OpenCV 라이브러리를 cv라는 이름으로 불러옴
import numpy as np                # NumPy 라이브러리를 np라는 이름으로 불러옴

# 이미지 불러오기
img = cv.imread('soccer.jpg')     # 'soccer.jpg' 이미지를 컬러로 불러옴 (기본 BGR 형식)

# 이미지 크기 축소 (원래 크기의 40%)
img = cv.resize(img, dsize=(0, 0), fx=0.4, fy=0.4)

# 이미지 그레이스케일로 변환 (흑백 처리)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 흑백 이미지 위에 'soccer' 텍스트 추가
cv.putText(gray, 'soccer', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

# 원본(흑백+텍스트 포함) 이미지 띄우기
cv.imshow('Original', gray)

# 가우시안 블러 처리한 이미지 3개를 좌우로 나란히 붙이기
smooth = np.hstack((
    cv.GaussianBlur(gray, (5,5), 0.0),   # 작은 블러 (약한 흐림)
    cv.GaussianBlur(gray, (9,9), 0.0),   # 중간 블러
    cv.GaussianBlur(gray, (15,15), 0.0)  # 큰 블러 (강한 흐림)
))

# 블러 결과 이미지 띄우기
cv.imshow('Smooth', smooth)

# 엠보싱 필터 커널 정의 (대각선 방향 강조)
femboss = np.array([
    [-1.0, 0.0, 0.0],
    [ 0.0, 0.0, 0.0],
    [ 0.0, 0.0, 1.0]
])

# 그레이스케일 이미지를 int16으로 변환 (계산 중 음수 방지용)
gray16 = np.int16(gray)

# 엠보싱 필터 적용 + 128 더해서 밝기 조절 + 범위 클리핑 → 최종 결과
emboss = np.uint8(np.clip(cv.filter2D(gray16, -1, femboss) + 128, 0, 255))

# (❌) 클리핑 없이 강제로 uint8 변환 → 오버플로우/언더플로우 가능
emboss_bad = np.uint8(cv.filter2D(gray16, -1, femboss) + 128)

# (❌❌) 애초에 uint8로 필터 적용 → 음수 결과가 제대로 표현되지 않음
emboss_worse = cv.filter2D(gray, -1, femboss)

# 각각의 결과 이미지 띄우기
cv.imshow('Emboss', emboss)
cv.imshow('Emboss_bad', emboss_bad)
cv.imshow('Emboss_worse', emboss_worse)

# 키 입력 기다렸다가 창 닫기
cv.waitKey()
cv.destroyAllWindows()
