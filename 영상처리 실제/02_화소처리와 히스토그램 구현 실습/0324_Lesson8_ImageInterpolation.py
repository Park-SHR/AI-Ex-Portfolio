import cv2 as cv                      # OpenCV 라이브러리 import

# 이미지 불러오기
img = cv.imread('soccer.jpg')        # 'soccer.jpg' 이미지를 컬러로 불러옴 (BGR)

# 특정 영역 잘라내기 (세로 250~350, 가로 270~370 범위)
patch = img[250:350, 270:370, :]     # 관심영역(ROI): 100x100 크기 추출

# 원본 이미지에 ROI 영역 사각형 그리기 (파란색, 두께 3)
img = cv.rectangle(img, (270, 250), (370, 350), (255, 0, 0), 3)

# (1) 최근접 이웃 보간법으로 확대 (계단형 느낌, 빠름)
patch1 = cv.resize(patch, dsize=(0,0), fx=5, fy=5, interpolation=cv.INTER_NEAREST)

# (2) 양선형 보간법으로 확대 (부드럽고 자연스러움)
patch2 = cv.resize(patch, dsize=(0,0), fx=5, fy=5, interpolation=cv.INTER_LINEAR)

# (3) 3차 보간법으로 확대 (가장 부드럽고 고급 방식, 속도 느림)
patch3 = cv.resize(patch, dsize=(0,0), fx=5, fy=5, interpolation=cv.INTER_CUBIC)

# 원본 이미지(사각형 표시된) 출력
cv.imshow('Original', img)

# 각각의 보간 방법으로 확대된 결과 출력
cv.imshow('Resize nearest', patch1)     # 최근접 보간
cv.imshow('Resize bilinear', patch2)    # 양선형 보간
cv.imshow('Resize bicubic', patch3)     # 3차 보간

# 키 입력 대기 후 모든 창 닫기
cv.waitKey()
cv.destroyAllWindows()
