import cv2 as cv                  # OpenCV 라이브러리 import
import sys                        # 예외 처리용 시스템 라이브러리 import

# 이미지 불러오기
img = cv.imread('soccer.jpg')     # 'soccer.jpg'를 컬러 이미지로 불러옴 (BGR 순서)

# 이미지 불러오기 실패 시 프로그램 종료
if img is None:
    sys.exit('파일을 읽을 수 없습니다.')  # 이미지가 없거나 경로가 잘못되었을 때 메시지 출력 후 종료

# 원본 이미지 출력
cv.imshow('original_RGB', img)

# 이미지의 왼쪽 위 절반 출력
cv.imshow('Upper left half', img[
    0:img.shape[0]//2,            # 세로 절반 (위쪽)
    0:img.shape[1]//2,            # 가로 절반 (왼쪽)
    :
])

# 이미지의 정중앙 부분 (가로, 세로 각각 중앙 50%) 출력
cv.imshow('Center half', img[
    img.shape[0]//4 : 3*img.shape[0]//4,   # 세로 중앙 부분
    img.shape[1]//4 : 3*img.shape[1]//4,   # 가로 중앙 부분
    :
])

# 이미지의 세로, 가로 크기 출력
print(img.shape[0], img.shape[1])  # (높이, 너비)

# R 채널만 추출해서 출력
cv.imshow('R channel', img[:,:,2])  # OpenCV는 BGR 순서이므로 R은 2번 인덱스

# G 채널만 추출해서 출력
cv.imshow('G channel', img[:,:,1])

# B 채널만 추출해서 출력
cv.imshow('B channel', img[:,:,0])

# 키 입력 대기 후 모든 창 닫기
cv.waitKey()
cv.destroyAllWindows()
