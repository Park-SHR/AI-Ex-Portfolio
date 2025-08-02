import cv2 as cv                        # OpenCV 라이브러리 import
import matplotlib.pyplot as plt        # (plt는 여기선 사용되지 않지만 import 되어 있음)

# 이미지 불러오기
img = cv.imread('soccer.jpg')          # 컬러 이미지 불러오기 (기본 BGR 형식)

# R 채널만 선택하여 Otsu 알고리즘으로 이진화 수행
t, bin_img = cv.threshold(
    img[:, :, 2],                      # R 채널 선택 (B=0, G=1, R=2)
    0, 255,                            # 임계값 범위 (0~255)
    cv.THRESH_BINARY + cv.THRESH_OTSU # Otsu 알고리즘 적용하여 최적 임계값 자동 계산
)

# 계산된 Otsu 임계값 출력
print('오츄 알고리즘이 찾은 최적 임곗값 =', t)

# 원본 R 채널 이미지 창에 표시
cv.imshow('R chaanel', img[:, :, 2])   # 오타 주의: chaanel → channel

# 이진화된 결과 이미지 창에 표시
cv.imshow('R channel binalrization', bin_img)  # 오타 주의: binalrization → binarization

# 이진화된 이미지를 파일로 저장
cv.imwrite('soccer_binarization.jpg', bin_img)

# 키 입력 대기 후 창 닫기
cv.waitKey()
cv.destroyAllWindows()
