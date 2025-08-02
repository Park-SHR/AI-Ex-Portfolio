import skimage             # scikit-image 라이브러리
import numpy as np         # 수치 연산을 위한 numpy
import cv2 as cv           # OpenCV 라이브러리

img = skimage.data.coffee()  # 예제 이미지 로드 (RGB 형식)
cv.imshow('Coffee image', cv.cvtColor(img, cv.COLOR_RGB2BGR))  # OpenCV는 BGR이므로 변환 후 출력

# SLIC 알고리즘 적용 (compactness=20, 세그먼트 수=600)
slic1 = skimage.segmentation.slic(img, compactness=20, n_segments=600)

# 분할된 영역의 경계선을 이미지 위에 표시
sp_img1 = skimage.segmentation.mark_boundaries(img, slic1)

# float 타입 결과를 0~255 정수형으로 변환 (OpenCV에 맞게)
sp_img1 = np.uint8(sp_img1 * 255.0)

# 다른 compactness 값으로 SLIC 재적용 (compactness=40)
slic2 = skimage.segmentation.slic(img, compactness=40, n_segments=600)

# 경계선 시각화 및 정수형 변환
sp_img2 = skimage.segmentation.mark_boundaries(img, slic2)
sp_img2 = np.uint8(sp_img2 * 255.0)

# 결과 출력 (RGB → BGR 변환해서 OpenCV로 시각화)
cv.imshow('Super pixels (compact 20)', cv.cvtColor(sp_img1, cv.COLOR_RGB2BGR))
cv.imshow('Super pixels (compact 40)', cv.cvtColor(sp_img2, cv.COLOR_RGB2BGR))

# 키 입력 대기 및 모든 창 닫기
cv.waitKey()
cv.destroyAllWindows()
