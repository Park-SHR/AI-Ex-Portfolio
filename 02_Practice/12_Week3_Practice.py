import cv2
import numpy as np
import matplotlib.pyplot as plt

def normalize_to_255(channel):
    norm = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)

# Lena 이미지를 컬러영상으로 읽고 화면에 출력
image = cv2.imread('../03_data/Lena.png', cv2.IMREAD_COLOR)
cv2.imshow('Original Image', image)
cv2.waitKey()
cv2.destroyWindow('Original Image')

# 영상을 흑백으로 변환하여 화면에 출력
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', gray)
cv2.waitKey()

## 흑백으로 변환된 영상에 Histogram Equalization 적용
gray_eq = cv2.equalizeHist(gray)
hist, bins = np.histogram(gray_eq, 256, [0, 255])
plt.fill_between(range(256), hist, 0)
plt.xlabel('pixel value')
plt.show()

cv2.imshow('Histogram Equalized Gray', gray_eq)
cv2.waitKey()

## 흑백으로 변환된 영상에 Gamma Correction 적용
gamma = 0.5
gamma_img = np.power(gray.astype(np.float32) / 255, gamma)
cv2.imshow('Gamma Corrected Gray', gamma_img)
cv2.imwrite('/tmp/gamma_img.png', gamma_img * 255)
cv2.waitKey()
cv2.destroyAllWindows()

# 영상 HSV 컬러 스페이스 변환 및 각각 화면을 출력(0부터 255정규화)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h = hsv[:, :, 0]
s = hsv[:, :, 1]
v = hsv[:, :, 2]

h_norm = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
s_norm = cv2.normalize(s, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
v_norm = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

cv2.imshow('H Channel (Normalized 0~255)', h_norm)
cv2.imshow('S Channel (Normalized 0~255)', s_norm)
cv2.imshow('V Channel (Normalized 0~255)', v_norm)
cv2.waitKey()
cv2.destroyAllWindows()

## H 채널 Median Filter 적용 및 출력
h_median = cv2.medianBlur((h_norm * 255).astype(np.uint8), 7)
plt.imshow(h_median)
plt.show()

## S 채널 Gaussian Filter 적용 및 출력
s_gaussian = cv2.GaussianBlur(s_norm, (7, 7), 0)
plt.imshow(s_gaussian)
plt.show()

## V 채널 Bilateral Filter 적용 및 출력
v_bilateral = cv2.bilateralFilter(v_norm, -1, 0.3, 10)
plt.imshow(v_bilateral)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()