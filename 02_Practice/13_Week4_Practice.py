import cv2
import math
import numpy as np
from scipy import signal

import matplotlib.pyplot as plt

#영상 결과 화면 출력
image = cv2.imread('../03_data/Lena.png')
image_g = cv2.imread('../03_data/Lena.png',0)
#(1) Unsharp mask 적용하고 출력
KSIZE = 11
ALPHA = 2

kernel = cv2.getGaussianKernel(KSIZE, 0)
kernel = -ALPHA * kernel @ kernel.T
kernel[KSIZE//2, KSIZE//2] += 1 + ALPHA

unsharp = cv2.filter2D(image_g, -1, kernel)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(unsharp, cmap='gray')
plt.title('Unsharp Mask Result')
plt.axis('off')

plt.tight_layout()
plt.draw()

plt.waitforbuttonpress()
plt.close()

# (1)의 결과에 Sobel filter를 적용하고 출력하시오. (2)
dx = cv2.Sobel(unsharp, cv2.CV_32F, 1, 0)
dy = cv2.Sobel(unsharp, cv2.CV_32F, 0, 1)
sobel = cv2.magnitude(dx, dy)
sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(unsharp, cmap='gray')
plt.title('Unsharp Mask Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(sobel, cmap='gray')
plt.title('Gabor Result')
plt.axis('off')

plt.tight_layout()
plt.draw()

plt.waitforbuttonpress()
plt.close()

# (1)의 결과에 Gabor filter를 적용하고 출력하시오. (3)
src_f = unsharp.astype(np.float32) / 255

gabor_kernel = cv2.getGaborKernel((21, 21), 5, 1, 10, 1, 0, cv2.CV_32F)
gabor_kernel /= math.sqrt((gabor_kernel * gabor_kernel).sum())

gabor = cv2.filter2D(src_f, -1, gabor_kernel)
gabor = cv2.normalize(gabor, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(unsharp, cmap='gray')
plt.title('Unsharp Mask Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(gabor, cmap='gray')
plt.title('Gabor Result')
plt.axis('off')

plt.tight_layout()
plt.draw()

plt.waitforbuttonpress()
plt.close()

# (2)와 (3)의 차이를 Threshold의 변화에 따라서 출력하시오. (트랙바) (4)
diff = cv2.absdiff(sobel, gabor)

morph_kernel = np.ones((3, 3), np.uint8)
threshold_result = np.zeros_like(diff)

def on_trackbar(pos):
    global threshold_result

    thr = cv2.getTrackbarPos('Threshold', 'Difference Threshold')
    _, threshold_result = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)

    cv2.imshow('Difference Threshold', threshold_result)

cv2.namedWindow('Difference Threshold')
cv2.createTrackbar('Threshold', 'Difference Threshold', 50, 255, on_trackbar)
on_trackbar(0)

cv2.waitKey()
cv2.destroyAllWindows()

# (4)의 결과에 Opening과 Closing을 적용하시오.
opening = cv2.morphologyEx(threshold_result, cv2.MORPH_OPEN, morph_kernel)
closing = cv2.morphologyEx(threshold_result, cv2.MORPH_CLOSE, morph_kernel)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(threshold_result, cmap='gray')
plt.title('Threshold_result')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(opening, cmap='gray')
plt.title('Opening')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(closing, cmap='gray')
plt.title('Closing')
plt.axis('off')

plt.tight_layout()
plt.draw()

plt.waitforbuttonpress()
plt.close()
# Frequency-based Filtering
# 영상을 화면에 출력하고 DFT를 적용한 결과를 출력하시오.
# 중심으로부터 원 모양과 사각형 모양의 필터링을 적용하고 결과를 출력하시오
image_f = cv2.imread('../03_data/Lena.png',0).astype(np.float32) / 255
fft = cv2.dft(image_f, flags=cv2.DFT_COMPLEX_OUTPUT)
fft_shift = np.fft.fftshift(fft, axes=[0, 1])

magnitude = cv2.magnitude(fft_shift[:, :, 0], fft_shift[:, :, 1])
magnitude = np.log(magnitude + 1)
magnitude_show = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

h, w = image_f.shape
cy, cx = h // 2, w // 2

radius = 25
mask_circle = np.zeros((h, w, 2), np.uint8)

for y in range(h):
    for x in range(w):
        if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
            mask_circle[y, x] = 1

fft_circle = fft_shift * mask_circle
ifft_circle = np.fft.ifftshift(fft_circle, axes=[0, 1])
filtered_circle = cv2.idft(ifft_circle, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

size = 25
mask_rect = np.zeros(fft.shape, np.uint8)
mask_rect[cy-size:cy+size, cx-size:cx+size] = 1

fft_shift *= mask_rect
ifft_rect = np.fft.ifftshift(fft_shift, axes=[0, 1])
filtered_rect = cv2.idft(ifft_rect, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

plt.figure(figsize=(10, 10))

image_rgb = cv2.cvtColor(image_f, cv2.COLOR_BGR2RGB)
plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(magnitude_show, cmap='gray')
plt.title('DFT Spectrum')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(filtered_circle, cmap='gray')
plt.title('Circle Filtered Result')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(filtered_rect, cmap='gray')
plt.title('Rectangle Filtered Result')
plt.axis('off')

plt.tight_layout()
plt.draw()

plt.waitforbuttonpress()
plt.close()