import cv2, numpy as np

image = cv2.imread('../03_data/Lena.png').astype(np.float32) / 255
print('Shape:', image.shape)
cv2.imshow('original image', image)

image[:, :, [0, 2]] = image[:, :, [2, 0]]
cv2.imshow('blue_and_red_swapped', image)
cv2.waitKey()

image[:, :, [0, 2]] = image[:, :, [2, 0]]
image[:, :, 0] = (image[:, :, 0] * 0.9).clip(0, 1)
image[:, :, 1] = (image[:, :, 1] * 1.1).clip(0, 1)
cv2.imshow('converted image', image)
cv2.waitKey()

cv2.destroyAllWindows()