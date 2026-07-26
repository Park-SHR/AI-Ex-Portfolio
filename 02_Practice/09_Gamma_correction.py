import cv2
import numpy as np

image = cv2.imread('../03_data/Lena.png', 0).astype(np.float32) / 255

gamma = 0.5
corrected_image = np.power(image, gamma)

cv2.imshow('image', image)
cv2.imshow('corrected_image', corrected_image)
cv2.waitKey()

cv2.imwrite('/tmp/image.png', image * 255)
cv2.imwrite('/tmp/corrected_image.png', corrected_image * 255)

cv2.destroyAllWindows()