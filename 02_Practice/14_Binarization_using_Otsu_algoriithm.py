import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('../03_data/Lena.png', 8)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

otsu_thr, otsu_mask= cv2.threshold(gray, -1, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print('Estumated threshold (Otsu):', otsu_thr)

plt.figure(figsize=(6,3))
plt.subplot(121)
plt.axis('off')
plt.title('original')
plt.imshow(image_rgb)

plt.subplot(122)
plt.axis('off')
plt.title('Otsu threshold')
plt.imshow(otsu_mask, cmap='gray')

plt.tight_layout()
plt.show()

plt.close()