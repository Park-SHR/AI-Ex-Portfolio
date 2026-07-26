import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('../03_data/Lena.png', 8)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# (1) Otsu의 알고리즘을 통해 thresholding
otsu_thr, otsu_mask= cv2.threshold(gray, -1, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
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

# (2) (1)의 영상에 대해서 External/Internal Contour 찾기
contours, hierarchy = cv2.findContours(otsu_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

image_external = np.zeros(otsu_mask.shape, otsu_mask.dtype)
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(image_external, contours, i, 255, -1)

image_internal = np.zeros(otsu_mask.shape, otsu_mask.dtype)
for i in range(len(contours)):
    if hierarchy[0][i][3] != -1:
        cv2.drawContours(image_internal, contours, i, 255, -1)

plt.figure(figsize=(10,3))
plt.subplot(131)
plt.axis('off')
plt.title('Otsu threshold')
plt.imshow(otsu_mask, cmap='gray')
plt.subplot(132)
plt.axis('off')
plt.title('external')
plt.imshow(image_external, cmap='gray')
plt.subplot(133)
plt.axis('off')
plt.title('internal')
plt.imshow(image_internal, cmap='gray')
plt.tight_layout()
plt.show()
plt.close()

# (3) (1)의 영상에 대해서 Connected COmponent를 찾고 
# 스페이스를 누를 때마다 Random하게 5개의 component를 보여주시오
# 각 component의 색은 random 하게 정하시오
output = cv2.connectedComponentsWithStats(otsu_mask, 8, cv2.CV_32S)
num_labels, labelmap, stats, centers = output

valid_labels = []
for l in range(1, num_labels):
    if stats[l][4] > 200:
        valid_labels.append(l)

img = cv2.cvtColor(otsu_mask*255, cv2.COLOR_GRAY2BGR)

while True:
    colored = np.full((img.shape[0], img.shape[1], 3), 0, np.uint8)

    if len(valid_labels) >= 5:
        selected = np.random.choice(valid_labels, 5, replace=False)
    else:
        selected = valid_labels

    for l in selected:
        b = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        r = np.random.randint(0, 256)

        colored[labelmap == l] = (b, g, r)
        cv2.circle(colored, 
                   (int(centers[l][0]), int(centers[l][1])), 5, (255, 0, 0), cv2.FILLED)

    cv2.imshow('Connected components', np.hstack((img, colored)))

    key = cv2.waitKey(0)

    if key == 32:
        continue
    else:
        break

cv2.destroyAllWindows()

# (4) (1)의 영상에 대해서 Distance Transform을 계산해서 보여주시오
image = np.full((480, 640), 255, np.uint8)
cv2.circle(otsu_mask, (320, 240), 100, 0)

distmap = cv2.distanceTransform(otsu_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
plt.figure()
plt.imshow(distmap, cmap='gray')
plt.show()