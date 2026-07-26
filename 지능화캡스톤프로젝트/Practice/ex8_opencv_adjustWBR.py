import cv2
import numpy as np

image = cv2.imread("Lenna.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("이미지를 열 수 없습니다. 파일 경로를 확인하세요.")
else:
    alpha = 1.0
    func = (1 + alpha) * image - (alpha * 128)
    dst = np.clip(func, 0, 255).astype(np.uint8)

    cv2.imshow("Original", image)
    cv2.imshow("Contrast Adjusted", dst)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
