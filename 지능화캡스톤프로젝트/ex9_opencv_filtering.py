import cv2
import numpy as np

def apply_filter(image, kernel, filter_name):
    filtered_image = cv2.filter2D(image, -1, kernel)
    cv2.imshow(filter_name, filtered_image)
    cv2.imwrite(f"{filter_name}.jpg", filtered_image)
    return filtered_image

image = cv2.imread("Lenna.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("이미지를 열 수 없습니다. 파일 경로를 확인하세요.")
else:
    cv2.imshow("Original", image)

    average_filter = np.ones((3, 3), np.float32) / 9.0

    sharpening_filter = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], np.float32)

    laplacian_filter = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], np.float32)

    apply_filter(image, average_filter, "Average Filter")
    apply_filter(image, sharpening_filter, "Sharpening Filter")
    apply_filter(image, laplacian_filter, "Laplacian Filter")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
