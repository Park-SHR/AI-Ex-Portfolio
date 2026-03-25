import cv2

image = cv2.imread("Hawkes.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("이미지를 열 수 없습니다. 파일 경로를 확인하세요.")
else:
    dst = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    cv2.imshow("Original", image)
    cv2.imshow("Normalized", dst)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
