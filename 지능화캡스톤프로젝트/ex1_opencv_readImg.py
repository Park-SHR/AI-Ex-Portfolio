import cv2

image = cv2.imread("Lenna.png")

if image is None:
    print("이미지를 읽을 수 없습니다. 파일 경로를 확인하세요.")
else:
    cv2.imshow("Lenna", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
