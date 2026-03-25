import cv2
import numpy as np

image = cv2.imread("Candies.png")

if image is None:
    print("이미지를 열 수 없습니다. 파일 경로를 확인하세요.")
else:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 120, 70])
    upper_blue = np.array([140, 255, 255])

    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    blue_candies = cv2.bitwise_and(image, image, mask=blue_mask)

    cv2.imshow("Original Image", image)
    cv2.imshow("Blue Mask", blue_mask)
    cv2.imshow("Blue Candies", blue_candies)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
