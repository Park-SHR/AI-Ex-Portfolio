import cv2
import numpy as np

image = cv2.imread("Candies.png")

if image is None:
    print("이미지를 열 수 없습니다. 파일 경로를 확인하세요.")
else:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    red_candies = cv2.bitwise_and(image, image, mask=red_mask)

    cv2.imshow("Original Image", image)
    cv2.imshow("Red Mask", red_mask)
    cv2.imshow("Red Candies", red_candies)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
