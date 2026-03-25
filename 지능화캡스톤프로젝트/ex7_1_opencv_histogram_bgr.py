import cv2
import matplotlib.pyplot as plt

image = cv2.imread("Lenna.png")

if image is None:
    print("이미지를 열 수 없습니다. 파일 경로를 확인하세요.")
else:
    colors = ('b', 'g', 'r')
    channel_names = ("Blue", "Green", "Red")

    cv2.imshow("Original Image", image)

    plt.figure()
    plt.title("BGR Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, label=channel_names[i])

    plt.xlim([0, 256])
    plt.legend()
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
