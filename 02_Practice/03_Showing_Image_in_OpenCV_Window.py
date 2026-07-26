import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='../03_data/Lena.png', help='Image path.')
parser.add_argument('--iter', default=50, help='Downsampling-upsampling Iterations number')
params = parser.parse_args()
orig = cv2.imread(params.path)
orig_size = orig.shape[0:2]

cv2.imshow("Original Image", orig)
cv2.waitKey(2000)

resized = orig

for i in range(params.iter):
    resized = cv2.resize(cv2.resize(resized, (256,256)), orig_size)
    cv2.imshow("downsized&restored", resized)
    cv2.waitKey(100)

cv2.destroyWindow("downsized&restored")

cv2.namedWindow("original", cv2.WINDOW_NORMAL)
cv2.namedWindow("result")
cv2.imshow("original", orig)
cv2.imshow("result", resized)
cv2.waitKey(0)

cv2.destroyAllWindows()