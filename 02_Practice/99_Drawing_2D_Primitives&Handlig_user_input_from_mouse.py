import argparse
import cv2, numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='../03_data/Lena.png', help='Image path.')
params = parser.parse_args()
image = cv2.imread(params.path)
image_to_show = image.copy()

def mouse_callback(event, x, y, flags, param):
    global image_to_show

    if event == cv2.EVENT_LBUTTONDOWN:
        size = 40
        cv2.rectangle(image_to_show,
                      (x - size, y - size),
                      (x + size, y + size),
                      (0, 255, 0), 2)

    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(image_to_show,
                   (x, y),
                   30,
                   (255, 0, 0),
                   2)

    elif event == cv2.EVENT_MBUTTONDOWN:
        size = 40
        pts = np.array([
            [x, y - size],
            [x - size, y + size],
            [x + size, y + size]
        ], np.int32)

        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image_to_show, [pts], True, (0, 0, 255), 2)


cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)

while True:
    cv2.imshow('image', image_to_show)
    key = cv2.waitKey(1)

    if key == 27: 
        break

cv2.destroyAllWindows()