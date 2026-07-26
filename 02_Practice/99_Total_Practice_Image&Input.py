import argparse
import cv2
import numpy as np
import os
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='../03_data/Lena.png', help='Image path')
params = parser.parse_args()

image = cv2.imread(params.path)
image_to_show = np.copy(image)
temp_image = np.copy(image)

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

drawing = False
start_pt = (-1, -1)
end_pt = (-1, -1)
draw_mode = 'r'

def draw_mode_label(mode):
    if mode == 'r':
        return 'MODE: RECTANGLE'
    elif mode == 'l':
        return 'MODE: LINE'
    elif mode == 'a':
        return 'MODE: ARROW'
    return 'MODE: NONE'

def draw_status_box(img, mode):
    result = np.copy(img)
    text = draw_mode_label(mode)

    x1, y1 = 10, 10
    x2, y2 = 260, 50

    cv2.rectangle(result, (x1, y1), (x2, y2), (255, 255, 255), -1)
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 0), 2)
    cv2.putText(result, text, (20, 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return result


def mouse_callback(event, x, y, flags, param):
    global drawing, start_pt, end_pt, image_to_show, temp_image, draw_mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_pt = (x, y)
        end_pt = (x, y)
        temp_image = np.copy(image_to_show)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_pt = (x, y)
            preview = np.copy(temp_image)

            if draw_mode == 'r':
                cv2.rectangle(preview, start_pt, end_pt, (0, 255, 0), 2)
            elif draw_mode == 'l':
                cv2.line(preview, start_pt, end_pt, (255, 0, 0), 2)
            elif draw_mode == 'a':
                cv2.arrowedLine(preview, start_pt, end_pt, (0, 0, 255), 2)

            image_to_show = preview

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            end_pt = (x, y)

            if draw_mode == 'r':
                cv2.rectangle(temp_image, start_pt, end_pt, (0, 255, 0), 2)
            elif draw_mode == 'l':
                cv2.line(temp_image, start_pt, end_pt, (255, 0, 0), 2)
            elif draw_mode == 'a':
                cv2.arrowedLine(temp_image, start_pt, end_pt, (0, 0, 255), 2)

            image_to_show = np.copy(temp_image)

cv2.namedWindow("result")
cv2.setMouseCallback("result", mouse_callback)

finish = False

while not finish:
    display_image = draw_status_box(image_to_show, draw_mode)
    cv2.imshow("result", display_image)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        draw_mode = 'r'

    elif key == ord('l'):
        draw_mode = 'l'

    elif key == ord('a'):
        draw_mode = 'a'

    elif key == ord('w'):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{now}_lena_draw.png"
        save_path = os.path.join(output_dir, filename)

        cv2.imwrite(save_path, image_to_show, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print("Saved:", save_path)

    elif key == ord('c'):
        image_to_show = np.copy(image)
        temp_image = np.copy(image)

    elif key == 27:
        finish = True

cv2.destroyAllWindows()