import cv2

src = cv2.imread("desert.jpg", cv2.IMREAD_COLOR)

if src is None:
    print("이미지를 열 수 없습니다. 파일 경로를 확인하세요.")
    exit()

x, y, w, h = cv2.selectROI("Select ROI", src, showCrosshair=True, fromCenter=False)
cv2.destroyWindow("Select ROI")

src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
roi = src_ycrcb[y:y+h, x:x+w]

hist = cv2.calcHist([roi], [1, 2], None, [128, 128], [0, 256, 0, 256])
cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

backproj = cv2.calcBackProject([src_ycrcb], [1, 2], hist, [0, 256, 0, 256], 1)

dst = cv2.copyTo(src, backproj)

cv2.imshow("Source", src)
cv2.imshow("BackProjection", backproj)
cv2.imshow("Result", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
