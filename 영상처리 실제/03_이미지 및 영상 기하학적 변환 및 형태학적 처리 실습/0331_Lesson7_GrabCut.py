import cv2 as cv                         # OpenCV 라이브러리 불러오기
import numpy as np                      # NumPy 불러오기

img = cv.imread('soccer.jpg')           # 영상 읽기
img_show = np.copy(img)                 # 마우스 브러시 표시용 이미지 복사

mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)  # 마스크 초기화 (0으로 채움)
mask[:,:] = cv.GC_PR_BGD                  # 모든 화소를 '아마 배경'으로 초기 설정

BrushSiz = 9                            # 브러시 크기 설정 (반지름)
LColor, RColor = (255, 0, 0), (0, 0, 255)  # 파란색(물체), 빨간색(배경)

# 마우스 이벤트 콜백 함수 정의
def painting(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:   # 왼쪽 클릭하면 파란색 표시 + 물체 영역으로 마스크 수정
        cv.circle(img_show, (x, y), BrushSiz, LColor, -1)
        cv.circle(mask, (x, y), BrushSiz, cv.GC_FGD, -1)
    elif event == cv.EVENT_RBUTTONDOWN: # 오른쪽 클릭하면 빨간색 표시 + 배경 영역으로 마스크 수정
        cv.circle(img_show, (x, y), BrushSiz, RColor, -1)
        cv.circle(mask, (x, y), BrushSiz, cv.GC_BGD, -1)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:  # 왼쪽 클릭 후 드래그
        cv.circle(img_show, (x, y), BrushSiz, LColor, -1)
        cv.circle(mask, (x, y), BrushSiz, cv.GC_FGD, -1)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:  # 오른쪽 클릭 후 드래그
        cv.circle(img_show, (x, y), BrushSiz, RColor, -1)
        cv.circle(mask, (x, y), BrushSiz, cv.GC_BGD, -1)

cv.imshow('Painting', img_show)                    # 브러시 마킹 이미지 창 생성
cv.namedWindow('Painting')                         # 창 이름 등록
cv.setMouseCallback('Painting', painting)          # 마우스 콜백 등록

while True:                                        # 'q' 키 누를 때까지 브러시 마킹 대기
    cv.imshow('Painting', img_show) #콜백 받으면서 화면 갱신 코드 추가
    if cv.waitKey(1) == ord('q'):
        break

# GrabCut 적용
background = np.zeros((1, 65), np.float64)         # 배경 히스토그램 (grabCut 내부에서 사용)
foreground = np.zeros((1, 65), np.float64)         # 전경 히스토그램

# GrabCut 실행 (마스크 기반, 반복 횟수 5회)
cv.grabCut(img, mask, None, background, foreground, 5, cv.GC_INIT_WITH_MASK)

# 결과 마스크: 확실한 전경 또는 아마 전경이면 1, 나머지는 0
mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')

# 원본 이미지에 마스크 적용해서 전경만 추출
grab = img * mask2[:, :, np.newaxis]

cv.imshow('Grab cut image', grab)                 # 최종 분리된 결과 출력
cv.waitKey()
cv.destroyAllWindows()
