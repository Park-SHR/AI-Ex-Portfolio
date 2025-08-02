import skimage             # scikit-image (영상 처리)
import numpy as np         # 수치 계산용 numpy
import cv2 as cv           # OpenCV (이미지 출력용)
import time                # 실행 시간 측정용

coffee = skimage.data.coffee()  # 내장된 예제 이미지 (RGB 형식)
start = time.time()  # 분할 시간 측정 시작

# SLIC 알고리즘으로 초기 슈퍼픽셀 생성
slic = skimage.segmentation.slic(
    coffee,
    compactness=20,        # 색상 vs 위치 가중치 (작을수록 색상 중심)
    n_segments=600,        # 원하는 슈퍼픽셀 수
    start_label=1          # 라벨 번호 1부터 시작
)

#2.0 이후로 구조가 변경됨됨
# 변경 전
# 슈퍼픽셀들을 그래프로 구성 (각 노드는 평균 색상 기준 유사도 기반 연결)
#g = skimage.future.graph.rag_mean_color(coffee, slic, mode='similarity')
# 정규화 절단(Normalized Cut) 알고리즘으로 영역 분할 수행
#ncut = skimage.future.graph.cut_normalized(slic, g)

# 변경 후
from skimage import graph  # 이 줄 꼭 추가!
g = graph.rag_mean_color(coffee, slic, mode='similarity')
ncut = graph.cut_normalized(slic, g)

# 시간 측정 종료 및 출력
print(coffee.shape, ' Coffee 영상을 분할하는 데 ', time.time() - start, '초 소요')
# 분할된 영역의 경계 시각화
marking = skimage.segmentation.mark_boundaries(coffee, ncut)

# float → uint8 정수형으로 변환 (OpenCV 출력용)
ncut_coffee = np.uint8(marking * 255.0)
# 결과 이미지 출력 (RGB → BGR 변환해서 OpenCV에 맞춤)
cv.imshow('Normalized cut', cv.cvtColor(ncut_coffee, cv.COLOR_RGB2BGR))
# 키 입력 대기 후 창 닫기
cv.waitKey()
cv.destroyAllWindows()
