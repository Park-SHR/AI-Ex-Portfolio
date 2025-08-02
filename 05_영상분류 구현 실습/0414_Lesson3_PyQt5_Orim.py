import cv2 as cv              # OpenCV 라이브러리
import numpy as np           # NumPy 라이브러리
import sys                   # 시스템 종료용
from PyQt5.QtWidgets import *  # PyQt5 위젯 전체 임포트

class Orim(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('오림')                     # 윈도우 제목
        self.setGeometry(200, 200, 700, 200)            # 윈도우 위치 및 크기

        # 버튼 생성
        fileButton = QPushButton('파일', self)
        paintButton = QPushButton('페인팅', self)
        cutButton = QPushButton('오림', self)
        incButton = QPushButton('+', self)
        decButton = QPushButton('-', self)
        saveButton = QPushButton('저장', self)
        quitButton = QPushButton('나가기', self)

        # 버튼 위치 및 크기
        fileButton.setGeometry(10, 10, 100, 30)
        paintButton.setGeometry(110, 10, 100, 30)
        cutButton.setGeometry(210, 10, 100, 30)
        incButton.setGeometry(310, 10, 50, 30)
        decButton.setGeometry(360, 10, 50, 30)
        saveButton.setGeometry(410, 10, 100, 30)
        quitButton.setGeometry(510, 10, 100, 30)

        # 버튼 클릭 연결
        fileButton.clicked.connect(self.fileOpenFunction)
        paintButton.clicked.connect(self.paintFunction)
        cutButton.clicked.connect(self.cutFunction)
        incButton.clicked.connect(self.incFunction)
        decButton.clicked.connect(self.decFunction)
        saveButton.clicked.connect(self.saveFunction)
        quitButton.clicked.connect(self.quitFunction)

        # 초기값 설정
        self.BrushSiz = 5                                     # 붓 크기
        self.LColor, self.RColor = (255, 0, 0), (0, 0, 255)   # 파란색 물체, 빨간색 배경

    def fileOpenFunction(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', './')
        self.img = cv.imread(fname[0])                        # 이미지 읽기
        if self.img is None:
            sys.exit('파일을 찾을 수 없습니다.')               # 파일 없으면 종료

        self.img_show = np.copy(self.img)                     # 표시용 이미지 복사
        cv.imshow('Painting', self.img_show)

        self.mask = np.zeros((self.img.shape[0], self.img.shape[1]), np.uint8)
        self.mask[:, :] = cv.GC_PR_BGD                        # 초기 배경 설정

    def paintFunction(self):
        cv.setMouseCallback('Painting', self.painting)        # 마우스 이벤트 연결

    def painting(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            cv.circle(self.img_show, (x, y), self.BrushSiz, self.LColor, -1)
            cv.circle(self.mask, (x, y), self.BrushSiz, cv.GC_FGD, -1)

        elif event == cv.EVENT_RBUTTONDOWN:
            cv.circle(self.img_show, (x, y), self.BrushSiz, self.RColor, -1)
            cv.circle(self.mask, (x, y), self.BrushSiz, cv.GC_BGD, -1)

        elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
            cv.circle(self.img_show, (x, y), self.BrushSiz, self.LColor, -1)
            cv.circle(self.mask, (x, y), self.BrushSiz, cv.GC_FGD, -1)

        elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:
            cv.circle(self.img_show, (x, y), self.BrushSiz, self.RColor, -1)
            cv.circle(self.mask, (x, y), self.BrushSiz, cv.GC_BGD, -1)

        cv.imshow('Painting', self.img_show)

    def cutFunction(self):
        background = np.zeros((1, 65), np.float64)
        foreground = np.zeros((1, 65), np.float64)

        cv.grabCut(self.img, self.mask, None, background, foreground, 5, cv.GC_INIT_WITH_MASK)
        mask2 = np.where((self.mask == 2) | (self.mask == 0), 0, 1).astype('uint8')
        self.grabImg = self.img * mask2[:, :, np.newaxis]     # 마스크 적용된 최종 이미지
        cv.imshow('Scissoring', self.grabImg)

    def incFunction(self):
        self.BrushSiz = min(20, self.BrushSiz + 1)            # 브러시 최대 크기 제한

    def decFunction(self):
        self.BrushSiz = max(1, self.BrushSiz - 1)             # 브러시 최소 크기 제한

    def saveFunction(self):
        fname = QFileDialog.getSaveFileName(self, '파일 저장', './')
        cv.imwrite(fname[0], self.grabImg)                    # 최종 오림 이미지 저장

    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()

# 메인 루프 실행
app = QApplication(sys.argv)
win = Orim()
win.show()
app.exec_()
