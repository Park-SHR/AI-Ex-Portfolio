import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
import sys
import winsound

class TrafficWeak(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('교통약자 보호')              # 윈도우 제목
        self.setGeometry(200, 200, 700, 200)             # 위치 및 크기 설정

        # 버튼 생성
        signButton = QPushButton('표지판 등록', self)
        roadButton = QPushButton('도로 영상 불러옴', self)
        recognitionButton = QPushButton('인식', self)
        quitButton = QPushButton('나가기', self)
        self.label = QLabel('환영합니다!', self)

        # 버튼 위치 설정
        signButton.setGeometry(10, 10, 100, 30)
        roadButton.setGeometry(110, 10, 100, 30)
        recognitionButton.setGeometry(210, 10, 100, 30)
        quitButton.setGeometry(510, 10, 100, 30)
        self.label.setGeometry(10, 40, 600, 170)

        # 버튼 기능 연결
        signButton.clicked.connect(self.signFunction)
        roadButton.clicked.connect(self.roadFunction)
        recognitionButton.clicked.connect(self.recognitionFunction)
        quitButton.clicked.connect(self.quitFunction)

        # 표지판 이미지 파일명 및 이름
        self.signFiles = [['child.png', '어린이'], ['elder.png', '노인'], ['disabled.png', '장애인']]
        self.signImgs = []  # 실제 이미지 객체 저장용 리스트

    # 1. 표지판 이미지 등록
    def signFunction(self):
        self.label.clear()
        self.label.setText('교통약자 표지판을 등록합니다.')
        for fname, _ in self.signFiles:
            self.signImgs.append(cv.imread(fname))  # 이미지 읽기
            cv.imshow(fname, self.signImgs[-1])     # 이미지 표시

    # 2. 도로 영상 불러오기
    def roadFunction(self):
        if self.signImgs == []:
            self.label.setText('먼저 표지판을 등록하세요.')
        else:
            fname = QFileDialog.getOpenFileName(self, '파일 읽기', './')
            self.roadImg = cv.imread(fname[0])
            if self.roadImg is None:
                sys.exit('파일을 찾을 수 없습니다.')
            cv.imshow('Road scene', self.roadImg)

    # 3. 인식 기능 (SIFT 사용)
    def recognitionFunction(self):
        if self.roadImg is None:
            self.label.setText('먼저 도로 영상을 입력하세요.')
        else:
            sift = cv.SIFT_create()
            KD = []  # 키포인트/디스크립터 저장

            # 표지판 이미지의 특징 추출
            for img in self.signImgs:
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                KD.append(sift.detectAndCompute(gray, None))

            # 도로 영상 특징 추출
            grayRoad = cv.cvtColor(self.roadImg, cv.COLOR_BGR2GRAY)
            road_kp, road_des = sift.detectAndCompute(grayRoad, None)

            # 매칭 수행
            matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
            GM = []  # good match 목록
            for sign_kp, sign_des in KD:
                knn_match = matcher.knnMatch(sign_des, road_des, 2)
                T = 0.7
                good_match = []
                for nearest1, nearest2 in knn_match:
                    if nearest1.distance / nearest2.distance < T:
                        good_match.append(nearest1)
                GM.append(good_match)

            # 가장 매칭 수가 많은 표지판 선택
            best = GM.index(max(GM, key=len))

            # 결과 출력 및 경고음
            cv.imshow('Matches and Homography', self.roadImg)
            self.label.setText(self.signFiles[best][1] + ' 보호구역입니다. 30km로 서행하세요.')
            winsound.Beep(3000, 500)

    # 종료 버튼
    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()

# 메인 실행
app = QApplication(sys.argv)
win = TrafficWeak()
win.show()
app.exec_()
