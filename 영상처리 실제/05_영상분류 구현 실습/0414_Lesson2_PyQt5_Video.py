from PyQt5.QtWidgets import *  # PyQt5 위젯 임포트
import sys                    # 시스템 인자용
import cv2 as cv              # OpenCV 라이브러리 임포트

class Video(QMainWindow):     # QMainWindow 상속
    def __init__(self):
        super().__init__()  # 부모 생성자 호출

        self.setWindowTitle('비디오에서 프레임 수집')  # 윈도우 제목 설정
        self.setGeometry(200, 200, 500, 100)          # 위치와 크기 설정

        # 버튼 생성
        videoButton = QPushButton('비디오 켜기', self)
        captureButton = QPushButton('프레임 잡기', self)
        saveButton = QPushButton('프레임 저장', self)
        quitButton = QPushButton('나가기', self)

        # 버튼 위치 지정
        videoButton.setGeometry(10, 10, 100, 30)
        captureButton.setGeometry(110, 10, 100, 30)
        saveButton.setGeometry(210, 10, 100, 30)
        quitButton.setGeometry(310, 10, 100, 30)

        # 콜백 함수 연결
        videoButton.clicked.connect(self.videoFunction)
        captureButton.clicked.connect(self.captureFunction)
        saveButton.clicked.connect(self.saveFunction)
        quitButton.clicked.connect(self.quitFunction)

    def videoFunction(self):
        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)  # 카메라 연결 시도
        if not self.cap.isOpened():  # 연결 실패 시
            self.close()

        while True:
            ret, self.frame = self.cap.read()  # 프레임 읽기
            if not ret:
                break  # 읽기 실패 시 종료
            cv.imshow('video display', self.frame)  # 영상 출력
            cv.waitKey(1)  # 빠르게 다음 프레임으로 넘어감

    def captureFunction(self):
        self.capturedFrame = self.frame  # 현재 프레임을 저장
        cv.imshow('Captured Frame', self.capturedFrame)  # 캡처된 프레임 보여줌

    def saveFunction(self):
        fname = QFileDialog.getSaveFileName(self, '파일 저장', './')  # 저장할 파일 경로 선택
        cv.imwrite(fname[0], self.capturedFrame)  # 이미지 파일로 저장

    def quitFunction(self):
        self.cap.release()  # 카메라 해제
        cv.destroyAllWindows()  # 모든 OpenCV 창 닫기
        self.close()  # 프로그램 종료

# 프로그램 실행
app = QApplication(sys.argv)  # QApplication 객체 생성
win = Video()                 # Video 클래스 인스턴스 생성
win.show()                    # 윈도우 보이기
app.exec_()                   # 이벤트 루프 실행
