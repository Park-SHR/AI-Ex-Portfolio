from PyQt5.QtWidgets import *  # PyQt5의 모든 위젯 클래스 임포트
import sys                    # 시스템 인자를 받기 위한 sys 모듈
import winsound              # 윈도우에서 삑(Beep) 소리 내기 위한 모듈

# MainWindow를 상속한 BeepSound 클래스 정의
class BeepSound(QMainWindow):
    def __init__(self):  # 생성자 함수
        super().__init__()  # 부모 클래스 초기화

        self.setWindowTitle('삑 소리 내기')  # 윈도우 제목 설정
        self.setGeometry(200, 200, 500, 100)  # 윈도우 위치(x, y) 및 크기(width, height) 설정

        # 버튼 생성
        shortBeepButton = QPushButton('짧게 삑', self)
        longBeepButton = QPushButton('길게 삑', self)
        quitButton = QPushButton('나가기', self)

        self.label = QLabel('환영합니다!', self)  # 라벨 생성

        # 버튼 위치 및 크기 설정 (x, y, width, height)
        shortBeepButton.setGeometry(10, 10, 100, 30)
        longBeepButton.setGeometry(110, 10, 100, 30)
        quitButton.setGeometry(210, 10, 100, 30)
        self.label.setGeometry(10, 40, 500, 70)

        # 버튼 클릭 시 연결될 함수(콜백 함수) 지정
        shortBeepButton.clicked.connect(self.shortBeepFunction)
        longBeepButton.clicked.connect(self.longBeepFunction)
        quitButton.clicked.connect(self.quitFunction)

    # 짧게 삑 함수 정의
    def shortBeepFunction(self):
        self.label.setText('주파수 1000으로 0.5초 동안 삑 소리를 냅니다.')  # 라벨 텍스트 변경
        winsound.Beep(1000, 500)  # 1000Hz, 0.5초 동안 소리 발생

    # 길게 삑 함수 정의
    def longBeepFunction(self):
        self.label.setText('주파수 1000으로 3초 동안 삑 소리를 냅니다.')
        winsound.Beep(1000, 3000)  # 1000Hz, 3초 동안 소리 발생

    # 종료 함수 정의
    def quitFunction(self):
        self.close()  # 윈도우 닫기

# 애플리케이션 실행 코드
app = QApplication(sys.argv)  # QApplication 객체 생성
win = BeepSound()             # BeepSound 클래스 객체 생성
win.show()                    # 윈도우 보이기
app.exec_()                   # 이벤트 루프 실행
