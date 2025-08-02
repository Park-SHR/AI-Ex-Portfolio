import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image, ImageTk

IMAGE_SIZE = 224  # 캔버스 크기
MODEL_INPUT_SIZE = 28  # MNIST 모델 입력 크기

import os
import sys

def get_executable_path():
    # .exe로 실행될 때는 _MEIPASS 경로 사용, 아니면 현재 디렉토리
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(__file__)

def preprocess_for_prediction(img):
    img = img.copy()

    # 흰 배경일 경우 숫자를 반전시켜 검정 배경 + 흰 숫자 구조로 (MNIST 스타일)
    if np.mean(img) > 127:
        img = 255 - img

    # 이진화 및 외곽 추출
    _, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)

    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = img[y:y + h, x:x + w]


        MIN_SIZE = 20
        if w < MIN_SIZE or h < MIN_SIZE:
            padded = np.zeros((MIN_SIZE, MIN_SIZE), dtype=np.uint8)
            pad_y = (MIN_SIZE - h) // 2
            pad_x = (MIN_SIZE - w) // 2
            padded[pad_y:pad_y+h, pad_x:pad_x+w] = cropped
            square_size = MIN_SIZE
        else:
            square_size = max(h, w)
            padded = np.zeros((square_size, square_size), dtype=np.uint8)
            pad_y = (square_size - h) // 2
            pad_x = (square_size - w) // 2
            padded[pad_y:pad_y+h, pad_x:pad_x+w] = cropped
    else:
        padded = img

    resized = cv2.resize(padded, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    return resized


class DigitRecognizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Digit Recognizer")

        self.model = None
        self.canvas_image = None
        self.drawing = False
        self.last_x, self.last_y = None, None

        tk.Button(master, text="모델 불러오기 (.h5)", command=self.load_model_file).pack(anchor='w')
        tk.Button(master, text="이미지 불러오기 (.png)", command=self.load_image_file).pack(anchor='w')
        tk.Button(master, text="초기화 (그림 및 이미지)", command=self.clear_canvas).pack(anchor='w')
        tk.Button(master, text="숫자 예측", command=self.predict_digit).pack(anchor='w')

        self.canvas = tk.Canvas(master, width=IMAGE_SIZE, height=IMAGE_SIZE, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        tk.Label(master, text="로그:").pack(anchor='w')
        self.log_box = scrolledtext.ScrolledText(master, width=70, height=10)
        self.log_box.pack(anchor='w')

        self.image_draw_buffer = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    def log(self, message):
        self.log_box.insert(tk.END, message + "\n")
        self.log_box.see(tk.END)
        self.master.update()

    def load_model_file(self):
        model_path = filedialog.askopenfilename(filetypes=[("H5 파일", "*.h5")])
        if model_path:
            try:
                self.model = load_model(model_path)
                self.log(f"✅ 모델 로드 완료: {model_path}")
            except Exception as e:
                self.log(f"❌ 모델 로드 실패: {e}")

    def load_image_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("PNG 파일", "*.png")])
        if file_path:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.image_draw_buffer = img.copy()
            self.show_image_on_canvas(img)
            self.log(f"📂 이미지 불러오기 완료: {file_path}")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image_draw_buffer = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        self.canvas_image = None
        self.log("🧼 캔버스 및 이미지 초기화 완료")

    def draw(self, event):
        x, y = event.x, event.y
        r = 6
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')
        if self.last_x is not None and self.last_y is not None:
            cv2.line(self.image_draw_buffer, (self.last_x, self.last_y), (x, y), 255, thickness=6)
        self.last_x, self.last_y = x, y

    def reset(self, event):
        self.last_x, self.last_y = None, None

    def predict_digit(self):
        if self.model is None:
            messagebox.showerror("오류", "먼저 모델을 불러오세요.")
            return

        try:
            processed = preprocess_for_prediction(self.image_draw_buffer)
            img_input = processed.astype("float32") / 255.0
            img_input = img_input.reshape(1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 1)
            prediction = self.model.predict(img_input, verbose=0)
            result = int(np.argmax(prediction[0]))

            save_path = os.path.join(get_executable_path(), "2025254017.txt")
            self.log(f"📄 저장 시도 경로: {save_path}")
        
            # 🔎 저장 테스트
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(str(result))

            self.show_image_on_canvas(self.image_draw_buffer)
            self.log(f"✅ 예측 결과: {result} → 저장 완료")
    
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.log(f"❌ 예측 실패: {e}\n{tb}")

    def show_image_on_canvas(self, img_array):
        img = Image.fromarray(img_array).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
        self.canvas_image = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.canvas_image)


if __name__ == '__main__':
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
