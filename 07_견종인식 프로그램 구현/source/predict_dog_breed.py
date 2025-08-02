import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import chardet
import os
import sys

# ✅ stdout/stderr None 방지 처리
if sys.stdout is None or sys.stderr is None:
    class DummyStream:
        def write(self, *args, **kwargs): pass
        def flush(self): pass
    sys.stdout = DummyStream()
    sys.stderr = DummyStream()

def select_model_file():
    global model_path
    model_path = filedialog.askopenfilename(title="모델(.h5) 파일 선택", filetypes=[("H5 files", "*.h5")])
    if model_path:
        model_label.config(text=os.path.basename(model_path))

def select_class_file():
    global class_txt_path
    class_txt_path = filedialog.askopenfilename(title="클래스 텍스트(.txt) 파일 선택", filetypes=[("Text files", "*.txt")])
    if class_txt_path:
        class_label.config(text=os.path.basename(class_txt_path))

def select_image_file():
    global input_img_path
    input_img_path = filedialog.askopenfilename(title="입력 이미지 선택", filetypes=[("Image files", "*.jpg *.png")])
    if input_img_path:
        image_label.config(text=os.path.basename(input_img_path))

def run_prediction():
    try:
        if not model_path or not class_txt_path or not input_img_path:
            messagebox.showerror("오류", "모든 파일을 선택해야 합니다.")
            return

        output_img_path = "2025254017.png"

        # 🔍 인코딩 자동 감지 후 클래스 파일 읽기
        with open(class_txt_path, 'rb') as f:
            raw = f.read()
            enc = chardet.detect(raw)['encoding']

        with open(class_txt_path, 'r', encoding=enc) as f:
            class_names = [line.strip() for line in f.readlines()]

        # 모델 로드
        model = load_model(model_path)

        # 이미지 전처리
        img = image.load_img(input_img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 예측
        pred = model.predict(img_array)[0]
        # 클래스 수 확인
        if len(pred) != len(class_names):
            raise ValueError(f"❗ 모델 출력({len(pred)}) ≠ 클래스 수({len(class_names)}) - class_txt를 확인하세요.")
        top_indices = pred.argsort()[-5:][::-1]
        top_classes = [(class_names[i], pred[i]) for i in top_indices]

        # 원본 이미지 로드
        orig = cv2.imread(input_img_path)
        y = 30
        for (cls, prob) in top_classes:
            text = f"({prob:.8f}){cls}"
            cv2.putText(orig, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
            y += 30

        # 결과 저장
        cv2.imwrite(output_img_path, orig)
        # 이미지 미리보기
        cv2.imshow("예측 결과 미리보기", orig)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        messagebox.showinfo("완료", f"예측 결과가 {output_img_path}로 저장되었습니다.")

    except Exception as e:
        messagebox.showerror("예외 발생", str(e))

# 🔧 GUI 창 구성
root = tk.Tk()
root.title("견종 인식 프로그램")
root.geometry("500x300")

model_path = class_txt_path = input_img_path = None

tk.Label(root, text="1. 모델(.h5) 파일 선택").pack(anchor='w', padx=10)
tk.Button(root, text="모델 선택", command=select_model_file).pack(anchor='w', padx=10)
model_label = tk.Label(root, text="선택된 파일 없음")
model_label.pack(anchor='w', padx=10)

tk.Label(root, text="2. 클래스 텍스트(.txt) 파일 선택").pack(anchor='w', padx=10, pady=(10,0))
tk.Button(root, text="클래스 파일 선택", command=select_class_file).pack(anchor='w', padx=10)
class_label = tk.Label(root, text="선택된 파일 없음")
class_label.pack(anchor='w', padx=10)

tk.Label(root, text="3. 입력 이미지 파일 선택").pack(anchor='w', padx=10, pady=(10,0))
tk.Button(root, text="이미지 선택", command=select_image_file).pack(anchor='w', padx=10)
image_label = tk.Label(root, text="선택된 파일 없음")
image_label.pack(anchor='w', padx=10)

tk.Button(root, text="견종 인식 실행", command=run_prediction, bg="#4CAF50", fg="white").pack(pady=20)

root.mainloop()