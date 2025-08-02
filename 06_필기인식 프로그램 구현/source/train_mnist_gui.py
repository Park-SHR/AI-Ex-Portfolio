import tkinter as tk
from tkinter import messagebox, scrolledtext
import threading
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback

class TrainMNISTApp:
    def __init__(self, master):
        self.master = master
        master.title("MNIST 학습 프로그램")

        # 에폭 입력
        tk.Label(master, text="에폭 수 입력:").pack(anchor='w')
        self.epoch_entry = tk.Entry(master)
        self.epoch_entry.insert(0, "5")
        self.epoch_entry.pack(anchor='w')

        # 학습 버튼
        tk.Button(master, text="학습 시작", command=self.start_training).pack(anchor='w', pady=5)

        # 로그 창
        tk.Label(master, text="로그 출력:").pack(anchor='w')
        self.log_box = scrolledtext.ScrolledText(master, width=80, height=15)
        self.log_box.pack(anchor='w')

    def log(self, message):
        self.log_box.insert(tk.END, message + "\n")
        self.log_box.see(tk.END)
        self.master.update()

    def start_training(self):
        try:
            epochs = int(self.epoch_entry.get())
        except ValueError:
            messagebox.showerror("입력 오류", "유효한 에폭 수를 입력하세요.")
            return

        threading.Thread(target=self.train_model, args=(epochs,), daemon=True).start()

    def train_model(self, epochs):
        self.log("MNIST 데이터 로딩 중...")
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        self.log("CNN 모델 구성 중...")
        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
            MaxPooling2D((2,2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        class LoggerCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                acc = logs.get("accuracy", 0)
                val_acc = logs.get("val_accuracy", 0)
                loss = logs.get("loss", 0)
                val_loss = logs.get("val_loss", 0)
                self.model.app.log(f"Epoch {epoch+1}/{epochs} - loss: {loss:.4f}, acc: {acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

        model.app = self

        self.log("모델 학습 시작...")
        model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), callbacks=[LoggerCallback()], verbose=0)

        model.save("2025254017.h5")
        self.log("✅ 학습 완료! 모델이 '2025254017.h5'로 저장되었습니다.")

if __name__ == '__main__':
    root = tk.Tk()
    app = TrainMNISTApp(root)
    root.mainloop()
