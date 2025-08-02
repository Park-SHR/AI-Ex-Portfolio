import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Rescaling
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import pathlib
import pickle
import threading
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class DogBreedTrainerApp:
    def __init__(self, master):
        self.master = master
        master.title("Dog Breed Classifier Trainer")
        self.dataset_path = None

        # UI
        tk.Label(master, text="데이터셋 폴더 선택:").pack(anchor='w')
        tk.Button(master, text="폴더 열기", command=self.select_folder).pack(anchor='w')

        tk.Label(master, text="에폭 수 입력:").pack(anchor='w')
        self.epoch_entry = tk.Entry(master)
        self.epoch_entry.insert(0, "50")
        self.epoch_entry.pack(anchor='w')

        tk.Label(master, text="배치 사이즈 입력:").pack(anchor='w')
        self.batch_entry = tk.Entry(master)
        self.batch_entry.insert(0, "16")
        self.batch_entry.pack(anchor='w')

        tk.Button(master, text="학습 시작", command=self.start_training).pack(anchor='w')

        self.progress = ttk.Progressbar(master, length=500, mode='determinate')
        self.progress.pack(anchor='w', pady=5)

        tk.Label(master, text="학습 로그:").pack(anchor='w')
        self.log_box = scrolledtext.ScrolledText(master, width=90, height=10)
        self.log_box.pack(anchor='w')

        self.graph_frame = tk.Frame(master)
        self.graph_frame.pack(fill=tk.BOTH, expand=True)

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.dataset_path = pathlib.Path(folder_path)
            self.log(f"선택된 폴더: {self.dataset_path}")

    def log(self, message):
        self.log_box.insert(tk.END, message + "\n")
        self.log_box.see(tk.END)
        self.master.update()

    def start_training(self):
        if not self.dataset_path:
            messagebox.showerror("오류", "데이터셋 폴더를 먼저 선택하세요.")
            return
        try:
            epochs = int(self.epoch_entry.get())
            batch_size = int(self.batch_entry.get())
            if epochs <= 0 or batch_size <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("오류", "유효한 정수를 입력하세요.")
            return

        self.progress['value'] = 0
        self.progress['maximum'] = epochs
        threading.Thread(target=self.train_model, args=(epochs, batch_size), daemon=True).start()

    def train_model(self, epochs, batch_size):
        self.log("TensorFlow 버전: " + tf.__version__)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.log("GPU 메모리 성장 허용됨.")
            except RuntimeError as e:
                self.log("GPU 설정 오류: " + str(e))

        self.log("데이터셋 로딩 중...")
        original_train_ds = image_dataset_from_directory(
            self.dataset_path,
            validation_split=0.2,
            subset='training',
            seed=123,
            image_size=(224, 224),
            batch_size=batch_size,
            label_mode='int'  # ✅ 정확도 문제 해결 핵심
        )
        test_ds = image_dataset_from_directory(
            self.dataset_path,
            validation_split=0.2,
            subset='validation',
            seed=123,
            image_size=(224, 224),
            batch_size=batch_size,
            label_mode='int'  # ✅ 여기에도 반드시 포함
        )

        class_names = original_train_ds.class_names
        train_ds = original_train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        model = Sequential([
            Rescaling(1./255),
            DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
            GlobalAveragePooling2D(),
            Dense(1024, activation='relu'),
            Dropout(0.5),
            Dense(len(class_names), activation='softmax')
        ])

        model.compile(optimizer=Adam(1e-6), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        class CustomCallback(tf.keras.callbacks.Callback):
            def __init__(self, app, epochs):
                super().__init__()
                self.app = app
                self.epochs = epochs

            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start = time.time()

            def on_epoch_end(self, epoch, logs=None):
                elapsed = time.time() - self.epoch_start
                steps = self.params.get('steps', 1)
                step_time = elapsed / steps * 1000
                self.app.progress['value'] = epoch + 1
                self.app.log(
                    f"Epoch {epoch+1}/{self.epochs} - {int(elapsed)}s - "
                    f"loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} - "
                    f"val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f} - "
                    f"{int(elapsed)}s/epoch - {int(step_time)}ms/step"
                )

        self.log(f"학습 시작... 에폭: {epochs}, 배치 사이즈: {batch_size}")
        history = model.fit(train_ds, epochs=epochs, validation_data=test_ds,
                            callbacks=[CustomCallback(self, epochs)], verbose=0)

        acc = model.evaluate(test_ds, verbose=0)[1] * 100
        self.log(f"최종 테스트 정확도: {acc:.2f}%")

        model.save("2025254017.h5")
        with open("dog_species_name.txt", "w", encoding="utf-8") as f:
            for name in class_names:
                f.write(name + "\n")
        self.log("모델 및 클래스 정보 저장 완료.")

        self.show_graphs(history)

    def show_graphs(self, history):
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        fig = Figure(figsize=(9, 4), dpi=100)
        acc_plot = fig.add_subplot(1, 2, 1)
        acc_plot.plot(history.history['accuracy'], label='Train')
        acc_plot.plot(history.history['val_accuracy'], label='Validation')
        acc_plot.set_title('Accuracy Graph')
        acc_plot.set_xlabel('Epoch')
        acc_plot.set_ylabel('Accuracy')
        acc_plot.grid()
        acc_plot.legend()

        loss_plot = fig.add_subplot(1, 2, 2)
        loss_plot.plot(history.history['loss'], label='Train')
        loss_plot.plot(history.history['val_loss'], label='Validation')
        loss_plot.set_title('Loss Graph')
        loss_plot.set_xlabel('Epoch')
        loss_plot.set_ylabel('Loss')
        loss_plot.grid()
        loss_plot.legend()

        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == '__main__':
    root = tk.Tk()
    app = DogBreedTrainerApp(root)
    root.mainloop()