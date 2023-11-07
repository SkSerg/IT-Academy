import sys
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget, QLabel
from PySide6.QtGui import QImage, QPixmap, QPainter
from PySide6.QtCore import QTimer, QDateTime

class App(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.cap = cv2.VideoCapture(0)
        self.net = cv2.dnn.readNetFromONNX('MobileNet_2_1.onnx')

        self.processing = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def initUI(self):
        layout = QVBoxLayout()
        
        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        self.camera_btn = QPushButton("Camera", self)
        self.camera_btn.clicked.connect(self.toggle_camera)
        layout.addWidget(self.camera_btn)

        self.work_btn = QPushButton("Work", self)
        self.work_btn.clicked.connect(self.toggle_work)
        layout.addWidget(self.work_btn)

        self.stop_btn = QPushButton("Stop", self)
        self.stop_btn.clicked.connect(self.stop)
        layout.addWidget(self.stop_btn)

        self.snapshot_left_btn = QPushButton("Snapshot_left", self)
        self.snapshot_left_btn.clicked.connect(lambda: self.take_snapshot('left'))
        layout.addWidget(self.snapshot_left_btn)

        self.snapshot_right_btn = QPushButton("Snapshot_right", self)
        self.snapshot_right_btn.clicked.connect(lambda: self.take_snapshot('right'))
        layout.addWidget(self.snapshot_right_btn)

        self.snapshot_center_btn = QPushButton("Snapshot_center", self)
        self.snapshot_center_btn.clicked.connect(lambda: self.take_snapshot('center'))
        layout.addWidget(self.snapshot_center_btn)

        self.setLayout(layout)

    def toggle_camera(self):
        if self.timer.isActive():
            self.timer.stop()
            self.cap.release()
            self.camera_btn.setText("Camera")
        else:
            self.cap.open(0)
            self.timer.start(30)
            self.camera_btn.setText("Stop Camera")

    def toggle_work(self):
        self.processing = not self.processing

    def stop(self):
        self.timer.stop()
        self.cap.release()
        self.camera_btn.setText("Camera")

    def update_frame(self):
        ret, frame = self.cap.read()
        if self.processing:
            # Resize frame to model input size
            resized = cv2.resize(frame, (224, 224))
            blob = cv2.dnn.blobFromImage(resized, 1.0, (224, 224), (127.5, 127.5, 127.5), swapRB=True, crop=False)
            self.net.setInput(blob)
            preds = self.net.forward()
            class_id = np.argmax(preds)
            cv2.putText(frame, str(class_id), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        self.show_image(frame)

    def show_image(self, frame):
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)

    def take_snapshot(self, position):
        ret, frame = self.cap.read()
        timestamp = QDateTime.currentDateTime().toString('yyyyMMddHHmmss')
        filename = f'img/{position}/{timestamp}.jpg'
        cv2.imwrite(filename, frame)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec())
