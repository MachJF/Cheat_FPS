import numpy as np
from PIL import ImageGrab, ImageQt
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QComboBox, QLabel
from PyQt5.QtCore import *
# from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog
# from PyQt6.QtCore import *
import win32gui
import sys
from ultralyticsplus import YOLO, render_result
import os
import time


def init_detector():
    model = YOLO('keremberke/yolov8m-csgo-player-detection')
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    return model


def detect(model, img):
    results = model.predict(img)
    render = render_result(model=model, result=results[0], image=img)
    return render


def pil_recorder():
    im = ImageGrab.grab()
    width, height = im.size
    print(width, height)
    fourcc = cv2.VideoWriter_fourcc(*'xvid')
    fps = 15
    video = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))
    while True:
        im = ImageGrab.grab()
        img_np = np.array(im)
        frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        video.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    video.release()


def qt2mat(qt_image):
    qt_image = qt_image.convertToFormat(4)
    width = qt_image.width()
    height = qt_image.height()
    ptr = qt_image.constBits()
    ptr.setsize(qt_image.byteCount())
    arr = np.array(ptr).reshape(height, width, 4)  # Copies the data
    return arr


# 使用 PyQt5 录制屏幕
def qt_recorder():
    def get_all_hwnd():
        hwnd_title = dict()

        def callback(hwnd, mouse):
            if win32gui.IsWindow(hwnd) and win32gui.IsWindowEnabled(hwnd) and win32gui.IsWindowVisible(hwnd):
                hwnd_title.update({hwnd: win32gui.GetWindowText(hwnd)})

        win32gui.EnumWindows(callback, 0)
        titles = [t for _, t in hwnd_title.items()]
        return titles

    def get_all_windows():
        def callback(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd):
                if win32gui.GetWindowText(hwnd) != '':
                    titles.append(win32gui.GetWindowText(hwnd))

        titles = []
        win32gui.EnumWindows(callback, 0)
        return titles

    class Window(QWidget):
        def __init__(self):
            super().__init__()
            self.lbl = None
            self.window_name = None
            self.setWindowTitle('录制屏幕')
            self.resize(600, 600)
            self.btn = QPushButton('开始录制')
            self.init_combo()
            # self.combo.textActivated[str].connect(self.on_activated)
            self.btn.clicked.connect(self.start_record)
            layout = QVBoxLayout()
            layout.addWidget(self.btn)
            self.setLayout(layout)
            # self.fourcc = cv2.VideoWriter_fourcc(*'xvid')
            # self.fps = 15
            # self.width = 1920
            # self.height = 1080
            # self.video = cv2.VideoWriter('output.avi', self.fourcc, self.fps, (self.width, self.height))

        def init_combo(self):
            self.lbl = QLabel('请选择窗口', self)
            combo = QComboBox(self)
            combo.resize(450, 30)
            combo.addItem('请选择窗口')
            combo.addItems(get_all_windows())

            combo.move(50, 50)
            self.lbl.move(50, 25)

            combo.textActivated[str].connect(self.on_activated)

        def on_activated(self, text):
            self.window_name = text

        def start_record(self):
            self.btn.setText('录制中')
            self.btn.setEnabled(False)
            self.record(self.window_name)

        def record(self, window_name):
            hwnd = win32gui.FindWindow(None, window_name)
            screen = app.primaryScreen()
            img = screen.grabWindow(hwnd).toImage()
            model = init_detector()
            pil_img = ImageQt.fromqimage(img)
            # pil_img.show()
            result = model.predict(pil_img)
            print(result[0].boxes)
            while True:
                img = screen.grabWindow(hwnd).toImage()
                pil_img = ImageQt.fromqimage(img)
                result = detect(model, pil_img)
                img_np = np.array(result)
                frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)                # img = qt2mat(img)
                # frame = img
                # self.video.write(frame)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    sys.exit()
                # QTimer.singleShot(1, self.record)

    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    # pil_recorder()
    qt_recorder()
