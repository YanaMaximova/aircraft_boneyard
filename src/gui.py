from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QRadioButton, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import numpy as np
from image_processing import *

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.image_path = False
        self.bin_image = None
        self.image = None
        self.morph_image = None
        self.label_image = None

        self.setWindowTitle("Planes recognition")
        #self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.count_label = QLabel("Количество объектов: 0")
        self.layout.addWidget(self.count_label)

        self.load_layout = QVBoxLayout()  # Макет для кнопки "Загрузить изображение"
        self.radio_layout = QHBoxLayout()

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)


        self.load_button = QPushButton("Загрузить изображение")
        self.load_button.clicked.connect(self.load_image)
        self.load_layout.addWidget(self.load_button)

        self.raw_button = QRadioButton("Raw")
        self.raw_button.setChecked(True)
        self.raw_button.clicked.connect(self.show_raw_image)
        self.radio_layout.addWidget(self.raw_button)

        self.bin_button = QRadioButton("Bin")
        self.bin_button.clicked.connect(self.show_bin_image)
        self.radio_layout.addWidget(self.bin_button)

        self.morph_button = QRadioButton("Morth")
        self.morph_button.clicked.connect(self.show_morph_image)
        self.radio_layout.addWidget(self.morph_button)

        self.label_button = QRadioButton("Labeled")
        self.label_button.clicked.connect(self.show_labeled_image)
        self.radio_layout.addWidget(self.label_button)

        self.layout.addLayout(self.load_layout)
        self.layout.addLayout(self.radio_layout)

        self.load_layout.setAlignment(Qt.AlignLeft)
        self.radio_layout.setAlignment(Qt.AlignLeft)

        self.layout.setStretchFactor(self.load_layout, 1)
        self.layout.setStretchFactor(self.radio_layout, 4)
    def load_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if filename:
            self.image_path = filename
            pixmap = QPixmap(filename)
            self.image = cv2.imread(self.image_path)
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.bin_image = None
            self.morph_image = None
            self.label_image = None
            self.show_image(img)

    def show_raw_image(self):
        if self.image_path:
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.show_image(img)

    def show_bin_image(self):
        if self.image_path:
            self.bin_image = binarization(self.image)
            self.show_image(self.bin_image)

    def show_morph_image(self):
        if self.image_path:
            if self.bin_image is None:
                self.bin_image = binarization(self.image)
            self.morph_image = morphological(self.bin_image)
            self.show_image(self.morph_image)

    def show_labeled_image(self):
        if self.image_path:
            if self.morph_image is None:
                if self.bin_image is None:
                    self.bin_image = binarization(self.image)
                self.morph_image = morphological(self.bin_image)
            image_copy = self.image.copy()
            morth_copy = self.morph_image.copy()
            self.label_image, contour_count = find_and_draw_connected_components(image_copy, morth_copy)
            self.count_label.setText(f"Количество объектов: {contour_count}")
            img = cv2.cvtColor(self.label_image, cv2.COLOR_BGR2RGB)
            self.show_image(img)


    def show_image(self, image):
        if len(image.shape) == 2:
            qImg = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], QImage.Format_Grayscale8))
        elif len(image.shape) == 3:
            height, width, channel = image.shape
            bytesPerLine = 3 * width
            qImg = QPixmap.fromImage(QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888))
        else:
            raise ValueError("Unsupported image shape")

        self.image_label.setPixmap(qImg)

def opening(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def closing(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def dilatation(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

