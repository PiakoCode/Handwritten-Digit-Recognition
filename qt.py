import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QGridLayout
from PyQt5.QtGui import QImage, QPainter, QPen, QPixmap
from PyQt5.QtCore import Qt, QPoint
import torch
from ResNet18 import ResNet, Block
import torchvision.transforms as transforms

class DrawingWidget(QWidget):
    def __init__(self, parent=None):
        super(DrawingWidget, self).__init__(parent)
        self.setFixedSize(280, 280)
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.black)
        self.drawing = False
        self.lastPoint = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.white, 20, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())

    def clearImage(self):
        self.image.fill(Qt.black)
        self.update()

    def getImage(self):
        return self.image

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.drawing_widget = DrawingWidget()
        self.predict_button = QPushButton("Predict")
        self.clear_button = QPushButton("Clear")
        self.result_label = QLabel("识别为: ")
        self.result_label.setStyleSheet("font-size: 20px;")

        self.predict_button.clicked.connect(self.predict)
        self.clear_button.clicked.connect(self.drawing_widget.clearImage)

        layout = QVBoxLayout()
        layout.addWidget(self.drawing_widget)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.result_label)
        self.setLayout(layout)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load("model.pth")
        self.model.to(self.device)
        
        self.model.eval()

    def predict(self):
        qimage = self.drawing_widget.getImage()
        image = qimage.convertToFormat(QImage.Format_Grayscale8)
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.frombuffer(ptr, np.uint8).reshape((image.height(), image.width()))

        arr = arr.astype(np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        arr = np.expand_dims(arr, axis=0)
        
        
        tensor = torch.from_numpy(arr)
        tensor =  transforms.Resize((28,28))(tensor)
        
        with torch.no_grad():
            
            tensor = tensor.to(self.device)
            output = self.model(tensor)
            
            
            pred = output.argmax(dim=1, keepdim=True).item()
            self.result_label.setText(f"识别为: {pred}")

app = QApplication(sys.argv)
main_window = MainWindow()
main_window.show()
sys.exit(app.exec_())