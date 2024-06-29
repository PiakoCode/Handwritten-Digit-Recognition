import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QGridLayout
from PyQt5.QtGui import QImage, QPainter, QPen, QPixmap
from PyQt5.QtCore import Qt, QPoint
import torch
from model import ResNet, Block, ResNet_with_Smaller
import torchvision.transforms as transforms
import time
from PyQt5.QtWidgets import QMessageBox
from torch.nn.functional import softmax

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
        self.device_label = QLabel("Device: ")
        self.result_label = QLabel("")
        self.prediction_time_label = QLabel("识别时间: 0ms")  
        self.correct_predictions = 0  
        self.total_predictions = 0  
        self.incorrect_predictions = 0
        self.total_prediction_time = 0.0
        self.accuracy_label = QLabel("正确率: N/A")  
        self.total_predictions_label = QLabel("总识别次数: 0")
        self.incorrect_predictions_label = QLabel("错误识别次数: 0")
        self.average_prediction_time_label = QLabel("平均识别时间: 0.0秒")

        self.predict_button.clicked.connect(self.predict)
        self.clear_button.clicked.connect(self.drawing_widget.clearImage)

        layout = QVBoxLayout()
        layout.addWidget(self.drawing_widget)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.device_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.prediction_time_label) 
        layout.addWidget(self.average_prediction_time_label)
        layout.addWidget(self.incorrect_predictions_label)
        layout.addWidget(self.total_predictions_label)
        layout.addWidget(self.accuracy_label) 

        self.setLayout(layout)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_label.setText(f"Device: {self.device}")
        self.model = ResNet_with_Smaller()
        self.model.load_state_dict(torch.load("ResNet_with_Smaller_mnist.pt"))

        self.model.to(self.device)

        self.model.eval()

    def predict(self):
        self.total_predictions += 1
        self.total_predictions_label.setText(f"总识别次数: {self.total_predictions}")
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

        # * MUST NORMALIZE THE IMAGE AS THE TRAINING DATA WAS NORMALIZED
        tensor = transforms.Normalize((0.1307,), (0.3081,))(tensor)

        with torch.no_grad():

            start_time = time.time()  
            tensor = tensor.to(self.device)
            output = self.model(tensor)
            print(output)
            end_time = time.time()  

            probabilities = softmax(output, dim=1)

            topk_probs, topk_inds = probabilities.topk(5, dim=1)

            topk_probs = topk_probs.squeeze().tolist()  
            topk_inds = topk_inds.squeeze().tolist()

            topk_results_str = "\n".join([f"{idx}: {prob*100:.2f}%" for idx, prob in zip(topk_inds, topk_probs)])
            self.result_label.setText(f"Top 5 识别结果:\n{topk_results_str}")

            pred = topk_inds[0] 

            prediction_time = (end_time - start_time) * 1000 
            self.prediction_time_label.setText(f"识别时间: {prediction_time:.2f}ms") 

            self.total_prediction_time += prediction_time
            average_prediction_time = self.total_prediction_time / self.total_predictions
            self.average_prediction_time_label.setText(f"平均识别时间: {average_prediction_time:.2f}ms")
        # 弹窗询问是否识别正确
        reply = QMessageBox.question(
            self,
            "识别结果确认",
            f"识别结果是: {pred}\n识别是否正确？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.correct_predictions += 1 
        else:
            self.incorrect_predictions += 1
            self.incorrect_predictions_label.setText(f"错误识别次数: {self.incorrect_predictions}")

        accuracy = (self.correct_predictions / self.total_predictions) * 100
        self.accuracy_label.setText(f"正确率: {accuracy:.2f}%")

app = QApplication(sys.argv)
main_window = MainWindow()
main_window.show()
sys.exit(app.exec_())
