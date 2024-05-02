import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit
from transformers import pipeline

# Pyqt app
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.classifier = pipeline("sentiment-analysis")
        self.setWindowTitle("Sentiment Analysis App")

        self.text_input = QLineEdit()
        self.predict_button = QPushButton("Predict")
        self.result_label = QLabel("")

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Enter text:"))
        layout.addWidget(self.text_input)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

        self.predict_button.clicked.connect(self.predict)

    def predict(self):
        text = self.text_input.text()
        cls_result = self.classifier(text)[0]
        label = f"This sentence is {cls_result['label'].lower()} with score {round(cls_result['score'] * 100, 2)}%"
        self.result_label.setText(label)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
