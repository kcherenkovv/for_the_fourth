import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit
from transformers import pipeline

# Основное окно PyQt приложения
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        # Инициализация модели для анализа настроений
        self.classifier = pipeline("sentiment-analysis")
        self.setWindowTitle("Приложение для анализа настроений")

        # Виджеты для пользовательского ввода и отображения результата
        self.text_input = QLineEdit()
        self.predict_button = QPushButton("Предсказать")
        self.result_label = QLabel("")

        # Создание вертикального макета и добавление виджетов
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Введите текст:"))
        layout.addWidget(self.text_input)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

        # Привязка события нажатия кнопки к методу predict
        self.predict_button.clicked.connect(self.predict)

    # Метод для предсказания настроения текста и обновления метки с результатом
    def predict(self):
        text = self.text_input.text()
        cls_result = self.classifier(text)[0]
        label = f"Это предложение {cls_result['label'].lower()} с вероятностью {round(cls_result['score'] * 100, 2)}%"
        self.result_label.setText(label)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
