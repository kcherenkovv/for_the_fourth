from fastapi.testclient import TestClient
from main import app

# Создаем клиент для тестирования FastAPI приложения
client = TestClient(app)

# Тест для проверки корневого маршрута
def test_read_main():
    # Отправляем GET запрос на корневой маршрут
    response = client.get("/")
    # Проверяем, что статус код ответа равен 200 (ОК)
    assert response.status_code == 200
    # Проверяем, что возвращаемые данные соответствуют ожидаемому словарю
    assert response.json() == {"message": "Hello World"}

# Тест для проверки предсказания позитивного настроения
def test_predict_positive():
    # Отправляем POST запрос с текстом, который вызывает позитивное настроение
    response = client.post("/predict/", json={"text": "I like machine learning!"})
    json_data = response.json()
    # Проверяем, что статус код ответа равен 200 (ОК)
    assert response.status_code == 200
    # Проверяем, что предсказанное настроение соответствует 'POSITIVE'
    assert json_data['label'] == 'POSITIVE'

# Тест для проверки предсказания негативного настроения
def test_predict_negative():
    # Отправляем POST запрос с текстом, который вызывает негативное настроение
    response = client.post("/predict/", json={"text": "I hate machine learning!"})
    json_data = response.json()
    # Проверяем, что статус код ответа равен 200 (ОК)
    assert response.status_code == 200
    # Проверяем, что предсказанное настроение соответствует 'NEGATIVE'
    assert json_data['label'] == 'NEGATIVE'
