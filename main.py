from fastapi import FastAPI
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from pydantic import BaseModel

# Создаем экземпляр FastAPI
app = FastAPI()

# Загружаем модели для анализа настроений и машинного перевода текста
classifier = pipeline("sentiment-analysis")
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru")

# Определяем базовую модель данных с помощью Pydantic
class Item(BaseModel):
    text: str

# Объявляем обработчик корневого маршрута
@app.get("/")
def root():
    return {"message": "Hello World"}

# Объявляем обработчик маршрута '/predict/', который принимает объект item и предсказывает настроения текста
@app.post("/predict/")
def predict(item: Item):
    return classifier(item.text)[0]

# Объявляем обработчик маршрута '/translate/', который принимает объект item и выполняет машинный перевод текста
@app.post("/translate/")
def translate_text(item: Item):
    inputs = tokenizer(item.text, return_tensors="pt")
    outputs = model.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"translated_text": translated_text}
