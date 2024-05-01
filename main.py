from fastapi import FastAPI
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from pydantic import BaseModel


class Item(BaseModel):
    text: str

# Cоздаем веб-приложение с использованием FastAPI. Загружаем модели для выполнения задач 
# анализа настроений, токенизации и машинного перевода текста с английского на русский.
app = FastAPI()
classifier = pipeline("sentiment-analysis")
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru")

# Объявляем обработчик корневого маршрута, который возвращает простое сообщение "hello world".
@app.get("/")
def root():
    return {"message": "Hello World"}

# Объявляем обработчик маршрута '/predict/', который принимает объект item и предсказывает 
# настроения текста item.text с помощью модели classifier. Возвращается результат предсказания.
@app.post("/predict/")
def predict(item: Item):
    return classifier(item.text)[0]

# Объявляем обработчик маршрута '/translate/', который принимает объект item и выполняет 
# машинный перевод текста item.text с английского на русский с помощью модели model и 
# токенизатора tokenizer. Полученный переведенный текст возвращается в виде словаря.
@app.post("/translate/")
def translate_text(item: Item):
    inputs = tokenizer(item.text, return_tensors="pt")
    outputs = model.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"translated_text": translated_text}
