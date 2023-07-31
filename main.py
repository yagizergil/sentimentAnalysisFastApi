# main.py
import sqlite3

import aiosqlite as aiosqlite
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .sentiment_model import predict_sentiment
from fastapi.middleware.cors import CORSMiddleware
from .sentiment_model import save_sentiment_to_database
from fastapi.responses import JSONResponse

app = FastAPI()

# Tüm kökenlere izin ver (üretimde React uygulamanızın belirli kökenini buraya ekleyin)
origins = ["*"]

# Uygulamaya CORS ara yazılımını ekle
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # OPTIONS yöntemini ekleyin
    allow_headers=["*"],
)


vectorizer = joblib.load('model/vectorizer.pkl')
class SentimentInput(BaseModel):
    text: str

class SentimentOutput(BaseModel):
    prediction: str
    lr_model_proba: list

@app.post("/predict_sentiment/", response_model=SentimentOutput)
async def predict_sentiment_endpoint(input_data: SentimentInput):
    new_text_vector = vectorizer.transform([input_data.text])
    sentiment_prediction, lr_model_proba = predict_sentiment(input_data.text, new_text_vector)

    # Veritabanına kaydet
    label = sentiment_prediction.split()[0]
    save_sentiment_to_database(input_data.text, label)

    return SentimentOutput(prediction=sentiment_prediction, lr_model_proba=lr_model_proba)


@app.get("/get_sentiment_data/")
async def get_sentiment_data():
    async with aiosqlite.connect("sentiment_analysis.db") as conn:
        c = await conn.cursor()
        await c.execute('SELECT * FROM sentiment_analysis')
        data = await c.fetchall()
    return JSONResponse(content={"data": data})

@app.post("/save_sentiment/")
async def save_sentiment_endpoint(input_data: SentimentInput):
    new_text_vector = vectorizer.transform([input_data.text])
    sentiment_prediction, _ = predict_sentiment(input_data.text, new_text_vector)
    label = sentiment_prediction.split()[0]
    await save_sentiment_to_database(input_data.text, label)
    return {"mesaj": "Veriler başarıyla veritabanına kaydedildi."}