# sentiment_model.py

import string
import sqlite3

import aiosqlite
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError

nltk.download('punkt')
nltk.download('stopwords')

# sentiment_analysis.db dosyası adıyla bir veritabanı oluştur. Sqlite3 kullanıyoruz.
conn = sqlite3.connect('sentiment_analysis.db')
c = conn.cursor()

# Tabloyu oluştur autoincrement id, text ve label sütunları olsun.
c.execute('''CREATE TABLE IF NOT EXISTS sentiment_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                label TEXT NOT NULL
             )''')

# Değişiklikleri kaydet ve bağlantıyı kapat
conn.commit()
conn.close()

async def save_sentiment_to_database(text, label):
    async with aiosqlite.connect('sentiment_analysis.db') as conn:
        c = await conn.cursor()
        await c.execute('INSERT INTO sentiment_analysis (text, label) VALUES (?, ?)', (text, label))
        await conn.commit()

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    words = [word for word in words if len(word) > 1]
    return " ".join(words)

def load_lr_model():
    return joblib.load('model/lr_model.pkl')

def load_vectorizer():
    return CountVectorizer()

def predict_sentiment(text, new_text_vector=None):
    try:
        vectorizer = load_vectorizer()
        vectorizer.fit([text])  # Vektörleme yapılıyor ve öğreniliyor
        if new_text_vector is None:  # Eğer yeni metin verilmediyse vektöre dönüştürüyoruz
            new_text_vector = vectorizer.transform([text])  # Yeni metin vektöre dönüştürülüyor
    except NotFittedError:  # Vektörleme yapılacaksa
        pass

    lr_model = load_lr_model()
    lr_model_proba = lr_model.predict_proba(new_text_vector)[0].tolist()

    class_labels = ['negative', 'neutral', 'positive']
    max_prob_index = lr_model_proba.index(max(lr_model_proba))
    sentiment_prediction = f"{class_labels[max_prob_index]} %{max(lr_model_proba) * 100:.2f}"

    return sentiment_prediction, lr_model_proba
