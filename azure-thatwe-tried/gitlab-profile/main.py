from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import os

nltk.download('stopwords')

app = FastAPI()

# Load pre-trained model and tokenizer
model = load_model('disaster_tweet_lstm_model.h5')
tokenizer = joblib.load('tokenizer.pkl')

# Define text preprocessing function
def text_preprocessing(text, language='english', minWordSize=2):
    text_no_html = BeautifulSoup(text, "html.parser").get_text()
    text_no_url = re.sub(r'http\S+', ' ', text_no_html)
    text_no_at = ' '.join([word for word in text_no_url.split() if not word.startswith('@')])
    text_no_RT = ' '.join([word for word in text_no_at.split() if not word.startswith('RT')])
    text_alpha_chars = re.sub("[^a-zA-Z']", " ", text_no_RT)
    text_lower = text_alpha_chars.lower()
    stops = set(stopwords.words(language))
    whitelist = ["n't", "not", "no"]
    text_no_stop_words = ' '.join([word for word in text_lower.split() if word not in stops or word in whitelist])
    stemmer = SnowballStemmer(language)
    text_stemmer = ' '.join([stemmer.stem(word) for word in text_no_stop_words.split()])
    text_no_short_words = ' '.join([word for word in text_stemmer.split() if len(word) >= minWordSize])
    return text_no_short_words

# Define request body model
class TextData(BaseModel):
    text: str

# Define prediction function
def predict(text):
    processed_text = text_preprocessing(text)
    sequences = tokenizer.texts_to_sequences([processed_text])
    padded_sequences = pad_sequences(sequences, maxlen=200)
    prediction = model.predict(padded_sequences)
    prediction_label = int(np.argmax(prediction, axis=1)[0])
    return prediction_label

# Define endpoint to make predictions
@app.post("/predict")
def make_prediction(text_data: TextData):
    prediction = predict(text_data.text)
    return {"prediction": prediction}
