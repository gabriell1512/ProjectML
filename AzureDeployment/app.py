import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template
from flask_swagger_ui import get_swaggerui_blueprint

app = Flask(__name__)

# Load tokenizer and model
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = load_model('disaster_tweet_lstm_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    tweet = data['text']
    
    # Tokenize and pad sequences
    sequences = tokenizer.texts_to_sequences([tweet])
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=50)
    
    # Predict
    prediction = model.predict(padded_sequences)[0][0]
    result = 1 if prediction > 0.5 else 0
    
    return jsonify({'result': result})

# Swagger configuration
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Disaster Tweet Classifier"
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

if __name__ == '__main__':
    app.run(debug=True)
