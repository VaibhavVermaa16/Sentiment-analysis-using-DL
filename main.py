from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Load the saved model
model = load_model('models/best_sentiment_model.h5')


# Load the tokenizer
with open('models/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to preprocess the review using the loaded tokenizer
def preprocess_review(review):
    tokens = tokenizer.texts_to_sequences([review])  # Convert review to sequences
    padded_tokens = pad_sequences(tokens, maxlen=10, padding='post')  # Pad/truncate to match model input length
    return padded_tokens

@app.route('/')
def index():
    return render_template('template/index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    review = request.form['review']
    if review == '':
        # sentiment = "Please enter a review."
        return jsonify({"sentiment": "Neutral"})

    # Preprocess the review text
    processed_review = preprocess_review(review)

    # Perform the prediction
    prediction = model.predict(processed_review)
    
    # Since this is a binary classification (output between 0 and 1), classify sentiment
    sentiment = 'positive' if prediction[0][0] >= 0.5 else 'negative'
    print(sentiment)

    return jsonify({"sentiment": sentiment})

if __name__ == '__main__':
    app.run(debug=True)
