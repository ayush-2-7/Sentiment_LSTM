#import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.datasets import imdb # type: ignore
from tensorflow.keras.preprocessing import sequence # type: ignore
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
try:
    model = load_model('./sentiment_analysis_model_lstm.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    st.error("Error loading the model. Please check the file path or model compatibility.")
    st.stop()

# Function to decode values
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess the data
def preprocess_text(text):
    print(f"Original text: {text}")  # Debugging
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    print(f"Encoded Review: {encoded_review}")  # Debugging
    padded_review = sequence.pad_sequences([encoded_review], maxlen=200)
    print(f"Padded Review Shape: {padded_review.shape}")  # Debugging
    return padded_review

# Prediction function
def prediction_sentiment(review):
    preprocessed_input = preprocess_text(review)
    print(f"Preprocessed Input Shape: {preprocessed_input.shape}")  # Debugging
    prediction = model.predict(preprocessed_input)
    print(f"Prediction: {prediction}")  # Debugging
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return sentiment, prediction[0][0]

# Streamlit App
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative')

user_input = st.text_area('Movie Review')

if st.button('Submit'):
    if not user_input.strip():
        st.error("Please enter a valid review.")
    else:
        sentiment, prediction = prediction_sentiment(user_input)
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Prediction Score: {prediction}")
else:
    st.write('Please enter a movie review')
