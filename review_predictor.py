import streamlit as st
import tensorflow as tf
import tensorflow.keras.backend as K 
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, Flatten, Activation, RepeatVector, Permute, Dropout, BatchNormalization, Lambda
from tensorflow.keras.models import Model, Sequential
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer + models
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


def sum_axis_1(x):
    return K.sum(x, axis=1)

rnn_model = tf.keras.models.load_model("rnn_model.h5", safe_mode=False)
lstm_model = tf.keras.models.load_model("lstm_model.h5", safe_mode=False)
gru_model = tf.keras.models.load_model("gru_model.h5", safe_mode=False)
gru_attention_model = tf.keras.models.load_model(
    "gru_attention_model.h5",
    custom_objects={"sum_axis_1": sum_axis_1}
)
max_len = 100


# Text Cleaning 
def clean_text(text):

    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


# Prediction Function
def predict_sentiment(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')

    preds = {
        "Vanilla RNN": float(rnn_model.predict(padded, verbose=0)[0][0]),
        "LSTM": float(lstm_model.predict(padded, verbose=0)[0][0]),
        "GRU": float(gru_model.predict(padded, verbose=0)[0][0]),
        "GRU + Attention": float(gru_attention_model.predict(padded, verbose=0)[0][0])
    }

    return preds


# Streamlit UI
st.title(" IMDB Movie Review Sentiment Analyzer")
st.write("Enter a movie review and compare predictions from **4 different models**.")

review_text = st.text_area("Enter your review here:", height=200)

if st.button("Predict Sentiment"):
    if review_text.strip() == "":
        st.warning("Please enter a review.")
    else:
        with st.spinner("Analyzing..."):
            preds = predict_sentiment(review_text)

        st.success("Done!")

        for model_name, score in preds.items():
            sentiment = "Positive " if score >= 0.5 else "Negative "
            st.write(f"### **{model_name}**")
            st.write(f"Prediction Score: **{score:.4f}** → {sentiment}")
            st.write("---")
