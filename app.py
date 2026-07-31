import streamlit as st
import pickle
import re
import librosa
import numpy as np

st.set_page_config(
    page_title="Indian Language Identifier",
    page_icon="",
    layout="centered"
)

with open("model/language_model.pkl", "rb") as f:
    text_model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model/audio_language_model.pkl", "rb") as f:
    audio_model = pickle.load(f)

def clean_text(text):
    text = re.sub(r"[^\w\s]", "", text)
    return text.lower().strip()

def predict_text_language(text):
    X = vectorizer.transform([clean_text(text)])
    probs = text_model.predict_proba(X)[0]
    langs = text_model.classes_
    return sorted(zip(langs, probs), key=lambda x: x[1], reverse=True)

def predict_audio_language(audio_file):
    audio, sr = librosa.load(audio_file, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc = np.mean(mfcc.T, axis=0).reshape(1, -1)

    probs = audio_model.predict_proba(mfcc)[0]
    langs = audio_model.classes_

    return sorted(zip(langs, probs), key=lambda x: x[1], reverse=True)

st.title("🗣️ Indian Language Identification System")
st.caption("Text + Audio | Low-Resource Indian Languages")

tab1, tab2 = st.tabs([" Text Input", "Audio Input (.mp3)"])

with tab1:
    text = st.text_area("Enter text")

    if st.button("Detect Text Language"):
        results = predict_text_language(text)

        for lang, prob in results:
            st.progress(float(prob))
            st.write(f"{lang}: {prob*100:.2f}%")

        st.success(f"Detected Language: {results[0][0]}")

with tab2:
    audio_file = st.file_uploader("Upload MP3 audio", type=["mp3"])

    if audio_file:
        st.audio(audio_file)

        if st.button("Detect Audio Language"):
            results = predict_audio_language(audio_file)

            for lang, prob in results:
                st.progress(float(prob))
                st.write(f"{lang}: {prob*100:.2f}%")

            st.success(f"Detected Language: {results[0][0]}")
