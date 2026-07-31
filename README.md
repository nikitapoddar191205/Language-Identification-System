🗣️ Language Identification System (Text + Audio)
An end-to-end Indian Language Identification System that detects language from both text input and spoken audio (.mp3) using Machine Learning.
The system is designed especially for low-resource Indian languages, without relying on speech-to-text conversion.

📌 Project Overview
This project implements a dual-pipeline architecture:

📝 Text Language Identification using Character-level TF-IDF + Logistic Regression
🎧 Audio Language Identification using MFCC features + Random Forest
The final models are deployed using an interactive Streamlit web application.

🎓 “We implemented a dual-pipeline system: character-level TF-IDF for text language identification and MFCC-based spoken language identification trained directly on MP3 audio files, avoiding speech-to-text bias and improving robustness for low-resource Indian languages.”

📂 Project Structure
language-identification/
│
├── data/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
│
├── Indian_Languages_Audio_Dataset/
│   ├── Hindi/
│   │   ├── h1.mp3
│   │   └── h2.mp3
│   ├── Malayalam/
│   ├── Marathi/
│   ├── Odia/
│   ├── Urdu/
│   └── English/
│
├── notebooks/
│   ├── text_language_model.ipynb
│   └── audio_language_model.ipynb
│
├── model/
│   ├── language_model.pkl
│   ├── vectorizer.pkl
│   └── audio_language_model.pkl
│
├── app.py
└── requirements.txt

---
🧠 System Architecture
📝 Text Pipeline Input text cleaning (lowercase, punctuation removal)

Character-level TF-IDF vectorization (2–5 ngrams)

Logistic Regression classification

Probability-based language prediction

🎧 Audio Pipeline Load .mp3 audio at 16kHz

Extract MFCC (Mel-Frequency Cepstral Coefficients)

Train Random Forest classifier

Predict spoken language probabilities

🧪 Dataset Description
📄 Text Dataset Stored in data/

CSV format with:

Headline → Input text

Language → Target label

🎧 Audio Dataset Folder-based structure

Each language has its own directory

Raw .mp3 files used directly (no speech-to-text)

🧠 Model Training
Text Model (notebooks/text_language_model.ipynb)
Vectorizer: Character-level TF-IDF

Model: Logistic Regression

Advantages:

Language-independent

Robust to spelling errors

Works well for short text

Saved files:

model/language_model.pkl

model/vectorizer.pkl

Audio Model (notebooks/audio_language_model.ipynb)
Feature Extraction: MFCC (13 coefficients)

Model: Random Forest Classifier

Advantages:

No speech recognition bias

Works on raw audio

Suitable for low-resource languages

Saved file:

model/audio_language_model.pkl

🌐 Streamlit Web Application

Application Features
Dual input modes:

📝 Text input

🎧 Audio upload (.mp3)

Displays:

Language probabilities

Final detected language

Clean and interactive UI

Run the App Locally
pip install -r requirements.txt streamlit run app.py App will open at:

arduino Copy code http://localhost:8501 📸 Results & Screenshots

📝 Text Language Detection
Screenshot 2025-12-20 at 3 27 08 PM Screenshot 2025-12-20 at 3 27 33 PM Screenshot 2025-12-20 at 3 38 21 PM Screenshot 2025-12-20 at 6 39 14 PM Screenshot 2025-12-20 at 6 39 54 PM
Text Detection Result

🎧 Audio Language Detection
Audio Detection Result 📌 Create a screenshots/ folder and add your Streamlit output images.

📦 Requirements txt Copy code streamlit pandas numpy scikit-learn librosa soundfile 🚀 Key Highlights ✔️ Works for multiple Indian languages

✔️ Audio model trained directly on .mp3

✔️ Avoids speech-to-text dependency

✔️ Research + industry aligned

✔️ Deployable as a web app

🎓 Viva / Interview Explanation Problem Statement: Identifying language from both written and spoken input for Indian languages with limited resources.

Solution: We designed a dual ML pipeline:

Text → Character TF-IDF + Logistic Regression

Audio → MFCC + Random Forest

Outcome: The system accurately detects languages from both text and audio while remaining lightweight and scalable.

🔮 Future Improvements Add deep learning (CNN/LSTM) for audio

Increase dataset size

Deploy on Streamlit Cloud / AWS :
https://language-identification-system.streamlit.app/

Add real-time microphone input
