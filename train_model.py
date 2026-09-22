"""
Training script for the Emotion Classification Model based on finalproject.ipynb.
Preprocesses text, trains TF-IDF + LogisticRegression, and saves model artifacts.
"""

import os
import json
import string
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Ensure NLTK resources are available
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

STOP_WORDS = set(stopwords.words('english'))

def remove_punc(text: str) -> str:
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_numbers(text: str) -> str:
    return ''.join([ch for ch in text if not ch.isdigit()])

def remove_emojis(text: str) -> str:
    return ''.join([ch for ch in text if ch.isascii()])

def remove_stopwords(text: str) -> str:
    words = text.split()
    cleaned = [w for w in words if w not in STOP_WORDS]
    return ' '.join(cleaned)

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = remove_punc(text)
    text = remove_numbers(text)
    text = remove_emojis(text)
    text = remove_stopwords(text)
    return text.strip()

def main():
    # 1. Locate dataset
    data_path = 'train.txt'
    if not os.path.exists(data_path):
        data_path = os.path.join('..', 'train.txt')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not locate train.txt in current or parent directory.")
    
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path, sep=';', header=None, names=['text', 'emotion'])
    print(f"Loaded {len(df)} samples across {df['emotion'].nunique()} emotions.")

    # 2. Map emotion labels matching finalproject.ipynb
    unique_emotions = list(df['emotion'].unique())
    emotion_to_id = {emo: i for i, emo in enumerate(unique_emotions)}
    id_to_emotion = {i: emo for emo, i in emotion_to_id.items()}
    df['label'] = df['emotion'].map(emotion_to_id)

    # 3. Clean text using exact notebook preprocessing
    print("Preprocessing text data...")
    df['clean_text'] = df['text'].apply(preprocess_text)

    # 4. Train/Test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.20, random_state=42
    )

    # 5. Fit TF-IDF Vectorizer
    print("Fitting TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # 6. Train Logistic Regression
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)

    # 7. Evaluate
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Validation Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=[id_to_emotion[i] for i in range(len(unique_emotions))]))

    # 8. Train on full dataset for maximum deployment quality
    print("Retraining on 100% of the dataset for optimal production inference...")
    full_tfidf = TfidfVectorizer()
    X_full = full_tfidf.fit_transform(df['clean_text'])
    full_model = LogisticRegression(max_iter=1000, random_state=42)
    full_model.fit(X_full, df['label'])

    # 9. Save artifacts
    print("Saving model artifacts...")
    joblib.dump(full_model, 'model.joblib')
    joblib.dump(full_tfidf, 'tfidf.joblib')

    # Detailed emotion metadata (emojis, colors, descriptions)
    emotion_meta = {
        "sadness": {"emoji": "😢", "color": "#38bdf8", "bg": "#0284c7", "description": "Feeling sorrowful, down, or melancholy"},
        "anger": {"emoji": "😡", "color": "#f87171", "bg": "#dc2626", "description": "Feeling irritated, enraged, or aggrieved"},
        "love": {"emoji": "💖", "color": "#f472b6", "bg": "#db2777", "description": "Feeling affectionate, caring, or warm"},
        "surprise": {"emoji": "😲", "color": "#fbbf24", "bg": "#d97706", "description": "Feeling astonishment, shock, or wonder"},
        "fear": {"emoji": "😨", "color": "#a78bfa", "bg": "#7c3aed", "description": "Feeling anxious, threatened, or terrified"},
        "joy": {"emoji": "😊", "color": "#34d399", "bg": "#059669", "description": "Feeling cheerful, happy, or elated"}
    }

    config = {
        "unique_emotions": unique_emotions,
        "emotion_to_id": emotion_to_id,
        "id_to_emotion": {str(k): v for k, v in id_to_emotion.items()},
        "emotion_meta": emotion_meta,
        "accuracy": acc
    }

    with open('emotion_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    print("Training finished successfully! Saved: model.joblib, tfidf.joblib, emotion_config.json")

if __name__ == '__main__':
    main()
