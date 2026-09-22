# EmotionSense AI — Web UI & API

An interactive, responsive web application and REST API for real-time text emotion detection. Built with **Flask**, **HTML5**, **CSS3**, and **Vanilla JavaScript**, powered by a **scikit-learn** NLP model (TF-IDF + Logistic Regression).

---

## Features

- **Multi-Class Emotion Classification**: Detects 6 core affective states:
  - 😊 **Joy**
  - 😢 **Sadness**
  - 😡 **Anger**
  - 💖 **Love**
  - 😨 **Fear**
  - 😲 **Surprise**
- **Confidence & Probability Breakdown**: Displays full confidence spectrum and interactive probability bars across all 6 classes.
- **NLP Inspection**: View the raw sentence side-by-side with the preprocessed tokens (after lowercasing, punctuation/emoji/digit stripping, and stopword removal).
- **Instant Sample Chips**: One-click prompt pills for quick demonstration and testing.
- **Prediction History**: Automatically logs recent predictions locally with quick re-test capability.
- **Keyboard Shortcuts**: Press `Ctrl + Enter` (or `Cmd + Enter`) to instantly analyze text.

---

## Project Structure

```
New folder/
├── app.py                  # Flask backend & REST endpoints
├── train_model.py          # Machine learning model training pipeline
├── test_app.py             # Automated unit & integration tests
├── model.joblib            # Trained Logistic Regression classifier
├── tfidf.joblib            # Fitted TF-IDF vectorizer
├── emotion_config.json     # Class mappings, metadata, emojis, and accuracy stats
├── finalproject.ipynb      # Original experimental notebook
├── templates/
│   └── index.html          # Web UI markup
└── static/
    ├── css/
    │   └── style.css       # Custom modern dark-theme styling
    └── js/
        └── app.js          # Client-side reactivity, API calls, and history
```

---

## Quick Start

### 1. Run the Web Application

In your terminal, navigate to this folder and run:

```bash
python app.py
```

Then open your browser to:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

### 2. API Endpoints

#### Predict Emotion: `POST /api/predict`
**Request:**
```json
{
  "text": "I am feeling so grateful and happy today!"
}
```

**Response:**
```json
{
  "status": "success",
  "predicted_emotion": "joy",
  "confidence": 98.2,
  "emoji": "😊",
  "color": "#34d399",
  "breakdown": [
    { "emotion": "joy", "percentage": 98.2, "emoji": "😊", "color": "#34d399" },
    { "emotion": "love", "percentage": 1.1, "emoji": "💖", "color": "#f472b6" },
    ...
  ],
  "raw_text": "I am feeling so grateful and happy today!",
  "cleaned_text": "feeling grateful happy today"
}
```

#### Health Check: `GET /api/health`
Returns system status and model readiness.

#### Sample Examples: `GET /api/examples`
Returns pre-configured sample emotional statements.

---

### 3. Run Automated Tests

```bash
python test_app.py
```
