"""
Flask Backend for Emotion Detection Web Application.
Serves the web UI and provides RESTful prediction endpoints.
Optimized for local execution and serverless deployment (e.g. Vercel).
"""

import os
import json
import string
import joblib
from flask import Flask, render_template, request, jsonify

# Pre-defined NLTK English stopwords to prevent cold-start download failures in serverless environments
FALLBACK_STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren',
    "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
    'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don',
    "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn',
    "hasn't", 'have', 'haven', "haven't", 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself',
    'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'll', 'm',
    'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't",
    'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves',
    'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she's", 'should', "should've", 'shouldn',
    "shouldn't", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them',
    'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until',
    'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'what', 'when', 'where',
    'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you',
    "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'
}

try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords', quiet=True)
    STOP_WORDS = set(stopwords.words('english'))
except Exception:
    STOP_WORDS = FALLBACK_STOPWORDS

app = Flask(__name__)
# Serverless handler alias
handler = app

# Preprocessing helpers matching finalproject.ipynb
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

# Load model, vectorizer, and configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.joblib')
TFIDF_PATH = os.path.join(BASE_DIR, 'tfidf.joblib')
CONFIG_PATH = os.path.join(BASE_DIR, 'emotion_config.json')

model = None
tfidf = None
config = {}

def load_artifacts():
    global model, tfidf, config
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TFIDF_PATH):
        raise FileNotFoundError("Model artifacts not found. Please run train_model.py first.")
    
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)

load_artifacts()

SAMPLE_EXAMPLES = [
    {
        "category": "Joy",
        "emoji": "😊",
        "text": "I just received the acceptance letter for my dream internship! I am smiling ear to ear!"
    },
    {
        "category": "Sadness",
        "emoji": "😢",
        "text": "I feel so hopeless and empty today, staring out the window with tears in my eyes."
    },
    {
        "category": "Anger",
        "emoji": "😡",
        "text": "I am absolutely disgusted and furious at how unfair and rude they were to us."
    },
    {
        "category": "Love",
        "emoji": "💖",
        "text": "Spending quiet Sunday mornings cooking breakfast with you makes me feel so deeply blessed and cherished."
    },
    {
        "category": "Fear",
        "emoji": "😨",
        "text": "Hearing footsteps behind me in the pitch dark alley made my heart race with terror."
    },
    {
        "category": "Surprise",
        "emoji": "😲",
        "text": "I can't believe they threw me a secret party! I opened the door completely stunned!"
    }
]

@app.route('/')
def home():
    """Serves the single-page application UI."""
    return render_template('index.html', model_accuracy=round(config.get('accuracy', 0.8628) * 100, 1))

@app.route('/api/examples', methods=['GET'])
def get_examples():
    """Returns sample inputs for one-click testing."""
    return jsonify({
        "status": "success",
        "examples": SAMPLE_EXAMPLES
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Accepts JSON: {"text": "..."}
    Returns emotion classification, confidence, and full class probability breakdown.
    """
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({"status": "error", "message": "Missing 'text' field in JSON request body."}), 400

    raw_text = str(data.get('text', '')).strip()
    if not raw_text:
        return jsonify({"status": "error", "message": "Text cannot be empty."}), 400

    cleaned_text = preprocess_text(raw_text)

    # Vectorize
    features = tfidf.transform([cleaned_text if cleaned_text else raw_text.lower()])
    
    # Predict probabilities
    probabilities = model.predict_proba(features)[0]
    predicted_class_id = int(model.classes_[probabilities.argmax()])
    predicted_emotion = config['id_to_emotion'][str(predicted_class_id)]
    
    # Build breakdown list
    meta = config.get('emotion_meta', {})
    breakdown = []
    for idx, class_id in enumerate(model.classes_):
        emo_name = config['id_to_emotion'][str(class_id)]
        prob = float(probabilities[idx])
        emo_info = meta.get(emo_name, {})
        breakdown.append({
            "id": int(class_id),
            "emotion": emo_name,
            "probability": prob,
            "percentage": round(prob * 100, 1),
            "emoji": emo_info.get("emoji", "💭"),
            "color": emo_info.get("color", "#6366f1"),
            "bg": emo_info.get("bg", "#4f46e5"),
            "description": emo_info.get("description", "")
        })

    # Sort breakdown from highest probability to lowest
    breakdown.sort(key=lambda x: x['probability'], reverse=True)

    top_meta = meta.get(predicted_emotion, {})
    response_payload = {
        "status": "success",
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "predicted_emotion": predicted_emotion,
        "confidence": round(float(probabilities.max()) * 100, 1),
        "emoji": top_meta.get("emoji", "💭"),
        "color": top_meta.get("color", "#6366f1"),
        "bg": top_meta.get("bg", "#4f46e5"),
        "description": top_meta.get("description", ""),
        "breakdown": breakdown
    }

    return jsonify(response_payload)

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "classes": config.get('unique_emotions', [])
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask App on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
