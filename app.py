import os, re, sys, subprocess, joblib, nltk
from flask import Flask, request, jsonify, send_from_directory

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, 'intent_model.pkl')
VEC_PATH     = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')
STATIC_DIR   = os.path.join(BASE_DIR, 'static')
TRAIN_SCRIPT = os.path.join(BASE_DIR, 'train_model.py')

app = Flask(__name__, static_folder=STATIC_DIR)

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))

model      = None
vectorizer = None

def clean_text(text):
    text   = str(text).lower()
    text   = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

def load_models():
    global model, vectorizer
    if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
        try:
            model      = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VEC_PATH)
            print("  Model loaded successfully.")
            return True
        except Exception as e:
            print(f"  Error loading model: {e}")
            return False
    print("  Model files not found. Use the web UI to train.")
    return False

@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')

@app.route('/status')
def status():
    return jsonify({'model_loaded': model is not None and vectorizer is not None})

@app.route('/train', methods=['POST'])
def train():
    if not os.path.exists(TRAIN_SCRIPT):
        return jsonify({'success': False, 'message': 'train_model.py not found'}), 500
    try:
        result = subprocess.run(
            [sys.executable, TRAIN_SCRIPT],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            ok = load_models()
            return jsonify({'success': ok, 'message': 'Training complete!', 'log': result.stdout})
        else:
            return jsonify({'success': False, 'message': 'Training failed.', 'log': result.stderr or result.stdout}), 500
    except subprocess.TimeoutExpired:
        return jsonify({'success': False, 'message': 'Timed out (>10 min).'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'MODEL_NOT_LOADED'}), 503
    data    = request.get_json() or {}
    message = data.get('message', '').strip()
    if not message:
        return jsonify({'error': 'Empty message'}), 400
    cleaned    = clean_text(message)
    vec        = vectorizer.transform([cleaned])
    prediction = int(model.predict(vec)[0])
    proba      = model.predict_proba(vec)[0].tolist()
    return jsonify({
        'prediction': prediction,
        'label':      'Interested' if prediction == 1 else 'Not Interested',
        'confidence': round(max(proba) * 100, 1),
        'prob_0':     round(proba[0] * 100, 1),
        'prob_1':     round(proba[1] * 100, 1),
        'cleaned':    cleaned,
    })

@app.route('/batch', methods=['POST'])
def batch():
    if model is None:
        return jsonify({'error': 'MODEL_NOT_LOADED'}), 503
    data     = request.get_json() or {}
    messages = data.get('messages', [])
    results  = []
    for msg in messages[:50]:
        msg = str(msg).strip()
        if not msg: continue
        cleaned    = clean_text(msg)
        vec        = vectorizer.transform([cleaned])
        prediction = int(model.predict(vec)[0])
        proba      = model.predict_proba(vec)[0].tolist()
        results.append({
            'message':    msg,
            'prediction': prediction,
            'label':      'Interested' if prediction == 1 else 'Not Interested',
            'confidence': round(max(proba) * 100, 1),
            'prob_0':     round(proba[0] * 100, 1),
            'prob_1':     round(proba[1] * 100, 1),
        })
    return jsonify({'results': results})

if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  BuySignal - Customer Intent Detection")
    print("=" * 55)
    load_models()
    if model is None:
        print("\n  No model found.")
        print("  Open http://localhost:5000 and click Train Model.\n")
    print("\n  Server: http://localhost:5000")
    print("  Press Ctrl+C to stop.\n")
    app.run(debug=False, port=5000, host='0.0.0.0')
