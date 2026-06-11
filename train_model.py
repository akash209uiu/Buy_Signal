import warnings
warnings.filterwarnings('ignore')
import os, re, joblib, nltk
import matplotlib
matplotlib.use('Agg')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'intent_model.pkl')
VEC_PATH   = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')

print("=" * 55)
print("  BUYING INTENT MODEL - TRAINING")
print("=" * 55)

print("\n[1/7] Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)
print("      OK")

print("\n[2/7] Loading dataset from Hugging Face...")
print("      (First run downloads ~10MB, takes 1-2 min)")
from datasets import load_dataset
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
df = dataset['train'].to_pandas()
print(f"      Loaded {len(df)} rows")

print("\n[3/7] Labelling buying intent...")
buying_intents = [
    'place_order', 'check_invoice', 'delivery_options',
    'check_payment_methods', 'delivery_period',
    'track_order', 'change_order', 'check_cancellation_fee'
]
df['label'] = df['intent'].apply(lambda x: 1 if x in buying_intents else 0)
c = df['label'].value_counts()
print(f"      Interested (1): {c.get(1,0)}  |  Not Interested (0): {c.get(0,0)}")

print("\n[4/7] Cleaning text...")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))

def clean_text(text):
    text   = str(text).lower()
    text   = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

df['clean_text'] = df['instruction'].apply(clean_text)
print("      OK")

print("\n[5/7] Splitting 80/20 train/test...")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)
print(f"      Train: {len(X_train)}  |  Test: {len(X_test)}")

print("\n[6/7] TF-IDF + Training models...")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)
Xtr = vectorizer.fit_transform(X_train)
Xte = vectorizer.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(Xtr, y_train)
acc_lr = accuracy_score(y_test, lr.predict(Xte))
print(f"      Logistic Regression: {acc_lr*100:.2f}%")

nb = MultinomialNB(alpha=0.1)
nb.fit(Xtr, y_train)
acc_nb = accuracy_score(y_test, nb.predict(Xte))
print(f"      Naive Bayes:         {acc_nb*100:.2f}%")

best_model = lr if acc_lr >= acc_nb else nb
best_acc   = max(acc_lr, acc_nb)
best_name  = "Logistic Regression" if acc_lr >= acc_nb else "Naive Bayes"

print("\n[7/7] Saving model files...")
joblib.dump(best_model, MODEL_PATH)
joblib.dump(vectorizer, VEC_PATH)
print(f"      Saved: {MODEL_PATH}")
print(f"      Saved: {VEC_PATH}")

print("\n" + "=" * 55)
print(f"  DONE!  Best: {best_name}  Accuracy: {best_acc*100:.2f}%")
print("=" * 55)
print("\nModel is ready. The web app will load it automatically.\n")
