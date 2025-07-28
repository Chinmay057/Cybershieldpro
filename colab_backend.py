# colab_backend.py
# Backend for Phishing Email Detector

import pandas as pd
import numpy as np
import re
import string
import pickle
import warnings
from collections import Counter
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

warnings.filterwarnings('ignore')

try:
    import nltk
    from packaging import version
    if version.parse(nltk.__version__) < version.parse('3.8.1'):
        raise ImportError(f"Your NLTK version is {nltk.__version__}. Please run 'pip install --upgrade nltk==3.8.1' to fix the punkt_tab bug.")
except ImportError as e:
    raise ImportError("The 'nltk' package is not installed or is the wrong version. Please run 'pip install --upgrade nltk==3.8.1' in your current environment.\n" + str(e))

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

# --- Data Loading and Preprocessing ---
def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    column_mapping = {
        'subject': 'Subject',
        'body': 'Body',
        'label': 'Label'
    }
    df_fixed = df.rename(columns=column_mapping)
    return df_fixed

# --- Preprocessor Class ---
class AdvancedEmailPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.phishing_keywords = {
            'urgent': ['urgent', 'immediate', 'act now', 'limited time', 'expires', 'deadline', 'hurry', 'rush'],
            'money': ['money', 'cash', 'prize', 'winner', 'lottery', 'million', 'thousand', '$', '€', '£', 'refund', 'claim'],
            'personal': ['password', 'account', 'social security', 'credit card', 'verify', 'confirm', 'update', 'suspend'],
            'action': ['click here', 'download', 'install', 'open attachment', 'follow link', 'visit'],
            'threats': ['suspend', 'terminate', 'close', 'lock', 'freeze', 'penalty', 'legal action'],
            'authority': ['irs', 'fbi', 'police', 'government', 'tax', 'court', 'legal', 'official'],
            'emotional': ['congratulations', 'selected', 'chosen', 'lucky', 'special', 'exclusive']
        }

    def clean_text(self, text):
        if pd.isna(text) or text == '':
            return ''
        text = str(text).lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', ' URL_TOKEN ', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', ' EMAIL_TOKEN ', text)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', ' PHONE_TOKEN ', text)
        text = re.sub(r'[^\w\s!?$%]', ' ', text)
        text = ' '.join(text.split())
        tokens = word_tokenize(text)
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                if token in ['URL_TOKEN', 'EMAIL_TOKEN', 'PHONE_TOKEN']:
                    processed_tokens.append(token)
                else:
                    processed_tokens.append(self.lemmatizer.lemmatize(token))
        return ' '.join(processed_tokens)

    def extract_advanced_features(self, text):
        if pd.isna(text) or text == '':
            return self._get_empty_features()
        text = str(text)
        text_lower = text.lower()
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(sent_tokenize(text)),
            'capital_ratio': sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if len(text) > 0 else 0,
        }
        features.update({
            'url_count': len(re.findall(r'http\S+|www\S+|https\S+', text)),
            'email_count': len(re.findall(r'\S+@\S+', text)),
            'phone_count': len(re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)),
            'ip_count': len(re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text)),
        })
        for category, keywords in self.phishing_keywords.items():
            features[f'{category}_words'] = sum(1 for word in keywords if word in text_lower)
        words = text.split()
        if words:
            features.update({
                'avg_word_length': np.mean([len(word) for word in words]),
                'unique_word_ratio': len(set(words)) / len(words),
                'repeated_chars': len(re.findall(r'(.)\1{2,}', text)),
            })
        else:
            features.update({
                'avg_word_length': 0,
                'unique_word_ratio': 0,
                'repeated_chars': 0,
            })
        features.update({
            'has_attachments': int(any(word in text_lower for word in ['attachment', 'download', 'zip', 'exe'])),
            'has_urgency': int(any(word in text_lower for word in ['urgent', 'immediate', 'asap', 'quickly'])),
            'has_money_symbols': int(any(symbol in text for symbol in ['$', '€', '£', '¥'])),
            'has_personal_request': int(any(word in text_lower for word in ['password', 'ssn', 'social security'])),
        })
        return features

    def _get_empty_features(self):
        empty_features = {
            'length': 0, 'word_count': 0, 'sentence_count': 0, 'capital_ratio': 0,
            'exclamation_count': 0, 'question_count': 0, 'digit_ratio': 0,
            'url_count': 0, 'email_count': 0, 'phone_count': 0, 'ip_count': 0,
            'avg_word_length': 0, 'unique_word_ratio': 0, 'repeated_chars': 0,
            'has_attachments': 0, 'has_urgency': 0, 'has_money_symbols': 0, 'has_personal_request': 0
        }
        for category in self.phishing_keywords.keys():
            empty_features[f'{category}_words'] = 0
        return empty_features

# --- Detector Class ---
class EnhancedPhishingDetector:
    def __init__(self, use_balancing=True, feature_selection=True):
        self.preprocessor = AdvancedEmailPreprocessor()
        self.use_balancing = use_balancing
        self.feature_selection = feature_selection
        self.tfidf_vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 3), min_df=2, max_df=0.95, sublinear_tf=True)
        self.count_vectorizer = CountVectorizer(max_features=10000, ngram_range=(1, 2), binary=True)
        self.models = {}
        self.ensemble_model = None
        self.is_trained = False
        self.feature_selector = None
        self.scaler = StandardScaler()

    def train_models(self, df):
        df['combined_text'] = df['Subject'].fillna('') + ' ' + df['Body'].fillna('')
        df['cleaned_text'] = df['combined_text'].apply(self.preprocessor.clean_text)
        feature_data = df['combined_text'].apply(self.preprocessor.extract_advanced_features)
        feature_df = pd.DataFrame(feature_data.tolist())
        X_tfidf = self.tfidf_vectorizer.fit_transform(df['cleaned_text'])
        X_count = self.count_vectorizer.fit_transform(df['cleaned_text'])
        X_text = hstack([X_tfidf, X_count])
        X_features = feature_df.values
        X_features_scaled = self.scaler.fit_transform(X_features)
        X_combined = hstack([X_text, X_features_scaled])
        if self.feature_selection:
            self.feature_selector = SelectKBest(f_classif, k=min(20000, X_combined.shape[1]))
            X_combined = self.feature_selector.fit_transform(X_combined, df['Label'])
        X = X_combined
        y = df['Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        balance_ratio = len(df[df['Label'] == 1]) / max(1, len(df[df['Label'] == 0]))
        if self.use_balancing and balance_ratio < 0.8:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=2000, C=10, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1, class_weight='balanced'),
            'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1),
            'LightGBM': lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        }
        estimators = [(name, model) for name, model in self.models.items() if name not in ['XGBoost', 'LightGBM']]
        self.ensemble_model = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1, weights=[2, 2, 1])
        for name, model in self.models.items():
            model.fit(X_train, y_train)
        self.ensemble_model.fit(X_train, y_train)
        self.is_trained = True
        self.X_test = X_test
        self.y_test = y_test

    def predict_email(self, subject, body):
        if not self.is_trained:
            raise ValueError("Models not trained yet!")
        combined_text = f"{subject} {body}"
        cleaned_text = self.preprocessor.clean_text(combined_text)
        features = self.preprocessor.extract_advanced_features(combined_text)
        feature_array = np.array(list(features.values())).reshape(1, -1)
        feature_array_scaled = self.scaler.transform(feature_array)
        X_tfidf = self.tfidf_vectorizer.transform([cleaned_text])
        X_count = self.count_vectorizer.transform([cleaned_text])
        X_text = hstack([X_tfidf, X_count])
        X_combined = hstack([X_text, feature_array_scaled])
        if self.feature_selection and self.feature_selector:
            X_combined = self.feature_selector.transform(X_combined)
        ensemble_pred = self.ensemble_model.predict(X_combined)[0]
        ensemble_prob = self.ensemble_model.predict_proba(X_combined)[0]
        ensemble_prob_phishing = ensemble_prob[1]
        ensemble_confidence = max(ensemble_prob) - min(ensemble_prob)
        return int(ensemble_pred), float(ensemble_prob_phishing), float(ensemble_confidence), features

# --- Flask API ---
app = Flask(__name__)
CORS(app)

detector = None

@app.route('/train', methods=['POST'])
def train():
    global detector
    data = request.get_json()
    csv_path = data.get('csv_path')
    if not csv_path:
        return jsonify({'error': 'csv_path is required'}), 400
    df_fixed = load_and_prepare_data(csv_path)
    # Convert label to numeric if needed
    if df_fixed['Label'].dtype == 'object':
        label_mapping = {'ham': 0, 'spam': 1, 'legitimate': 0, 'phishing': 1, 'normal': 0, 'malicious': 1}
        df_fixed['Label'] = df_fixed['Label'].str.lower().map(label_mapping).fillna(df_fixed['Label'])
    detector = EnhancedPhishingDetector()
    detector.train_models(df_fixed)
    return jsonify({'status': 'Model trained successfully'})

@app.route('/predict', methods=['POST'])
def predict():
    global detector
    if detector is None or not detector.is_trained:
        return jsonify({'error': 'Model not trained yet'}), 400
    data = request.get_json()
    subject = data.get('subject', '')
    body = data.get('body', '')
    try:
        pred, prob, conf, features = detector.predict_email(subject, body)
        return jsonify({
            'ensemble_prediction': pred,
            'ensemble_probability': prob,
            'ensemble_confidence': conf,
            'features': features
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 