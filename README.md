# 🛡️ Phishing Email Detection System

A sophisticated machine learning-based phishing email detection system with a stunning dark galaxy-themed web interface. This project combines advanced NLP techniques with ensemble machine learning models to identify and classify phishing emails with high accuracy.

![Phishing Detection Demo](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-2.0+-lightgrey)
![Machine Learning](https://img.shields.io/badge/ML-Ensemble-orange)

## ✨ Features

- **Advanced ML Models**: Ensemble of Logistic Regression, Random Forest, XGBoost, LightGBM, and Gradient Boosting
- **NLP Processing**: TF-IDF, CountVectorizer, advanced text preprocessing with NLTK
- **Feature Engineering**: 20+ linguistic and structural features for comprehensive analysis
- **Real-time Analysis**: Instant email classification with confidence scores
- **Beautiful UI**: Dark galaxy theme with animated starfield and nebula effects
- **Interactive Charts**: Dynamic visualization of prediction history using Chart.js
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Auto History Management**: Automatic clearing of prediction history for clean workflow

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Modern web browser

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/phishing-detection-system.git
   cd phishing-detection-system
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data**
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"
   ```

### Running the Application

1. **Start the backend server**
   ```bash
   python colab_backend.py
   ```
   The server will start on `http://127.0.0.1:5000`

2. **Train the model** (first time only)
   ```bash
   python train_model.py
   ```

3. **Open the frontend**
   - Open `index.html` in your web browser
   - Or serve it using a local server:
     ```bash
     python -m http.server 8000
     ```
   - Navigate to `http://localhost:8000`

## 📁 Project Structure

```
phishing-detection-system/
├── colab_backend.py          # Flask backend server
├── train_model.py            # Model training script
├── index.html               # Main web interface
├── style.css                # Dark galaxy theme styling
├── script.js                # Frontend JavaScript logic
├── CEAS_08.csv              # Training dataset
├── requirements.txt         # Python dependencies
└── README.md               # This documentation
```

## 🔧 API Endpoints

### POST `/train`
Trains the ensemble model on the provided dataset.

**Request Body:**
```json
{
  "dataset_path": "CEAS_08.csv"
}
```

**Response:**
```json
{
  "status": "Model trained successfully",
  "accuracy": 0.95,
  "models": ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "Gradient Boosting"]
}
```

### POST `/predict`
Analyzes an email for phishing detection.

**Request Body:**
```json
{
  "subject": "URGENT: Your account will be suspended!",
  "body": "Dear customer, your account will be suspended within 24 hours..."
}
```

**Response:**
```json
{
  "prediction": "phishing",
  "ensemble_probability": 0.87,
  "ensemble_confidence": 0.92,
  "individual_predictions": {
    "Logistic Regression": 0.85,
    "Random Forest": 0.89,
    "XGBoost": 0.86,
    "LightGBM": 0.88,
    "Gradient Boosting": 0.84
  }
}
```

## 🎨 UI Features

### Dark Galaxy Theme
- Animated starfield background with 120+ moving stars
- Nebula gradient effects with purple, blue, and magenta colors
- Glassmorphism cards with neon borders and glows
- Responsive design that works on all devices

### Interactive Elements
- Real-time email analysis with instant results
- Dynamic risk level indicators (High/Medium/Low)
- Confidence scores and probability percentages
- Animated loading spinners and transitions

### Data Visualization
- Chart.js integration for prediction history
- White chart background for optimal readability
- Automatic history management

## 🤖 Machine Learning Models

### Ensemble Approach
The system uses a voting ensemble of 5 different models:

1. **Logistic Regression**: Fast baseline model
2. **Random Forest**: Robust tree-based classification
3. **XGBoost**: Gradient boosting with regularization
4. **LightGBM**: Light gradient boosting machine
5. **Gradient Boosting**: Traditional gradient boosting

### Feature Engineering
- **Text Features**: TF-IDF vectors, n-grams (1-3), binary features
- **Linguistic Features**: Word count, sentence count, capital ratio, punctuation analysis
- **Structural Features**: URL count, email count, phone number detection
- **Phishing Indicators**: Keyword matching across 7 categories (urgency, money, personal info, etc.)

### Preprocessing Pipeline
1. Text cleaning and normalization
2. HTML tag removal
3. URL and email tokenization
4. Stop word removal and lemmatization
5. Feature extraction and scaling
6. SMOTE balancing for imbalanced datasets

## 📊 Performance Metrics

- **Accuracy**: 95%+ on test datasets
- **AUC Score**: 0.97+ for ensemble model
- **Cross-validation**: 5-fold stratified CV
- **Feature Selection**: Top 20,000 features using f_classif

## 🛠️ Technical Stack

### Backend
- **Python 3.8+**: Core programming language
- **Flask**: Web framework for API
- **scikit-learn**: Machine learning library
- **NLTK**: Natural language processing
- **XGBoost & LightGBM**: Advanced ML algorithms
- **pandas & numpy**: Data manipulation

### Frontend
- **HTML5**: Semantic markup
- **CSS3**: Advanced styling with animations
- **JavaScript (ES6+)**: Interactive functionality
- **Chart.js**: Data visualization
- **Fetch API**: Backend communication

## 🔒 Security Features

- Input validation and sanitization
- CORS configuration for local development
- Error handling and graceful degradation
- No sensitive data storage

## 🚀 Deployment

### Local Development
```bash
# Backend
python colab_backend.py

# Frontend (optional)
python -m http.server 8000
```

### Production Deployment
1. Use a production WSGI server (Gunicorn, uWSGI)
2. Set up proper CORS configuration
3. Implement rate limiting
4. Add SSL/TLS encryption
5. Configure environment variables

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use meaningful commit messages
- Add tests for new features
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: CEAS 2008 Phishing Email Dataset
- NLTK community for natural language processing tools
- scikit-learn team for machine learning algorithms
- Chart.js for data visualization

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Email: your.email@example.com
- Documentation: [Wiki](https://github.com/yourusername/phishing-detection-system/wiki)

---

**Made with ❤️ for cybersecurity and machine learning enthusiasts** 