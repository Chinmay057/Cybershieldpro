Phishing Email Detector - Backend & Frontend Integration
======================================================

1. Backend (Python Flask API)
----------------------------
- File: colab_backend.py
- Requirements: Python 3.8+, Flask, flask-cors, scikit-learn, imbalanced-learn, xgboost, lightgbm, nltk, pandas, numpy, etc.
- To train the model, send a POST request to /train with JSON: {"csv_path": "path/to/your/CEAS_08_fixed.csv"}
- To predict, send a POST request to /predict with JSON: {"subject": "...", "body": "..."}
- The backend must be running for the frontend to work.

2. Frontend (HTML/CSS/JS)
-------------------------
- Files: index.html, style.css, script.js
- Open index.html in your browser.
- In script.js, set BACKEND_URL to your backend's /predict endpoint (e.g., http://127.0.0.1:5000/predict or your ngrok URL).

3. Running Locally
------------------
- Install dependencies: pip install -r requirements.txt (create one if needed)
- Run: python colab_backend.py
- Use Postman or curl to test /train and /predict endpoints.
- Open index.html and use the web interface.

4. Notes
--------
- The backend must be trained before predictions can be made.
- For Colab, use flask-ngrok and update BACKEND_URL accordingly.
- For local use, update csv_path in /train to your local CSV file. 