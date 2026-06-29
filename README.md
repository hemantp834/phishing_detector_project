# phishing_detector_project
A Machine Learning-based Phishing URL Detection system that analyzes URL features and classifies websites as Safe or Phishing using multiple trained ML models. Built with Python, scikit-learn, and Flask for real-time URL prediction.

# PhishGuard AI –  Phishing URL Detection System

Ensemble machine-learning web application that detects phishing URLs in real time
using **5 trained models**, 28 extracted features, and explainable AI indicators.

## Features
- **5 ML Models**: Random Forest, Decision Tree, KNN, Naive Bayes, XGBoost
- **Ensemble voting** with majority-decision logic
- **28 URL features** extracted automatically (entropy, TLD, IP, keywords, …)
- **Risk score** 0–100, confidence %, suspicious indicator list
- **Dashboard** with Chart.js charts (pie, bar, radar)
- **Scan history** with CSV export — no database required
- **Dark cybersecurity UI** — Bootstrap 5 + custom CSS

## Quick Start

```bash
# 1. Clone / extract the project
cd phishing_detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train all 5 models  (one-time, ~30 seconds)
python train_models.py

# 4. Start the web server
python app.py
```

Open **http://localhost:5000** in your browser.

## Project Structure

```
phishing_detector/
├── app.py                 # Flask backend
├── train_models.py        # Dataset generation + model training
├── feature_extractor.py   # URL feature extraction module
├── requirements.txt
├── README.md
├── history.json           # Scan history (auto-created)
├── statistics.json        # Aggregate stats (auto-created)
├── model_results.json     # Accuracy/F1 per model (auto-created)
├── models/                # .pkl files (created by train_models.py)
│   ├── scaler.pkl
│   ├── random_forest.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   └── xgboost.pkl
├── dataset/               # CSV dataset (created by train_models.py)
│   └── phishing_dataset.csv
└── templates/
    ├── base.html
    ├── index.html
    ├── result.html
    ├── dashboard.html
    ├── history.html
    └── models.html
```

## Pages

| Route | Page |
|---|---|
| `/` | URL scanner (home) |
| `/dashboard` | Analytics dashboard |
| `/history` | Scan history + CSV export |
| `/models` | Model accuracy comparison |
| `/api/scan` | JSON REST endpoint |

## REST API

```bash
curl -X POST http://localhost:5000/api/scan \
  -H "Content-Type: application/json" \
  -d '{"url": "https://paypal-login.suspicious.tk", "model": "random_forest"}'
```

Response:
```json
{"url": "...", "label": "Phishing", "confidence": 97.3}
```

## Models Trained On

A synthetic dataset of 6,000 labelled URLs (3,000 safe / 3,000 phishing) generated
from statistical patterns observed in real phishing datasets.  
Replace `dataset/phishing_dataset.csv` with the
[PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
from UCI for production accuracy.

<img width="1917" height="1030" alt="Screenshot 2026-06-29 160505" src="https://github.com/user-attachments/assets/753d67f2-383a-48c7-a268-15b4fa070778" />


