"""
We flip to our convention:      0 = safe,               1 = phishing
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATASET_DIR= os.path.join(BASE_DIR, 'dataset')
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# ── Exact column names from PhiUSIIL dataset that we use as features ──────
# These 24 columns are numeric, meaningful and cover URL + page structure.
PHIUSIIL_FEATURE_COLS = [
    'URLLength',
    'DomainLength',
    'IsDomainIP',
    'TLDLegitimateProb',
    'URLCharProb',
    'TLDLength',
    'NoOfSubDomain',
    'HasObfuscation',
    'NoOfObfuscatedChar',
    'ObfuscationRatio',
    'NoOfLettersInURL',
    'LetterRatioInURL',
    'NoOfDegitsInURL',
    'DegitRatioInURL',
    'NoOfEqualsInURL',
    'NoOfQMarkInURL',
    'NoOfAmpersandInURL',
    'NoOfOtherSpecialCharsInURL',
    'SpacialCharRatioInURL',
    'IsHTTPS',
    'NoOfURLRedirect',
    'HasObfuscation',
    'Bank',
    'Pay',
    'HasPasswordField',
    'HasHiddenFields',
    'HasExternalFormSubmit',
    'URLSimilarityIndex',
    'CharContinuationRate',
]
# Deduplicate while preserving order
seen = set()
PHIUSIIL_FEATURE_COLS = [c for c in PHIUSIIL_FEATURE_COLS
                          if not (c in seen or seen.add(c))]


def load_dataset():
    csv_path = os.path.join(DATASET_DIR, 'PhiUSIIL_Phishing_URL_Dataset.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}\n"
            "Download from: https://archive.ics.uci.edu/dataset/967"
        )

    print(f"[*] Loading PhiUSIIL dataset … ({csv_path})")
    df = pd.read_csv(csv_path)
    print(f"    Rows: {len(df)}   Columns: {len(df.columns)}")
    print(f"    Raw label counts: {df['label'].value_counts().to_dict()}")

    # PhiUSIIL: label 1 = legitimate → flip → 0 = safe, 1 = phishing
    df['label'] = 1 - df['label']
    print(f"    After flip  → Safe(0): {(df['label']==0).sum()}  Phishing(1): {(df['label']==1).sum()}")

    # Keep only the feature columns we need + label
    missing = [c for c in PHIUSIIL_FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"    [!] Missing columns (will fill with 0): {missing}")
    for c in missing:
        df[c] = 0

    X = df[PHIUSIIL_FEATURE_COLS].fillna(0).values
    y = df['label'].values
    return X, y


def train_and_save():
    X, y = load_dataset()

    # Save the feature column names so app.py can use them
    joblib.dump(PHIUSIIL_FEATURE_COLS, os.path.join(MODELS_DIR, 'feature_cols.pkl'))

    print(f"\n[*] Features: {len(PHIUSIIL_FEATURE_COLS)}  Samples: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    print("[✓] Scaler saved")

    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_leaf=2,
            random_state=42, n_jobs=-1),
        'decision_tree': DecisionTreeClassifier(
            max_depth=15, min_samples_leaf=2, random_state=42),
        'knn':           KNeighborsClassifier(n_neighbors=9, n_jobs=-1),
        'naive_bayes':   GaussianNB(),
        'xgboost':       XGBClassifier(
            n_estimators=300, max_depth=7, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
            random_state=42, eval_metric='logloss', verbosity=0),
    }
    NEEDS_SCALE = {'knn', 'naive_bayes'}
    results = {}

    print("\n[*] Training …\n")
    for name, model in models.items():
        Xtr = X_tr_sc if name in NEEDS_SCALE else X_train
        Xte = X_te_sc if name in NEEDS_SCALE else X_test
        model.fit(Xtr, y_train)
        yp   = model.predict(Xte)
        acc  = round(accuracy_score(y_test, yp)*100, 2)
        prec = round(precision_score(y_test, yp, zero_division=0)*100, 2)
        rec  = round(recall_score(y_test, yp, zero_division=0)*100, 2)
        f1   = round(f1_score(y_test, yp, zero_division=0)*100, 2)
        cm   = confusion_matrix(y_test, yp).tolist()
        results[name] = {
            'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1_score': f1, 'confusion_matrix': cm
        }
        joblib.dump(model, os.path.join(MODELS_DIR, f'{name}.pkl'))
        print(f"  ✓ {name:<18}  Acc={acc}%  Prec={prec}%  Rec={rec}%  F1={f1}%")

    with open(os.path.join(BASE_DIR, 'model_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\n[✓] model_results.json saved")

    for fname, default in [
        ('history.json', []),
        ('statistics.json', {'total': 0, 'safe': 0, 'phishing': 0})
    ]:
        fpath = os.path.join(BASE_DIR, fname)
        if not os.path.exists(fpath):
            with open(fpath, 'w') as f:
                json.dump(default, f)

    print("\n Training complete!  Run:  python app.py\n")


if __name__ == '__main__':
    train_and_save()
