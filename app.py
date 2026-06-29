"""
app.py — PhishGuard AI Flask backend
"""
import os, json, csv, io, datetime
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response
from feature_extractor import (extract_features, features_to_vector,
                                get_suspicious_indicators, calculate_risk_score)

app = Flask(__name__)
app.jinja_env.globals['enumerate'] = enumerate

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
HISTORY_F  = os.path.join(BASE_DIR, 'history.json')
STATS_F    = os.path.join(BASE_DIR, 'statistics.json')
RESULTS_F  = os.path.join(BASE_DIR, 'model_results.json')

MODEL_NAMES  = ['random_forest','decision_tree','knn','naive_bayes','xgboost']
MODEL_LABELS = {
    'random_forest': 'Random Forest',
    'decision_tree': 'Decision Tree',
    'knn':           'K-Nearest Neighbors',
    'naive_bayes':   'Naive Bayes',
    'xgboost':       'XGBoost',
}
NEEDS_SCALE = {'knn','naive_bayes'}

_models      = {}
_scaler      = None
_feature_cols = None   # loaded from feature_cols.pkl

def load_models():
    global _scaler, _feature_cols
    sp = os.path.join(MODELS_DIR, 'scaler.pkl')
    fp = os.path.join(MODELS_DIR, 'feature_cols.pkl')
    if os.path.exists(sp): _scaler = joblib.load(sp)
    if os.path.exists(fp): _feature_cols = joblib.load(fp)
    for name in MODEL_NAMES:
        p = os.path.join(MODELS_DIR, f'{name}.pkl')
        if os.path.exists(p): _models[name] = joblib.load(p)
    print(f"[+] Loaded {len(_models)} models | Features: {len(_feature_cols) if _feature_cols else 'N/A'}")

load_models()

def read_json(path, default):
    try:
        with open(path) as f: return json.load(f)
    except: return default

def write_json(path, data):
    with open(path,'w') as f: json.dump(data, f, indent=2)

def get_vector(features):
    if _feature_cols:
        return features_to_vector(features, _feature_cols)
    return []

def predict_single(model_name, feature_vec):
    if model_name not in _models or not feature_vec: return None, None
    model = _models[model_name]
    vec   = np.array(feature_vec, dtype=float).reshape(1,-1)
    if model_name in NEEDS_SCALE and _scaler:
        vec = _scaler.transform(vec)
    pred  = model.predict(vec)[0]
    try:    conf = float(max(model.predict_proba(vec)[0]))*100
    except: conf = 100.0
    return ('Phishing' if pred == 1 else 'Safe'), round(conf, 2)

def ensemble_predict(feature_vec):
    individual = {}
    votes = {'Safe':0,'Phishing':0}
    for name in MODEL_NAMES:
        label, conf = predict_single(name, feature_vec)
        if label:
            individual[name] = {'label':label,'confidence':conf}
            votes[label] += 1
    final = 'Phishing' if votes['Phishing'] > votes['Safe'] else 'Safe'
    return individual, final, votes

@app.route('/')
def index():
    return render_template('index.html', models_loaded=len(_models)>0,
                           model_names=MODEL_NAMES, model_labels=MODEL_LABELS)

@app.route('/scan', methods=['POST'])
def scan():
    url        = request.form.get('url','').strip()
    model_name = request.form.get('model','random_forest')
    if not url: return redirect(url_for('index'))
    if not _models or not _feature_cols:
        return render_template('result.html',
            error="Models not trained yet. Run: python train_models.py",
            result=None, model_labels=MODEL_LABELS)

    features   = extract_features(url)
    feat_vec   = get_vector(features)
    indicators = get_suspicious_indicators(features, url)

    if model_name == 'ensemble':
        individual, final_label, votes = ensemble_predict(feat_vec)
        confidence = round(
            sum(v['confidence'] for v in individual.values()) / max(len(individual),1), 2)
        risk_score = calculate_risk_score(features, final_label, confidence/100)
        result = dict(url=url, model='ensemble', label=final_label,
                      confidence=confidence, risk_score=risk_score,
                      individual=individual, votes=votes,
                      indicators=indicators, features=features, is_ensemble=True)
    else:
        label, confidence = predict_single(model_name, feat_vec)
        if label is None: label, confidence = 'Unknown', 0.0
        risk_score = calculate_risk_score(features, label, confidence/100)
        result = dict(url=url, model=model_name,
                      model_label=MODEL_LABELS.get(model_name, model_name),
                      label=label, confidence=confidence,
                      risk_score=risk_score, indicators=indicators,
                      features=features, is_ensemble=False)

    history = read_json(HISTORY_F, [])
    history.insert(0, dict(url=url, label=result['label'],
                           confidence=result['confidence'],
                           risk_score=risk_score, model=model_name,
                           timestamp=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    write_json(HISTORY_F, history[:500])

    stats = read_json(STATS_F, {'total':0,'safe':0,'phishing':0})
    stats['total'] += 1
    stats['safe' if result['label']=='Safe' else 'phishing'] += 1
    write_json(STATS_F, stats)

    return render_template('result.html', result=result, model_labels=MODEL_LABELS)

@app.route('/dashboard')
def dashboard():
    stats     = read_json(STATS_F, {'total':0,'safe':0,'phishing':0})
    history   = read_json(HISTORY_F, [])
    m_results = read_json(RESULTS_F, {})
    best = max(m_results, key=lambda k: m_results[k].get('accuracy',0), default='N/A') if m_results else 'N/A'
    return render_template('dashboard.html', stats=stats, recent=history[:10],
                           m_results=m_results, best_model=best, model_labels=MODEL_LABELS)

@app.route('/history')
def history_page():
    return render_template('history.html', history=read_json(HISTORY_F,[]))

@app.route('/history/clear', methods=['POST'])
def clear_history():
    write_json(HISTORY_F, [])
    write_json(STATS_F, {'total':0,'safe':0,'phishing':0})
    return redirect(url_for('history_page'))

@app.route('/history/export')
def export_history():
    history = read_json(HISTORY_F, [])
    out = io.StringIO()
    w   = csv.DictWriter(out, fieldnames=['url','label','confidence','risk_score','model','timestamp'])
    w.writeheader(); w.writerows(history); out.seek(0)
    return Response(out.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition':'attachment; filename=scan_history.csv'})

@app.route('/models')
def model_comparison():
    return render_template('models.html',
                           m_results=read_json(RESULTS_F,{}), model_labels=MODEL_LABELS)

@app.route('/api/scan', methods=['POST'])
def api_scan():
    data = request.get_json(force=True) or {}
    url  = data.get('url','').strip()
    if not url: return jsonify({'error':'URL required'}), 400
    features = extract_features(url)
    feat_vec = get_vector(features)
    label, conf = predict_single(data.get('model','random_forest'), feat_vec)
    return jsonify({'url':url,'label':label,'confidence':conf})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
