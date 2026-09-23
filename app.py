"""
CardioML — Flask Web Application Backend
Compliant with Darshan University MLDL SOP Specification (Weeks 6 & 7)
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Base directory for resolving artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)

def get_file_path(filename):
    """Finds file in root or cardio_frontend folder."""
    root_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(root_path):
        return root_path
    sub_path = os.path.join(BASE_DIR, "cardio_frontend", filename)
    if os.path.exists(sub_path):
        return sub_path
    return root_path

model_error = None
# Load ML artifacts
try:
    with open(get_file_path("model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(get_file_path("scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(get_file_path("feature_order.pkl"), "rb") as f:
        feature_order = pickle.load(f)
    with open(get_file_path("model_metadata.json"), "r", encoding="utf-8") as f:
        model_metadata = json.load(f)
    print("Successfully loaded model, scaler, feature order, and metadata.")
except Exception as e:
    model_error = f"{type(e).__name__}: {str(e)}"
    print(f"Warning loading artifacts: {model_error}")
    model = None
    scaler = None
    feature_order = [
        'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
        'cholesterol', 'gluc', 'smoke', 'alco', 'active',
        'age_years', 'bmi', 'pulse_pressure'
    ]
    model_metadata = {
        "model_name": "GradientBoostingClassifier",
        "library": "scikit-learn",
        "trained_at": "September 2026",
        "feature_count": 13,
        "hyperparameters": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_samples_leaf": 3
        },
        "performance": {
            "accuracy": 73.0,
            "f1_score": 71.5,
            "roc_auc": 79.9
        },
        "feature_importance": {
            "ap_hi": 70.4,
            "age_years": 13.5,
            "cholesterol": 7.4,
            "bmi": 2.8,
            "weight": 1.3,
            "ap_lo": 1.2,
            "height": 0.8,
            "gluc": 0.7,
            "pulse_pressure": 0.7,
            "active": 0.7,
            "smoke": 0.3,
            "alco": 0.2,
            "gender": 0.1
        }
    }

continuous_cols = ['age_years', 'height', 'weight', 'bmi', 'ap_hi', 'ap_lo', 'pulse_pressure']

def perform_prediction(patient_dict):
    """Executes validation, derived feature calculation, scaling, and inference."""
    # Derived calculations
    height = float(patient_dict.get('height', 170))
    weight = float(patient_dict.get('weight', 70))
    ap_hi = float(patient_dict.get('ap_hi', 120))
    ap_lo = float(patient_dict.get('ap_lo', 80))

    height_m = height / 100.0 if height > 0 else 1.70
    bmi = round(weight / (height_m ** 2), 2)
    pulse_pressure = int(round(ap_hi - ap_lo))

    patient_dict['bmi'] = bmi
    patient_dict['pulse_pressure'] = pulse_pressure

    # Build DataFrame matching feature order
    input_df = pd.DataFrame([patient_dict])[feature_order]

    # Standard scale continuous features
    input_df[continuous_cols] = scaler.transform(input_df[continuous_cols])

    # Run inference
    pred_class = int(model.predict(input_df)[0])
    pred_proba = float(model.predict_proba(input_df)[0][1])

    # Identify contributing factors
    risk_factors = []
    if ap_hi >= 140 or ap_lo >= 90:
        risk_factors.append(f"Stage 1/2 Hypertension detected (BP {int(ap_hi)}/{int(ap_lo)} mmHg)")
    elif ap_hi >= 120:
        risk_factors.append(f"Elevated blood pressure (Systolic {int(ap_hi)} mmHg)")

    if bmi >= 30.0:
        risk_factors.append(f"Obesity range Body Mass Index ({bmi} kg/m^2)")
    elif bmi >= 25.0:
        risk_factors.append(f"Overweight Body Mass Index ({bmi} kg/m^2)")

    if int(patient_dict.get('cholesterol', 1)) == 2:
        risk_factors.append("Cholesterol level above normal (200-239 mg/dL)")
    elif int(patient_dict.get('cholesterol', 1)) == 3:
        risk_factors.append("Cholesterol level well above normal (>= 240 mg/dL)")

    if int(patient_dict.get('gluc', 1)) == 2:
        risk_factors.append("Fasting blood glucose above normal (pre-diabetic range)")
    elif int(patient_dict.get('gluc', 1)) == 3:
        risk_factors.append("Fasting blood glucose well above normal (>= 126 mg/dL)")

    if int(patient_dict.get('smoke', 0)) == 1:
        risk_factors.append("Active tobacco smoking (accelerates arterial plaque)")

    if int(patient_dict.get('alco', 0)) == 1:
        risk_factors.append("Regular alcohol intake")

    if int(patient_dict.get('active', 1)) == 0:
        risk_factors.append("Sedentary lifestyle (insufficient physical cardiovascular activity)")

    if pulse_pressure > 50:
        risk_factors.append(f"Widened pulse pressure ({pulse_pressure} mmHg, indicates arterial stiffness)")

    return {
        "prediction": pred_class,
        "probability": round(pred_proba * 100, 1),
        "risk_level": "High Risk" if pred_class == 1 else "Low Risk",
        "bmi": bmi,
        "pulse_pressure": pulse_pressure,
        "risk_factors": risk_factors
    }

# ----------------- Web Routes -----------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = None
    form_data = None
    if request.method == 'POST':
        form_data = {
            'age_years': float(request.form.get('age_years', 50)),
            'gender': int(request.form.get('gender', 1)),
            'height': float(request.form.get('height', 165)),
            'weight': float(request.form.get('weight', 70)),
            'ap_hi': float(request.form.get('ap_hi', 120)),
            'ap_lo': float(request.form.get('ap_lo', 80)),
            'cholesterol': int(request.form.get('cholesterol', 1)),
            'gluc': int(request.form.get('gluc', 1)),
            'smoke': int(request.form.get('smoke', 0)),
            'alco': int(request.form.get('alco', 0)),
            'active': int(request.form.get('active', 1))
        }
        if model is not None and scaler is not None:
            result = perform_prediction(form_data.copy())

    return render_template('predict.html', result=result, form_data=form_data)

@app.route('/model')
def model_info():
    return render_template('model.html', metadata=model_metadata)

@app.route('/insights')
def insights():
    return render_template('insights.html')

@app.route('/disclaimer')
def disclaimer():
    return render_template('disclaimer.html')

# ----------------- REST API Endpoints (Week 7) -----------------

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if model is None or scaler is None:
        return jsonify({"status": "error", "message": "Model not loaded on server"}), 500

    try:
        data = request.get_json(force=True)
        required_keys = ['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        for k in required_keys:
            if k not in data:
                return jsonify({"status": "error", "message": f"Missing input parameter: {k}"}), 400

        result = perform_prediction(data)
        result["status"] = "success"
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/model-info', methods=['GET'])
def api_model_info():
    return jsonify(model_metadata)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "CardioML",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "model_error": model_error,
        "base_dir": BASE_DIR,
        "files": os.listdir(BASE_DIR) if os.path.exists(BASE_DIR) else []
    })

if __name__ == '__main__':
    # Local development server
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
