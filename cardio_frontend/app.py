import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Cardio Risk Predictor", page_icon="❤️")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("feature_order.pkl", "rb") as f:
    feature_order = pickle.load(f)

continuous_cols = ['age_years', 'height', 'weight', 'bmi', 'ap_hi', 'ap_lo', 'pulse_pressure']

st.title("Cardiovascular Disease Risk Predictor")
st.write("Enter the patient details below to get a prediction from the model.")

col1, col2 = st.columns(2)

with col1:
    age_years = st.number_input("Age (years)", min_value=18, max_value=100, value=50)
    gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
    height = st.number_input("Height (cm)", min_value=120, max_value=220, value=165)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=70, max_value=250, value=120)
    ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=180, value=80)
    active = st.selectbox("Physically active?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    cholesterol = st.selectbox("Cholesterol", options=[1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Above normal", 3: "Well above normal"}[x])
    gluc = st.selectbox("Glucose", options=[1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Above normal", 3: "Well above normal"}[x])
    smoke = st.selectbox("Smokes?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    alco = st.selectbox("Drinks alcohol?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

bmi = weight / ((height / 100) ** 2)
pulse_pressure = ap_hi - ap_lo

st.caption(f"Calculated BMI: {bmi:.1f}  |  Pulse pressure: {pulse_pressure}")

if st.button("Predict"):
    row = {
        "gender": gender,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active,
        "age_years": age_years,
        "bmi": bmi,
        "pulse_pressure": pulse_pressure,
    }
    input_df = pd.DataFrame([row])[feature_order]
    input_df[continuous_cols] = scaler.transform(input_df[continuous_cols])

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.divider()
    if pred == 1:
        st.error(f"Model prediction: Higher risk of cardiovascular disease ({proba*100:.1f}% probability)")
    else:
        st.success(f"Model prediction: Lower risk of cardiovascular disease ({proba*100:.1f}% probability)")

    st.progress(min(max(proba, 0.0), 1.0))
    st.caption("This is a prediction from a Gradient Boosting model trained on the cardio dataset. Not medical advice.")
