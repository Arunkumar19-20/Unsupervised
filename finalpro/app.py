# ==========================================================
# TEAM 8 – Employee Attrition Prediction App
# FINAL PROFESSIONAL HR VERSION
# ==========================================================

import streamlit as st
import pandas as pd
import joblib
import os

# ==========================================================
# Page Config
# ==========================================================

st.set_page_config(
    page_title="AI Employee Attrition Predictor",
    page_icon="🧠",
    layout="wide"
)

# ==========================================================
# Simple Clean UI
# ==========================================================

st.title("🧠 AI Employee Attrition Predictor")
st.subheader("Smart HR Analytics Dashboard")

# ==========================================================
# Load Model Files
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model = joblib.load(os.path.join(BASE_DIR, "team8_employee_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "team8_scaler.pkl"))
    feature_names = joblib.load(os.path.join(BASE_DIR, "team8_feature_names.pkl"))
except Exception:
    st.error("❌ Model files not found!")
    st.stop()

# ==========================================================
# Layout
# ==========================================================

left, right = st.columns([2,1])

# ==========================================================
# LEFT SIDE – PROFESSIONAL FORM
# ==========================================================

with left:
    st.subheader("📋 Employee Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 60, 30)
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        job_level = st.selectbox("Job Level", [1,2,3,4,5])

    with col2:
        monthly_income = st.number_input("Monthly Income", value=5000)
        years_company = st.number_input("Years At Company", value=5)
        overtime = st.selectbox("OverTime", ["No", "Yes"])
        environment_sat = st.selectbox("Environment Satisfaction", [1,2,3,4])
        work_life = st.selectbox("Work Life Balance", [1,2,3,4])

    # ==========================================================
    # Convert Categorical to Numeric (Must Match Training)
    # ==========================================================

    department_map = {
        "Sales": 0,
        "Research & Development": 1,
        "Human Resources": 2
    }

    gender_map = {
        "Male": 0,
        "Female": 1
    }

    marital_map = {
        "Single": 0,
        "Married": 1,
        "Divorced": 2
    }

    overtime_map = {
        "No": 0,
        "Yes": 1
    }

    input_data = {
        "Age": age,
        "Department": department_map.get(department, 0),
        "Gender": gender_map.get(gender, 0),
        "MaritalStatus": marital_map.get(marital_status, 0),
        "JobLevel": job_level,
        "MonthlyIncome": monthly_income,
        "YearsAtCompany": years_company,
        "OverTime": overtime_map.get(overtime, 0),
        "EnvironmentSatisfaction": environment_sat,
        "WorkLifeBalance": work_life
    }

    # Fill missing features safely
    for feature in feature_names:
        if feature not in input_data:
            input_data[feature] = 0

    input_df = pd.DataFrame([input_data])[feature_names]

    predict = st.button("🚀 Run AI Prediction")

# ==========================================================
# RIGHT SIDE – AI INSIGHTS
# ==========================================================

with right:
    st.subheader("📊 AI Insights")

    if predict:

        input_scaled = scaler.transform(input_df)

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_scaled)[0][1]
        else:
            probability = 0.5

        # Custom Threshold
        threshold = 0.30
        prediction = 1 if probability > threshold else 0

        # Risk Levels
        if probability < 0.25:
            risk = "Low"
        elif probability < 0.40:
            risk = "Medium"
        else:
            risk = "High"

        st.write("### Prediction:", "Leave" if prediction==1 else "Stay")
        st.write("### Risk Level:", risk)
        st.write("### Attrition Probability:", f"{probability:.2%}")

        st.progress(int(probability * 100))

    else:
        st.write("Run prediction to see AI insights.")
