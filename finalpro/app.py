# ==========================================================
# TEAM 8 – Employee Attrition Prediction App
# PREMIUM UI + PROFESSIONAL HR FORM
# ==========================================================

import streamlit as st
import pandas as pd
import joblib
import os

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="AI Employee Attrition Predictor",
    page_icon="🧠",
    layout="wide"
)

# ==========================================================
# Premium Dashboard CSS
# ==========================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #1c3b47, #274c5e);
}
h1, h2, h3, h4 {
    color: white !important;
}
label {
    color: #e2e8f0 !important;
    font-weight: 500;
}
.card {
    background: rgba(255,255,255,0.10);
    backdrop-filter: blur(18px);
    padding: 35px;
    border-radius: 20px;
    box-shadow: 0 0 25px rgba(0,191,255,0.2);
}
.kpi {
    background: linear-gradient(145deg, #1e293b, #334155);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    color: white;
    box-shadow: 0 0 20px rgba(0,191,255,0.4);
    margin-bottom: 20px;
}
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    font-weight: bold;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# Header
# ==========================================================

st.markdown("<h1 style='text-align:center;'>🧠 AI Employee Attrition Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:#cbd5e1;'>Smart HR Analytics Dashboard</h4>", unsafe_allow_html=True)
st.write("")

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
# LEFT SIDE – PREMIUM HR FORM
# ==========================================================

with left:
    st.markdown('<div class="card"><h3>📋 Employee Information</h3>', unsafe_allow_html=True)

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
    # Encoding Maps (Must Match Training Encoding)
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

    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# RIGHT SIDE – PREMIUM AI INSIGHTS
# ==========================================================

with right:
    st.markdown('<div class="card"><h3>📊 AI Insights</h3>', unsafe_allow_html=True)

    if predict:

        input_scaled = scaler.transform(input_df)

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_scaled)[0][1]
        else:
            probability = 0.5

        threshold = 0.30
        prediction = 1 if probability > threshold else 0

        # Risk Levels adjusted for 0.30–0.38 range
        if probability < 0.30:
            risk = "Low"
            color = "#16a34a"
        elif probability < 0.35:
            risk = "Medium"
            color = "#f59e0b"
        else:
            risk = "High"
            color = "#ef4444"

        st.markdown(f"""
        <div class="kpi">
            <h3>Prediction</h3>
            <h2>{'Leave' if prediction==1 else 'Stay'}</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="kpi">
            <h3>Risk Level</h3>
            <h2 style='color:{color};'>{risk}</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="kpi">
            <h3>Attrition Probability</h3>
            <h2>{probability:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(probability * 100))

    else:
        st.write("Run prediction to see AI insights.")

    st.markdown('</div>', unsafe_allow_html=True)
