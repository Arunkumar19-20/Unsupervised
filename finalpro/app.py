# ==========================================================
# TEAM 8 – Employee Attrition Prediction App
# FINAL AI-Powered Smart Dashboard UI (Improved Prediction)
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
h1, h2, h3 {
    color: white !important;
}
label {
    color: #e2e8f0 !important;
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
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
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
except Exception as e:
    st.error("❌ Model files not found! Keep all .pkl files in same folder.")
    st.stop()

# ==========================================================
# Layout
# ==========================================================

left, right = st.columns([2, 1])

# ==========================================================
# LEFT SIDE – EMPLOYEE FORM
# ==========================================================

with left:
    st.markdown('<div class="card"><h3>📋 Employee Information Form</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # -------- Personal Details --------
    with col1:
        age = st.slider("Age", 18, 60, 30)
        distance = st.number_input("Distance From Home", min_value=0, value=5)
        education = st.selectbox("Education Level (1-5)", [1,2,3,4,5])
        job_level = st.selectbox("Job Level (1-5)", [1,2,3,4,5])
        monthly_income = st.number_input("Monthly Income", min_value=1000, value=5000)

    # -------- Work Details --------
    with col2:
        years_company = st.number_input("Years At Company", min_value=0, value=5)
        years_role = st.number_input("Years In Current Role", min_value=0, value=3)
        years_promotion = st.number_input("Years Since Last Promotion", min_value=0, value=1)
        overtime = st.selectbox("OverTime", ["No", "Yes"])
        job_satisfaction = st.selectbox("Job Satisfaction (1-4)", [1,2,3,4])

    overtime = 1 if overtime == "Yes" else 0

    # Create input dictionary
    input_data = {
        "Age": age,
        "DistanceFromHome": distance,
        "Education": education,
        "JobLevel": job_level,
        "MonthlyIncome": monthly_income,
        "YearsAtCompany": years_company,
        "YearsInCurrentRole": years_role,
        "YearsSinceLastPromotion": years_promotion,
        "OverTime": overtime,
        "JobSatisfaction": job_satisfaction
    }

    # Fill missing features
    for feature in feature_names:
        if feature not in input_data:
            input_data[feature] = 0

    input_df = pd.DataFrame([input_data])[feature_names]

    predict = st.button("🚀 Run AI Prediction")

    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# RIGHT SIDE – AI INSIGHTS (IMPROVED LOGIC)
# ==========================================================

with right:
    st.markdown('<div class="card"><h3>📊 AI Insights</h3>', unsafe_allow_html=True)

    if predict:

        input_scaled = scaler.transform(input_df)

        # Always use probability
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_scaled)[0][1]
        else:
            probability = 0.5

        # 🔥 Custom threshold (more sensitive)
        threshold = 0.35
        prediction = 1 if probability > threshold else 0

        # Debug Probability
        st.write("Raw Probability:", round(probability, 4))

        # Risk Classification
        if probability < 0.30:
            risk = "Low"
            recommendation = "Employee is stable. Maintain engagement."
            color = "#16a34a"
        elif probability < 0.60:
            risk = "Medium"
            recommendation = "Monitor performance and increase engagement."
            color = "#f59e0b"
        else:
            risk = "High"
            recommendation = "Immediate HR intervention recommended."
            color = "#ef4444"

        # KPI Cards
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
        st.success(f"🧠 AI Recommendation: {recommendation}")

    else:
        st.write("Run prediction to see AI insights.")

    st.markdown('</div>', unsafe_allow_html=True)
