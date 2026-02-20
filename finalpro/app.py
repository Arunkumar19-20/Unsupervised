# ==========================================================
# TEAM 8 – Employee Attrition Prediction App
# FINAL SMART VERSION (Auto Categorical + Balanced Risk)
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
except Exception:
    st.error("❌ Model files not found! Keep all .pkl files in same folder.")
    st.stop()

# ==========================================================
# Layout
# ==========================================================

left, right = st.columns([2, 1])

# ==========================================================
# LEFT SIDE – SMART FORM
# ==========================================================

with left:
    st.markdown('<div class="card"><h3>📋 Employee Information Form</h3>', unsafe_allow_html=True)

    input_data = {}
    cols = st.columns(2)

    # Known categorical columns (IBM dataset style)
    categorical_cols = [
        "Department",
        "Gender",
        "MaritalStatus",
        "OverTime"
    ]

    # Satisfaction scale columns (1–4)
    satisfaction_cols = [
        "EnvironmentSatisfaction",
        "JobInvolvement",
        "JobSatisfaction",
        "WorkLifeBalance"
    ]

    for i, feature in enumerate(feature_names):
        with cols[i % 2]:

            # Categorical fields (encoded)
            if feature in categorical_cols:
                input_data[feature] = st.selectbox(feature, [0, 1, 2])

            # Satisfaction scale (1–4)
            elif feature in satisfaction_cols:
                input_data[feature] = st.selectbox(feature, [1, 2, 3, 4])

            # JobLevel (1–5)
            elif feature == "JobLevel":
                input_data[feature] = st.selectbox(feature, [1, 2, 3, 4, 5])

            # StockOptionLevel (0–3)
            elif feature == "StockOptionLevel":
                input_data[feature] = st.selectbox(feature, [0, 1, 2, 3])

            # Default numeric
            else:
                input_data[feature] = st.number_input(feature, value=0.0)

    input_df = pd.DataFrame([input_data])

    predict = st.button("🚀 Run AI Prediction")

    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# RIGHT SIDE – AI INSIGHTS
# ==========================================================

with right:
    st.markdown('<div class="card"><h3>📊 AI Insights</h3>', unsafe_allow_html=True)

    if predict:

        input_scaled = scaler.transform(input_df)

        # Get probability
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_scaled)[0][1]
        else:
            probability = 0.5

        # Custom threshold (adjusted)
        threshold = 0.30
        prediction = 1 if probability > threshold else 0

        st.write("Raw Probability:", round(probability, 4))

        # Adjusted risk classification
        if probability < 0.25:
            risk = "Low"
            recommendation = "Employee is stable. Maintain engagement."
            color = "#16a34a"

        elif probability < 0.40:
            risk = "Medium"
            recommendation = "Moderate attrition risk. Monitor employee."
            color = "#f59e0b"

        else:
            risk = "High"
            recommendation = "High attrition risk. HR intervention required."
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
