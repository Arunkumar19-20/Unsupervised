# ==========================================================
# TEAM 8 – Employee Attrition Prediction App
# FINAL AI-Powered Smart Dashboard UI
# ==========================================================

import streamlit as st
import pandas as pd
import joblib

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="AI Employee Attrition Predictor",
    page_icon="🧠",
    layout="wide"
)

# ==========================================================
# Premium AI Dashboard CSS
# ==========================================================

st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #1c3b47, #274c5e);
}

/* Headings */
h1, h2, h3, h4 {
    color: #ffffff !important;
}

/* Labels */
label {
    color: #e2e8f0 !important;
    font-weight: 500;
}

/* Glass Card */
.card {
    background: rgba(255, 255, 255, 0.10);
    backdrop-filter: blur(18px);
    padding: 40px 35px;   /* Increased padding */
    border-radius: 20px;
    box-shadow: 0 0 25px rgba(0, 191, 255, 0.2);
}


/* KPI Cards */
.kpi {
    background: linear-gradient(145deg, #1e293b, #334155);
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    color: #ffffff;
    box-shadow: 0 0 20px rgba(0, 191, 255, 0.4);
    margin-bottom: 25px;   /* Adds vertical spacing */
}


/* Inputs */
.stNumberInput input {
    background-color: #0f172a !important;
    color: white !important;
    border-radius: 8px !important;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    font-weight: bold;
    border-radius: 12px;
    height: 55px;
    font-size: 18px;
    border: none;
    box-shadow: 0 0 20px rgba(0, 114, 255, 0.6);
}

.stButton>button:hover {
    background: linear-gradient(90deg, #0072ff, #00c6ff);
    box-shadow: 0 0 25px rgba(0, 255, 255, 0.8);
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

try:
    model = joblib.load("team8_employee_model.pkl")
    scaler = joblib.load("team8_scaler.pkl")
    feature_names = joblib.load("team8_feature_names.pkl")
except:
    st.error("❌ Model files not found! Keep all .pkl files in same folder.")
    st.stop()

# ==========================================================
# Layout
# ==========================================================

left, right = st.columns([2, 1])

# ================= LEFT SIDE – INPUT =================

with left:
    st.markdown('<div class="card"><h3>Enter Employee Details</h3>', unsafe_allow_html=True)


    input_data = {}
    cols = st.columns(2)

    for i, feature in enumerate(feature_names):
        with cols[i % 2]:
            input_data[feature] = st.number_input(feature, value=0.0)

    input_df = pd.DataFrame([input_data])

    predict = st.button("🚀 Run AI Prediction")

    st.markdown('</div>', unsafe_allow_html=True)

# ================= RIGHT SIDE – AI INSIGHTS =================

with right:
    st.markdown('<div class="card"><h3>📊 AI Insights</h3>', unsafe_allow_html=True)

    if predict:
        try:
            # Scale
            input_scaled = scaler.transform(input_df)

            # Predict
            prediction = model.predict(input_scaled)[0]

            # Probability
            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(input_scaled)[0][1]
            else:
                probability = 0.5

            # Risk Logic
            if probability < 0.30:
                risk = "Low"
                recommendation = "Employee is stable. Maintain engagement & motivation."
                color = "#16a34a"
            elif probability < 0.60:
                risk = "Medium"
                recommendation = "Monitor performance & increase engagement initiatives."
                color = "#f59e0b"
            else:
                risk = "High"
                recommendation = "Immediate HR intervention recommended."
                color = "#ef4444"

            # KPI Row
            k1, k2 = st.columns(2)

            with k1:
                st.markdown(f"""
                <div class="kpi">
                    <h3>Prediction</h3>
                    <h2>{'Leave' if prediction==1 else 'Stay'}</h2>
                </div>
                """, unsafe_allow_html=True)

            with k2:
                st.markdown(f"""
                <div class="kpi">
                    <h3>Risk Level</h3>
                    <h2 style='color:{color};'>{risk}</h2>
                </div>
                """, unsafe_allow_html=True)

            # Probability Card
            st.markdown(f"""
            <div class="kpi">
                <h3>Attrition Probability</h3>
                <h2>{probability:.2%}</h2>
            </div>
            """, unsafe_allow_html=True)

            # Animated Progress
            st.progress(int(probability * 100))

            # AI Recommendation
            st.success(f"🧠 AI Recommendation: {recommendation}")

        except Exception as e:
            st.error("Prediction failed.")
            st.write(str(e))

    else:
        st.write("Run prediction to see AI insights.")

    st.markdown('</div>', unsafe_allow_html=True)
