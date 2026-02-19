# =========================================
# TITANIC FEATURE SELECTION APP
# =========================================

import streamlit as st
import numpy as np
import joblib

# =========================================
# Page Configuration
# =========================================
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# =========================================
# Custom CSS (Beautiful Gradient UI)
# =========================================
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #1e3c72, #2a5298);
}
h1 {
    color: white;
    text-align: center;
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
}
.prediction-box {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =========================================
# Title
# =========================================
st.markdown("<h1>🚢 Titanic Survival Prediction App</h1>", unsafe_allow_html=True)

# =========================================
# Load Models
# =========================================
models = {
    "Chi-Square": "model_chi_square.pkl",
    "Information Gain": "model_information_gain.pkl",
    "Fisher Score": "model_fisher.pkl",
    "Forward Selection": "model_forward.pkl",
    "Backward Elimination": "model_backward.pkl",
    "RFE": "model_rfe.pkl",
    "Exhaustive Selection": "model_exhaustive.pkl"
}

selected_model_name = st.selectbox("🔍 Select Feature Selection Method", list(models.keys()))

model, selector = joblib.load(models[selected_model_name])

# =========================================
# Input Section
# =========================================
st.subheader("Enter Passenger Details")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Pclass", [1, 2, 3])
    sex = st.selectbox("Sex", ["Male", "Female"])
    age = st.slider("Age", 1, 80, 25)

with col2:
    sibsp = st.slider("Siblings/Spouses Aboard", 0, 8, 0)
    parch = st.slider("Parents/Children Aboard", 0, 6, 0)
    fare = st.slider("Fare", 0.0, 600.0, 50.0)

# Encode Sex
sex = 1 if sex == "Male" else 0

# Full feature array (original order)
input_data = np.array([[pclass, sex, age, sibsp, parch, fare]])

# =========================================
# Prediction Button
# =========================================
if st.button("🚀 Predict Survival"):

    # Apply selector if needed
    if hasattr(selector, "transform"):
        input_selected = selector.transform(input_data)
    else:
        # For forward/backward/exhaustive (stored feature names)
        feature_indices = []
        original_features = ['Pclass', 'Sex', 'Age',
                             'Siblings/Spouses Aboard',
                             'Parents/Children Aboard',
                             'Fare']

        for feature in selector:
            feature_indices.append(original_features.index(feature))

        input_selected = input_data[:, feature_indices]

    prediction = model.predict(input_selected)[0]

    if prediction == 1:
        st.markdown('<div class="prediction-box">🎉 Passenger SURVIVED</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="prediction-box">💀 Passenger DID NOT SURVIVE</div>', unsafe_allow_html=True)
