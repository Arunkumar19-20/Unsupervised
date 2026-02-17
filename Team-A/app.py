import streamlit as st
import pandas as pd
import joblib

# --------------------------------
# Page Config
# --------------------------------
st.set_page_config(
    page_title="AI Performance Predictor",
    page_icon="🚀",
    layout="wide"
)

# --------------------------------
# Clean Glass UI (No div wrapping)
# --------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    font-family: 'Segoe UI', sans-serif;
}

/* Target streamlit containers directly */
section.main > div > div > div > div {
    background: rgba(255, 255, 255, 0.07);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

/* Title */
h1 {
    text-align: center;
    font-size: 40px;
    background: linear-gradient(to right, #00f5ff, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #00f5ff, #7b2ff7);
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 100%;
    border: none;
    font-size: 16px;
}

/* Labels */
label {
    color: #00f5ff !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🚀 AI Employee Performance Predictor</h1>", unsafe_allow_html=True)

# --------------------------------
# Load Model
# --------------------------------
model = joblib.load("random_forest_model.pkl")
le = joblib.load("label_encoder.pkl")

# --------------------------------
# Layout
# --------------------------------
col1, col2 = st.columns(2)

# ---------------- LEFT PANEL ----------------
with col1:
    st.markdown(
        "<h3 style='color:#00f5ff;'>👤 Employee Input</h3>",
        unsafe_allow_html=True
    )

    age = st.slider("Age", 20, 60, 30)
    experience = st.slider("Experience Years", 0, 20, 5)
    department = st.selectbox("Department", ["IT", "HR", "Sales", "Finance"])
    salary = st.number_input("Salary", 30000, 100000, 50000)
    work_hours = st.slider("Work Hours", 30, 60, 40)
    projects = st.slider("Projects Handled", 1, 10, 3)
    training = st.slider("Training Hours", 5, 50, 20)

    predict_button = st.button("Predict with AI 🚀")

# Encode department
dept_encoded = le.transform([department])[0]

# ---------------- RIGHT PANEL ----------------
with col2:
    st.markdown(
        "<h3 style='color:#00f5ff;'>📊 AI Prediction</h3>",
        unsafe_allow_html=True
    )

    if predict_button:

        input_data = pd.DataFrame({
            "Age": [age],
            "Experience_Years": [experience],
            "Department": [dept_encoded],
            "Salary": [salary],
            "Work_Hours": [work_hours],
            "Projects_Handled": [projects],
            "Training_Hours": [training]
        })

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        if prediction[0] == 1:
            st.success("⭐ High Performance Employee")
        else:
            st.error("⚠ Needs Improvement")

        st.write("### Confidence Score")
        st.progress(float(probability[0][1]))

    else:
        st.info("Enter employee details and click Predict.")
