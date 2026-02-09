import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="centered"
)

# ---------------- LOAD MODEL & SCALER ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
kmeans = joblib.load(os.path.join(BASE_DIR, "cluster_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scalar.pkl"))

# ---------------- ADVANCED UI CSS ----------------
st.markdown("""
<style>

/* Animated gradient background */
.stApp {
    background: linear-gradient(-45deg, #141e30, #243b55, #1f4037, #99f2c8);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Glass card */
.block-container {
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(12px);
    padding: 2rem;
    border-radius: 20px;
    animation: fadeIn 1.2s ease-in-out;
}

/* Fade-in animation */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(15px);}
    to {opacity: 1; transform: translateY(0);}
}

/* Button animation */
.stButton > button {
    background: linear-gradient(135deg, #ff512f, #dd2476);
    color: white;
    font-size: 17px;
    padding: 0.6rem 1.8rem;
    border-radius: 12px;
    border: none;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: scale(1.08);
    box-shadow: 0px 8px 20px rgba(0,0,0,0.35);
}

/* Input boxes */
input {
    border-radius: 10px !important;
}

/* Text color */
h1, h2, h3, p, label {
    color: #ffffff !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🛍️ Mall Customer Segmentation")
st.write("Predict customer group using **K-Means Clustering**")

# ---------------- INPUT SECTION ----------------
st.subheader("📥 Enter Customer Details")

annual_income = st.number_input(
    "Annual Income (k$)",
    min_value=0.0,
    value=50.0
)

spending_score = st.number_input(
    "Spending Score (1–100)",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Cluster"):
    input_data = np.array([[annual_income, spending_score]])
    scaled_data = scaler.transform(input_data)
    cluster = kmeans.predict(scaled_data)[0]

    st.success(f"✅ Customer belongs to **Cluster {cluster}**")

    cluster_info = {
        0: "💸 Low Income – Low Spending",
        1: "💰 High Income – High Spending",
        2: "💰 low Income – high Spending",
        3: "💸 high Income – low Spending",
        4: "⚖️ Average Income – Average Spending"
    }

    st.info(cluster_info.get(cluster, "Cluster description not available"))

    ''''''
# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("✨ K-Means Customer Segmentation | Streamlit UI Enhanced")
