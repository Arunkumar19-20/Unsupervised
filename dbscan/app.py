# =========================================
# DBSCAN Cluster Predictor (Meaningful Version)
# =========================================

import streamlit as st
import numpy as np
import joblib
import os

# =========================================
# Page Configuration
# =========================================

st.set_page_config(
    page_title="DBSCAN Blob Cluster Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 DBSCAN Cluster Interpretation App")
st.write("Enter feature values to understand which region the point belongs to.")

# =========================================
# Load Saved DBSCAN Model
# =========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "dbscan.pkl")

try:
    dbscan = joblib.load(model_path)
except Exception as e:
    st.error("❌ dbscan.pkl not found in the same folder.")
    st.write(e)
    st.stop()

# =========================================
# Input Section
# =========================================

st.subheader("Enter Feature Values")

col1, col2 = st.columns(2)

with col1:
    f1 = st.number_input("Feature 1 (f1)", value=0.0)

with col2:
    f2 = st.number_input("Feature 2 (f2)", value=0.0)

input_point = np.array([[f1, f2]])

# =========================================
# Predict Button
# =========================================

if st.button("🚀 Predict Cluster"):

    try:
        # DBSCAN has no direct predict(), so we simulate
        combined_data = np.vstack([dbscan.components_, input_point])

        new_model = dbscan
        new_model.fit(combined_data)

        prediction = new_model.labels_[-1]

        st.subheader("📊 Result")

        if prediction == -1:
            st.warning("⚠️ This point is classified as an OUTLIER.")
            st.info("It does not belong to any dense region in the dataset.")

        else:
            # Meaning based on feature values
            if f1 > 0 and f2 > 0:
                meaning = "High Feature 1 & High Feature 2 Region"
            elif f1 < 0 and f2 < 0:
                meaning = "Low Feature 1 & Low Feature 2 Region"
            elif f1 > 0 and f2 < 0:
                meaning = "High Feature 1 & Low Feature 2 Region"
            else:
                meaning = "Low Feature 1 & High Feature 2 Region"

            st.success(f"✅ This point belongs to Cluster {prediction}")
            st.info(f"📌 Interpretation: {meaning}")

    except Exception as e:
        st.error("Prediction failed.")
        st.write(e)
