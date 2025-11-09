import streamlit as st
import pickle
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Student Performance Predictor")
st.markdown("Predict whether a student will **pass or fail** based on personal and study-related factors.")

# Load trained model (resolve path relative to this file; allow upload as fallback)
model_path = Path(__file__).resolve().parents[1] / "models" / "decision_tree.pkl"

if model_path.exists():
    with open(model_path, "rb") as f:
        model = pickle.load(f)
else:
    st.error(f"Model not found at: {model_path}")
    st.info("Run `python train.py` from the project root to create the model, or upload a trained model file (.pkl) below.")
    uploaded = st.file_uploader("Upload decision_tree.pkl", type=["pkl"])
    if uploaded is not None:
        model = pickle.load(uploaded)
    else:
        st.stop()

# Input fields
gender = st.selectbox("Gender", ["female", "male"])
race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parent_edu = st.selectbox("Parental Education Level", [
    "some high school", "high school", "some college", "bachelor's degree", "master's degree"])
lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
prep = st.selectbox("Test Preparation Course", ["none", "completed"])

# Encoding manually (must match training LabelEncoder mapping)
def encode_inputs(gender, race, parent_edu, lunch, prep):
    gender_map = {"female": 0, "male": 1}
    race_map = {"group A": 0, "group B": 1, "group C": 2, "group D": 3, "group E": 4}
    edu_map = {"some high school": 5, "high school": 3, "some college": 4, 
               "bachelor's degree": 0, "master's degree": 2}
    lunch_map = {"standard": 1, "free/reduced": 0}
    prep_map = {"none": 0, "completed": 1}
    return np.array([[gender_map[gender], race_map[race], edu_map[parent_edu], lunch_map[lunch], prep_map[prep]]])

if st.button("Predict Performance"):
    X = encode_inputs(gender, race, parent_edu, lunch, prep)
    prediction = model.predict(X)[0]
    if prediction == 1:
        st.success("✅ The student is likely to **Pass**.")
    else:
        st.error("❌ The student is likely to **Fail**.")
