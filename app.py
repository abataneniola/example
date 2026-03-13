import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ------------------------
# Load model, scaler, and encoders
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "best_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
le_gender = joblib.load(os.path.join(BASE_DIR, "label_encoder_gender.pkl"))
le_diabetic = joblib.load(os.path.join(BASE_DIR, "label_encoder_diabetic.pkl"))
le_smoker = joblib.load(os.path.join(BASE_DIR, "label_encoder_smoker.pkl"))

# ------------------------
# Streamlit app config
# ------------------------
st.set_page_config(page_title="Insurance Claim Predictor", layout="centered")
st.title("Health Insurance Payment Prediction App")
st.info("This model predicts insurance cost based on your health and demographic data.")
st.info("Note: Use 0 = Male / No, 1 = Female / Yes for selections below")

# ------------------------
# Default values for inputs
# ------------------------
DEFAULTS = {
    "age": 30,
    "bmi": 25.0,
    "children": 0,
    "bloodpressure": 120,
    "gender_input": 0,
    "diabetic_input": 0,
    "smoker_input": 0
}

# Initialize session state if not present
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ------------------------
# Reset callback
# ------------------------
def reset_inputs():
    for key, val in DEFAULTS.items():
        st.session_state[key] = val

# ------------------------
# User input form
# ------------------------
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, key="age")
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, key="bmi")
        children = st.number_input("Number of Children", min_value=0, max_value=10, key="children")
        
    with col2:
        bloodpressure = st.number_input("Blood Pressure", min_value=60, max_value=200, key="bloodpressure")
        gender_input = st.selectbox("Gender (0=Male, 1=Female)", options=[0,1], key="gender_input")
        diabetic_input = st.selectbox("Diabetic (0=No, 1=Yes)", options=[0,1], key="diabetic_input")
        smoker_input = st.selectbox("Smoker (0=No, 1=Yes)", options=[0,1], key="smoker_input")
    
    submitted = st.form_submit_button("Predict Payment")
    reset = st.form_submit_button("Reset Inputs", on_click=reset_inputs)

# ------------------------
# Prediction logic
# ------------------------
if submitted:
    input_data = pd.DataFrame({
        "age": [st.session_state.age],
        "gender": [st.session_state.gender_input],
        "bmi": [st.session_state.bmi],
        "bloodpressure": [st.session_state.bloodpressure],
        "diabetic": [st.session_state.diabetic_input],
        "children": [st.session_state.children],
        "smoker": [st.session_state.smoker_input]
    })
    
    # Scale numeric columns
    num_cols = ["age", "bmi", "bloodpressure", "children"]
    input_data[num_cols] = scaler.transform(input_data[num_cols])
    
    # Predict
    prediction = model.predict(input_data)[0]
    st.success(f"Your Estimated Insurance Payment Cost is : ${prediction:,.2f}")