import streamlit as st
import numpy as np
import pickle

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("❤️ Heart Disease Prediction App")

st.markdown("### Enter the patient's details below:")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, step=1)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220)
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.number_input("ST depression induced by exercise", min_value=0.0, max_value=10.0, step=0.1)
slope = st.selectbox("Slope of peak exercise ST segment", ["Upsloping", "Flat", "Downsloping"])
ca = st.selectbox("Number of major vessels (0–3) colored by fluoroscopy", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", ["Normal", "Fixed defect", "Reversible defect"])

# Encoding categorical values
sex = 1 if sex == "Male" else 0
cp = {"Typical angina": 0, "Atypical angina": 1, "Non-anginal pain": 2, "Asymptomatic": 3}[cp]
fbs = 1 if fbs == "Yes" else 0
restecg = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}[restecg]
exang = 1 if exang == "Yes" else 0
slope = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}[slope]
thal = {"Normal": 1, "Fixed defect": 2, "Reversible defect": 3}[thal]

# Final input
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                        exang, oldpeak, slope, ca, thal]])
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("🚨 High Risk: The person is likely to have heart disease.")
    else:
        st.success("✅ Low Risk: The person is unlikely to have heart disease.")
