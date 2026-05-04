import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD FILES (SAFE)
# =========================
try:
    model = joblib.load("L_G_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# =========================
# UI
# =========================
st.set_page_config(page_title="Heart Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction")
st.markdown("Fill the details below:")

# =========================
# INPUTS
# =========================
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.slider("Cholesterol (mg/dl)", 80, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["N", "Y"])
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Flat", "Up", "Down"])

# =========================
# PREDICTION
# =========================
if st.button("Predict"):

    try:
        # Base numeric features
        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak
        }

        # One-hot encoding
        raw_input[f'Sex_{sex}'] = 1
        raw_input[f'ChestPainType_{chest_pain}'] = 1
        raw_input[f'RestingECG_{resting_ecg}'] = 1
        raw_input[f'ExerciseAngina_{exercise_angina}'] = 1
        raw_input[f'ST_Slope_{st_slope}'] = 1

        # Convert to DataFrame
        input_df = pd.DataFrame([raw_input])

        # Missing columns add karo
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Order fix karo
        input_df = input_df[expected_columns]

        # Scale
        scaled_input = scaler.transform(input_df)

        # Predict
        prediction = model.predict(scaled_input)[0]

        # Result
        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

    except Exception as e:
        st.error(f"Prediction Error: {e}")