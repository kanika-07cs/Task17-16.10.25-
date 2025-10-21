# app.py
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Load pre-trained objects
# -----------------------------
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("dbscan_model.pkl", "rb") as f:
    dbscan = pickle.load(f)

st.title("Patient Clustering App")
st.write("Enter patient details to predict cluster assignment using KMeans and DBSCAN.")

# -----------------------------
# Input fields for all features
# -----------------------------
def user_input_features():
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    gender = st.selectbox("Gender", ["Male", "Female"])
    chest_pain_type = st.selectbox("Chest Pain Type", ["Type1", "Type2", "Type3", "Type4"])
    blood_pressure = st.number_input("Blood Pressure", min_value=50, max_value=200, value=120)
    cholesterol = st.number_input("Cholesterol", min_value=100, max_value=500, value=200)
    max_heart_rate = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)
    exercise_angina = st.selectbox("Exercise Angina", ["Yes", "No"])
    plasma_glucose = st.number_input("Plasma Glucose", min_value=50, max_value=500, value=120)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=80)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    diabetes_pedigree = st.number_input("Diabetes Pedigree", min_value=0.0, max_value=5.0, value=0.5)
    hypertension = st.selectbox("Hypertension", ["Yes", "No"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    smoking_status = st.selectbox("Smoking Status", ["Smoker", "Non-Smoker"])

    data = {
        'age': age,
        'gender': gender,
        'chest_pain_type': chest_pain_type,
        'blood_pressure': blood_pressure,
        'cholesterol': cholesterol,
        'max_heart_rate': max_heart_rate,
        'exercise_angina': exercise_angina,
        'plasma_glucose': plasma_glucose,
        'skin_thickness': skin_thickness,
        'insulin': insulin,
        'bmi': bmi,
        'diabetes_pedigree': diabetes_pedigree,
        'hypertension': hypertension,
        'residence_type': residence_type,
        'smoking_status': smoking_status
    }
    return pd.DataFrame([data])

df_input = user_input_features()

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict Clusters"):

    # Encode categorical features
    categorical_cols = ['gender', 'chest_pain_type', 'exercise_angina', 
                        'hypertension','residence_type','smoking_status']

    for col in categorical_cols:
        le = LabelEncoder()
        df_input[col] = le.fit_transform(df_input[col])

    # Ensure columns order matches training
    features = ['age','gender','chest_pain_type','blood_pressure','cholesterol',
                'max_heart_rate','exercise_angina','plasma_glucose','skin_thickness',
                'insulin','bmi','diabetes_pedigree','hypertension',
                'residence_type','smoking_status']

    X_scaled = scaler.transform(df_input[features])

    # Predict clusters
    kmeans_cluster = kmeans.predict(X_scaled)[0]
    dbscan_cluster = dbscan.fit_predict(X_scaled)[0]

    # Display results
    st.success(f"KMeans Cluster: {kmeans_cluster}")
    if dbscan_cluster == -1:
        st.warning("DBSCAN Cluster: Noise")
    else:
        st.info(f"DBSCAN Cluster: {dbscan_cluster}")
