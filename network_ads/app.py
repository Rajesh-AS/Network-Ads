import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("knn_model.pkl")

st.set_page_config(page_title="Social Network Ads Prediction", layout="centered")

st.title("📊 Social Network Ads Prediction")
st.write("Predict whether a user will purchase a product.")

# User inputs
age = st.number_input("Age", min_value=18, max_value=60, value=25)
salary = st.number_input(
    "Estimated Salary", min_value=10000, max_value=200000, value=50000
)

# Prediction
if st.button("Predict"):
    input_data = np.array([[age, salary]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ User WILL purchase the product")
    else:
        st.error("❌ User will NOT purchase the product")
