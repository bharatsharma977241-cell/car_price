import pandas as pd
import streamlit as st
import joblib

# Load trained model
model_pipeline = joblib.load(r"C:\Users\bhara\OneDrive\Desktop\car project 1\car_price_model.h5")

# Title
st.title("🚗 Car Price Prediction App")

# Show model accuracy
st.subheader("Model Performance")
st.write("The model was trained with R² Score and RMSE calculated during training.")
st.info("Note: These metrics are based on the training/testing phase and not updated live here.")

# Input fields
st.subheader("Enter Car Details")

# Numeric fields
symboling = st.number_input("Symboling", min_value=-3, max_value=3, step=1)
enginesize = st.number_input("Engine Size", min_value=50, max_value=400, step=1)
horsepower = st.number_input("Horsepower", min_value=40, max_value=300, step=1)
peakrpm = st.number_input("Peak RPM", min_value=4000, max_value=7000, step=100)
highwaympg = st.number_input("Highway MPG", min_value=10, max_value=60, step=1)
citympg = st.number_input("City MPG", min_value=5, max_value=50, step=1)

# Categorical fields
enginetype = st.selectbox("Engine Type", ["dohc", "ohcv", "ohc", "rotor", "l", "dohcv"])
cylindernumber = st.selectbox("Cylinder Number", ["two", "three", "four", "five", "six", "eight", "twelve"])
fueltype = st.selectbox("Fuel Type", ["gas", "diesel"])
carbody = st.selectbox("Car Body", ["convertible", "hatchback", "sedan", "wagon", "hardtop"])
CarName = st.text_input("Car Name (e.g., bmw 320i, audi 100ls)")

# Predict button
if st.button("Predict Price"):
    new_data = pd.DataFrame({
        'symboling': [symboling],
        'enginetype': [enginetype],
        'enginesize': [enginesize],
        'horsepower': [horsepower],
        'peakrpm': [peakrpm],
        'highwaympg': [highwaympg],
        'citympg': [citympg],
        'cylindernumber': [cylindernumber],
        'CarName': [CarName],
        'fueltype': [fueltype],
        'carbody': [carbody]
    })

    prediction = model_pipeline.predict(new_data)[0]
    rupees = prediction * 88.18  # assuming 1 USD ≈ 88.18 INR
    st.success(f"Estimated Car Price: ${prediction:,.2f} or ₹{rupees:,.2f}")
