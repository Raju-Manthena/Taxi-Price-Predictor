import streamlit as st
import pandas as pd
import pickle


# Load trained model pipeline
with open("taxi_fare_model.pkl", "rb") as file:
    model = pickle.load(file)


# Streamlit page configuration
st.set_page_config(
    page_title="Taxi Fare Predictor",
    page_icon="🚕",
    layout="centered"
)


st.title("Taxi Fare Predictor")
st.write(
    "Enter the trip details below to predict the taxi fare."
)


# Input fields
trip_distance = st.number_input(
    "Trip Distance (km)",
    min_value=0.0,
    value=5.0,
    step=0.1
)

time_of_day = st.selectbox(
    "Time of Day",
    ["Morning", "Afternoon", "Evening", "Night"]
)

day_of_week = st.selectbox(
    "Day of Week",
    ["Weekday", "Weekend"]
)

passenger_count = st.number_input(
    "Passenger Count",
    min_value=1,
    value=1,
    step=1
)

traffic_conditions = st.selectbox(
    "Traffic Conditions",
    ["Low", "Medium", "High"]
)

weather = st.selectbox(
    "Weather",
    ["Clear", "Rain", "Snow"]
)

base_fare = st.number_input(
    "Base Fare",
    min_value=0.0,
    value=3.0,
    step=0.1
)

per_km_rate = st.number_input(
    "Per Km Rate",
    min_value=0.0,
    value=1.0,
    step=0.1
)

per_minute_rate = st.number_input(
    "Per Minute Rate",
    min_value=0.0,
    value=0.2,
    step=0.01
)

trip_duration = st.number_input(
    "Trip Duration (minutes)",
    min_value=0.0,
    value=20.0,
    step=1.0
)


# Prediction
if st.button("Predict Fare"):
    input_data = pd.DataFrame({
        "Trip_Distance_km": [trip_distance],
        "Time_of_Day": [time_of_day],
        "Day_of_Week": [day_of_week],
        "Passenger_Count": [passenger_count],
        "Traffic_Conditions": [traffic_conditions],
        "Weather": [weather],
        "Base_Fare": [base_fare],
        "Per_Km_Rate": [per_km_rate],
        "Per_Minute_Rate": [per_minute_rate],
        "Trip_Duration_Minutes": [trip_duration]
    })

    prediction = model.predict(input_data)[0]

    st.subheader("Predicted Taxi Fare")
    st.success(f"${prediction:.2f}")


# Model information
st.divider()

st.subheader("Model Information")

col1, col2 = st.columns(2)

with col1:
    st.write("**Model:** Tuned Random Forest")
    st.write("**Features:** Trip-related features")

with col2:
    st.write("**CV RMSE:** 10.27")
    st.write("**Test RMSE:** 11.19")

st.write("**Test R²:** 0.95")