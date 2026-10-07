
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained Random Forest model
model_package = joblib.load(
    "capstone_outputs/respiratory_random_forest_model.joblib"
)

model = model_package["model"]

# Page configuration
st.set_page_config(
    page_title="Respiratory Illness Prediction",
    page_icon="🫁",
    layout="centered"
)

# Title
st.title("Respiratory Illness Prediction")
st.write(
    "Predict the estimated probability of a respiratory-related "
    "health-unit visit among university students."
)

st.divider()

# User inputs
age = st.number_input(
    "Age",
    min_value=15,
    max_value=100,
    value=20,
    step=1
)

gender = st.selectbox(
    "Gender",
    ["F", "M"]
)

month = st.selectbox(
    "Month",
    list(range(1, 13)),
    index=0
)

# Month names
month_names = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

st.write(f"Selected month: **{month_names[month]}**")

# Prediction
if st.button("Predict", type="primary"):

    # Create seasonal features
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    # Prepare input
    new_student = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "month_sin": [month_sin],
        "month_cos": [month_cos]
    })

    # Predict probability
    probability = model.predict_proba(new_student)[0, 1]

    # Classification threshold
    if probability >= 0.50:
        prediction = "Respiratory-related"
    else:
        prediction = "Other"

    st.divider()

    st.subheader("Prediction Result")

    st.metric(
        "Respiratory-related probability",
        f"{probability * 100:.2f}%"
    )

    if prediction == "Respiratory-related":
        st.warning(
            "The model predicts a respiratory-related health-unit visit."
        )
    else:
        st.info(
            "The model predicts a non-respiratory health-unit visit."
        )

    st.write(f"**Predicted class:** {prediction}")

    st.caption(
        "This is a machine-learning prediction based on patterns "
        "in the training data and is not a medical diagnosis."
    )
