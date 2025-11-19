import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Title
st.title("📈 Stock Price Prediction App")
st.write("Predict next-day stock closing price using a trained ML model.")

# Load model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.subheader("Enter Today's Stock Data")

# User inputs
open_price = st.number_input("Open Price", min_value=0.0)
high_price = st.number_input("High Price", min_value=0.0)
low_price = st.number_input("Low Price", min_value=0.0)
close_price = st.number_input("Close Price", min_value=0.0)
volume = st.number_input("Volume", min_value=0.0)

# Generate additional features like the training notebook
def compute_features(open_p, high_p, low_p, close_p, vol):
    df = pd.DataFrame({
        "open": [open_p],
        "high": [high_p],
        "low": [low_p],
        "close": [close_p],
        "volume": [vol]
    })

    # Daily return
    df["return"] = (df["close"] - df["open"]) / df["open"]

    # Moving averages (for a single input → use close price itself)
    df["ma5"] = df["close"]
    df["ma10"] = df["close"]

    return df

if st.button("Predict Next-Day Close"):
    # Prepare feature row
    features = compute_features(open_price, high_price, low_price, close_price, volume)

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]

    st.success(f"Predicted Next-Day Closing Price: ₹{prediction:.2f}")
