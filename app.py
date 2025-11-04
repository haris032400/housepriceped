import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load model and preprocessing data
model = tf.keras.models.load_model('model/house_price_model.h5', compile=False)
X_mean, X_std, y_mean, y_std = joblib.load('model/preprocessing.pkl')

st.title("House Price Prediction")
st.write("Enter the details of the house:")

# User inputs (text boxes, empty by default)
marlas_input = st.text_input("Marlas")
portions_input = st.text_input("Portions")
age_input = st.text_input("Age of the house (years)")

if st.button("Predict"):
    try:
        # Convert inputs to float
        marlas = float(marlas_input)
        portions = float(portions_input)
        age = float(age_input)

        # Normalize input
        input_data = np.array([[marlas, portions, age]])
        input_norm = (input_data - X_mean) / X_std

        # Predict
        prediction_norm = model.predict(input_norm)
        prediction = prediction_norm[0][0] * y_std + y_mean

        st.success(f"Predicted House Price: Rs {prediction:.2f}M")
    except ValueError:
        st.error("Please enter valid numbers in all fields.")
    except Exception as e:
        st.error(f"Error: {e}")
