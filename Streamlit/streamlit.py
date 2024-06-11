import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from scipy.special import boxcox1p
import requests
import io

# Load pickle files from URLs
url_lambda = 'https://raw.githubusercontent.com/juanvalno/SEC/e63f5341a8298d5e5b3a9da976689526f21c5bf1/Model/transformation_params.pkl'
url_model = 'https://raw.githubusercontent.com/juanvalno/SEC/62c1cceec41ade73e7876f45f1071fe8832eb312/Model/model.pkl'

response_lambda = requests.get(url_lambda)
response_model = requests.get(url_model)

# Check if the requests were successful
if response_lambda.status_code == 200 and response_model.status_code == 200:
    # Write the content to a buffer
    lambda_buffer = io.BytesIO(response_lambda.content)
    model_buffer = io.BytesIO(response_model.content)

    # Load the model and transformation parameters
    optimal_lambdas = pickle.load(lambda_buffer)
    model = joblib.load(model_buffer)

    # Debugging: Check the type and a few key attributes of the loaded model
    st.write(f"Model type: {type(model)}")

    # Define all expected features
    expected_features = ['POV', 'FOOD', 'ELEC', 'WATER', 'LIFE', 'HEALTH', 'SCHOOL', 'STUNTING']

    # Streamlit app
    st.title('FARM: Food Availability and Security Monitor')

    # Input fields
    input_data = {}
    col1, col2 = st.columns(2)
    for i, feature in enumerate(expected_features):
        if i < 4:
            input_data[feature] = col1.number_input(feature, value=0.0)
        else:
            input_data[feature] = col2.number_input(feature, value=0.0)

    # Create DataFrame with input data
    input_df = pd.DataFrame([input_data])

    # Ensure all inputs are numeric
    input_df = input_df.apply(pd.to_numeric)

    # Transform the inputs
    for feature, optimal_lambda in optimal_lambdas.items():
        if feature in input_df:
            input_df[feature] = boxcox1p(input_df[feature], optimal_lambda)

    # Reorder the columns to match the expected feature order
    input_df = input_df[expected_features]

    # Debugging: Check the transformed input data
    st.write("Transformed input data:")
    st.write(input_df)

    # Make predictions
    if st.button('Predict'):
        try:
            prediction = model.predict(input_df)
            inverse_prediction = np.expm1(prediction)
            st.write('Predicted IKP: {:.2f}'.format(inverse_prediction[0]))
        except Exception as e:
            st.error(f"Prediction error: {e}")
else:
    st.error("Failed to load model or transformation parameters. Please check the URLs.")
