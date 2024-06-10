import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from scipy.special import boxcox1p
import requests
import io

# Load pickle files from URLs
url_lambda = 'https://raw.githubusercontent.com/juanvalno/SEC/6d0553bca78ed9b7479eb6f103ebcb1c2dca79b0/Model/model.pkl'
url_model = 'https://raw.githubusercontent.com/juanvalno/SEC/6d0553bca78ed9b7479eb6f103ebcb1c2dca79b0/Model/transformation_params.pkl'

response_lambda = requests.get(url_lambda)
response_model = requests.get(url_model)

# Check if the requests were successful
if response_lambda.status_code == 200 and response_model.status_code == 200:
    # Write the content to a buffer
    lambda_buffer = io.BytesIO(response_lambda.content)
    model_buffer = io.BytesIO(response_model.content)

    # Load the model and transformation parameters
    model = joblib.load(lambda_buffer)
    optimal_lambdas = pickle.load(model_buffer)

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

    # Make predictions
    if st.button('Predict'):
        prediction = model.predict(input_df)
        inverse_prediction = np.expm1(prediction)
        st.write('Predicted IKP: {:.2f}'.format(inverse_prediction[0]))
else:
    st.error("Failed to load model or transformation parameters. Please check the URLs.")
