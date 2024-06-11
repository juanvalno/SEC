import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from scipy.special import boxcox1p
import requests
import io
import pickle

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Function to load the model and transformation parameters
def load_model_and_params(url_lambda, url_model):
    response_lambda = requests.get(url_lambda)
    response_model = requests.get(url_model)

    if response_lambda.status_code == 200 and response_model.status_code == 200:
        lambda_buffer = io.BytesIO(response_lambda.content)
        model_buffer = io.BytesIO(response_model.content)

        optimal_lambdas = pickle.load(lambda_buffer)

        model_file_path = 'Model/model.json'
        with open(model_file_path, 'wb') as file:
            file.write(model_buffer.getvalue())

        model = XGBRegressor()
        model.load_model(model_file_path)
        return model, optimal_lambdas
    else:
        st.error("Failed to load model or transformation parameters. Please check the URLs.")
        return None, None

# Define all expected features
expected_features = ['POV', 'FOOD', 'ELEC', 'WATER', 'LIFE', 'HEALTH', 'SCHOOL', 'STUNTING']

# Home page
if st.session_state.page == 'home':
    st.title('Welcome to FARM: Food Availability and Security Monitor')
    st.write('This application helps monitor food availability and security based on various indicators.')
    if st.button('Go to Prediction Page'):
        st.session_state.page = 'predict'

# Prediction page
elif st.session_state.page == 'predict':
    # Load the model and parameters
    model, optimal_lambdas = load_model_and_params(
        'https://raw.githubusercontent.com/juanvalno/SEC/6d0553bca78ed9b7479eb6f103ebcb1c2dca79b0/Model/transformation_params.pkl',
        'https://raw.githubusercontent.com/juanvalno/SEC/22d581a130216da15bff6c439d5cd7819258332d/Model/model.json'
    )

    if model is not None and optimal_lambdas is not None:
        st.title('Prediction Page')

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

        # Button to make predictions
        if st.button('Predict'):
            prediction = model.predict(input_df)
            inverse_prediction = np.expm1(prediction)
            st.write('Predicted IKP: {:.2f}'.format(inverse_prediction[0]))

        # Button to go back to the home page
        if st.button('Back to Home Page'):
            st.session_state.page = 'home'
