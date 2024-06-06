import streamlit as st
import pandas as pd
import pickle

# Load the preprocessing pipeline
with open('../Model/preprocessing_pipeline.pkl', 'rb') as file:
    preprocessing_pipeline = pickle.load(file)

# Load the trained model
with open('../Model/xgb_tuned_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Predictive Model")

# Input fields for each feature
input_data = {}
columns = ['POV', 'FOOD', 'ELEC', 'WATER', 'LIFE', 'HEALTH', 'SCHOOL', 'STUNTING', 'IKP']

for col in columns:
    input_data[col] = st.number_input(f"Enter value for {col}", value=0.0)

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Apply preprocessing
preprocessed_data = preprocessing_pipeline.transform(input_df)

# Make predictions
prediction = model.predict(preprocessed_data)

st.write("Prediction:", prediction)
