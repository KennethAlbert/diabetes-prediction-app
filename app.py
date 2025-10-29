import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# Load the saved scaler and classifier
try:
    scaler = joblib.load('models/scaler.pkl')
    classifier = joblib.load('models/classifier.pkl')
except FileNotFoundError:
    st.error("Scaler and classifier files not found. Please run the saving cell first.")
    st.stop()

def diabetes_prediction(input_data):
    """Predicts diabetes based on input features."""
    # Standardize the input data
    std_data = scaler.transform(np.array(input_data).reshape(1, -1))

    # Make prediction
    prediction = classifier.predict(std_data)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'


# Streamlit app
st.set_page_config(page_title='Diabetes Prediction App', layout='wide')

st.title('Diabetes Prediction using SVM')

# Input fields for features
pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=17, value=0)
glucose = st.number_input('Glucose Level', min_value=0, max_value=200, value=0)
blood_pressure = st.number_input('Blood Pressure value', min_value=0, max_value=122, value=0)
skin_thickness = st.number_input('Skin Thickness value', min_value=0, max_value=100, value=0)
insulin = st.number_input('Insulin Level', min_value=0, max_value=850, value=0)
bmi = st.number_input('BMI value', min_value=0.0, max_value=70.0, value=0.0)
diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function value', min_value=0.0, max_value=2.5, value=0.0)
age = st.number_input('Age of the Person', min_value=0, max_value=120, value=0)

# Prediction button
if st.button('Diabetes Test Result'):
    input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
    prediction_result = diabetes_prediction(input_data)

    if prediction_result == 'The person is not diabetic':
        st.success(prediction_result)
    else:
        st.error(prediction_result)
