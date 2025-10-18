import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_model, preprocess_data


@st.cache_resource
def load_resources():
    try:
        model = load_model('models/trained_RFC.joblib')

        return model

    except FileNotFoundError:
        st.error(f"Model not found. Please run the training_pipeline.py first or verify the given model path.")
        return None

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

model = load_resources()

FEATURE_ORDER = ['person_income', 
                 'person_home_ownership', 
                 'person_emp_length', 
                 'loan_intent', 
                 'loan_grade', 
                 'loan_amnt', 
                 'loan_int_rate', 
                 'loan_percent_income', 
                 'cb_defaulter_on_file'
                 ]


st.set_page_config(page_title="Loan Approval Predictor", layout='wide')
st.title("Loan Approval Prediction")
st.write("Enter the applicant's details in the sidebar to get a loan approval prediction.")

st.sidebar.header("Applicant Information")

person_income = st.sidebar.number_input('Annual Income', min_value=0, value=50000)
person_emp_length = st.sidebar.number_input('Employment Length (Years)', min_value=0, value=5)
loan_amnt = st.sidebar.number_input('Loan Amount', min_value=0, value=10000)
loan_int_rate = st.sidebar.number_input('Interest rate (%)', min_value=0.0, value=11.0, format="%.2f")
loan_percent_income = round((loan_amnt / person_income) * 100, 2)
# loan_percent_income = st.sidebar.number_input('Loan as % of Income', min_value=0.0, max_value=1.0, value=0.2, format="%.2f")
loan_intent = st.sidebar.selectbox('Loan Intent', ['EDUCATION', 'MEDICAL', 'PERSONAL', 'VENTURE', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'])
loan_grade = st.sidebar.selectbox('Loan Grade', ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
person_home_ownership = st.sidebar.selectbox('Home Ownership', ['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
cb_defaulter_on_file = st.sidebar.selectbox('Has Defaulted On File?', ['N', 'Y'])


if st.sidebar.button('Predict Loan Status'):
    try:
        raw_data = {
            'person_income': person_income, 
            'person_emp_length': person_emp_length,
            'loan_intent': loan_intent,
            'loan_grade': loan_grade,
            'loan_amnt': loan_amnt,
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': loan_percent_income,
            'cb_defaulter_on_file': cb_defaulter_on_file,
            'person_home_ownership': person_home_ownership
        }

        input_df = pd.DataFrame(raw_data, index=[0])
        st.write("Accepted Input details.")

        processed_df = preprocess_data(input_df, mode='inference')
        st.write("Processed the Input details.")
        

        prediction = model.predict(processed_df)
        prediction_proba = model.predict_proba(processed_df)


        st.subheader('Prediction Result')
        if prediction[0] == 1:
            st.success('**Loan Approved!**')
            st.progress(prediction_proba[0][1])
            st.write(f"Confidence: **{prediction_proba[0][1]*100:.2f}%**")
        
        else:
            st.error('**Loan Denied**')
            st.progress(prediction_proba[0][0])
            st.write(f"Confidence: **{prediction_proba[0][0]*100:.2f}%**")


    except Exception as e:
        st.error(f"Error occurred during prediction: {e}")
        st.write("Please check the input values and ensure all components are correctly loaded.")

