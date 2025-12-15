import streamlit as st
import joblib
import pandas as pd

# 1. Load the saved model pipeline
# The pipeline handles both the preprocessing and the prediction
try:
    # Ensure this filename exactly matches the file you downloaded from Colab
    pipeline = joblib.load('churn_prediction_pipeline.joblib')
except FileNotFoundError:
    st.error("Error: 'churn_prediction_pipeline.joblib' not found. Ensure it is uploaded to the same directory.")
    st.stop()


st.set_page_config(page_title="Bank Churn Predictor", layout="wide")
st.title('🏦 Bank Customer Churn Prediction Model')
st.markdown('### Predict a customer\'s likelihood of leaving the bank.')

# --- 2. Input Fields using Streamlit Widgets ---
st.header("Customer Input Data")

col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.slider('Credit Score', 300, 900, 650, help="FICO score of the customer.")
    age = st.slider('Age', 18, 92, 40, help="The age of the customer.")
    tenure = st.slider('Tenure (Years as customer)', 0, 10, 5, help="Number of years customer has been with the bank.")
    gender = st.selectbox('Gender', ('Female', 'Male'))

with col2:
    balance = st.number_input('Account Balance ($)', 0.00, 250000.00, 50000.00, help="Current balance in the customer's account.")
    estimated_salary = st.number_input('Estimated Salary ($)', 0.00, 250000.00, 100000.00, help="The estimated salary of the customer.")
    country = st.selectbox('Country', ('France', 'Germany', 'Spain'))
    
with col3:
    products_number = st.selectbox('Number of Products', (1, 2, 3, 4), help="Number of bank products the customer holds (e.g., savings, checking, loan).")
    # Use 1 and 0 internally, but show Yes/No to the user
    credit_card = st.selectbox('Has Credit Card?', (1, 0), format_func=lambda x: 'Yes' if x == 1 else 'No', help="1 if customer has a credit card, 0 otherwise.")
    active_member = st.selectbox('Is an Active Member?', (1, 0), format_func=lambda x: 'Yes' if x == 1 else 'No', help="1 if customer is an active member, 0 otherwise.")


# --- 3. Prediction Logic ---
if st.button('Predict Churn Risk', key='predict_button'):
    # Create a DataFrame from the inputs, matching the exact column order used for training
    input_data = pd.DataFrame({
        'credit_score': [credit_score],
        'country': [country],
        'gender': [gender],
        'age': [age],
        'tenure': [tenure],
        'balance': [balance],
        'products_number': [products_number],
        'credit_card': [credit_card],
        'active_member': [active_member],
        'estimated_salary': [estimated_salary]
    })
    
    # Use the pipeline to preprocess the data and make a prediction
    # predict_proba returns the probability for class 0 (non-churn) and class 1 (churn)
    prediction_proba = pipeline.predict_proba(input_data)[0, 1]
    churn_percentage = prediction_proba * 100

    st.subheader(f"Prediction Result:")
    
    # Display results with visual cues
    if prediction_proba >= 0.5:
        st.error(f'🚨 High Risk of Churn: {churn_percentage:.2f}%')
        st.markdown("**Action Required:** Customer should be contacted by a retention specialist.")
        st.balloons()
    elif prediction_proba >= 0.2:
        st.warning(f'⚠️ Moderate Risk of Churn: {churn_percentage:.2f}%')
        st.markdown("**Consider:** Offer a loyalty incentive or improved product features.")
    else:
        st.success(f'✅ Low Risk of Churn: {churn_percentage:.2f}%')
        st.markdown("**Status:** Customer is likely to be retained.")

    # Display a clear progress bar for the probability
    st.progress(prediction_proba, text=f"**Churn Probability:** {churn_percentage:.2f}%")
