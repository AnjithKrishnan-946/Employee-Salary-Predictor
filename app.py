import streamlit as st
import pandas as pd
import joblib

# Load the saved model and label encoder
model = joblib.load("salary_prediction_pipeline.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Employee Salary Predictor (XGBoost)", layout="centered")
st.title("🧑‍💼 Employee Salary Predictor")
st.markdown("Predict if a person earns more than 50K/year based on demographics.")

# Sidebar inputs
st.sidebar.header("User Input Features")

def get_user_input():
    age = st.sidebar.slider("Age", 17, 90, 30)
    workclass = st.sidebar.selectbox("Workclass", [
        'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 
        'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'
    ])
    fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", value=200000)
    education_num = st.sidebar.slider("Education Number", 1, 16, 10)
    marital_status = st.sidebar.selectbox("Marital Status", [
        'Never-married', 'Married-civ-spouse', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent'
    ])
    occupation = st.sidebar.selectbox("Occupation", [
        'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
        'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
        'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'
    ])
    relationship = st.sidebar.selectbox("Relationship", [
        'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'
    ])
    race = st.sidebar.selectbox("Race", [
        'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'
    ])
    gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
    capital_gain = st.sidebar.number_input("Capital Gain", value=0)
    capital_loss = st.sidebar.number_input("Capital Loss", value=0)
    hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)
    native_country = st.sidebar.selectbox("Native Country", [
        'United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'India', 'England', 'China', 'Others'
    ])

    # Return data as a dataframe
    return pd.DataFrame([{
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'educational-num': education_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }])

# Get input
input_df = get_user_input()

# Display input
st.subheader("📄 Input Summary")
st.write(input_df)

# Predict button
if st.button("Predict Salary"):
    prediction = model.predict(input_df)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    st.success(f"💰 Predicted Income Category: **{predicted_label}**")
