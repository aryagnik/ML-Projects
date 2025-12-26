import pandas as pd
import streamlit as st
import joblib
import os

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "bank_customer_churn_model.pkl")

model = joblib.load(MODEL_PATH)

# Page config
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

# Title & subtitle
st.title("🏦 Bank Customer Churn Prediction")
st.markdown(
    "Predict whether a customer is likely to **leave the bank** based on their profile."
)

# Sidebar for inputs
st.sidebar.header("🧾 Customer Details")

# --- Personal Info ---
st.sidebar.subheader("👤 Personal Information")
credit_score = st.sidebar.number_input(
    "Credit Score", min_value=300, max_value=850, value=650
)
age = st.sidebar.number_input(
    "Age", min_value=18, max_value=100, value=35
)
gender = st.sidebar.selectbox(
    "Gender", ["Male", "Female"]
)
country = st.sidebar.selectbox(
    "Country", ["France", "Germany", "Spain"]
)

# --- Account Info ---
st.sidebar.subheader("🏦 Account Information")
tenure = st.sidebar.number_input(
    "Tenure (years)", min_value=0, max_value=20, value=5
)
products_number = st.sidebar.number_input(
    "Number of Products", min_value=1, max_value=10, value=1
)
credit_card = st.sidebar.selectbox(
    "Has Credit Card?", ["Yes", "No"]
)
active_member = st.sidebar.selectbox(
    "Is Active Member?", ["Yes", "No"]
)

# --- Financial Info ---
st.sidebar.subheader("💰 Financial Information")
balance = st.sidebar.number_input(
    "Account Balance", min_value=0.0, value=50000.0
)
estimated_salary = st.sidebar.number_input(
    "Estimated Salary", min_value=0.0, value=60000.0
)

# Convert categorical binary fields
credit_card = 1 if credit_card == "Yes" else 0
active_member = 1 if active_member == "Yes" else 0

# Prepare input dataframe - FIXED COLUMN NAMES (use exact names from training)
# Common column name patterns in bank churn datasets:
# Try these column names (match your training data exactly):
input_data = pd.DataFrame(
    [[
        credit_score, country, gender, age, tenure,
        balance, products_number, credit_card,
        active_member, estimated_salary
    ]],
    columns=[
        "CreditScore", "Geography", "Gender", "Age", "Tenure",
        "Balance", "NumOfProducts", "HasCrCard",
        "IsActiveMember", "EstimatedSalary"
    ]
)

# Prediction section
st.divider()
st.subheader("📊 Prediction Result")

if st.button("🔮 Predict Churn", use_container_width=True):
    try:
        prediction = model.predict(input_data)
        
        if prediction[0] == 1:
            st.error("⚠️ Customer is **likely to churn**")
        else:
            st.success("✅ Customer is **unlikely to churn**")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("Debug information:")
        st.write("Input data columns:", input_data.columns.tolist())
        st.write("Input data shape:", input_data.shape)
        st.write("Input data:", input_data)