# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("telecom_churn.csv")

    # Convert TotalCharges to numeric
    data['TotalCharges'] = pd.to_numeric(
        data['TotalCharges'],
        errors='coerce'
    )

    data = data.dropna()

    return data

data = load_data()

# -------------------------------
# Train Model
# -------------------------------
@st.cache_resource
def train_model(data):

    df = data.copy()

    # Remove customerID
    df = df.drop("customerID", axis=1)

    # Encode categorical columns
    encoders = {}

    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Features and Target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100)

    model.fit(X_train, y_train)

    return model, encoders, X.columns

model, encoders, feature_columns = train_model(data)

# -------------------------------
# Streamlit UI
# -------------------------------

st.title("Telecom Customer Churn Prediction Dashboard")

st.sidebar.header("Enter Customer Details")

# User Inputs
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])

senior = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.sidebar.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.sidebar.slider(
    "Tenure (months)",
    0, 72, 12
)

phone = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple = st.sidebar.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

internet = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_sec = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device_protection = st.sidebar.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

tv = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly = st.sidebar.slider(
    "Monthly Charges",
    0.0, 200.0, 50.0
)

total = st.sidebar.number_input(
    "Total Charges",
    0.0, 10000.0, 500.0
)

# -------------------------------
# Create Input Data
# -------------------------------

input_dict = {
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple,
    "InternetService": internet,
    "OnlineSecurity": online_sec,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": tv,
    "StreamingMovies": movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total
}

input_df = pd.DataFrame([input_dict])

# -------------------------------
# Apply Encoders
# -------------------------------

for col in input_df.columns:

    if col in encoders:

        try:
            input_df[col] = encoders[col].transform(input_df[col])

        except:
            input_df[col] = 0

# Ensure correct order
input_df = input_df[feature_columns]

# -------------------------------
# Prediction
# -------------------------------

prediction = model.predict(input_df)[0]

probability = model.predict_proba(input_df)[0][1]

# -------------------------------
# Output
# -------------------------------

st.subheader("Prediction Result")

if prediction == 1:
    st.error("Customer is likely to CHURN")
else:
    st.success("Customer is likely to STAY")

st.write(f"Churn Probability: {probability:.2f}")

# -------------------------------
# KPI Dashboard
# -------------------------------

st.subheader("KPI Dashboard")

total_customers = len(data)

churn_rate = (data["Churn"] == "Yes").mean()

st.metric("Total Customers", total_customers)

st.metric("Churn Rate", f"{churn_rate:.2%}")

# -------------------------------
# Visualization
# -------------------------------

st.subheader("Churn Distribution")

chart_data = data["Churn"].value_counts()

st.bar_chart(chart_data)