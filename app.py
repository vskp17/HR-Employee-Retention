import streamlit as st
import pandas as pd
import joblib

# Load the model and feature list
model = joblib.load("random_forest_model.pkl")
feature_names = joblib.load("model_features.pkl")  # Save this during model training

# App title
st.title("Employee Retention Predictor")

# Input fields
satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
last_evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.5)
number_project = st.number_input("Number of Projects", 1, 10, 3)
average_monthly_hours = st.number_input("Average Monthly Hours", 50, 400, 160)
time_spend_company = st.number_input("Years at Company", 1, 10, 3)
work_accident = st.selectbox("Had Work Accident?", ["No", "Yes"])
promotion_last_5years = st.selectbox("Promoted in Last 5 Years?", ["No", "Yes"])
department = st.selectbox("Department", ['sales', 'technical', 'support', 'IT', 'product_mng',
                                         'marketing', 'RandD', 'accounting', 'hr', 'management'])
salary = st.selectbox("Salary Level", ["low", "medium", "high"])

# Process input
input_dict = {
    'satisfaction_level': satisfaction_level,
    'last_evaluation': last_evaluation,
    'number_project': number_project,
    'average_montly_hours': average_monthly_hours,
    'time_spend_company': time_spend_company,
    'Work_accident': 1 if work_accident == "Yes" else 0,
    'promotion_last_5years': 1 if promotion_last_5years == "Yes" else 0,
    'Department': department,
    'salary': salary
}
input_df = pd.DataFrame([input_dict])

# One-hot encode categorical features
input_encoded = pd.get_dummies(input_df)

# Ensure columns match training data
input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_encoded)[0]
    if prediction == 1:
        st.error("🔴 This employee is likely to leave the company.")
    else:
        st.success("🟢 This employee is likely to stay with the company.")
