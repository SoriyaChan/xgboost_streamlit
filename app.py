import streamlit as st
import joblib
import pandas as pd

def main():
    st.title("Banking Target Marketing with XGBoost")

    age = st.number_input("The age of the customer (age):", value=41)
    balance = st.number_input("average yearly balance, in euros (balance):", value=1362)
    day = st.number_input("Last contact day of the month (day):", value=16)
    duration = st.number_input("Last contact duration, in seconds (duration):", value=258)
    campaign = st.number_input("Number of contacts performed during this campaign and for this client (campaign):", value=3)
    pdays = st.number_input("Number of days that passed by after the client was last contacted from a previous campaign (pdays):", value=40)
    previous = st.number_input("Number of contacts performed before this campaign and for this client (previous)", value=1)

    job = st.selectbox("Type of job (job):", ['management', 'technician', 'entrepreneur', 'blue-collar', 'unknown',
                                    'retired', 'admin.', 'services', 'self-employed', 'unemployed', 'housemaid',
                                    'student'])
    marital = st.radio("Marital status (marital):", ['married', 'single', 'divorced'])
    education = st.radio("Education (education):", ['tertiary', 'secondary', 'unknown', 'primary'])
    default = st.radio("Has credit in default (default):", ['no', 'yes'])
    housing = st.radio("Has housing loan (housing):", ['no', 'yes'])
    loan = st.radio("Has personal loan (loan):", ['no', 'yes'])
    contact = st.radio("Contact communication type (contact):", ['unknown', 'cellular', 'telephone'])
    month = st.selectbox("Last contact month of year (month):", ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                                        'jul', 'aug', 'oct', 'nov', 'sep', 'dec']) 
    poutcome = st.radio("Outcome of the previous marketing campaign (poutcome):", ['unknown', 'failure', 'other', 'success'])

    if st.button("Predict"): # Add a button to trigger prediction
        prediction = predict(age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome)
        if prediction == 0:
          st.write("The Customer might not subscribe")
        elif prediction == 1:
          st.write("The Customer might subscribe!")

def predict(age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome):

    column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']

    input_data = [[age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome]]

    input_df = pd.DataFrame(input_data, columns=column_names)

    loaded_pipeline = joblib.load("my_model.pkl")

    predictions = loaded_pipeline.predict(input_df)

    return predictions

if __name__ == "__main__":
    main()
