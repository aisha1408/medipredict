import pandas as pd 
import numpy as np
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import matplotlib.pyplot as plt
import json
import hashlib
import os

# --- Authentication ---
USER_FILE = "users.json"

def load_users():
    if not os.path.exists(USER_FILE):
        return {}
    with open(USER_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    users = load_users()
    if username in users:
        return False, "Username already exists!"
    users[username] = hash_password(password)
    save_users(users)
    return True, "Account created successfully."

def authenticate_user(username, password):
    users = load_users()
    return username in users and users[username] == hash_password(password)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

if not st.session_state.logged_in:
    st.title("🩺 Welcome to MediPredict!")
    st.subheader("\U0001F511 Login / Register")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        login_user = st.text_input("Username")
        login_pass = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate_user(login_user, login_pass):
                st.success(f"Welcome back, {login_user}!")
                st.session_state.logged_in = True
                st.session_state.username = login_user
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab2:
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        if st.button("Create Account"):
            success, msg = register_user(new_user, new_pass)
            if success:
                st.success(msg)
            else:
                st.error(msg)

    st.stop()

# --- App Content Starts ---
st.set_page_config(page_title="Hospital Resource Dashboard", layout="wide")
st.title("🩺Medipredict")
st.subheader("Where we predict the resources for your hospital.")

st.sidebar.markdown(f"👋 Welcome back *{st.session_state.username}*")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

st.markdown("""
This AI-powered dashboard helps hospitals predict patient admissions, estimate average length of stay (LOS), and forecast bed/staff needs using previous existing hospital data.

Upload the following datasets:
- Admissions data
- Patient demographics
- Discharge logs
- ICU equipment usage
- Staff roster
- Emergency case logs
- Department-wise patient logs

The system uses machine learning (Prophet + Random Forest) to visualize and forecast future resource demands, helping administrators make better decisions in advance.
""")

# --- File Uploads ---
st.sidebar.header("Upload your CSV data")
admissions_file = st.sidebar.file_uploader("Admissions logs (date, admissions)", type="csv")
demographics_file = st.sidebar.file_uploader("Patient demographics (patient_id, age, gender, ...)", type="csv")
discharge_file = st.sidebar.file_uploader("Discharge summaries (patient_id, admission_date, discharge_date)", type="csv")
icu_file = st.sidebar.file_uploader("ICU equipment usage (date, ventilators_used, beds_occupied, ...)", type="csv")
staff_file = st.sidebar.file_uploader("Staff rosters (date, staff_count)", type="csv")
emergency_file = st.sidebar.file_uploader("Emergency cases (date, emergency_cases)", type="csv")
dept_file = st.sidebar.file_uploader("Department-wise patient data (date, department, patient_count)", type="csv")

if not (admissions_file and demographics_file and discharge_file and icu_file and staff_file):
    st.warning("Please upload all five core CSV files to proceed.")
    st.stop()

@st.cache_data
def load_csv(f): return pd.read_csv(f)

# Load core datasets
ad_df = load_csv(admissions_file)
demo_df = load_csv(demographics_file)
disc_df = load_csv(discharge_file)
icu_df = load_csv(icu_file)
staff_df = load_csv(staff_file)

# ----------------------------
# 1. Patient Admissions Forecast
# ----------------------------
st.header("1. Patient Admissions Forecast")
ad_df.columns = ad_df.columns.str.lower()
if {'date', 'admissions'}.issubset(ad_df.columns):
    ad_df = ad_df.rename(columns={'date': 'ds', 'admissions': 'y'})
else:
    st.error("Admissions CSV must have date and admissions columns.")
    st.stop()

ad_df['ds'] = pd.to_datetime(ad_df['ds'])
ad_model = Prophet()
ad_model.fit(ad_df)
future_ad = ad_model.make_future_dataframe(periods=30)
forecast_ad = ad_model.predict(future_ad)
st.line_chart(forecast_ad.set_index('ds')['yhat'])

# ----------------------------
# 2. Length of Stay (LOS) Prediction
# ----------------------------
st.header("2. Length‑of‑Stay (LOS) Prediction")
demo_df.columns = demo_df.columns.str.lower()
disc_df.columns = disc_df.columns.str.lower()
required = {'patient_id', 'admission_date', 'discharge_date'}
if not required.issubset(disc_df.columns):
    st.error("Discharge CSV must have patient_id, admission_date, discharge_date.")
    st.stop()

disc_df['admission_date'] = pd.to_datetime(disc_df['admission_date'])
disc_df['discharge_date'] = pd.to_datetime(disc_df['discharge_date'])
disc_df['los'] = (disc_df['discharge_date'] - disc_df['admission_date']).dt.days
los_df = pd.merge(disc_df[['patient_id', 'los']], demo_df, on='patient_id')
if 'gender' in los_df.columns:
    los_df['gender'] = los_df['gender'].map({'M': 0, 'F': 1}).fillna(0)

features = [c for c in los_df.columns if c not in ('patient_id', 'los')]
X = los_df[features]
y = los_df['los']
los_model = RandomForestRegressor(n_estimators=100, random_state=42)
los_model.fit(X, y)
avg_feat = X.mean().to_frame().T
pred_avg_los = los_model.predict(avg_feat)[0]
st.metric("Predicted Avg LOS (days)", f"{pred_avg_los:.2f}")

# ----------------------------
# 3. Bed & Staff Needs Forecast
# ----------------------------
st.header("3. Bed & Staff Needs (Next 30 days)")
next_ad = forecast_ad[['ds', 'yhat']].tail(30).rename(columns={'yhat': 'pred_admissions'})
next_ad['beds_needed'] = np.ceil(next_ad['pred_admissions'])

staff_df.columns = staff_df.columns.str.lower()
if {'date', 'staff_count'}.issubset(staff_df.columns):
    staff_df = staff_df.rename(columns={'date': 'ds'})
    staff_df['ds'] = pd.to_datetime(staff_df['ds'])
    hist = pd.merge(ad_df, staff_df, on='ds', how='inner')
    avg_staff = hist['staff_count'].mean()
    avg_beds = hist['y'].mean() * pred_avg_los
    ratio = avg_staff / avg_beds if avg_beds > 0 else 0.5
    next_ad['staff_needed'] = np.ceil(2 * next_ad['pred_admissions'] * ratio)
else:
    st.error("Staff CSV must have date and staff_count.")
    st.stop()

st.dataframe(
    next_ad
    .rename(columns={
        'ds': 'Date',
        'pred_admissions': 'Admissions',
        'beds_needed': 'Beds Needed',
        'staff_needed': 'Staff Needed'
    })
    .set_index('Date')
)
st.line_chart(
    next_ad
    .set_index('ds')[['beds_needed', 'staff_needed']]
)

# ----------------------------
# 4. ICU Equipment Forecast
# ----------------------------
st.header("4. ICU Equipment Usage Forecast")
icu_df.columns = icu_df.columns.str.lower()
if 'date' not in icu_df.columns:
    st.error("ICU CSV must have a date column.")
    st.stop()

icu_df = icu_df.rename(columns={'date': 'ds'})
icu_df['ds'] = pd.to_datetime(icu_df['ds'])

for col in icu_df.columns:
    if col == 'ds': continue
    st.subheader(f"Forecast for {col}")
    temp = icu_df[['ds', col]].rename(columns={col: 'y'})
    m = Prophet()
    m.fit(temp)
    future = m.make_future_dataframe(periods=30)
    fc = m.predict(future)
    st.line_chart(fc.set_index('ds')['yhat'].tail(30))

# ----------------------------
# 5. Emergency Case Forecast
# ----------------------------
if emergency_file:
    st.header("5. Emergency Case Forecasting")
    emer_df = load_csv(emergency_file)
    emer_df.columns = emer_df.columns.str.lower()
    if {'date', 'emergency_cases'}.issubset(emer_df.columns):
        emer_df = emer_df.rename(columns={'date': 'ds', 'emergency_cases': 'y'})
        emer_df['ds'] = pd.to_datetime(emer_df['ds'])
        emer_model = Prophet()
        emer_model.fit(emer_df)
        future_emer = emer_model.make_future_dataframe(periods=30)
        forecast_emer = emer_model.predict(future_emer)
        st.line_chart(forecast_emer.set_index('ds')['yhat'])
    else:
        st.error("Emergency CSV must have date and emergency_cases columns.")

# ----------------------------
# 6. Department-wise Patient Forecast
# ----------------------------
if dept_file is not None:
    st.header("6. Department-wise Patient Forecast")
    dept_df = pd.read_csv(dept_file)
    dept_df.columns = dept_df.columns.str.lower()

    if {'date', 'department', 'patient_count'}.issubset(dept_df.columns):
        dept_df['date'] = pd.to_datetime(dept_df['date'])
        departments = dept_df['department'].unique()

        for dept in departments:
            st.subheader(f"{dept.capitalize()} Department")
            sub_df = dept_df[dept_df['department'] == dept][['date', 'patient_count']]
            sub_df = sub_df.rename(columns={'date': 'ds', 'patient_count': 'y'})
            model = Prophet()
            model.fit(sub_df)
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            st.line_chart(forecast.set_index('ds')['yhat'].tail(30))
            forecast_table = forecast[['ds', 'yhat']].tail(30).rename(columns={'ds': 'Date', 'yhat': 'Predicted Patients'})
            st.dataframe(forecast_table.set_index('Date'))
    else:
        st.error("Department CSV must have 'date', 'department', and 'patient_count' columns.")

