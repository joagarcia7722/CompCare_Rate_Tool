import streamlit as st
from streamlit_elements import elements, mui, html
import pandas as pd
import plotly.express as px
import numpy as np
from google.cloud import storage
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import io

# -------------------------------
# 1. LOAD DATA FROM GOOGLE CLOUD STORAGE
# -------------------------------
def load_excel_from_gcs(bucket_name, file_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    data = blob.download_as_bytes()
    return pd.read_excel(io.BytesIO(data))

# Load data
bucket_name = "healthcare_data_clean"
travel_df = load_excel_from_gcs(bucket_name, "travel_cleaned_data8.xlsx")
local_df = load_excel_from_gcs(bucket_name, "local_cleaned_data8.xlsx")
gsa_df = load_excel_from_gcs(bucket_name, "FY2025_ZipCodeFile_080824 (2).xlsx")

data = pd.concat([travel_df, local_df], ignore_index=True)
data['City'] = data['City'].str.title()
data['State'] = data['State'].str.title()
data['location'] = data['City'] + ', ' + data['State']

# Remove outliers
if 'Weekly Pay' in data.columns:
    data = data[(np.abs(stats.zscore(data['Weekly Pay'])) < 3)]

# Encode job titles and cities
if 'Job Title' in data.columns:
    data['job_title_encoded'] = LabelEncoder().fit_transform(data['Job Title'].astype(str))
if 'City' in data.columns:
    data['city_encoded'] = LabelEncoder().fit_transform(data['City'].astype(str))

# -------------------------------
# 2. STREAMLIT CLOUD UI
# -------------------------------
st.set_page_config(page_title="CompCare Rate Insights", layout="wide")

with elements("main"):
    with mui.Grid(container=True, spacing=3):

        with mui.Grid(item=True, xs=6):
            mui.Card(
                mui.CardContent(
                    mui.Typography("CompCare Rate Insights", variant="h4"),
                    mui.Typography("Healthcare Bill Rate Analysis Tool", variant="body2")
                )
            )

        with mui.Grid(item=True, xs=6):
            mui.Card(
                mui.CardContent(
                    mui.Typography("Subscription Status", variant="h6"),
                    mui.LinearProgress(value=50),
                    mui.Typography("50% of quota used", variant="body2")
                )
            )

# Sidebar inputs
st.sidebar.header("Configuration")
worker_type = st.sidebar.radio("Select Worker Type", ["Travel", "Local"])
markup_percentage = st.sidebar.slider("Enter Markup (%)", 0, 100, 60)
weekly_hours = st.sidebar.number_input("Enter Weekly Hours", 20, 60, 36)

# Job title and state selection
data['Job Title'] = data['Job Title'].fillna('Unknown')
job_title = st.selectbox("Select Job Title", data['Job Title'].unique())
state = st.selectbox("Select State", data['State'].unique())

# -------------------------------
# 3. PREDICT BILL RATES
# -------------------------------
def calculate_bill_rate(row, markup, hours):
    travel_hourly_pay = row['Weekly Pay'] / hours
    stipend_hourly = row.get('Hourly Stipend', 0)
    if travel_hourly_pay <= stipend_hourly:
        return None
    return (travel_hourly_pay - stipend_hourly) * (1 + markup / 100) + stipend_hourly

filtered_df = data[(data['Worker Type'] == worker_type) &
                   (data['Job Title'] == job_title) &
                   (data['State'] == state)]

if filtered_df.empty:
    st.warning("No data found for the selected filters.")
else:
    filtered_df['Bill Rate'] = filtered_df.apply(lambda x: calculate_bill_rate(x, markup_percentage, weekly_hours), axis=1)

    st.subheader("Bill Rate Distribution")
    fig = px.scatter(filtered_df, x='City', y='Bill Rate', color='Job Title', title='Bill Rate Distribution')
    st.plotly_chart(fig)

    # Display summary metrics
    st.write(f"Min: ${filtered_df['Bill Rate'].min():.2f}, Avg: ${filtered_df['Bill Rate'].mean():.2f}, Max: ${filtered_df['Bill Rate'].max():.2f}")

# -------------------------------
# 4. EXPORT OPTIONS
# -------------------------------
if not filtered_df.empty:
    st.download_button("Download Detailed Summary", filtered_df.to_csv(index=False), "detailed_bill_rates.csv", "text/csv")
    st.download_button("Download Raw Data", data.to_csv(index=False), "raw_data.csv", "text/csv")

st.success("App ready for deployment on Streamlit Cloud!")
