import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from google.cloud import storage
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import io

# -------------------------------
# 1. APPLY CUSTOM THEME
# -------------------------------
def apply_custom_theme():
    st.set_page_config(
        page_title="CompCare Rate Insights",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown(
        """
        <style>
            /* Global background color */
            html, body, [data-testid="stAppViewContainer"] {
                background-color: #f3f6fc !important;
                color: #000 !important;
            }
            /* Sidebar customization */
            [data-testid="stSidebar"] {
                background-color: #1f3c88 !important; /* a deep bluish color */
                color: white !important;
            }
            /* Top bar */
            .top-bar {
                background: linear-gradient(135deg, #4285F4, #34A853);
                color: #fff;
                padding: 1rem;
                text-align: center;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .brand-name {
                font-size: 2rem;
                font-weight: 700;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

apply_custom_theme()

# -------------------------------
# 2. TOP BAR
# -------------------------------
st.markdown('<div class="top-bar"><span class="brand-name">CompCare Rate Insights</span></div>', unsafe_allow_html=True)

# -------------------------------
# 3. LOAD DATA FROM GCS (FIXED FOR XLSX)
# -------------------------------
def load_xlsx_from_gcs(bucket_name, file_name):
    """
    Downloads an .xlsx file from GCS and returns a DataFrame.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    data_bytes = blob.download_as_bytes()  # download as bytes to handle Excel
    return pd.read_excel(io.BytesIO(data_bytes))

# Attempt to load data from your bucket
bucket_name = "healthcare_data_clean"

try:
    travel_df = load_xlsx_from_gcs(bucket_name, "travel_cleaned_data8.xlsx")
    st.success("Loaded travel_cleaned_data8.xlsx")
except Exception as e:
    st.error(f"Failed to load travel data: {e}")
    travel_df = pd.DataFrame()

try:
    local_df = load_xlsx_from_gcs(bucket_name, "local_cleaned_data8.xlsx")
    st.success("Loaded local_cleaned_data8.xlsx")
except Exception as e:
    st.error(f"Failed to load local data: {e}")
    local_df = pd.DataFrame()

try:
    gsa_df = load_xlsx_from_gcs(bucket_name, "FY2025_ZipCodeFile_080824 (2).xlsx")
    st.success("Loaded FY2025_ZipCodeFile_080824 (2).xlsx")
except Exception as e:
    st.error(f"Failed to load GSA data: {e}")
    gsa_df = pd.DataFrame()

# Combine travel & local
combined_df = pd.concat([travel_df, local_df], ignore_index=True)

# -------------------------------
# 4. BASIC DATA CLEANING
# -------------------------------
if not combined_df.empty:
    # Title-case city/state if columns exist
    if "City" in combined_df.columns:
        combined_df["City"] = combined_df["City"].astype(str).str.title()
    if "State" in combined_df.columns:
        combined_df["State"] = combined_df["State"].astype(str).str.title()

    # Remove outliers on Weekly Pay if it exists
    if "Weekly Pay" in combined_df.columns:
        combined_df = combined_df[combined_df["Weekly Pay"].notnull()]
        combined_df = combined_df[(np.abs(stats.zscore(combined_df["Weekly Pay"])) < 3)]

    # Encode job/city
    if "Job Title" in combined_df.columns:
        combined_df["job_title_encoded"] = LabelEncoder().fit_transform(combined_df["Job Title"].astype(str))
    if "City" in combined_df.columns:
        combined_df["city_encoded"] = LabelEncoder().fit_transform(combined_df["City"].astype(str))

# -------------------------------
# 5. SIDEBAR CONFIG
# -------------------------------
st.sidebar.header("Configuration")
markup_percentage = st.sidebar.slider("Markup (%)", 0, 100, 60)
weekly_hours = st.sidebar.number_input("Weekly Hours", 20, 60, 36)

# Worker Type selection if it exists, else fallback
if "Worker Type" in combined_df.columns:
    worker_types = combined_df["Worker Type"].dropna().unique().tolist()
    selected_worker_type = st.sidebar.selectbox("Worker Type", worker_types)
else:
    selected_worker_type = "Travel"  # fallback

# Job Title selection
job_titles = combined_df["Job Title"].dropna().unique().tolist() if "Job Title" in combined_df.columns else []
if job_titles:
    selected_job_title = st.sidebar.selectbox("Job Title", job_titles)
else:
    selected_job_title = None

# State selection
states = combined_df["State"].dropna().unique().tolist() if "State" in combined_df.columns else []
if states:
    selected_state = st.sidebar.selectbox("Select State", states)
else:
    selected_state = None

# -------------------------------
# 6. FILTER & MODEL
# -------------------------------
st.subheader("Data & Predictions")

# Filter
if not combined_df.empty and selected_worker_type and selected_job_title and selected_state:
    filtered_data = combined_df[
        (combined_df["Worker Type"] == selected_worker_type) &
        (combined_df["Job Title"] == selected_job_title) &
        (combined_df["State"] == selected_state)
    ]
else:
    filtered_data = pd.DataFrame()

st.write("Filtered Data (preview):", filtered_data.head())

# Simple ML Approach (if data is enough)
if len(filtered_data) < 5:
    model = RandomForestRegressor(n_estimators=50)
else:
    model = KNeighborsRegressor(n_neighbors=min(20, len(filtered_data)-1))

def calculate_bill_rate(weekly_pay, stipend_hourly, markup, hours):
    """
    Basic Travel Bill Rate formula.
    """
    travel_hourly_pay = weekly_pay / hours
    if travel_hourly_pay <= stipend_hourly:
        return None
    return (travel_hourly_pay - stipend_hourly) * (1 + markup / 100) + stipend_hourly

if st.button("Predict & Calculate Rates"):
    if filtered_data.empty:
        st.error("No data for this selection.")
    else:
        # Ensure columns exist
        if "Weekly Pay" in filtered_data.columns:
            X = filtered_data[["job_title_encoded", "city_encoded"]].dropna()
            y = filtered_data["Weekly Pay"].dropna()

            # Fit Model
            if not X.empty and len(X) == len(y):
                model.fit(X, y)
                predicted_weekly_pay = model.predict(X)
                if "Hourly Stipend" not in filtered_data.columns:
                    filtered_data["Hourly Stipend"] = 0

                # Apply formula
                bill_rates = []
                for i, pay in enumerate(predicted_weekly_pay):
                    row_index = X.index[i]
                    stipend = filtered_data.loc[row_index, "Hourly Stipend"]
                    br = calculate_bill_rate(pay, stipend, markup_percentage, weekly_hours)
                    bill_rates.append(br)

                # Merge predictions back
                filtered_data["predicted_weekly_pay"] = pd.Series(predicted_weekly_pay, index=X.index)
                filtered_data["bill_rate"] = pd.Series(bill_rates, index=X.index)

                # Show results
                st.write("**Results**:")
                st.write(filtered_data[["Job Title", "City", "State", "predicted_weekly_pay", "bill_rate"]].head())

                if filtered_data["bill_rate"].notnull().any():
                    fig = px.scatter(
                        filtered_data.dropna(subset=["bill_rate"]),
                        x="City",
                        y="bill_rate",
                        color="Job Title",
                        title="Bill Rate Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.write(f"Min: ${filtered_data['bill_rate'].min():.2f}, "
                             f"Avg: ${filtered_data['bill_rate'].mean():.2f}, "
                             f"Max: ${filtered_data['bill_rate'].max():.2f}")
                else:
                    st.warning("No valid bill rates calculated.")
            else:
                st.warning("Check if features or weekly_pay are missing.")
        else:
            st.warning("No 'Weekly Pay' column found in data.")

# -------------------------------
# 7. OPTIONAL EXPORT
# -------------------------------
if not filtered_data.empty:
    csv = filtered_data.to_csv(index=False).encode("utf-8")
    st.download_button("Download Filtered Data", csv, file_name="filtered_data.csv", mime="text/csv")
EOF

