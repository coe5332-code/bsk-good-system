import streamlit as st
import requests
import pandas as pd
import os

# Configure the page
st.set_page_config(
    page_title="BSK Training Optimization",
    page_icon="🎓",
    layout="wide"
)

st.title("BSK Training Optimization System")
st.markdown("""
Welcome to the BSK Training Optimization System!\
Use the sidebar to navigate to different data views and analytics.\
Each section provides interactive tables and visualizations for your data.\
""") 

# Fetch data for overview
# Use environment variable for API URL (for Hugging Face Spaces deployment)
import os
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:54300")

# Local parquet filenames mapping (same names as backend dataset)
PARQUET_FILES = {
    "bsk_master": "dbo_ml_bsk_master.parquet",
    "deo_master": "dbo_ml_deo_master.parquet",
    "service_master": "dbo_ml_service_master.parquet",
}


def load_local_table(table_key: str):
    """Try to load a parquet file from common local locations.
    Returns DataFrame or None if not found/failed.
    """
    filename = PARQUET_FILES.get(table_key)
    if not filename:
        return None

    candidates = [
        os.path.join(".parquet_cache", filename),
        os.path.join("data", filename),
        filename,
    ]

    for path in candidates:
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path, engine="pyarrow")
                return df
            except Exception as e:
                st.warning(f"Failed to read local parquet {path}: {e}")
                return None
    return None

# Check backend health
def check_backend_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except Exception as e:
        return False, str(e)

def fetch_all_data(endpoint):
    try:
        response = requests.get(f"{API_BASE_URL}/{endpoint}", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching {endpoint}: {e}")
        return []

# Try loading local parquet files first (so frontend can run without backend)
local_bsk = load_local_table("bsk_master")
local_deo = load_local_table("deo_master")
local_service = load_local_table("service_master")

if local_bsk is not None and local_deo is not None and local_service is not None:
    st.info("Using local parquet files for data (no backend required)")
    bsk_centers = local_bsk.to_dict("records")
    deos = local_deo.to_dict("records")
    services = local_service.to_dict("records")
else:
    # Health check
    health_ok, health_data = check_backend_health()
    if not health_ok:
        st.warning(f"⚠️ Backend service is not available at {API_BASE_URL}. Please wait a few seconds for the backend to start.")
        if health_data:
            st.error(f"Error: {health_data}")
        st.stop()
    else:
        if health_data and health_data.get("initialized"):
            st.success("✅ Backend connected and data loaded successfully!")
        else:
            st.warning("⚠️ Backend is running but data may not be fully loaded yet.")

    bsk_centers = fetch_all_data("bsk/")
    deos = fetch_all_data("deo/")
    services = fetch_all_data("services/")

num_bsks = len(bsk_centers) if bsk_centers else 0
num_deos = len(deos) if deos else 0
num_services = len(services) if services else 0

# Display summary info at the top
st.markdown("### System Overview")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Total BSKs", num_bsks)
col_b.metric("Total DEOs", num_deos)
col_c.metric("Total Services", num_services) 