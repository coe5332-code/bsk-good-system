import streamlit as st
import requests

import os

def fetch_data_with_controls(endpoint):
    st.sidebar.markdown(f"---\n**{endpoint.replace('/', '').capitalize()} Controls**")
    limit = st.sidebar.number_input(f"Limit for {endpoint}", min_value=1, value=100, step=1, key=f"limit_{endpoint}")
    skip = st.sidebar.number_input(f"Skip for {endpoint}", min_value=0, value=0, step=1, key=f"skip_{endpoint}")
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:54300")
    params = {"limit": limit, "skip": skip}
    try:
        response = requests.get(f"{API_BASE_URL}/{endpoint}", params=params)
        response.raise_for_status()
        return response.json(), limit, skip
    except Exception as e:
        st.error(f"Error fetching {endpoint}: {e}")
        return [], limit, skip 