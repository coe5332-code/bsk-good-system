import streamlit as st
import pandas as pd
import plotly.express as px
from utils import fetch_data_with_controls

def show_provisions():
    st.header("Provisions")
    provisions_data, limit, skip = fetch_data_with_controls("provisions/")
    if provisions_data:
        df = pd.DataFrame(provisions_data)
        st.dataframe(df)
        if "service_name" in df.columns:
            st.subheader("Provisions by Service")
            fig = px.bar(df, x="service_name", title="Provisions by Service")
            st.plotly_chart(fig, use_container_width=True)
        if "prov_date" in df.columns:
            st.subheader("Provisions Over Time")
            df['prov_date'] = pd.to_datetime(df['prov_date'], errors='coerce')
            fig = px.histogram(df, x="prov_date", title="Provisions Over Time")
            st.plotly_chart(fig, use_container_width=True)

show_provisions() 