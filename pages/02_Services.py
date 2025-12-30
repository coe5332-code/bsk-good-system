import streamlit as st
import pandas as pd
import plotly.express as px
from utils import fetch_data_with_controls

def show_services():
    st.header("Services")
    services_data, limit, skip = fetch_data_with_controls("services/")
    if services_data:
        df = pd.DataFrame(services_data)
        st.dataframe(df)
        if "department_name" in df.columns:
            st.subheader("Services by Department")
            fig = px.bar(df, x="department_name", title="Services by Department")
            st.plotly_chart(fig, use_container_width=True)
        if "service_type" in df.columns:
            st.subheader("Service Type Distribution")
            fig = px.pie(df, names="service_type", title="Service Type Distribution")
            st.plotly_chart(fig, use_container_width=True)

show_services() 