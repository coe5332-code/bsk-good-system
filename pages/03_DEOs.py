import streamlit as st
import pandas as pd
import plotly.express as px
from utils import fetch_data_with_controls

def show_deos():
    st.header("Data Entry Operators (DEOs)")
    deo_data, limit, skip = fetch_data_with_controls("deo/")
    if deo_data:
        df = pd.DataFrame(deo_data)
        st.dataframe(df)
        if "is_active" in df.columns:
            st.subheader("Active vs Inactive DEOs")
            fig = px.pie(df, names="is_active", title="Active vs Inactive DEOs")
            st.plotly_chart(fig, use_container_width=True)
        if "bsk_name" in df.columns:
            st.subheader("DEOs by BSK Center")
            fig = px.bar(df, x="bsk_name", title="DEOs by BSK Center")
            st.plotly_chart(fig, use_container_width=True)

show_deos() 