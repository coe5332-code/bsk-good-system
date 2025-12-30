import streamlit as st
import pandas as pd
import plotly.express as px
from utils import fetch_data_with_controls

def show_bsk_centers():
    st.header("BSK Centers")
    bsk_data, limit, skip = fetch_data_with_controls("bsk/")
    if bsk_data:
        df = pd.DataFrame(bsk_data)
        st.dataframe(df)
        if "district_name" in df.columns:
            st.subheader("BSK Centers by District")
            fig = px.bar(df, x="district_name", title="BSK Centers by District")
            st.plotly_chart(fig, use_container_width=True)
        if "bsk_type" in df.columns:
            st.subheader("BSK Type Distribution")
            fig = px.pie(df, names="bsk_type", title="BSK Type Distribution")
            st.plotly_chart(fig, use_container_width=True)

show_bsk_centers() 