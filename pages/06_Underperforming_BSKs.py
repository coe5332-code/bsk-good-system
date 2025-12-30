import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# Add AI service to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Import local data loader
from data_loader_local import (
    get_bsk_master,
    get_service_master,
    get_provisions,
    get_deo_master,
    is_initialized
)

# Import analytics functions
try:
    from ai_service.bsk_analytics import find_underperforming_bsks
    ANALYTICS_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Analytics not available: {e}")
    ANALYTICS_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="BSK Performance Analysis",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.8;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 BSK Performance Analysis")
st.markdown("**Identify Training Needs and Underperforming BSKs**")

# Check data
if not is_initialized():
    st.error("❌ Data not initialized. Please restart the app.")
    st.stop()

if not ANALYTICS_AVAILABLE:
    st.error("❌ Analytics functions not available")
    st.stop()

# Filters
st.markdown("## 🔧 Analysis Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    period_start = st.date_input("Start Date", value=None)

with col2:
    period_end = st.date_input("End Date", value=None)

with col3:
    num_bsks = st.number_input("Number of BSKs", min_value=1, max_value=200, value=50)

analyze_btn = st.button("🔍 Analyze Performance", type="primary", use_container_width=True)

# Session state for results
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Run analysis
if analyze_btn:
    with st.spinner("Analyzing BSK performance..."):
        try:
            # Load data
            bsks_df = get_bsk_master()
            services_df = get_service_master()
            provisions_df = get_provisions()
            deos_df = get_deo_master()
            
            if any(df.empty for df in [bsks_df, services_df, provisions_df, deos_df]):
                st.error("❌ Required data not available")
            else:
                # Run analytics
                results = find_underperforming_bsks(
                    bsks_df=bsks_df,
                    provisions_df=provisions_df,
                    deos_df=deos_df,
                    services_df=services_df,
                    period_start=period_start.strftime('%Y-%m-%d') if period_start else None,
                    period_end=period_end.strftime('%Y-%m-%d') if period_end else None
                )
                
                if results is not None and not results.empty:
                    st.session_state.analysis_results = results
                    st.success(f"✅ Analysis complete! Found {len(results)} BSKs")
                else:
                    st.warning("⚠️ No underperforming BSKs found")
                    st.session_state.analysis_results = None
                    
        except Exception as e:
            st.error(f"❌ Analysis error: {e}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

# Display results
if st.session_state.analysis_results is not None:
    df = st.session_state.analysis_results
    
    st.divider()
    
    # Sort options
    sort_order = st.selectbox(
        "Sort BSKs by Score",
        ["Lowest Score (Training Priority)", "Highest Score"],
        index=0
    )
    
    if sort_order == "Lowest Score (Training Priority)":
        df = df.sort_values('score', ascending=True)
    else:
        df = df.sort_values('score', ascending=False)
    
    # Limit results
    df = df.head(num_bsks)
    
    # Sidebar filters
    with st.sidebar:
        st.markdown("### 🔍 Filters")
        
        # District filter
        if 'district_name' in df.columns:
            districts = df['district_name'].dropna().unique().tolist()
            selected_district = st.selectbox("District", ["All"] + districts)
            if selected_district != "All":
                df = df[df['district_name'] == selected_district]
        
        # Block filter
        if 'block_municipalty_name' in df.columns:
            blocks = df['block_municipalty_name'].dropna().unique().tolist()
            selected_block = st.selectbox("Block/Municipality", ["All"] + blocks)
            if selected_block != "All":
                df = df[df['block_municipalty_name'] == selected_block]
        
        # BSK search
        bsk_search = st.text_input("Search BSK Name")
        if bsk_search:
            df = df[df['bsk_name'].str.contains(bsk_search, case=False, na=False)]
    
    # Summary metrics
    st.markdown("## 📊 Performance Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{len(df)}</p>
            <p class="metric-label">Total BSKs</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{df['district_name'].nunique()}</p>
            <p class="metric-label">Districts</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{df['block_municipalty_name'].nunique()}</p>
            <p class="metric-label">Blocks/Municipalities</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    st.markdown("## 📈 Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # District distribution
        district_counts = df['district_name'].value_counts().reset_index()
        district_counts.columns = ['District', 'Count']
        
        fig = px.bar(
            district_counts,
            x='District',
            y='Count',
            title='BSKs by District',
            color='Count',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Block distribution
        block_counts = df['block_municipalty_name'].value_counts().head(10).reset_index()
        block_counts.columns = ['Block/Municipality', 'Count']
        
        fig = px.bar(
            block_counts,
            x='Block/Municipality',
            y='Count',
            title='Top 10 Blocks/Municipalities',
            color='Count',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Map
    if 'bsk_lat' in df.columns and 'bsk_long' in df.columns:
        st.markdown("## 🗺️ BSK Locations")
        
        map_df = df.copy()
        map_df['bsk_lat'] = pd.to_numeric(map_df['bsk_lat'], errors='coerce')
        map_df['bsk_long'] = pd.to_numeric(map_df['bsk_long'], errors='coerce')
        map_df = map_df.dropna(subset=['bsk_lat', 'bsk_long'])
        
        if not map_df.empty:
            fig_map = px.scatter_mapbox(
                map_df,
                lat='bsk_lat',
                lon='bsk_long',
                color='district_name',
                hover_name='bsk_name',
                hover_data=['district_name', 'block_municipalty_name', 'score'],
                zoom=6,
                height=600,
                title='BSK Geographic Distribution'
            )
            fig_map.update_layout(mapbox_style="carto-darkmatter")
            st.plotly_chart(fig_map, use_container_width=True)
    
    # BSK Details
    st.markdown("## 📋 BSK Details")
    
    for _, row in df.iterrows():
        with st.expander(f"**{row['bsk_name']}** (Score: {row['score']:.3f})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📍 Location**")
                st.write(f"District: {row['district_name']}")
                st.write(f"Block: {row['block_municipalty_name']}")
            
            with col2:
                st.markdown("**📊 Performance**")
                st.write(f"Score: {row['score']:.3f}")
                st.write(f"Reason: {row.get('reason', 'N/A')}")
            
            # Recommended services
            st.markdown("**🎯 Recommended Training**")
            rec_services = row.get('recommended_services', [])
            if rec_services:
                for service in rec_services[:5]:  # Show top 5
                    st.write(f"• {service}")
            else:
                st.info("No specific recommendations")
    
    # Export
    st.markdown("## 💾 Export Results")
    
    csv = df.to_csv(index=False)
    st.download_button(
        "📄 Download Analysis Results",
        data=csv,
        file_name="bsk_performance_analysis.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    # Welcome screen
    st.info("👆 Configure analysis parameters above and click 'Analyze Performance' to begin")
