import streamlit as st
import pandas as pd
from utils import fetch_data_with_controls
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Try to import BSK analytics for fallback functionality
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ai_service'))
    from bsk_analytics import (
        find_underperforming_bsks, 
        analyze_bsk_performance_trends,
        get_top_performing_bsks,
        calculate_district_benchmarks
    )
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="BSK Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    
    .bsk-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .bsk-header {
        background: linear-gradient(45deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.25rem;
    }
    
    .status-underperforming {
        background-color: #ff4757;
        color: white;
    }
    
    .status-good {
        background-color: #2ed573;
        color: white;
    }
    
    .info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .info-item {
        background: #f8f9fa;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 4px solid #667eea;
    }
    
    .sidebar-section {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("Identify Training Needs in BSKs")
st.markdown("**Monitoring and Analysis of Training Requirements in BSKs**")

# Main page filters
st.markdown("## Analysis Parameters")

col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

with col1:
    period_start = st.date_input("Start Date", value=None, key="start_date")

with col2:
    period_end = st.date_input("End Date", value=None, key="end_date")

with col3:
    num_bsks = st.number_input("Number of BSKs", min_value=1, max_value=200, value=50)

with col4:
    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
    check_btn = st.button("Analyze Performance", type="primary", use_container_width=True)

# State for filters
if 'filters' not in st.session_state:
    st.session_state.filters = {}

if check_btn:
    st.session_state.filters = {
        'period_start': period_start.strftime('%Y-%m-%d') if period_start else None,
        'period_end': period_end.strftime('%Y-%m-%d') if period_end else None,
        'num_bsks': num_bsks
    }

if st.session_state.filters:
    with st.spinner("Fetching BSK performance data..."):
        params = {k: v for k, v in st.session_state.filters.items() if v}
        
        # Sorting option for score
        sort_option = st.selectbox(
            "Sort BSKs by Score",
            ["Lowest Score", "Highest Score"],
            index=0
        )
        sort_order = 'desc' if sort_option == "Highest Score" else 'asc'
        
        # Fetch data from backend (move this here to use sort_order)
        params['sort_order'] = sort_order
        API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:54300")
        df = pd.DataFrame()
        try:
            import requests
            resp = requests.get(f"{API_BASE_URL}/underperforming_bsks/", params=params)
            resp.raise_for_status()
            data = resp.json()
            df = pd.DataFrame(data)
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            df = pd.DataFrame()
        # Compute score range for context (after df is loaded)
        score_min = df['score'].min() if 'score' in df.columns else None
        score_max = df['score'].max() if 'score' in df.columns else None
        score_mean = df['score'].mean() if 'score' in df.columns else None
    
    if not df.empty:
        # Additional Filters
        with st.sidebar:
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### Filters")
            
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
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Summary Metrics
        st.markdown("## Performance Overview")
        
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
        
        # Charts Section
        st.markdown("## Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # District distribution
            district_counts = df['district_name'].value_counts().reset_index()
            district_counts.columns = ['District', 'Count']
            
            fig_district = px.bar(
                district_counts, 
                x='District', 
                y='Count',
                title='BSKs by District',
                color='Count',
                color_continuous_scale='Reds'
            )
            fig_district.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12)
            )
            st.plotly_chart(fig_district, use_container_width=True)
        
        with col2:
            # Block distribution
            block_counts = df['block_municipalty_name'].value_counts().head(10).reset_index()
            block_counts.columns = ['Block/Municipality', 'Count']
            
            fig_block = px.bar(
                block_counts, 
                x='Block/Municipality', 
                y='Count',
                title='Top 10 Blocks/Municipalities',
                color='Count',
                color_continuous_scale='Blues'
            )
            fig_block.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_block, use_container_width=True)
        
        # Debug: Show available columns
        st.markdown("## Data Analysis")
        
        with st.expander("Debug: Available Columns", expanded=False):
            st.write("**All available columns:**")
            st.write(list(df.columns))
            
            # Look for potential coordinate columns
            coord_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['lat', 'long', 'lng', 'latitude', 'longitude', 'coord', 'location'])]
            if coord_columns:
                st.write("**Potential coordinate columns found:**")
                st.write(coord_columns)
                
                # Show sample data for coordinate columns
                st.write("**Sample coordinate data:**")
                st.dataframe(df[coord_columns].head(10))
            else:
                st.write("**No coordinate columns found**")
        
        # Interactive Map Section
        st.markdown("## BSK Locations Map")
        
        # Check for various possible coordinate column names
        lat_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['lat', 'latitude'])]
        lng_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['long', 'lng', 'longitude'])]
        
        if lat_cols and lng_cols:
            # Use the first found coordinate columns
            lat_col = lat_cols[0]
            lng_col = lng_cols[0]
            
            st.info(f"Using columns: {lat_col} (latitude) and {lng_col} (longitude)")
            
            # Prepare map data
            required_cols = [lat_col, lng_col, 'bsk_name', 'district_name', 'block_municipalty_name']
            available_cols = [col for col in required_cols if col in df.columns]
            
            map_df = df[available_cols].dropna()
            map_df[lat_col] = pd.to_numeric(map_df[lat_col], errors='coerce')
            map_df[lng_col] = pd.to_numeric(map_df[lng_col], errors='coerce')
            map_df = map_df.dropna(subset=[lat_col, lng_col])
            
            if not map_df.empty:
                # Create an interactive map using plotly scatter_mapbox
                fig_map = px.scatter_mapbox(
                    map_df,
                    lat=lat_col,
                    lon=lng_col,
                    color='district_name' if 'district_name' in available_cols else None,
                    hover_name='bsk_name' if 'bsk_name' in available_cols else None,
                    hover_data={
                        'district_name': True if 'district_name' in available_cols else False,
                        'block_municipalty_name': True if 'block_municipalty_name' in available_cols else False
                    },
                    zoom=6,
                    height=600,
                    title='BSK Geographic Distribution',
                )
                fig_map.update_layout(
                    mapbox_style="carto-darkmatter",
                    margin={"r":0,"t":40,"l":0,"b":0},
                    legend_title_text='District'
                )
                st.plotly_chart(fig_map, use_container_width=True)
                # Display coordinate statistics
                st.markdown("### Coordinate Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total BSKs with Coordinates", len(map_df))
                    st.metric("Latitude Range", f"{map_df[lat_col].min():.4f} to {map_df[lat_col].max():.4f}")
                with col2:
                    st.metric("Unique Districts", map_df['district_name'].nunique() if 'district_name' in map_df.columns else "N/A")
                    st.metric("Longitude Range", f"{map_df[lng_col].min():.4f} to {map_df[lng_col].max():.4f}")
            else:
                st.warning("No valid coordinates found in the data")
        else:
            st.warning("No coordinate columns found in the data")
            # Show what columns are actually available
            st.markdown("### Available Columns")
            st.write("The following columns are available in your dataset:")
            for i, col in enumerate(df.columns, 1):
                st.write(f"{i}. {col}")
            st.info("💡 **Tip:** Make sure your dataset includes latitude and longitude columns. Common names include: lat, latitude, long, lng, longitude, bsk_lat, bsk_long, etc.")
        
        # BSK Details Section
        st.markdown("## BSK Details")
        
        # Group by BSK for detailed view
        for bsk_id, group in df.groupby('bsk_id'):
            bsk_info = group.iloc[0]
            
            with st.expander(f"{bsk_info['bsk_name']}", expanded=False):
                st.markdown(f"""
                <div class="bsk-card">
                    <div class="bsk-header">
                        <h3 style="margin: 0;">{bsk_info['bsk_name']}</h3>
                        <span class="status-badge status-underperforming">Underperforming</span>
                    </div>
                """, unsafe_allow_html=True)
                
                # Basic Info Grid
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Location Information**")
                    st.write(f"District: {bsk_info['district_name']}")
                    st.write(f"Block: {bsk_info['block_municipalty_name']}")
                    st.write(f"Cluster: {bsk_info.get('cluster_id', 'N/A')}")
                
                with col2:
                    st.markdown("**Performance Details**")
                    st.write(f"Reason: {bsk_info.get('reason', 'Not specified')}")
                    if 'bsk_lat' in bsk_info and 'bsk_long' in bsk_info:
                        st.write(f"Coordinates: {bsk_info['bsk_lat']}, {bsk_info['bsk_long']}")
                    # Show score if available
                    if 'score' in bsk_info:
                        st.write(f"Score: **{bsk_info['score']:.3f}**")
                        # if score_min is not None and score_max is not None and score_mean is not None:
                        #     st.write(f"Score Range: min={score_min:.3f}, max={score_max:.3f}, mean={score_mean:.3f}")
                
                # Recommended Training
                st.markdown("**Recommended Training**")
                rec_services = bsk_info.get('recommended_services', [])
                if rec_services:
                    for service in rec_services:
                        st.markdown(f"• {service}")
                else:
                    st.info("No specific recommendations available")
                
                # DEO Information
                st.markdown("**DEO Information**")
                deo_cols = ['agent_id', 'user_name', 'agent_code', 'agent_email', 'agent_phone', 'date_of_engagement', 'bsk_post']
                available_deo_cols = [col for col in deo_cols if col in group.columns]
                
                if available_deo_cols:
                    deo_df = group[available_deo_cols].drop_duplicates().reset_index(drop=True)
                    if not deo_df.empty:
                        st.dataframe(deo_df, use_container_width=True)
                    else:
                        st.info("No DEO data available for this BSK")
                else:
                    st.info("DEO columns not found in data")
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        st.warning("No data available for the selected criteria")

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        
    </div>
    """, unsafe_allow_html=True)

