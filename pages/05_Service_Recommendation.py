import streamlit as st
import pandas as pd
import sys
import os
import numpy as np
from sklearn.cluster import KMeans
import pydeck as pdk
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add paths for AI service imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ai_service')))

# Import local data loader
from data_loader_local import (
    get_bsk_master, 
    get_service_master, 
    get_provisions,
    is_initialized
)

# Import AI service functions
try:
    from ai_service.service_recommendation import (
        recommend_bsk_for_service, 
        initialize_service_embeddings,
        get_embedding_manager
    )
    AI_FUNCTIONS_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Could not import AI recommendation functions: {e}")
    AI_FUNCTIONS_AVAILABLE = False
    
    # Create dummy functions
    def recommend_bsk_for_service(*args, **kwargs):
        st.error("❌ Recommendation service not available")
        return None
    def initialize_service_embeddings(*args, **kwargs):
        return False

# Page config
st.set_page_config(page_title="Service Recommendation", layout="wide")

# Initialize session state
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1
if 'selected_bsk' not in st.session_state:
    st.session_state.selected_bsk = None

def get_color_rgba(score):
    """Return RGBA color based on score"""
    if score >= 0.7:
        return [46, 204, 113, 220]  # Green
    elif score >= 0.4:
        return [241, 196, 15, 220]  # Yellow
    else:
        return [231, 76, 60, 220]   # Red

# Main UI
st.title("🚀 Service Recommendation System")
st.markdown("Find the most suitable BSKs for launching new services")
st.divider()

# Check if data is loaded
if not is_initialized():
    st.error("❌ Data not initialized. Please restart the app.")
    st.stop()

# System Status
st.subheader("🔧 System Status")

# Initialize embeddings once
if 'embeddings_initialized' not in st.session_state:
    if AI_FUNCTIONS_AVAILABLE:
        with st.spinner("🔄 Initializing AI recommendation system..."):
            try:
                services_df = get_service_master()
                if not services_df.empty:
                    initialize_service_embeddings(services_df, force_rebuild=False)
                    st.session_state.embeddings_initialized = True
                    st.success("✅ AI recommendation system ready!")
                else:
                    st.warning("⚠️ No services data available")
                    st.session_state.embeddings_initialized = False
            except Exception as e:
                st.error(f"❌ Error initializing embeddings: {e}")
                st.session_state.embeddings_initialized = False
    else:
        st.warning("⚠️ AI functions not available")
        st.session_state.embeddings_initialized = False

# Service Input Form
with st.form("service_form"):
    st.subheader("📝 New Service Details")
    
    service_name = st.text_input(
        "Service Name*", 
        placeholder="e.g., Digital Ration Card Application"
    )
    service_type = st.text_input(
        "Service Type", 
        placeholder="e.g., Government Service, Certification"
    )
    service_desc = st.text_area(
        "Service Description*", 
        placeholder="Describe what this service does, who it's for, and relevant details...",
        height=100
    )
    
    st.subheader("⚙️ Options")
    col1, col2 = st.columns(2)
    
    with col1:
        # Check if embeddings are available
        embeddings_available = st.session_state.get('embeddings_initialized', False)
        if not embeddings_available and AI_FUNCTIONS_AVAILABLE:
            try:
                manager = get_embedding_manager()
                embeddings_available = manager.get_service_count() > 0
            except:
                embeddings_available = False
        
        use_precomputed = st.checkbox(
            "Use AI embeddings", 
            value=embeddings_available,
            help="Use precomputed embeddings for faster, more accurate recommendations",
            disabled=not embeddings_available
        )
    
    with col2:
        top_n = st.number_input(
            "Number of recommendations", 
            min_value=1, 
            max_value=500, 
            value=50
        )
    
    submitted = st.form_submit_button("🎯 Get Recommendations", type="primary")

# Process form submission
if submitted:
    if not service_name.strip() or not service_desc.strip():
        st.error("❌ Please provide service name and description")
    else:
        with st.spinner("🤖 Generating recommendations..."):
            try:
                # Prepare service data
                new_service = {
                    "service_name": service_name.strip(),
                    "service_type": service_type.strip() or "General Service",
                    "service_desc": service_desc.strip()
                }
                
                # Load data
                services_df = get_service_master()
                bsk_df = get_bsk_master()
                provisions_df = get_provisions()
                
                if services_df.empty or bsk_df.empty or provisions_df.empty:
                    st.error("❌ Required data not available")
                else:
                    # Get recommendations
                    result = recommend_bsk_for_service(
                        new_service=new_service,
                        services_df=services_df,
                        provisions_df=provisions_df,
                        bsk_df=bsk_df,
                        top_n=top_n,
                        target_location=None,
                        use_precomputed_embeddings=use_precomputed
                    )
                    
                    # Handle result (could be tuple or dataframe)
                    if isinstance(result, tuple) and len(result) == 2:
                        recommendations, similar_services = result
                    else:
                        recommendations = result
                        similar_services = None
                    
                    if recommendations is not None and not recommendations.empty:
                        st.session_state.recommendations = recommendations
                        st.session_state.current_page = 1
                        st.session_state.selected_bsk = None
                        
                        st.success(f"🎯 Found {len(recommendations)} BSK recommendations!")
                        
                        # Preview top 3
                        st.write("**🏆 Top 3 Recommendations:**")
                        for i, (_, row) in enumerate(recommendations.head(3).iterrows(), 1):
                            score_emoji = "🟢" if row['score'] >= 0.7 else "🟡" if row['score'] >= 0.4 else "🔴"
                            st.write(f"{i}. {score_emoji} **{row['bsk_name']}** - Score: {row['score']:.4f}")
                            if 'reason' in row:
                                st.write(f"   💡 {row['reason']}")
                        
                        # Show similar services if available
                        if similar_services is not None and len(similar_services) > 0:
                            with st.expander("🔎 View Similar Services"):
                                if isinstance(similar_services, pd.DataFrame):
                                    st.dataframe(
                                        similar_services.head(5)[['service_name', 'service_type', 'total_similarity']],
                                        use_container_width=True,
                                        hide_index=True
                                    )
                    else:
                        st.error("❌ No recommendations generated. Try adjusting your description.")
                        
            except Exception as e:
                st.error(f"❌ Error generating recommendations: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())

# Display Results
if st.session_state.recommendations is not None:
    recommendations = st.session_state.recommendations
    
    st.divider()
    st.subheader("📊 BSK Recommendations")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total BSKs", len(recommendations))
    with col2:
        st.metric("Average Score", f"{recommendations['score'].mean():.3f}")
    with col3:
        high_score = len(recommendations[recommendations['score'] >= 0.7])
        st.metric("High Score (≥0.7)", high_score)
    with col4:
        if 'usage_count' in recommendations.columns:
            st.metric("Avg Usage", f"{recommendations['usage_count'].mean():.1f}")
    
    # Filters
    st.subheader("🔍 Filter Results")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        min_score = st.slider("Minimum Score", 0.0, 1.0, 0.0, 0.05)
    
    with filter_col2:
        if 'district_name' in recommendations.columns:
            districts = ['All'] + sorted(recommendations['district_name'].dropna().unique().tolist())
            selected_district = st.selectbox("District", districts)
        else:
            selected_district = 'All'
    
    with filter_col3:
        score_range = st.select_slider(
            "Score Range",
            options=['All', 'High (≥0.7)', 'Medium (0.4-0.7)', 'Low (<0.4)'],
            value='All'
        )
    
    # Apply filters
    filtered = recommendations[recommendations['score'] >= min_score].copy()
    
    if selected_district != 'All':
        filtered = filtered[filtered['district_name'] == selected_district]
    
    if score_range == 'High (≥0.7)':
        filtered = filtered[filtered['score'] >= 0.7]
    elif score_range == 'Medium (0.4-0.7)':
        filtered = filtered[(filtered['score'] >= 0.4) & (filtered['score'] < 0.7)]
    elif score_range == 'Low (<0.4)':
        filtered = filtered[filtered['score'] < 0.4]
    
    st.info(f"Showing {len(filtered)} of {len(recommendations)} BSKs")
    
    # Pagination
    items_per_page = 20
    total_pages = (len(filtered) + items_per_page - 1) // items_per_page
    
    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ Previous") and st.session_state.current_page > 1:
                st.session_state.current_page -= 1
                st.rerun()
        with col2:
            st.write(f"Page {st.session_state.current_page} of {total_pages}")
        with col3:
            if st.button("Next ➡️") and st.session_state.current_page < total_pages:
                st.session_state.current_page += 1
                st.rerun()
    
    # Display page data
    start = (st.session_state.current_page - 1) * items_per_page
    end = min(start + items_per_page, len(filtered))
    
    if len(filtered) > 0:
        display_data = filtered.iloc[start:end]
        
        # Select columns to display
        exclude_cols = ['cluster', 'cluster_size', 'avg_score', 'color']
        display_cols = [col for col in display_data.columns if col not in exclude_cols]
        
        st.dataframe(
            display_data[display_cols],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("No BSKs match current filters")
    
    # Map visualization
    if 'bsk_lat' in filtered.columns and 'bsk_long' in filtered.columns:
        st.subheader("🗺️ Geographic Distribution")
        
        # Map options
        col1, col2 = st.columns(2)
        with col1:
            use_clustering = st.checkbox("Enable clustering", help="Group nearby BSKs")
        with col2:
            dot_size = st.slider("Dot size", 500, 5000, 2000, 250)
        
        # Prepare map data
        map_df = filtered.copy()
        map_df = map_df.rename(columns={'bsk_lat': 'lat', 'bsk_long': 'lon'})
        map_df['lat'] = pd.to_numeric(map_df['lat'], errors='coerce')
        map_df['lon'] = pd.to_numeric(map_df['lon'], errors='coerce')
        map_df = map_df.dropna(subset=['lat', 'lon'])
        map_df = map_df[(map_df['lat'].between(-90, 90)) & (map_df['lon'].between(-180, 180))]
        
        # Clustering if enabled
        if use_clustering and len(map_df) > 50:
            n_clusters = min(50, len(map_df) // 2)
            coords = map_df[['lat', 'lon']].values
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            map_df['cluster'] = kmeans.fit_predict(coords)
            
            # Keep best from each cluster
            cluster_stats = []
            for cid in range(n_clusters):
                cluster_data = map_df[map_df['cluster'] == cid]
                if len(cluster_data) > 0:
                    best = cluster_data.loc[cluster_data['score'].idxmax()].copy()
                    best['cluster_size'] = len(cluster_data)
                    best['avg_score'] = cluster_data['score'].mean()
                    best['bsk_name'] = f"{best['bsk_name']} (+{len(cluster_data)-1} others)"
                    cluster_stats.append(best)
            
            map_df = pd.DataFrame(cluster_stats)
            st.info(f"Clustered into {len(map_df)} groups")
        
        if not map_df.empty:
            # Add colors
            map_df['color'] = map_df['score'].apply(get_color_rgba)
            map_df['score_formatted'] = map_df['score'].apply(lambda x: f"{x:.4f}")
            
            # Create map layer
            layer = pdk.Layer(
                'ScatterplotLayer',
                data=map_df,
                get_position=['lon', 'lat'],
                get_color='color',
                get_radius=dot_size,
                pickable=True,
                auto_highlight=True,
                radius_scale=1,
                radius_min_pixels=3,
                radius_max_pixels=50,
            )
            
            # View state
            view_state = pdk.ViewState(
                latitude=map_df['lat'].mean(),
                longitude=map_df['lon'].mean(),
                zoom=7,
                pitch=0,
            )
            
            # Tooltip
            tooltip_html = '<b>BSK:</b> {bsk_name}<br/><b>Score:</b> {score_formatted}'
            if 'district_name' in map_df.columns:
                tooltip_html += '<br/><b>District:</b> {district_name}'
            
            # Display map
            st.pydeck_chart(pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={'html': tooltip_html, 'style': {'backgroundColor': 'steelblue', 'color': 'white'}}
            ))
            
            # Legend
            st.markdown("""
            **Legend:**
            - 🟢 Green: High score (≥0.7)
            - 🟡 Yellow: Medium score (0.4-0.7)
            - 🔴 Red: Low score (<0.4)
            """)
    
    # Export
    st.subheader("💾 Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        csv = filtered.to_csv(index=False)
        st.download_button(
            "📄 Download All as CSV",
            data=csv,
            file_name="bsk_recommendations.csv",
            mime="text/csv"
        )
    
    with col2:
        top_50_csv = filtered.head(50).to_csv(index=False)
        st.download_button(
            "📄 Download Top 50 as CSV",
            data=top_50_csv,
            file_name="top_50_recommendations.csv",
            mime="text/csv"
        )
