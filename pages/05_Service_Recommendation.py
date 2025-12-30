import streamlit as st
import pandas as pd
import requests
import sys
import os
import numpy as np
from sklearn.cluster import KMeans
import pydeck as pdk
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add paths for backend and AI service imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ai_service')))

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://bsk-backend-uywi.onrender.com")

# Import AI service functions with error handling and direct database access
try:
    # First try to import the AI functions
    from ai_service.service_recommendation import recommend_bsk_for_service_from_db, initialize_embeddings_from_db
    from ai_service.database_service import fetch_all_data_for_recommendations
    AI_FUNCTIONS_AVAILABLE = True
    DATABASE_AVAILABLE = True
    
except ImportError as e:
    DATABASE_AVAILABLE = False
    try:
        # Fallback to basic recommendation without database
        from ai_service.service_recommendation import recommend_bsk_for_service, initialize_service_embeddings
        AI_FUNCTIONS_AVAILABLE = True
        st.warning(f"⚠️ Database not available, using basic recommendation system. Error: {e}")
        
        # Create wrapper function that uses API instead of direct DB access
        def recommend_bsk_for_service_from_db(new_service, top_n=10, target_location=None, use_precomputed_embeddings=True, include_inactive=False):
            """Wrapper that uses API endpoints instead of direct database access"""
            try:
                # Get data from the FastAPI backend
                services = fetch_data("services/")
                provisions = fetch_data("provisions/")
                bsk_centers = fetch_data("bsk/")
                
                if not all([services, provisions, bsk_centers]):
                    st.error("❌ Could not fetch data from API. Please ensure backend is running.")
                    return None
                
                # Convert to DataFrames 
                services_df = pd.DataFrame(services)
                provisions_df = pd.DataFrame(provisions)
                bsk_df = pd.DataFrame(bsk_centers)
                
                # Call the recommendation function
                return recommend_bsk_for_service(
                    new_service=new_service,
                    services_df=services_df,
                    provisions_df=provisions_df,
                    bsk_df=bsk_df,
                    top_n=top_n,
                    target_location=target_location,
                    use_precomputed_embeddings=use_precomputed_embeddings
                )
            except Exception as e:
                st.error(f"❌ Error in API-based recommendation: {e}")
                return None
        
        def initialize_embeddings_from_db(*args, **kwargs):
            """Initialize embeddings using API data when database is not available"""
            try:
                from ai_service.service_recommendation import initialize_service_embeddings
                
                # Get services data from API
                services = fetch_data("services/")
                if not services:
                    st.info("📡 Cannot fetch services data from API for embedding initialization.")
                    return False
                
                services_df = pd.DataFrame(services)
                initialize_service_embeddings(services_df, force_rebuild=kwargs.get('force_rebuild', False))
                st.info("✅ Embeddings initialized using API data.")
                return True
                
            except Exception as e:
                st.info(f"📡 Embedding initialization with API data failed: {e}")
                return False
            
    except ImportError as e2:
        st.error(f"❌ Critical error: Cannot import recommendation functions. Error: {e2}")
        AI_FUNCTIONS_AVAILABLE = False
        DATABASE_AVAILABLE = False
        
        # Create minimal fallback functions
        def recommend_bsk_for_service_from_db(*args, **kwargs):
            st.error("❌ No recommendation service available. Please check your setup.")
            return None
        def initialize_embeddings_from_db(*args, **kwargs):
            return False

# Initialize session state variables
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1
if 'selected_bsk' not in st.session_state:
    st.session_state.selected_bsk = None

def fetch_data(endpoint):
    """Fetch data from API with improved error handling"""
    try:
        response = requests.get(f"{API_BASE_URL}/{endpoint}", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.warning(f"🔌 Cannot connect to backend service at {API_BASE_URL}")
        return []
    except requests.exceptions.Timeout:
        st.warning(f"⏱️ Request timeout for {endpoint}")
        return []
    except requests.exceptions.HTTPError as e:
        st.warning(f"📡 HTTP error for {endpoint}: {e}")
        return []
    except Exception as e:
        st.warning(f"⚠️ Error fetching {endpoint}: {e}")
        return []

def cluster_locations(df, n_clusters=81):
    """Cluster locations to reduce the number of points on the map"""
    if len(df) <= n_clusters:
        return df
    
    # Prepare coordinates for clustering
    coords = df[['bsk_lat', 'bsk_long']].values
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(coords)
    
    # Get the top BSK from each cluster based on score
    clustered_df = df.sort_values('score', ascending=False).groupby('cluster').first().reset_index()
    return clustered_df.drop('cluster', axis=1)

def get_color_rgba(score):
    """Return RGBA color based on score"""
    if score >= 0.7:
        return [46, 204, 113, 220]  # Green with transparency
    elif score >= 0.4:
        return [241, 196, 15, 220]  # Yellow with transparency
    else:
        return [231, 76, 60, 220]   # Red with transparency

# Main Application UI
st.title("🚀 Recommendation of Relevant BSKs for New Services")
st.markdown("Find the most suitable BSKs for launching new services")

st.divider()

# System Status Display
st.subheader("🔧 System Status")

# Initialize embeddings on first load
if 'embeddings_initialized' not in st.session_state:
    if AI_FUNCTIONS_AVAILABLE and DATABASE_AVAILABLE:
        with st.spinner("🔄 Initializing recommendation system with database..."):
            try:
                success = initialize_embeddings_from_db(force_rebuild=False)
                if success:
                    st.session_state.embeddings_initialized = True
                    st.success("  recommendation system with embeddings initialized successfully!")
                else:
                    st.warning("⚠️ system initialization incomplete. Continuing without precomputed embeddings.")
                    st.session_state.embeddings_initialized = False
            except Exception as e:
                st.error(f"❌ Error initializing embeddings: {e}")
                st.info("💡 Continuing without precomputed embeddings.")
                st.session_state.embeddings_initialized = False
    elif AI_FUNCTIONS_AVAILABLE:
        st.info("💡 Using API-based recommendation system. Embeddings will be computed on-demand.")
        st.session_state.embeddings_initialized = False
    else:
        st.warning("⚠️ Limited recommendation functionality available.")
        st.session_state.embeddings_initialized = False

# Service Input Form
with st.form("new_service_form"):
    service_name = st.text_input("Service Name", placeholder="e.g., Digital Ration Card Application")
    service_type = st.text_input("Service Type", placeholder="e.g., Government Service, Certification, Application")
    service_desc = st.text_area("Service Description", placeholder="Describe what this service does, who it's for, and any relevant details...")
    
    st.subheader("Options")
    col1, col2 = st.columns(2)
    
    with col1:
        if AI_FUNCTIONS_AVAILABLE:
            # Check if embeddings are available (either initialized or can be initialized)
            embeddings_available = st.session_state.get('embeddings_initialized', False)
            if not embeddings_available and DATABASE_AVAILABLE:
                # Check if we can access ChromaDB for existing embeddings
                try:
                    from ai_service.service_recommendation import get_embedding_manager
                    manager = get_embedding_manager()
                    embeddings_available = manager.get_service_count() > 0
                except:
                    embeddings_available = False
            
            use_precomputed = st.checkbox("Use embeddings", 
                                        value=embeddings_available,
                                        help="Use precomputed embeddings for faster recommendations")
        else:
            use_precomputed = False
            st.info("embeddings not available")
    
    with col2:
        top_n = st.number_input("Number of recommendations", min_value=1, max_value=1000, value=100)
        
        if DATABASE_AVAILABLE:
            include_inactive = st.checkbox("Include inactive BSKs", value=False)
        else:
            include_inactive = False
    
    submitted = st.form_submit_button("Get Recommendations")

# Process form submission
if submitted:
    if not service_name.strip() or not service_desc.strip():
        st.error("❌ Please provide at least a service name and description.")
        similar_services = None  # Ensure variable is defined for later display
    else:
        with st.spinner("🤖 Generating recommendations..."):
            try:
                new_service = {
                    "service_name": service_name.strip(),
                    "service_type": service_type.strip() if service_type.strip() else "General Service",
                    "service_desc": service_desc.strip()
                }
                target_location = None
                
                recommendations = recommend_bsk_for_service_from_db(
                    new_service=new_service,
                    top_n=top_n,
                    target_location=target_location,
                    use_precomputed_embeddings=use_precomputed,
                    include_inactive=include_inactive
                )
                # If recommendations is a dict, extract both
                if isinstance(recommendations, dict) and 'recommendations' in recommendations:
                    similar_services = recommendations.get('similar_services', None)
                    recommendations = recommendations['recommendations']
                else:
                    similar_services = None
                
                if recommendations is not None and not recommendations.empty:
                    st.session_state.recommendations = recommendations
                    st.session_state.current_page = 1
                    st.session_state.selected_bsk = None
                    st.success(f"🎯 Found {len(recommendations)} BSK recommendations!")
                    
                    # Show a quick preview of the top 3
                    st.write("**🏆 Top 3 Recommendations Preview:**")
                    top_3 = recommendations.head(3)
                    for i, (_, row) in enumerate(top_3.iterrows(), 1):
                        score_emoji = "🟢" if row['score'] >= 0.7 else "🟡" if row['score'] >= 0.4 else "🔴"
                        st.write(f"{i}. {score_emoji} **{row['bsk_name']}** - Score: {row['score']:.5f}")
                        if 'reason' in row:
                            st.write(f"   💡 {row['reason']}")
                else:
                    st.error("❌ No recommendations found. Please try adjusting your service description.")
            except ImportError as e:
                st.error("❌ Database connection issue. Please ensure the backend is running.")
                st.code(str(e))
                similar_services = None
            except Exception as e:
                st.error(f"❌ Error generating recommendations: {e}")
                similar_services = None

    # Always display the Top 5 Related Services section after form submission
    st.subheader("🔎 Related Services Found")
    if similar_services is not None and hasattr(similar_services, '__len__') and len(similar_services) > 0:
        if isinstance(similar_services, pd.DataFrame):
            top_similar = similar_services.head(5).reset_index(drop=True)
            st.dataframe(top_similar, use_container_width=True, hide_index=True)
        elif isinstance(similar_services, list) and isinstance(similar_services[0], dict):
            top_similar = similar_services[:5]
            st.dataframe(pd.DataFrame(top_similar), use_container_width=True, hide_index=True)
        elif isinstance(similar_services, list):
            top_similar = similar_services[:5]
            st.write(top_similar)
        else:
            st.write(similar_services)
    else:
        st.info("No similar services found for your input.")

# Display results if we have recommendations
if st.session_state.recommendations is not None:
    recommendations = st.session_state.recommendations
    st.subheader(" BSK Recommendations")
    
    # Display summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total BSKs", len(recommendations))
    with col2:
        avg_score = recommendations['score'].mean()
        st.metric("Average Score", f"{avg_score:.2f}")
    with col3:
        high_score_count = len(recommendations[recommendations['score'] >= 0.7])
        st.metric("High Score BSKs", high_score_count)
    with col4:
        if 'usage_count' in recommendations.columns:
            avg_usage = recommendations['usage_count'].mean()
            st.metric("Avg Usage Count", f"{avg_usage:.1f}")
    
    # Add filters
    st.subheader("🔍 Filter Results")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        min_score = st.slider("Minimum Score", 0.0, 1.0, 0.0, 0.05)
    
    with filter_col2:
        # District filter if available
        if 'district_name' in recommendations.columns:
            districts = ['All'] + sorted(recommendations['district_name'].dropna().unique().tolist())
            selected_district = st.selectbox("District", districts)
        else:
            selected_district = 'All'
    
    with filter_col3:
        # Score range filter
        score_range = st.select_slider(
            "Score Range",
            options=['All', 'High (≥0.7)', 'Medium (0.4-0.7)', 'Low (<0.4)'],
            value='All'
        )
    
    # Apply filters
    filtered_recommendations = recommendations[recommendations['score'] >= min_score].copy()
    
    if selected_district != 'All':
        filtered_recommendations = filtered_recommendations[
            filtered_recommendations['district_name'] == selected_district
        ]
    
    if score_range != 'All':
        if score_range == 'High (≥0.7)':
            filtered_recommendations = filtered_recommendations[filtered_recommendations['score'] >= 0.7]
        elif score_range == 'Medium (0.4-0.7)':
            filtered_recommendations = filtered_recommendations[
                (filtered_recommendations['score'] >= 0.4) & (filtered_recommendations['score'] < 0.7)
            ]
        elif score_range == 'Low (<0.4)':
            filtered_recommendations = filtered_recommendations[filtered_recommendations['score'] < 0.4]
    
    st.info(f"Showing {len(filtered_recommendations)} of {len(recommendations)} BSKs after filtering")
    
    # Add pagination controls
    items_per_page = 20
    total_pages = (len(filtered_recommendations) + items_per_page - 1) // items_per_page
    
    if total_pages > 1:
        # Create columns for pagination controls
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("Previous Page") and st.session_state.current_page > 1:
                st.session_state.current_page -= 1
        with col2:
            st.write(f"Page {st.session_state.current_page} of {total_pages}")
        with col3:
            if st.button("Next Page") and st.session_state.current_page < total_pages:
                st.session_state.current_page += 1
        
        # Ensure current page is within bounds
        if st.session_state.current_page > total_pages:
            st.session_state.current_page = total_pages
    
    # Display paginated data
    start_idx = (st.session_state.current_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(filtered_recommendations))
    
    if len(filtered_recommendations) > 0:
        display_data = filtered_recommendations.iloc[start_idx:end_idx]
        
        # Select which columns to display (exclude some technical columns)
        exclude_cols = ['cluster', 'cluster_size', 'avg_score', 'color']
        display_cols = [col for col in display_data.columns if col not in exclude_cols]
        
        st.dataframe(
            display_data[display_cols],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("No BSKs match the current filters.")
        
    # Map display
    recommendations_for_map = filtered_recommendations
    
    if 'bsk_lat' in recommendations_for_map.columns and 'bsk_long' in recommendations_for_map.columns:
        # Add map display options
        st.subheader("🗺️ Geographic Distribution")
        col1, col2 = st.columns(2)
        with col1:
            use_clustering = st.checkbox("Enable clustering ", value=False, 
                                       help="Group nearby BSKs to improve map performance")
        with col2:
            dot_size = st.slider("Dot size", min_value=500, max_value=5000, value=2000, step=250)
        
        # Prepare data for map
        map_df = recommendations_for_map.copy()
        map_df = map_df.rename(columns={'bsk_lat': 'lat', 'bsk_long': 'lon'})
        map_df = map_df.dropna(subset=['lat', 'lon'])
        
        # Convert to numeric and filter out invalid coordinates
        map_df['lat'] = pd.to_numeric(map_df['lat'], errors='coerce')
        map_df['lon'] = pd.to_numeric(map_df['lon'], errors='coerce')
        map_df = map_df.dropna(subset=['lat', 'lon'])
        
        # Filter out clearly invalid coordinates
        map_df = map_df[
            (map_df['lat'].between(-90, 90)) & 
            (map_df['lon'].between(-180, 180))
        ]

        # Apply clustering if enabled
        if use_clustering and len(map_df) > 50:
            # Determine number of clusters based on data size
            n_clusters = min(50, len(map_df) // 2)
            
            # Prepare coordinates for clustering
            coords = map_df[['lat', 'lon']].values
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            map_df['cluster'] = kmeans.fit_predict(coords)
            
            # For each cluster, keep the highest scoring BSK and calculate cluster stats
            cluster_stats = []
            for cluster_id in range(n_clusters):
                cluster_data = map_df[map_df['cluster'] == cluster_id]
                if len(cluster_data) > 0:
                    # Get the best BSK in this cluster
                    best_bsk = cluster_data.loc[cluster_data['score'].idxmax()]
                    
                    # Calculate cluster statistics
                    cluster_info = best_bsk.copy()
                    cluster_info['cluster_size'] = len(cluster_data)
                    cluster_info['avg_score'] = cluster_data['score'].mean()
                    cluster_info['bsk_name'] = f"{best_bsk['bsk_name']} (+{len(cluster_data)-1} others)"
                    
                    cluster_stats.append(cluster_info)
            
            map_df = pd.DataFrame(cluster_stats)
            st.info(f"Clustered {len(recommendations_for_map)} BSKs into {len(map_df)} clusters")

        if not map_df.empty:
            # Apply colors to each row
            map_df['color'] = map_df['score'].apply(get_color_rgba)
            
            # Format score for display in tooltip
            map_df['score_formatted'] = map_df['score'].apply(lambda x: f"{x:.5f}")
            
            # Convert to list of lists for PyDeck
            map_df['color'] = map_df['color'].tolist()

            # Create tooltip content based on clustering mode
            if use_clustering and 'cluster_size' in map_df.columns:
                # Format avg_score for display as well
                map_df['avg_score_formatted'] = map_df['avg_score'].apply(lambda x: f"{x:.5f}")
                tooltip_html = '''
                <b>BSK:</b> {bsk_name}<br/>
                <b>Score:</b> {score_formatted}<br/>
                <b>Cluster Size:</b> {cluster_size}<br/>
                <b>Avg Score:</b> {avg_score_formatted}
                '''
            else:
                # Add more details to tooltip
                tooltip_html = '<b>BSK:</b> {bsk_name}<br/><b>Score:</b> {score_formatted}'
                if 'district_name' in map_df.columns:
                    tooltip_html += '<br/><b>District:</b> {district_name}'
                if 'usage_count' in map_df.columns:
                    map_df['usage_formatted'] = map_df['usage_count'].apply(lambda x: f"{int(x)}")
                    tooltip_html += '<br/><b>Usage Count:</b> {usage_formatted}'
                if 'reason' in map_df.columns:
                    tooltip_html += '<br/><b>Reason:</b> {reason}'
            
            # Create the scatterplot layer
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
            
            # Set up the view state
            view_state = pdk.ViewState(
                latitude=map_df['lat'].mean(),
                longitude=map_df['lon'].mean(),
                zoom=7,
                pitch=0,
            )
            
            # Create and display the deck
            deck = pdk.Deck(
                layers=[layer], 
                initial_view_state=view_state,
                tooltip={
                    'html': tooltip_html,
                    'style': {
                        'backgroundColor': 'steelblue',
                        'color': 'white'
                    }
                }
            )
            
            st.pydeck_chart(deck)
            
            # Add legend
            legend_text = """
            **Map Legend:**
            - 🟢 Green: High score (≥ 0.7) - Excellent fit for new service
            - 🟡 Yellow: Medium score (0.4 - 0.7) - Good potential
            - 🔴 Red: Low score (< 0.4) - Limited potential
            """
            if use_clustering:
                legend_text += "\n- **Clustering enabled:** Each dot represents the best BSK in a cluster"
            
            st.markdown(legend_text)
        else:
            st.warning('No BSKs with valid coordinates to display on the map.')

        # BSK Details Section
        st.subheader("🔍 BSK Details")
        bsk_options = recommendations_for_map['bsk_name'].tolist() if 'bsk_name' in recommendations_for_map.columns else []
        if bsk_options:
            selected_bsk = st.selectbox("Choose a BSK to view detailed information", bsk_options, key='bsk_selector')
            
            # Update selected BSK in session state
            if selected_bsk != st.session_state.selected_bsk:
                st.session_state.selected_bsk = selected_bsk
            
            # Show details for selected BSK
            if st.session_state.selected_bsk:
                selected_data = recommendations_for_map[recommendations_for_map['bsk_name'] == st.session_state.selected_bsk]
                if not selected_data.empty:
                    st.write("### 📋 Selected BSK Details")
                    
                    # Create a nice display of the BSK details
                    row = selected_data.iloc[0]
                    
                    # Basic info in columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Score", f"{row['score']:.5f}")
                        if 'usage_count' in row:
                            st.metric("Usage Count", int(row['usage_count']))
                    with col2:
                        if 'district_name' in row:
                            st.write(f"**District:** {row['district_name']}")
                        if 'block_municipalty_name' in row:
                            st.write(f"**Block/Municipality:** {row['block_municipalty_name']}")
                    with col3:
                        if 'reason' in row:
                            st.write(f"**Recommendation Reason:** {row['reason']}")
                    
                    # Display all other relevant columns in an expandable section
                    with st.expander("View all details"):
                        # Exclude some columns from detailed view
                        exclude_detail_cols = ['bsk_lat', 'bsk_long', 'color', 'cluster', 'cluster_size', 'avg_score', 'lat', 'lon']
                        detail_cols = [col for col in selected_data.columns if col not in exclude_detail_cols]
                        st.dataframe(selected_data[detail_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No BSKs available for detailed view.")
    else:
        st.info("Geographic coordinates not available for map display.")
    
    # Analytics and Visualizations
    st.subheader("Recommendation Analytics")
    
    if 'bsk_name' in recommendations_for_map.columns and not recommendations_for_map.empty:
        tab1, tab2, tab3 = st.tabs(["Score Distribution", "Top BSKs", "Geographic Analysis"])
        
        with tab1:
            # Score distribution histogram using simple binning
            st.write("**Score Distribution:**")
            
            # Create bins for the scores
            score_bins = pd.cut(recommendations_for_map['score'], bins=10)
            score_counts = score_bins.value_counts().sort_index()
            
            # Create a simple bar chart
            bin_labels = [f"{interval.left:.1f}-{interval.right:.1f}" for interval in score_counts.index]
            score_df = pd.DataFrame({
                'Score Range': bin_labels,
                'Count': score_counts.values
            })
            
            st.bar_chart(score_df.set_index('Score Range')['Count'])
            
            # Show summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Score", f"{recommendations_for_map['score'].mean():.5f}")
            with col2:
                st.metric("Median Score", f"{recommendations_for_map['score'].median():.5f}")
            with col3:
                st.metric("Std Dev", f"{recommendations_for_map['score'].std():.5f}")
        
        with tab2:
            # Top 20 BSKs bar chart
            top_bsks = recommendations_for_map.head(20)
            st.bar_chart(
                top_bsks.set_index('bsk_name')['score'],
                height=600
            )
            st.caption("Top 20 BSKs by recommendation score")
        
        with tab3:
            # Geographic analysis
            if 'district_name' in recommendations_for_map.columns:
                district_scores = recommendations_for_map.groupby('district_name').agg({
                    'score': ['mean', 'count'],
                    'usage_count': 'mean' if 'usage_count' in recommendations_for_map.columns else 'size'
                }).round(3)
                district_scores.columns = ['Avg Score', 'BSK Count', 'Avg Usage']
                district_scores = district_scores.sort_values('Avg Score', ascending=False)
                
                st.write("**Performance by District:**")
                st.dataframe(district_scores, use_container_width=True)
    
    # Export functionality
    st.subheader("💾 Export Results")
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        if st.button("📄 Download as CSV"):
            csv = recommendations_for_map.to_csv(index=False)
            st.download_button(
                label="Click to download CSV",
                data=csv,
                file_name="bsk_recommendations.csv",
                mime="text/csv"
            )
    
    with export_col2:
        if st.button("Download Top 50 as CSV"):
            top_50 = recommendations_for_map.head(50)
            csv = top_50.to_csv(index=False)
            st.download_button(
                label="Click to download Top 50 CSV",
                data=csv,
                file_name="top_50_bsk_recommendations.csv",
                mime="text/csv"
            )