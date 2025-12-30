import streamlit as st
import pandas as pd
import requests
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pydeck as pdk
import warnings
warnings.filterwarnings('ignore')

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://bsk-backend-uywi.onrender.com")

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
        response = requests.get(f"{API_BASE_URL}/{endpoint}", timeout=30)
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

def extract_keywords(text):
    """Extract keywords from text"""
    if pd.isna(text) or not text:
        return set()
    text_lower = str(text).lower()
    # Simple keyword extraction - split by common delimiters
    words = text_lower.replace(',', ' ').replace('.', ' ').replace(';', ' ').split()
    # Filter out very short words
    return set([w for w in words if len(w) > 3])

def calculate_text_similarity(text1, text2):
    """Calculate simple text similarity using keyword overlap"""
    keywords1 = extract_keywords(text1)
    keywords2 = extract_keywords(text2)
    
    if not keywords1 or not keywords2:
        return 0.0
    
    intersection = keywords1.intersection(keywords2)
    union = keywords1.union(keywords2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

def recommend_bsk_for_service_api(new_service, top_n=100, include_inactive=False):
    """
    Recommend BSKs for a new service using API data only.
    This is a simplified version that works without ai_service module.
    """
    try:
        # Fetch data from API
        with st.spinner("📡 Fetching data from backend..."):
            services_data = fetch_data("services/")
            provisions_data = fetch_data("provisions/")
            bsk_data = fetch_data("bsk/")
        
        if not all([services_data, provisions_data, bsk_data]):
            st.error("❌ Could not fetch required data from API. Please ensure backend is running.")
            return None, None
        
        # Convert to DataFrames
        services_df = pd.DataFrame(services_data)
        provisions_df = pd.DataFrame(provisions_data)
        bsk_df = pd.DataFrame(bsk_data)
        
        if services_df.empty or bsk_df.empty:
            st.error("❌ No data available for recommendations.")
            return None, None
        
        # Filter inactive BSKs if needed
        if not include_inactive and 'is_active' in bsk_df.columns:
            bsk_df = bsk_df[bsk_df['is_active'] == True]
        
        # Create full description for new service
        new_service_desc = f"{new_service.get('service_name', '')} {new_service.get('service_type', '')} {new_service.get('service_desc', '')}"
        new_service_keywords = extract_keywords(new_service_desc)
        
        # Calculate similarity with existing services
        services_df = services_df.copy()
        
        # Create full description for each service
        if 'service_desc' in services_df.columns:
            services_df['full_desc'] = services_df.apply(
                lambda row: f"{row.get('service_name', '')} {row.get('service_type', '')} {row.get('service_desc', '')}",
                axis=1
            )
        else:
            services_df['full_desc'] = services_df.get('service_name', '') + ' ' + services_df.get('service_type', '')
        
        # Calculate keyword overlap
        services_df['keyword_overlap'] = services_df['full_desc'].apply(
            lambda desc: len(new_service_keywords.intersection(extract_keywords(desc)))
        )
        
        # Calculate text similarity
        services_df['text_similarity'] = services_df['full_desc'].apply(
            lambda desc: calculate_text_similarity(new_service_desc, desc)
        )
        
        # Use TF-IDF for better similarity if we have enough data
        try:
            if len(services_df) > 1:
                vectorizer = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))
                all_descriptions = [new_service_desc] + services_df['full_desc'].tolist()
                tfidf_matrix = vectorizer.fit_transform(all_descriptions)
                
                # Calculate cosine similarity with first vector (new service)
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
                services_df['tfidf_similarity'] = similarities
            else:
                services_df['tfidf_similarity'] = 0.0
        except Exception as e:
            st.warning(f"⚠️ TF-IDF calculation failed, using basic similarity: {e}")
            services_df['tfidf_similarity'] = services_df['text_similarity']
        
        # Combine similarities
        max_kw = max(services_df['keyword_overlap'].max(), 1) if services_df['keyword_overlap'].max() > 0 else 1
        services_df['total_similarity'] = (
            0.4 * services_df['tfidf_similarity'] +
            0.3 * services_df['text_similarity'] +
            0.3 * (services_df['keyword_overlap'] / max_kw)
        )
        
        # Get similar services (threshold > 0.1)
        similar_services = services_df[services_df['total_similarity'] > 0.1].copy()
        similar_services = similar_services.sort_values('total_similarity', ascending=False)
        similar_service_ids = similar_services['service_id'].tolist()
        
        if not similar_service_ids:
            # If no similar services found, use all services with lower threshold
            similar_services = services_df.nlargest(min(10, len(services_df)), 'total_similarity')
            similar_service_ids = similar_services['service_id'].tolist()
        
        # Usage analysis - find BSKs that provide similar services
        if not provisions_df.empty and 'service_id' in provisions_df.columns and 'bsk_id' in provisions_df.columns:
            relevant_provisions = provisions_df[provisions_df['service_id'].isin(similar_service_ids)]
            
            if not relevant_provisions.empty:
                bsk_counts = relevant_provisions['bsk_id'].value_counts().reset_index()
                bsk_counts.columns = ['bsk_id', 'usage_count']
            else:
                # If no provisions found, give all BSKs equal usage
                bsk_counts = pd.DataFrame({'bsk_id': bsk_df['bsk_id'].unique(), 'usage_count': 1})
        else:
            # If no provisions data, give all BSKs equal usage
            bsk_counts = pd.DataFrame({'bsk_id': bsk_df['bsk_id'].unique(), 'usage_count': 1})
        
        # Merge with BSK data
        recommended_bsk = bsk_counts.merge(bsk_df, on='bsk_id', how='left')
        
        # Calculate score based on usage (min-max normalization)
        if len(recommended_bsk) > 0:
            usage_min = recommended_bsk['usage_count'].min()
            usage_max = recommended_bsk['usage_count'].max()
            
            if usage_max > usage_min:
                recommended_bsk['score'] = (recommended_bsk['usage_count'] - usage_min) / (usage_max - usage_min)
            else:
                recommended_bsk['score'] = 1.0
            
            # Add reason text
            def get_reason(row):
                reasons = []
                if row['usage_count'] > recommended_bsk['usage_count'].quantile(0.75):
                    reasons.append("High usage of similar services")
                elif row['usage_count'] > recommended_bsk['usage_count'].quantile(0.5):
                    reasons.append("Moderate usage of similar services")
                else:
                    reasons.append("Some usage of similar services")
                
                if 'district_name' in row and pd.notna(row['district_name']):
                    reasons.append(f"Located in {row['district_name']}")
                
                return "; ".join(reasons)
            
            recommended_bsk['reason'] = recommended_bsk.apply(get_reason, axis=1)
            
            # Sort by score
            recommended_bsk = recommended_bsk.sort_values('score', ascending=False)
            
            # Limit to top_n
            recommended_bsk = recommended_bsk.head(top_n)
            
            return recommended_bsk, similar_services.head(10)
        else:
            st.error("❌ No BSKs found for recommendations.")
            return None, None
            
    except Exception as e:
        st.error(f"❌ Error generating recommendations: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None

def get_color_rgba(score):
    """Return RGBA color based on score"""
    if score >= 0.7:
        return [46, 204, 113, 220]  # Green
    elif score >= 0.4:
        return [241, 196, 15, 220]  # Yellow
    else:
        return [231, 76, 60, 220]   # Red

# Main Application UI
st.title("🚀 Recommendation of Relevant BSKs for New Services")
st.markdown("Find the most suitable BSKs for launching new services")

st.divider()

# System Status Display
st.subheader("🔧 System Status")
st.info("💡 Using API-based recommendation system. Recommendations are calculated using text similarity and usage patterns.")

# Service Input Form
with st.form("new_service_form"):
    service_name = st.text_input("Service Name", placeholder="e.g., Digital Ration Card Application")
    service_type = st.text_input("Service Type", placeholder="e.g., Government Service, Certification, Application")
    service_desc = st.text_area("Service Description", placeholder="Describe what this service does, who it's for, and any relevant details...")
    
    st.subheader("Options")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("💡 Recommendations use text similarity and usage analysis")
    
    with col2:
        top_n = st.number_input("Number of recommendations", min_value=1, max_value=1000, value=100)
        include_inactive = st.checkbox("Include inactive BSKs", value=False)
    
    submitted = st.form_submit_button("Get Recommendations", type="primary")

# Process form submission
if submitted:
    if not service_name.strip() or not service_desc.strip():
        st.error("❌ Please provide at least a service name and description.")
        similar_services = None
    else:
        with st.spinner("🤖 Generating recommendations..."):
            new_service = {
                "service_name": service_name.strip(),
                "service_type": service_type.strip() if service_type.strip() else "General Service",
                "service_desc": service_desc.strip()
            }
            
            recommendations, similar_services = recommend_bsk_for_service_api(
                new_service=new_service,
                top_n=top_n,
                include_inactive=include_inactive
            )
            
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
                    st.write(f"{i}. {score_emoji} **{row.get('bsk_name', 'N/A')}** - Score: {row['score']:.3f}")
                    if 'reason' in row:
                        st.write(f"   💡 {row['reason']}")
            else:
                st.error("❌ No recommendations found. Please try adjusting your service description.")

    # Display related services
    st.subheader("🔎 Related Services Found")
    if similar_services is not None and not similar_services.empty:
        display_cols = ['service_name', 'service_type', 'total_similarity']
        available_cols = [col for col in display_cols if col in similar_services.columns]
        if available_cols:
            st.dataframe(similar_services[available_cols].head(10), use_container_width=True, hide_index=True)
        else:
            st.dataframe(similar_services.head(10), use_container_width=True, hide_index=True)
    else:
        st.info("No similar services found for your input.")

# Display results if we have recommendations
if st.session_state.recommendations is not None:
    recommendations = st.session_state.recommendations
    st.subheader("📊 BSK Recommendations")
    
    # Display summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total BSKs", len(recommendations))
    with col2:
        avg_score = recommendations['score'].mean()
        st.metric("Average Score", f"{avg_score:.3f}")
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
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("Previous Page") and st.session_state.current_page > 1:
                st.session_state.current_page -= 1
        with col2:
            st.write(f"Page {st.session_state.current_page} of {total_pages}")
        with col3:
            if st.button("Next Page") and st.session_state.current_page < total_pages:
                st.session_state.current_page += 1
        
        if st.session_state.current_page > total_pages:
            st.session_state.current_page = total_pages
    
    # Display paginated data
    start_idx = (st.session_state.current_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(filtered_recommendations))
    
    if len(filtered_recommendations) > 0:
        display_data = filtered_recommendations.iloc[start_idx:end_idx]
        exclude_cols = ['cluster', 'cluster_size', 'avg_score', 'color']
        display_cols = [col for col in display_data.columns if col not in exclude_cols]
        st.dataframe(display_data[display_cols], use_container_width=True, hide_index=True)
    else:
        st.warning("No BSKs match the current filters.")
    
    # Map display
    recommendations_for_map = filtered_recommendations
    
    if 'bsk_lat' in recommendations_for_map.columns and 'bsk_long' in recommendations_for_map.columns:
        st.subheader("🗺️ Geographic Distribution")
        col1, col2 = st.columns(2)
        with col1:
            use_clustering = st.checkbox("Enable clustering", value=False, 
                                       help="Group nearby BSKs to improve map performance")
        with col2:
            dot_size = st.slider("Dot size", min_value=500, max_value=5000, value=2000, step=250)
        
        map_df = recommendations_for_map.copy()
        map_df = map_df.rename(columns={'bsk_lat': 'lat', 'bsk_long': 'lon'})
        map_df = map_df.dropna(subset=['lat', 'lon'])
        map_df['lat'] = pd.to_numeric(map_df['lat'], errors='coerce')
        map_df['lon'] = pd.to_numeric(map_df['lon'], errors='coerce')
        map_df = map_df.dropna(subset=['lat', 'lon'])
        map_df = map_df[
            (map_df['lat'].between(-90, 90)) & 
            (map_df['lon'].between(-180, 180))
        ]

        if use_clustering and len(map_df) > 50:
            n_clusters = min(50, len(map_df) // 2)
            coords = map_df[['lat', 'lon']].values
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            map_df['cluster'] = kmeans.fit_predict(coords)
            
            cluster_stats = []
            for cluster_id in range(n_clusters):
                cluster_data = map_df[map_df['cluster'] == cluster_id]
                if len(cluster_data) > 0:
                    best_bsk = cluster_data.loc[cluster_data['score'].idxmax()]
                    cluster_info = best_bsk.copy()
                    cluster_info['cluster_size'] = len(cluster_data)
                    cluster_info['avg_score'] = cluster_data['score'].mean()
                    cluster_info['bsk_name'] = f"{best_bsk.get('bsk_name', 'BSK')} (+{len(cluster_data)-1} others)"
                    cluster_stats.append(cluster_info)
            
            map_df = pd.DataFrame(cluster_stats)
            st.info(f"Clustered {len(recommendations_for_map)} BSKs into {len(map_df)} clusters")

        if not map_df.empty:
            map_df['color'] = map_df['score'].apply(get_color_rgba)
            map_df['score_formatted'] = map_df['score'].apply(lambda x: f"{x:.3f}")
            map_df['color'] = map_df['color'].tolist()

            if use_clustering and 'cluster_size' in map_df.columns:
                map_df['avg_score_formatted'] = map_df['avg_score'].apply(lambda x: f"{x:.3f}")
                tooltip_html = '''
                <b>BSK:</b> {bsk_name}<br/>
                <b>Score:</b> {score_formatted}<br/>
                <b>Cluster Size:</b> {cluster_size}<br/>
                <b>Avg Score:</b> {avg_score_formatted}
                '''
            else:
                tooltip_html = '<b>BSK:</b> {bsk_name}<br/><b>Score:</b> {score_formatted}'
                if 'district_name' in map_df.columns:
                    tooltip_html += '<br/><b>District:</b> {district_name}'
                if 'usage_count' in map_df.columns:
                    map_df['usage_formatted'] = map_df['usage_count'].apply(lambda x: f"{int(x)}")
                    tooltip_html += '<br/><b>Usage Count:</b> {usage_formatted}'
            
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
            
            view_state = pdk.ViewState(
                latitude=map_df['lat'].mean(),
                longitude=map_df['lon'].mean(),
                zoom=7,
                pitch=0,
            )
            
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
    else:
        st.info("Geographic coordinates not available for map display.")
    
    # Analytics
    st.subheader("📈 Recommendation Analytics")
    
    if 'bsk_name' in recommendations_for_map.columns and not recommendations_for_map.empty:
        tab1, tab2, tab3 = st.tabs(["Score Distribution", "Top BSKs", "Geographic Analysis"])
        
        with tab1:
            st.write("**Score Distribution:**")
            score_bins = pd.cut(recommendations_for_map['score'], bins=10)
            score_counts = score_bins.value_counts().sort_index()
            bin_labels = [f"{interval.left:.2f}-{interval.right:.2f}" for interval in score_counts.index]
            score_df = pd.DataFrame({
                'Score Range': bin_labels,
                'Count': score_counts.values
            })
            st.bar_chart(score_df.set_index('Score Range')['Count'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Score", f"{recommendations_for_map['score'].mean():.3f}")
            with col2:
                st.metric("Median Score", f"{recommendations_for_map['score'].median():.3f}")
            with col3:
                st.metric("Std Dev", f"{recommendations_for_map['score'].std():.3f}")
        
        with tab2:
            top_bsks = recommendations_for_map.head(20)
            if 'bsk_name' in top_bsks.columns:
                st.bar_chart(top_bsks.set_index('bsk_name')['score'], height=600)
                st.caption("Top 20 BSKs by recommendation score")
        
        with tab3:
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
        csv = recommendations_for_map.to_csv(index=False)
        st.download_button(
            label="📄 Download as CSV",
            data=csv,
            file_name="bsk_recommendations.csv",
            mime="text/csv"
        )
    
    with export_col2:
        top_50 = recommendations_for_map.head(50)
        csv = top_50.to_csv(index=False)
        st.download_button(
            label="📄 Download Top 50 as CSV",
            data=csv,
            file_name="top_50_bsk_recommendations.csv",
            mime="text/csv"
        )
