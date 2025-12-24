"""
Hierarchical Clustering Module
Visually group similar variables using dendrograms
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram, linkage
from scipy.spatial.distance import squareform

from core.utils import display_error
from core.viz.export import quick_export_buttons


def run_hierarchical_clustering(df_wide, feature_cols):
    """
    Main function to run Hierarchical Clustering tab
    
    Parameters:
    -----------
    df_wide : pd.DataFrame
        Wide format dataframe
    feature_cols : list
        List of feature column names to use
    """
    
    st.header("🌳 Hierarchical Clustering")
    st.markdown("*Visually group similar variables using dendrogram*")
    
    st.info("""
    💡 **Hierarchical Clustering:**
    - **Purpose:** Group variables based on similarity
    - **Output:** Tree diagram (dendrogram) showing relationships
    - **Best for:** Identifying natural variable groupings
    - **Benefit:** Very intuitive and visual - no math required!
    """)
    
    # Check for missing values
    if df_wide[feature_cols].isna().any().any():
        st.warning("⚠️ Data contains missing values. They will be handled using forward-fill.")
        df_clean = df_wide[feature_cols].fillna(method='ffill').fillna(method='bfill')
    else:
        df_clean = df_wide[feature_cols]
    
    st.subheader("Step 1: Configure Clustering")
    
    col1, col2 = st.columns(2)
    
    with col1:
        linkage_method = st.selectbox(
            "Linkage method",
            ["ward", "complete", "average", "single"],
            help="Ward: minimizes variance within clusters (recommended)"
        )
    
    with col2:
        distance_metric = st.selectbox(
            "Distance metric",
            ["euclidean", "correlation", "cityblock"],
            index=1,
            help="Correlation: based on variable relationships (recommended)"
        )
        
        # Display user-friendly name
        metric_display = {
            "euclidean": "Euclidean (straight-line distance)",
            "correlation": "Correlation (similarity-based)",
            "cityblock": "Manhattan/Cityblock (grid distance)"
        }
        st.caption(metric_display.get(distance_metric, distance_metric))
    
    # Number of clusters
    n_clusters = st.slider(
        "Number of clusters to form",
        min_value=2,
        max_value=min(8, len(feature_cols)),
        value=min(3, len(feature_cols)),
        help="Cut dendrogram to form this many clusters"
    )
    
    # Apply Clustering
    if st.button("🌳 Run Hierarchical Clustering", type="primary", key="run_clustering"):
        with st.spinner("Performing hierarchical clustering..."):
            try:
                # Prepare data
                data_for_clustering = df_clean[feature_cols].dropna()
                
                # Transpose: cluster variables (not time points)
                data_transposed = data_for_clustering.T
                
                # Standardize
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data_transposed)
                
                # Compute linkage
                if distance_metric == "correlation":
                    # Use correlation distance
                    corr_matrix = np.corrcoef(data_scaled)
                    distance_matrix = 1 - np.abs(corr_matrix)
                    
                    # Ensure perfect symmetry (fix numerical precision issues)
                    distance_matrix = (distance_matrix + distance_matrix.T) / 2
                    
                    # Ensure diagonal is exactly zero
                    np.fill_diagonal(distance_matrix, 0)
                    
                    # Ensure all values are non-negative (can happen with precision errors)
                    distance_matrix = np.maximum(distance_matrix, 0)
                    
                    # Convert to condensed form
                    try:
                        condensed_dist = squareform(distance_matrix, checks=False)
                        linkage_matrix = linkage(condensed_dist, method=linkage_method)
                    except Exception as e:
                        st.error(f"Error converting distance matrix: {str(e)}")
                        # Fallback: use pdist directly on scaled data
                        from scipy.spatial.distance import pdist
                        condensed_dist = pdist(data_scaled.T, metric='correlation')
                        linkage_matrix = linkage(condensed_dist, method=linkage_method)
                else:
                    # Use regular distance
                    linkage_matrix = linkage(data_scaled, method=linkage_method, metric=distance_metric)
                
                # Form clusters
                clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
                cluster_labels = clustering.fit_predict(data_scaled)
                
                # Group features by cluster
                clusters = {}
                for i in range(n_clusters):
                    clusters[f"Cluster {i+1}"] = [feature_cols[j] for j, label in enumerate(cluster_labels) if label == i]
                
                # Store results
                st.session_state.reduction_results['clustering'] = {
                    'linkage_matrix': linkage_matrix,
                    'linkage_method': linkage_method,
                    'distance_metric': distance_metric,
                    'n_clusters': n_clusters,
                    'cluster_labels': cluster_labels,
                    'clusters': clusters,
                    'features': feature_cols,
                    'timestamp': datetime.now()
                }
                
                st.success(f"✅ Hierarchical clustering completed with {n_clusters} clusters")
                st.rerun()
                
            except Exception as e:
                display_error(f"Error in hierarchical clustering: {str(e)}")
                st.exception(e)
    
    # Display results
    if st.session_state.get('reduction_results', {}).get('clustering'):
        display_clustering_results(df_clean)


def display_clustering_results(df_clean):
    """Display clustering analysis results"""
    
    st.markdown("---")
    st.subheader("📊 Clustering Results")
    
    results = st.session_state.reduction_results['clustering']
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Clusters Formed", results['n_clusters'])
    col2.metric("Linkage Method", results['linkage_method'])
    col3.metric("Distance Metric", results['distance_metric'])
    
    # Dendrogram
    st.markdown("#### Dendrogram (Tree Diagram)")
    st.markdown("*Shows hierarchical relationships between variables*")
    
    fig_dendrogram = create_dendrogram_plot(
        results['linkage_matrix'],
        results['features'],
        results['n_clusters']
    )
    
    if fig_dendrogram:
        st.plotly_chart(fig_dendrogram, use_container_width=True)
        
        with st.expander("💾 Export Dendrogram"):
            quick_export_buttons(fig_dendrogram, "hierarchical_dendrogram", ['png', 'pdf', 'html'])
    
    # Cluster composition
    st.markdown("#### Cluster Composition")
    st.markdown("*Variables grouped in each cluster*")
    
    for cluster_name, variables in results['clusters'].items():
        with st.expander(f"{cluster_name} ({len(variables)} variables)", expanded=True):
            for var in variables:
                st.text(f"  • {var}")
            
            if len(variables) > 1:
                st.info(f"💡 **Interpretation:** These {len(variables)} variables show similar patterns")
    
    # Cluster heatmap
    st.markdown("#### Cluster Visualization Heatmap")
    
    # Reorder features by cluster
    ordered_features = []
    for cluster in results['clusters'].values():
        ordered_features.extend(cluster)
    
    fig_heatmap = create_cluster_heatmap(df_clean, ordered_features, results['cluster_labels'], results['features'])
    if fig_heatmap:
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with st.expander("💾 Export Cluster Heatmap"):
            quick_export_buttons(fig_heatmap, "cluster_heatmap", ['png', 'pdf', 'html'])


def create_dendrogram_plot(linkage_matrix, labels, n_clusters):
    """Create interactive dendrogram using Plotly"""
    try:
        # Create dendrogram
        dend = scipy_dendrogram(linkage_matrix, labels=labels, no_plot=True)
        
        # Extract coordinates
        icoord = np.array(dend['icoord'])
        dcoord = np.array(dend['dcoord'])
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add dendrogram lines
        for i in range(len(icoord)):
            fig.add_trace(go.Scatter(
                x=icoord[i],
                y=dcoord[i],
                mode='lines',
                line=dict(color='steelblue', width=2),
                hoverinfo='skip',
                showlegend=False
            ))
        
        # Add labels
        fig.update_layout(
            title=f"Hierarchical Clustering Dendrogram ({n_clusters} clusters)",
            xaxis=dict(
                title="Variables",
                tickmode='array',
                tickvals=list(range(5, len(labels)*10+5, 10)),
                ticktext=dend['ivl'],
                tickangle=-45
            ),
            yaxis=dict(title="Distance"),
            height=600,
            template='plotly_white',
            hovermode='closest'
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating dendrogram: {str(e)}")
        return None


def create_cluster_heatmap(df, ordered_features, cluster_labels, original_features):
    """Create heatmap showing clusters"""
    try:
        # Get correlation matrix for ordered features
        corr_matrix = df[ordered_features].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=ordered_features,
            y=ordered_features,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Variable Correlation Grouped by Cluster",
            xaxis_title="Variables (ordered by cluster)",
            yaxis_title="Variables (ordered by cluster)",
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating cluster heatmap: {str(e)}")
        return None
