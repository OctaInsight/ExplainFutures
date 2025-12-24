"""
PCA Analysis Module
Principal Component Analysis with visualization and interpretation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from core.utils import display_error, display_success
from core.viz.export import quick_export_buttons


def run_pca_analysis(df_wide, feature_cols):
    """
    Main function to run PCA analysis tab
    
    Parameters:
    -----------
    df_wide : pd.DataFrame
        Wide format dataframe
    feature_cols : list
        List of feature column names to use
    """
    
    st.header("🧮 Principal Component Analysis (PCA)")
    st.markdown("*Identify dominant patterns and create orthogonal components*")
    
    # Check for missing values
    if df_wide[feature_cols].isna().any().any():
        st.warning("⚠️ Data contains missing values. They will be handled using forward-fill.")
        df_clean = df_wide[feature_cols].fillna(method='ffill').fillna(method='bfill')
    else:
        df_clean = df_wide[feature_cols]
    
    st.subheader("Step 1: Configure PCA")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Use selected features if available
        if st.session_state.selected_features:
            use_selected = st.checkbox(
                f"Use filtered features ({len(st.session_state.selected_features)})",
                value=True,
                help="Use features from previous filtering steps"
            )
            
            if use_selected:
                features_to_use = st.session_state.selected_features
                st.success(f"✅ Using {len(features_to_use)} filtered features")
            else:
                features_to_use = feature_cols
                st.info(f"Using all {len(features_to_use)} features")
        else:
            features_to_use = feature_cols
            st.info(f"💡 Tip: Apply correlation filtering first for better results")
    
    with col2:
        n_components = st.slider(
            "Number of components",
            min_value=2,
            max_value=min(10, len(features_to_use)),
            value=min(3, len(features_to_use)),
            help="Number of principal components to extract"
        )
    
    # Apply PCA button
    if st.button("🧮 Run PCA Analysis", type="primary", key="run_pca"):
        with st.spinner("Running PCA..."):
            try:
                # Prepare data
                data_for_pca = df_clean[features_to_use].dropna()
                
                if len(data_for_pca) < 2:
                    display_error("Not enough data points for PCA")
                    return
                
                # Standardize
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data_for_pca)
                
                # Run PCA
                pca = PCA(n_components=n_components)
                components = pca.fit_transform(data_scaled)
                
                # Store results
                st.session_state.pca_model = pca
                st.session_state.pca_scaler = scaler
                st.session_state.pca_components = components
                st.session_state.pca_features = features_to_use
                st.session_state.reduction_results['pca'] = {
                    'n_components': n_components,
                    'explained_variance': pca.explained_variance_ratio_,
                    'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
                    'loadings': pca.components_,
                    'features': features_to_use,
                    'timestamp': datetime.now()
                }
                
                st.success(f"✅ PCA completed with {n_components} components")
                st.rerun()
                
            except Exception as e:
                display_error(f"Error in PCA: {str(e)}")
                st.exception(e)
    
    # Display results
    if st.session_state.get('reduction_results', {}).get('pca'):
        display_pca_results()


def display_pca_results():
    """Display PCA analysis results"""
    
    st.markdown("---")
    st.subheader("📊 PCA Results")
    
    results = st.session_state.reduction_results['pca']
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Components", results['n_components'])
    col2.metric(
        "Total Variance Explained", 
        f"{results['cumulative_variance'][-1]*100:.1f}%"
    )
    col3.metric(
        "First Component", 
        f"{results['explained_variance'][0]*100:.1f}%"
    )
    
    # Variance explained plot
    st.markdown("#### Variance Explained by Component")
    fig = create_variance_plot(results)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("💾 Export Variance Plot"):
            quick_export_buttons(fig, "pca_variance", ['png', 'pdf', 'html'])
    
    # Component loadings
    st.markdown("#### Component Loadings")
    st.markdown("*Shows how original features contribute to each component*")
    
    loadings_df = pd.DataFrame(
        results['loadings'].T,
        columns=[f"PC{i+1}" for i in range(results['n_components'])],
        index=results['features']
    )
    
    # Display loadings
    st.dataframe(
        loadings_df.round(3),
        use_container_width=True
    )
    
    # Loadings heatmap
    fig = create_loadings_heatmap(loadings_df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("💾 Export Loadings Heatmap"):
            quick_export_buttons(fig, "pca_loadings", ['png', 'pdf', 'html'])
    
    # Component interpretation
    st.markdown("#### Component Interpretation")
    for i in range(results['n_components']):
        with st.expander(f"PC{i+1} - Top Contributing Features", expanded=(i==0)):
            loadings = loadings_df[f"PC{i+1}"].abs().sort_values(ascending=False)
            
            st.markdown(f"**Variance Explained:** {results['explained_variance'][i]*100:.2f}%")
            st.markdown("**Top 5 Contributors:**")
            
            for feat, loading in loadings.head(5).items():
                original_loading = loadings_df.loc[feat, f"PC{i+1}"]
                direction = "+" if original_loading > 0 else "-"
                st.text(f"  {direction} {feat}: {abs(original_loading):.3f}")
    
    st.markdown("---")
    
    # PCA SCATTER PLOT VISUALIZATION
    display_pca_scatter_plot(results)
    
    st.markdown("---")
    
    # Accept/Reject PCA for modeling
    display_pca_acceptance(results)


def display_pca_scatter_plot(results):
    """Display interactive PCA scatter plot"""
    
    st.markdown("#### PCA Scatter Plot Visualization")
    
    # Create scatter plot interface
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pc_x = st.selectbox(
            "X-axis",
            [f"PC{i+1}" for i in range(results['n_components'])],
            index=0,
            key="pca_scatter_x"
        )
    
    with col2:
        pc_y = st.selectbox(
            "Y-axis",
            [f"PC{i+1}" for i in range(results['n_components'])],
            index=min(1, results['n_components']-1),
            key="pca_scatter_y"
        )
    
    with col3:
        point_size = st.slider(
            "Point size",
            min_value=3,
            max_value=15,
            value=8,
            key="pca_point_size"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        point_color = st.color_picker(
            "Point color",
            value="#1f77b4",
            key="pca_point_color"
        )
    
    with col2:
        show_time_gradient = st.checkbox(
            "Color by time sequence",
            value=True,
            help="Color points by their position in time series",
            key="pca_time_gradient"
        )
    
    # Create scatter plot
    fig_scatter = create_pca_scatter_plot(
        st.session_state.pca_components,
        pc_x,
        pc_y,
        results,
        point_size,
        point_color,
        show_time_gradient
    )
    
    if fig_scatter:
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        with st.expander("💾 Export PCA Scatter Plot"):
            quick_export_buttons(fig_scatter, f"pca_scatter_{pc_x}_vs_{pc_y}", ['png', 'pdf', 'html'])


def display_pca_acceptance(results):
    """Display PCA acceptance interface"""
    
    st.subheader("✅ Accept PCA for Modeling?")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Decision Guide:**
        
        ✅ **Accept PCA if:**
        - Cumulative variance ≥ 85% (excellent)
        - Cumulative variance ≥ 70% (good)
        - Components have clear interpretation
        - Building multivariate models
        
        ❌ **Reject PCA if:**
        - Cumulative variance < 70% (too many dimensions lost)
        - Components are hard to interpret
        - Working with few variables already
        - Need to preserve original features
        """)
    
    with col2:
        current_status = st.session_state.get('pca_accepted', False)
        
        if current_status:
            st.success("✅ PCA Accepted for Modeling")
            if st.button("❌ Reject PCA", key="reject_pca"):
                st.session_state.pca_accepted = False
                st.success("PCA rejected - will use original features")
                st.rerun()
        else:
            st.info("🤔 PCA Not Yet Accepted")
            if st.button("✅ Accept PCA for Modeling", key="accept_pca", type="primary"):
                st.session_state.pca_accepted = True
                st.success("PCA accepted - components will be used in modeling")
                st.rerun()
    
    # Show what will be used
    if st.session_state.pca_accepted:
        st.success(f"""
        📊 **Modeling will use:** {results['n_components']} PCA components (PC1 to PC{results['n_components']})
        
        **Benefits:**
        - Orthogonal (uncorrelated) features
        - Reduced dimensionality
        - Captures {results['cumulative_variance'][-1]*100:.1f}% of variance
        """)
    else:
        st.info(f"""
        📊 **Modeling will use:** Original {len(results['features'])} features
        
        **Note:** PCA analysis available for reference, but won't be used in models
        """)


def create_variance_plot(results):
    """Create variance explained plot"""
    try:
        fig = go.Figure()
        
        # Individual variance
        fig.add_trace(go.Bar(
            x=[f"PC{i+1}" for i in range(results['n_components'])],
            y=results['explained_variance'] * 100,
            name='Individual',
            marker_color='steelblue',
            text=[f"{v*100:.1f}%" for v in results['explained_variance']],
            textposition='outside'
        ))
        
        # Cumulative variance
        fig.add_trace(go.Scatter(
            x=[f"PC{i+1}" for i in range(results['n_components'])],
            y=results['cumulative_variance'] * 100,
            name='Cumulative',
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=10),
            text=[f"{v*100:.1f}%" for v in results['cumulative_variance']],
            textposition='top center'
        ))
        
        fig.update_layout(
            title="Variance Explained by Principal Components",
            xaxis_title="Principal Component",
            yaxis_title="Variance Explained (%)",
            height=450,
            template='plotly_white',
            hovermode='x unified',
            showlegend=True,
            legend=dict(x=0.7, y=0.95)
        )
        
        # Add threshold lines
        fig.add_hline(y=85, line_dash="dash", line_color="green", 
                     annotation_text="85% (Excellent)", annotation_position="right")
        fig.add_hline(y=70, line_dash="dash", line_color="orange",
                     annotation_text="70% (Good)", annotation_position="right")
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating variance plot: {str(e)}")
        return None


def create_loadings_heatmap(loadings_df):
    """Create component loadings heatmap"""
    try:
        fig = go.Figure(data=go.Heatmap(
            z=loadings_df.T.values,
            x=loadings_df.index,
            y=loadings_df.columns,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(loadings_df.T.values, 2),
            texttemplate='%{text}',
            textfont={"size": 9},
            colorbar=dict(title="Loading")
        ))
        
        fig.update_layout(
            title="PCA Component Loadings",
            xaxis_title="Original Features",
            yaxis_title="Principal Components",
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating loadings heatmap: {str(e)}")
        return None


def create_pca_scatter_plot(components, pc_x, pc_y, results, point_size, point_color, show_time_gradient):
    """Create PCA scatter plot"""
    try:
        # Get indices
        x_idx = int(pc_x.replace("PC", "")) - 1
        y_idx = int(pc_y.replace("PC", "")) - 1
        
        # Extract data
        x_data = components[:, x_idx]
        y_data = components[:, y_idx]
        
        # Create figure
        fig = go.Figure()
        
        if show_time_gradient:
            # Color by time sequence
            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=np.arange(len(x_data)),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Time<br>Sequence")
                ),
                text=[f"Point {i+1}" for i in range(len(x_data))],
                hovertemplate=f'<b>%{{text}}</b><br>{pc_x}: %{{x:.3f}}<br>{pc_y}: %{{y:.3f}}<extra></extra>'
            ))
        else:
            # Single color
            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=point_color
                ),
                text=[f"Point {i+1}" for i in range(len(x_data))],
                hovertemplate=f'<b>%{{text}}</b><br>{pc_x}: %{{x:.3f}}<br>{pc_y}: %{{y:.3f}}<extra></extra>'
            ))
        
        # Layout
        var_x = results['explained_variance'][x_idx] * 100
        var_y = results['explained_variance'][y_idx] * 100
        
        fig.update_layout(
            title=f"PCA Scatter Plot: {pc_x} vs {pc_y}",
            xaxis_title=f"{pc_x} ({var_x:.1f}% variance)",
            yaxis_title=f"{pc_y} ({var_y:.1f}% variance)",
            height=600,
            template='plotly_white',
            hovermode='closest'
        )
        
        # Add zero lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating PCA scatter plot: {str(e)}")
        return None
