"""
Independent Component Analysis Module
Find statistically independent sources/drivers in your system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

from core.utils import display_error
from core.viz.export import quick_export_buttons


def run_ica_analysis(df_wide, feature_cols):
    """
    Main function to run ICA analysis tab
    
    Parameters:
    -----------
    df_wide : pd.DataFrame
        Wide format dataframe
    feature_cols : list
        List of feature column names to use
    """
    
    st.header("🔬 Independent Component Analysis (ICA)")
    st.markdown("*Discover independent sources/drivers in your system*")
    
    st.info("""
    💡 **ICA vs PCA:**
    - **PCA:** Finds orthogonal (uncorrelated) components - mathematical independence
    - **ICA:** Finds statistically independent sources - true independence
    - **Best for:** Systems with multiple independent driving forces
    - **Example:** Economic data might reveal independent drivers like "Policy", "Market Sentiment", "External Trade"
    """)
    
    # Check for missing values
    if df_wide[feature_cols].isna().any().any():
        st.warning("⚠️ Data contains missing values. They will be handled using forward-fill.")
        df_clean = df_wide[feature_cols].fillna(method='ffill').fillna(method='bfill')
    else:
        df_clean = df_wide[feature_cols]
    
    st.subheader("Step 1: Configure ICA")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Use selected features if available
        if st.session_state.selected_features:
            use_selected = st.checkbox(
                f"Use filtered features ({len(st.session_state.selected_features)})",
                value=True,
                key="ica_use_selected",
                help="Use features from previous filtering steps"
            )
            features_to_use = st.session_state.selected_features if use_selected else feature_cols
        else:
            features_to_use = feature_cols
        
        st.info(f"Using {len(features_to_use)} features")
    
    with col2:
        n_components = st.slider(
            "Number of components",
            min_value=2,
            max_value=min(8, len(features_to_use)),
            value=min(3, len(features_to_use)),
            help="Number of independent components to extract"
        )
    
    # Apply ICA
    if st.button("🔬 Run ICA", type="primary", key="run_ica"):
        with st.spinner("Running Independent Component Analysis..."):
            try:
                # Prepare data
                data_for_ica = df_clean[features_to_use].dropna()
                
                if len(data_for_ica) < 2:
                    display_error("Not enough data points for ICA")
                    return
                
                # Standardize
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data_for_ica)
                
                # Run ICA
                ica = FastICA(n_components=n_components, random_state=42, max_iter=500)
                components = ica.fit_transform(data_scaled)
                
                # Get mixing matrix (how sources mix to create observed data)
                mixing_matrix = ica.mixing_
                
                # Store results
                st.session_state.reduction_results['ica'] = {
                    'n_components': n_components,
                    'components': components,
                    'mixing_matrix': mixing_matrix,
                    'features': features_to_use,
                    'timestamp': datetime.now()
                }
                
                st.success(f"✅ ICA completed with {n_components} independent components")
                st.rerun()
                
            except Exception as e:
                display_error(f"Error in ICA: {str(e)}")
                st.exception(e)
    
    # Display results
    if st.session_state.get('reduction_results', {}).get('ica'):
        display_ica_results()


def display_ica_results():
    """Display ICA analysis results"""
    
    st.markdown("---")
    st.subheader("📊 ICA Results")
    
    results = st.session_state.reduction_results['ica']
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Independent Components", results['n_components'])
    col2.metric("Original Features", len(results['features']))
    col3.metric("Data Points", len(results['components']))
    
    # Mixing matrix (how independent sources combine to create observed variables)
    st.markdown("#### Mixing Matrix")
    st.markdown("*Shows how independent components mix to create observed variables*")
    
    mixing_df = pd.DataFrame(
        results['mixing_matrix'],
        columns=[f"IC{i+1}" for i in range(results['n_components'])],
        index=results['features']
    )
    
    st.dataframe(
        mixing_df.round(3),
        use_container_width=True
    )
    
    # Mixing matrix heatmap
    st.markdown("#### Mixing Matrix Heatmap")
    fig_mixing = create_ica_mixing_heatmap(mixing_df)
    if fig_mixing:
        st.plotly_chart(fig_mixing, use_container_width=True)
        
        with st.expander("💾 Export Mixing Matrix"):
            quick_export_buttons(fig_mixing, "ica_mixing_matrix", ['png', 'pdf', 'html'])
    
    # Component interpretation
    st.markdown("#### Independent Component Interpretation")
    st.markdown("*Name components based on their strongest influences*")
    
    for i in range(results['n_components']):
        with st.expander(f"IC{i+1} - Top Influences", expanded=(i==0)):
            ic_mixing = mixing_df.iloc[:, i].abs().sort_values(ascending=False)
            
            st.markdown("**Top 5 Influenced Variables:**")
            for feat, weight in ic_mixing.head(5).items():
                original_weight = mixing_df.loc[feat, f"IC{i+1}"]
                direction = "+" if original_weight > 0 else "-"
                st.text(f"  {direction} {feat}: {abs(original_weight):.3f}")
            
            st.info(f"💡 **Interpretation:** This independent source primarily affects {ic_mixing.index[0]}")
    
    # ICA component time series
    st.markdown("#### Independent Components Over Time")
    fig_timeseries = create_ica_timeseries_plot(results['components'], results['n_components'])
    if fig_timeseries:
        st.plotly_chart(fig_timeseries, use_container_width=True)
        
        with st.expander("💾 Export Time Series"):
            quick_export_buttons(fig_timeseries, "ica_timeseries", ['png', 'pdf', 'html'])
    
    # ICA scatter plot
    display_ica_scatter_plot(results)


def display_ica_scatter_plot(results):
    """Display ICA scatter plot"""
    
    st.markdown("#### ICA Scatter Plot")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ic_x = st.selectbox(
            "X-axis",
            [f"IC{i+1}" for i in range(results['n_components'])],
            index=0,
            key="ica_scatter_x"
        )
    
    with col2:
        ic_y = st.selectbox(
            "Y-axis",
            [f"IC{i+1}" for i in range(results['n_components'])],
            index=min(1, results['n_components']-1),
            key="ica_scatter_y"
        )
    
    with col3:
        ica_point_size = st.slider(
            "Point size",
            min_value=3,
            max_value=15,
            value=8,
            key="ica_point_size"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        ica_point_color = st.color_picker(
            "Point color",
            value="#4ECDC4",
            key="ica_point_color"
        )
    
    with col2:
        ica_time_gradient = st.checkbox(
            "Color by time sequence",
            value=True,
            key="ica_time_gradient"
        )
    
    # Create scatter
    fig_scatter = create_ica_scatter_plot(
        results['components'],
        ic_x,
        ic_y,
        ica_point_size,
        ica_point_color,
        ica_time_gradient
    )
    
    if fig_scatter:
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        with st.expander("💾 Export ICA Scatter Plot"):
            quick_export_buttons(fig_scatter, f"ica_scatter_{ic_x}_vs_{ic_y}", ['png', 'pdf', 'html'])


def create_ica_mixing_heatmap(mixing_df):
    """Create ICA mixing matrix heatmap"""
    try:
        fig = go.Figure(data=go.Heatmap(
            z=mixing_df.T.values,
            x=mixing_df.index,
            y=mixing_df.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(mixing_df.T.values, 2),
            texttemplate='%{text}',
            textfont={"size": 9},
            colorbar=dict(title="Mixing<br>Weight")
        ))
        
        fig.update_layout(
            title="ICA Mixing Matrix",
            xaxis_title="Original Variables",
            yaxis_title="Independent Components",
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating ICA mixing heatmap: {str(e)}")
        return None


def create_ica_timeseries_plot(components, n_components):
    """Create time series plot of independent components"""
    try:
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set2
        
        for i in range(n_components):
            fig.add_trace(go.Scatter(
                y=components[:, i],
                mode='lines',
                name=f'IC{i+1}',
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title="Independent Components Over Time",
            xaxis_title="Time Point",
            yaxis_title="Component Value",
            height=500,
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating ICA time series: {str(e)}")
        return None


def create_ica_scatter_plot(components, ic_x, ic_y, point_size, point_color, show_time_gradient):
    """Create ICA scatter plot"""
    try:
        # Get indices
        x_idx = int(ic_x.replace("IC", "")) - 1
        y_idx = int(ic_y.replace("IC", "")) - 1
        
        # Extract data
        x_data = components[:, x_idx]
        y_data = components[:, y_idx]
        
        # Create figure
        fig = go.Figure()
        
        if show_time_gradient:
            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=np.arange(len(x_data)),
                    colorscale='Turbo',
                    showscale=True,
                    colorbar=dict(title="Time<br>Sequence")
                ),
                text=[f"Point {i+1}" for i in range(len(x_data))],
                hovertemplate=f'<b>%{{text}}</b><br>{ic_x}: %{{x:.3f}}<br>{ic_y}: %{{y:.3f}}<extra></extra>'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=point_color
                ),
                text=[f"Point {i+1}" for i in range(len(x_data))],
                hovertemplate=f'<b>%{{text}}</b><br>{ic_x}: %{{x:.3f}}<br>{ic_y}: %{{y:.3f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"ICA Scatter Plot: {ic_x} vs {ic_y}",
            xaxis_title=ic_x,
            yaxis_title=ic_y,
            height=600,
            template='plotly_white',
            hovermode='closest'
        )
        
        # Add zero lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating ICA scatter plot: {str(e)}")
        return None
