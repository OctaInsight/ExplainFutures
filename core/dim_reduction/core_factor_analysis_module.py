"""
Factor Analysis Module
Discover meaningful latent factors driving your system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

from core.utils import display_error
from core.viz.export import quick_export_buttons


def run_factor_analysis(df_wide, feature_cols):
    """
    Main function to run Factor Analysis tab
    
    Parameters:
    -----------
    df_wide : pd.DataFrame
        Wide format dataframe
    feature_cols : list
        List of feature column names to use
    """
    
    st.header("🎯 Factor Analysis")
    st.markdown("*Discover meaningful latent factors driving your system*")
    
    st.info("""
    💡 **Factor Analysis vs PCA:**
    - **PCA:** Finds directions of maximum variance (mathematical)
    - **Factor Analysis:** Discovers underlying "causes" or "factors" (conceptual)
    - **Best for:** Understanding what drives your system
    - **Example:** Economic data might reveal factors like "Growth", "Stability", "Trade"
    """)
    
    # Check for missing values
    if df_wide[feature_cols].isna().any().any():
        st.warning("⚠️ Data contains missing values. They will be handled using forward-fill.")
        df_clean = df_wide[feature_cols].fillna(method='ffill').fillna(method='bfill')
    else:
        df_clean = df_wide[feature_cols]
    
    st.subheader("Step 1: Configure Factor Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Use selected features if available
        if st.session_state.selected_features:
            use_selected = st.checkbox(
                f"Use filtered features ({len(st.session_state.selected_features)})",
                value=True,
                key="fa_use_selected",
                help="Use features from previous filtering steps"
            )
            features_to_use = st.session_state.selected_features if use_selected else feature_cols
        else:
            features_to_use = feature_cols
        
        st.info(f"Using {len(features_to_use)} features")
    
    with col2:
        n_factors = st.slider(
            "Number of factors",
            min_value=2,
            max_value=min(8, len(features_to_use)),
            value=min(3, len(features_to_use)),
            help="Number of latent factors to extract"
        )
        
        rotation = st.selectbox(
            "Rotation method",
            ["varimax", "None"],
            help="Varimax makes factors more interpretable"
        )
    
    # Apply Factor Analysis
    if st.button("🎯 Run Factor Analysis", type="primary", key="run_fa"):
        with st.spinner("Running Factor Analysis..."):
            try:
                # Prepare data
                data_for_fa = df_clean[features_to_use].dropna()
                
                if len(data_for_fa) < 2:
                    display_error("Not enough data points for Factor Analysis")
                    return
                
                # Standardize
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data_for_fa)
                
                # Run Factor Analysis
                fa = FactorAnalysis(
                    n_components=n_factors, 
                    rotation=rotation if rotation != "None" else None, 
                    random_state=42
                )
                factors = fa.fit_transform(data_scaled)
                
                # Get loadings
                loadings = fa.components_.T
                
                # Calculate communalities (variance explained for each variable)
                communalities = np.sum(loadings**2, axis=1)
                
                # Store results
                st.session_state.reduction_results['factor_analysis'] = {
                    'n_factors': n_factors,
                    'rotation': rotation,
                    'loadings': loadings,
                    'factors': factors,
                    'features': features_to_use,
                    'communalities': communalities,
                    'noise_variance': fa.noise_variance_,
                    'timestamp': datetime.now()
                }
                
                st.success(f"✅ Factor Analysis completed with {n_factors} factors")
                st.rerun()
                
            except Exception as e:
                display_error(f"Error in Factor Analysis: {str(e)}")
                st.exception(e)
    
    # Display results
    if st.session_state.get('reduction_results', {}).get('factor_analysis'):
        display_factor_analysis_results()


def display_factor_analysis_results():
    """Display Factor Analysis results"""
    
    st.markdown("---")
    st.subheader("📊 Factor Analysis Results")
    
    results = st.session_state.reduction_results['factor_analysis']
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Factors Extracted", results['n_factors'])
    col2.metric("Rotation Method", results['rotation'])
    avg_communality = np.mean(results['communalities'])
    col3.metric("Avg Communality", f"{avg_communality:.3f}")
    
    # Factor loadings table
    st.markdown("#### Factor Loadings")
    st.markdown("*Shows how each variable loads onto each factor*")
    
    loadings_df = pd.DataFrame(
        results['loadings'],
        columns=[f"Factor {i+1}" for i in range(results['n_factors'])],
        index=results['features']
    )
    
    # Add communalities
    loadings_df['Communality'] = results['communalities']
    
    st.dataframe(
        loadings_df.round(3),
        use_container_width=True
    )
    
    # Factor loadings heatmap
    st.markdown("#### Factor Loadings Heatmap")
    fig_loadings = create_factor_loadings_heatmap(loadings_df.iloc[:, :-1])  # Exclude communality column
    if fig_loadings:
        st.plotly_chart(fig_loadings, use_container_width=True)
        
        with st.expander("💾 Export Factor Loadings Heatmap"):
            quick_export_buttons(fig_loadings, "factor_loadings", ['png', 'pdf', 'html'])
    
    # Factor interpretation
    st.markdown("#### Factor Interpretation")
    st.markdown("*Name your factors based on their top loading variables*")
    
    for i in range(results['n_factors']):
        with st.expander(f"Factor {i+1} - Top Loading Variables", expanded=(i==0)):
            factor_loadings = loadings_df.iloc[:, i].abs().sort_values(ascending=False)
            
            st.markdown("**Top 5 Loading Variables:**")
            for feat, loading in factor_loadings.head(5).items():
                original_loading = loadings_df.loc[feat, f"Factor {i+1}"]
                direction = "+" if original_loading > 0 else "-"
                st.text(f"  {direction} {feat}: {abs(original_loading):.3f}")
            
            # Suggested name
            top_var = factor_loadings.index[0]
            st.info(f"💡 **Suggested name:** Based on '{top_var}' and related variables")
    
    # Factor scatter plot
    display_factor_scatter_plot(results)


def display_factor_scatter_plot(results):
    """Display Factor scatter plot"""
    
    st.markdown("#### Factor Scatter Plot")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        factor_x = st.selectbox(
            "X-axis",
            [f"Factor {i+1}" for i in range(results['n_factors'])],
            index=0,
            key="fa_scatter_x"
        )
    
    with col2:
        factor_y = st.selectbox(
            "Y-axis",
            [f"Factor {i+1}" for i in range(results['n_factors'])],
            index=min(1, results['n_factors']-1),
            key="fa_scatter_y"
        )
    
    with col3:
        fa_point_size = st.slider(
            "Point size",
            min_value=3,
            max_value=15,
            value=8,
            key="fa_point_size"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        fa_point_color = st.color_picker(
            "Point color",
            value="#FF6B6B",
            key="fa_point_color"
        )
    
    with col2:
        fa_time_gradient = st.checkbox(
            "Color by time sequence",
            value=True,
            key="fa_time_gradient"
        )
    
    # Create scatter
    fig_scatter = create_factor_scatter_plot(
        results['factors'],
        factor_x,
        factor_y,
        fa_point_size,
        fa_point_color,
        fa_time_gradient
    )
    
    if fig_scatter:
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        with st.expander("💾 Export Factor Scatter Plot"):
            quick_export_buttons(fig_scatter, f"factor_scatter_{factor_x}_vs_{factor_y}", ['png', 'pdf', 'html'])


def create_factor_loadings_heatmap(loadings_df):
    """Create factor loadings heatmap"""
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
            title="Factor Loadings Heatmap",
            xaxis_title="Original Variables",
            yaxis_title="Factors",
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating factor loadings heatmap: {str(e)}")
        return None


def create_factor_scatter_plot(factors, factor_x, factor_y, point_size, point_color, show_time_gradient):
    """Create factor scatter plot"""
    try:
        # Get indices
        x_idx = int(factor_x.split()[-1]) - 1
        y_idx = int(factor_y.split()[-1]) - 1
        
        # Extract data
        x_data = factors[:, x_idx]
        y_data = factors[:, y_idx]
        
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
                    colorscale='Plasma',
                    showscale=True,
                    colorbar=dict(title="Time<br>Sequence")
                ),
                text=[f"Point {i+1}" for i in range(len(x_data))],
                hovertemplate=f'<b>%{{text}}</b><br>{factor_x}: %{{x:.3f}}<br>{factor_y}: %{{y:.3f}}<extra></extra>'
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
                hovertemplate=f'<b>%{{text}}</b><br>{factor_x}: %{{x:.3f}}<br>{factor_y}: %{{y:.3f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"Factor Scatter Plot: {factor_x} vs {factor_y}",
            xaxis_title=factor_x,
            yaxis_title=factor_y,
            height=600,
            template='plotly_white',
            hovermode='closest'
        )
        
        # Add zero lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating factor scatter plot: {str(e)}")
        return None
