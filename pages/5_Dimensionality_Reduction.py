"""
Page 5: Dimensionality Reduction
Reduce complexity, stabilize models, and improve explainability
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet, LassoCV, ElasticNetCV

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import get_config, initialize_session_state
from core.utils import display_error, display_success, display_warning, display_info
from core.shared_sidebar import render_app_sidebar
from core.viz.export import quick_export_buttons

# Initialize
initialize_session_state()
config = get_config()

# Page configuration
st.set_page_config(page_title="Dimensionality Reduction", page_icon="🔬", layout="wide")

# Render shared sidebar
render_app_sidebar()

st.title("🔬 Dimensionality Reduction")
st.markdown("*Simplify complex systems, stabilize models, and improve explainability*")
st.markdown("---")


def initialize_reduction_state():
    """Initialize session state for dimensionality reduction"""
    if "reduction_results" not in st.session_state:
        st.session_state.reduction_results = {}
    if "selected_features" not in st.session_state:
        st.session_state.selected_features = []
    if "pca_model" not in st.session_state:
        st.session_state.pca_model = None
    if "pca_accepted" not in st.session_state:
        st.session_state.pca_accepted = False
    if "reduced_data" not in st.session_state:
        st.session_state.reduced_data = None


def get_wide_format_data():
    """Convert long format to wide format for dimensionality reduction"""
    # Use cleaned data if available, otherwise original
    if (st.session_state.get('preprocessing_applied', False) and 
        st.session_state.get('df_clean') is not None):
        df_long = st.session_state.df_clean
        data_source = "cleaned"
    else:
        df_long = st.session_state.df_long
        data_source = "original"
    
    # Determine time column
    time_col = 'timestamp' if 'timestamp' in df_long.columns else 'time'
    
    # Pivot to wide format
    df_wide = df_long.pivot(
        index=time_col,
        columns='variable',
        values='value'
    ).reset_index()
    
    return df_wide, data_source


def main():
    """Main page function"""
    
    # Initialize state
    initialize_reduction_state()
    
    # Check if data is loaded
    if not st.session_state.data_loaded or st.session_state.df_long is None:
        st.warning("⚠️ No data loaded yet!")
        st.info("👈 Please go to **Upload & Data Diagnostics** to load your data first")
        
        if st.button("📁 Go to Upload Page"):
            st.switch_page("pages/1_Upload_and_Data_Diagnostics.py")
        return
    
    # Get data in wide format
    try:
        df_wide, data_source = get_wide_format_data()
        time_col = 'timestamp' if 'timestamp' in df_wide.columns else 'time'
        all_feature_cols = [col for col in df_wide.columns if col != time_col]
        
    except Exception as e:
        display_error(f"Error preparing data: {str(e)}")
        st.info("💡 Make sure your data has proper timestamp and variable columns")
        return
    
    # Display overview
    st.success(f"✅ Data loaded from {data_source} dataset")
    
    # === STEP 1: VARIABLE SELECTION (NEW - FIX #1) ===
    st.subheader("📋 Step 1: Select Variables for Analysis")
    
    # Separate original and cleaned/transformed variables
    original_vars = sorted(st.session_state.df_long['variable'].unique().tolist())
    cleaned_vars = [v for v in all_feature_cols if v not in original_vars]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if cleaned_vars:
            st.info(f"""
            💡 **Available variables:**
            - {len(original_vars)} original variables
            - {len(cleaned_vars)} cleaned/transformed variables
            
            **Tip:** Select either original OR cleaned versions of the same variable, not both.
            """)
            
            # Show variable types
            with st.expander("📊 Show Variable Details"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Original Variables:**")
                    for v in original_vars[:10]:
                        st.caption(f"• {v}")
                    if len(original_vars) > 10:
                        st.caption(f"... and {len(original_vars)-10} more")
                
                with col_b:
                    st.markdown("**Cleaned/Transformed:**")
                    for v in cleaned_vars[:10]:
                        st.caption(f"• {v}")
                    if len(cleaned_vars) > 10:
                        st.caption(f"... and {len(cleaned_vars)-10} more")
        
        # Multi-select for variable selection
        selected_features = st.multiselect(
            "Select variables to use in dimensionality reduction",
            all_feature_cols,
            default=all_feature_cols if len(all_feature_cols) <= 10 else all_feature_cols[:10],
            help="Choose which variables to include in the analysis",
            key="dim_reduction_variables"
        )
    
    with col2:
        st.metric("Total Available", len(all_feature_cols))
        st.metric("Selected", len(selected_features))
        
        if len(selected_features) < len(all_feature_cols):
            excluded = len(all_feature_cols) - len(selected_features)
            st.metric("Excluded", excluded)
    
    # Check minimum requirements
    if len(selected_features) < 2:
        st.error("❌ Please select at least 2 variables for dimensionality reduction")
        return
    
    # Use only selected features
    feature_cols = selected_features
    
    st.markdown("---")
    
    # Display current selection info
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Selected Variables", len(feature_cols))
    col2.metric("Time Points", len(df_wide))
    col3.metric("Missing Values", df_wide[feature_cols].isna().sum().sum())
    col4.metric("Data Source", data_source.capitalize())
    
    st.markdown("---")
    
    # Information panel
    with st.expander("ℹ️ Why Dimensionality Reduction?", expanded=False):
        st.markdown("""
        ### Purpose in ExplainFutures
        
        **Dimensionality reduction helps you:**
        1. 📊 **Stabilize multivariate models** - Reduce multicollinearity and overfitting
        2. 🎯 **Improve explainability** - Identify dominant patterns and components
        3. 🔮 **Support scenario analysis** - Map complex narratives to structured spaces
        4. ⚠️ **Diagnose model quality** - Early warning for unreliable forecasts
        
        ### When to Use
        
        ✅ **Use when:**
        - Working with 5+ correlated variables
        - Building multivariate forecasting models
        - Exploring scenario relationships
        - Variables show high correlation (>0.7)
        
        ❌ **Not needed when:**
        - Working with 2-4 distinct variables
        - Variables are clearly independent
        - Univariate time series modeling
        
        ### Philosophy
        This is an **optional analytical tool**, not a required step. You decide whether to apply it.
        """)
    
    # Create tabs (FIX #3: Variance Filtering removed)
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔗 Correlation Filtering",
        "🧮 PCA Analysis",
        "🎯 Regularization (Lasso/Elastic Net)",
        "📋 Summary & Export"
    ])
    
    with tab1:
        handle_correlation_filtering(df_wide, feature_cols)
    
    with tab2:
        handle_pca_analysis(df_wide, feature_cols)
    
    with tab3:
        handle_regularization(df_wide, feature_cols)
    
    with tab4:
        show_reduction_summary(df_wide, feature_cols, all_feature_cols)


def handle_correlation_filtering(df_wide, feature_cols):
    """Correlation-based feature filtering - Remove redundant variables"""
    
    st.header("🔗 Correlation-Based Feature Filtering")
    st.markdown("*Remove highly correlated (redundant) variables - Very explainable*")
    
    # Check for missing values
    if df_wide[feature_cols].isna().any().any():
        st.warning("⚠️ Data contains missing values. They will be handled using forward-fill.")
        df_clean = df_wide[feature_cols].fillna(method='ffill').fillna(method='bfill')
    else:
        df_clean = df_wide[feature_cols]
    
    st.subheader("Step 1: Analyze Correlations")
    
    # Show correlation matrix
    corr_matrix = df_clean.corr()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create correlation heatmap
        fig = create_correlation_heatmap(corr_matrix, feature_cols)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("💾 Export Correlation Matrix"):
                quick_export_buttons(fig, "correlation_matrix", ['png', 'pdf', 'html'])
    
    with col2:
        st.markdown("**Correlation Statistics**")
        
        # Find highly correlated pairs
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = []
        
        for column in upper.columns:
            for idx in upper.index:
                if abs(upper.loc[idx, column]) > 0.7:
                    high_corr_pairs.append((idx, column, upper.loc[idx, column]))
        
        st.metric("Total Variables", len(feature_cols))
        st.metric("High Correlations (>0.7)", len(high_corr_pairs))
        
        if high_corr_pairs:
            st.markdown("**Top Correlated Pairs:**")
            for var1, var2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]:
                st.caption(f"{var1} ↔ {var2}: {corr:.3f}")
    
    st.markdown("---")
    st.subheader("Step 2: Set Correlation Threshold")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        corr_threshold = st.slider(
            "Correlation threshold",
            min_value=0.5,
            max_value=0.99,
            value=0.85,
            step=0.05,
            help="Remove one feature from pairs with correlation above this threshold",
            key="corr_threshold"
        )
        
        st.info(f"""
        💡 **How it works:**
        - Identifies variable pairs with correlation > {corr_threshold}
        - Removes one variable from each pair
        - Keeps the variable that appears in fewer high-correlation pairs
        - Reduces multicollinearity for stable modeling
        """)
    
    with col2:
        # Preview impact
        to_drop = []
        for column in upper.columns:
            if any(abs(upper[column]) > corr_threshold):
                to_drop.append(column)
        
        kept = len(feature_cols) - len(set(to_drop))
        
        st.metric("Features to Keep", kept)
        st.metric("Features to Remove", len(set(to_drop)))
        
        if len(set(to_drop)) > 0:
            reduction_pct = (len(set(to_drop)) / len(feature_cols)) * 100
            st.metric("Reduction", f"{reduction_pct:.1f}%")
    
    # Apply button
    if st.button("🔗 Apply Correlation Filter", type="primary", key="apply_correlation"):
        with st.spinner("Filtering correlated features..."):
            try:
                selected_features, removed_features, filter_info = apply_correlation_filter(
                    df_clean, feature_cols, corr_threshold
                )
                
                # Store results
                st.session_state.selected_features = selected_features
                st.session_state.reduction_results['correlation'] = {
                    'method': 'Correlation Filtering',
                    'threshold': corr_threshold,
                    'selected': selected_features,
                    'removed': removed_features,
                    'correlation_matrix': corr_matrix,
                    'info': filter_info,
                    'timestamp': datetime.now()
                }
                
                st.success(f"✅ Kept {len(selected_features)} features, removed {len(removed_features)}")
                st.rerun()
                
            except Exception as e:
                display_error(f"Error in correlation filtering: {str(e)}")
                st.exception(e)
    
    # Display results if available
    if st.session_state.get('reduction_results', {}).get('correlation'):
        st.markdown("---")
        st.subheader("📊 Filtering Results")
        
        results = st.session_state.reduction_results['correlation']
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Original Features", len(feature_cols))
        col2.metric("Selected Features", len(results['selected']))
        col3.metric("Removed Features", len(results['removed']))
        
        # Show lists
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**✅ Kept Features:**")
            for feat in results['selected']:
                st.text(f"  • {feat}")
        
        with col2:
            st.markdown("**❌ Removed (Redundant):**")
            if results['removed']:
                for feat in results['removed']:
                    st.text(f"  • {feat}")
            else:
                st.caption("(none)")


def apply_correlation_filter(df, features, threshold):
    """Apply correlation-based filtering"""
    corr_matrix = df[features].corr().abs()
    
    # Upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features to drop
    to_drop = set()
    high_corr_info = []
    
    for column in upper.columns:
        correlated_features = upper.index[upper[column] > threshold].tolist()
        if correlated_features:
            # Drop the feature that has more high correlations
            for corr_feat in correlated_features:
                high_corr_info.append({
                    'feature1': column,
                    'feature2': corr_feat,
                    'correlation': corr_matrix.loc[column, corr_feat]
                })
                # Count high correlations for each
                col_count = (upper[column] > threshold).sum()
                feat_count = (upper[corr_feat] > threshold).sum()
                
                if col_count >= feat_count:
                    to_drop.add(column)
                else:
                    to_drop.add(corr_feat)
    
    selected = [f for f in features if f not in to_drop]
    removed = list(to_drop)
    
    info = {
        'high_correlations': high_corr_info,
        'threshold': threshold
    }
    
    return selected, removed, info


def create_correlation_heatmap(corr_matrix, features):
    """Create correlation heatmap"""
    try:
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=features,
            y=features,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 9},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            xaxis_title="Features",
            yaxis_title="Features",
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating heatmap: {str(e)}")
        return None


def create_pca_scatter_plot(components, pc_x, pc_y, results, point_size, point_color, show_time_gradient):
    """Create PCA scatter plot (FIX #4)"""
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


def handle_pca_analysis(df_wide, feature_cols):
    """Principal Component Analysis with accept/reject option"""
    
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
        
        # FIX #2: Display loadings without matplotlib styling
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
        
        # FIX #4: PCA SCATTER PLOT VISUALIZATION
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
        
        st.markdown("---")
        
        # Accept/Reject PCA for modeling
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


def handle_regularization(df_wide, feature_cols):
    """Feature selection using Lasso and Elastic Net regularization"""
    
    st.header("🎯 Regularization-Based Feature Selection")
    st.markdown("*Use Lasso or Elastic Net to identify important features*")
    
    # Check for missing values
    if df_wide[feature_cols].isna().any().any():
        st.warning("⚠️ Data contains missing values. They will be handled using forward-fill.")
        df_clean = df_wide[feature_cols].fillna(method='ffill').fillna(method='bfill')
    else:
        df_clean = df_wide[feature_cols]
    
    st.info("""
    💡 **Regularization methods:**
    - **Lasso (L1):** Drives weak feature coefficients to exactly zero
    - **Elastic Net:** Combines L1 and L2, handles correlated features better
    - Both methods automatically select important features
    """)
    
    st.subheader("Step 1: Select Target Variable")
    
    # Target selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Use selected features if available
        if st.session_state.selected_features:
            use_selected = st.checkbox(
                f"Use filtered features ({len(st.session_state.selected_features)})",
                value=True,
                key="reg_use_selected",
                help="Use features from previous filtering steps"
            )
            features_to_use = st.session_state.selected_features if use_selected else feature_cols
        else:
            features_to_use = feature_cols
        
        target_var = st.selectbox(
            "Target variable (to predict)",
            features_to_use,
            help="Select the variable you want to predict or model"
        )
        
        predictor_vars = [f for f in features_to_use if f != target_var]
        
        st.caption(f"Will use {len(predictor_vars)} features to predict {target_var}")
    
    with col2:
        st.metric("Predictor Features", len(predictor_vars))
        st.metric("Data Points", len(df_clean))
    
    if len(predictor_vars) < 2:
        st.warning("⚠️ Need at least 2 predictor features for regularization")
        return
    
    st.markdown("---")
    st.subheader("Step 2: Configure Regularization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        method = st.radio(
            "Regularization method",
            ["Lasso (L1)", "Elastic Net (L1 + L2)"],
            help="Lasso for sparse selection, Elastic Net for correlated features"
        )
    
    with col2:
        use_cv = st.checkbox(
            "Use cross-validation",
            value=True,
            help="Automatically find optimal regularization strength"
        )
        
        if not use_cv:
            alpha = st.slider(
                "Regularization strength (alpha)",
                min_value=0.001,
                max_value=10.0,
                value=1.0,
                step=0.1,
                format="%.3f",
                help="Higher = more regularization = fewer features"
            )
        
        if "Elastic" in method:
            l1_ratio = st.slider(
                "L1 ratio",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="1.0 = pure Lasso, 0.0 = pure Ridge, 0.5 = balanced"
            )
    
    # Apply regularization
    if st.button("🎯 Apply Regularization", type="primary", key="apply_regularization"):
        with st.spinner(f"Running {method}..."):
            try:
                # Prepare data
                X = df_clean[predictor_vars].values
                y = df_clean[target_var].values
                
                # Remove any remaining NaN
                mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                X = X[mask]
                y = y[mask]
                
                if len(X) < 10:
                    display_error("Not enough valid data points (need at least 10)")
                    return
                
                # Standardize
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                X_scaled = scaler_X.fit_transform(X)
                y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
                
                # Run regularization
                if "Lasso" in method and method == "Lasso (L1)":
                    if use_cv:
                        model = LassoCV(cv=5, random_state=42, max_iter=10000)
                        model.fit(X_scaled, y_scaled)
                        best_alpha = model.alpha_
                    else:
                        model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
                        model.fit(X_scaled, y_scaled)
                        best_alpha = alpha
                
                else:  # Elastic Net
                    if use_cv:
                        model = ElasticNetCV(l1_ratio=l1_ratio, cv=5, random_state=42, max_iter=10000)
                        model.fit(X_scaled, y_scaled)
                        best_alpha = model.alpha_
                    else:
                        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=10000)
                        model.fit(X_scaled, y_scaled)
                        best_alpha = alpha
                
                # Get feature importance
                coefficients = model.coef_
                
                # Select non-zero features
                selected_features = [predictor_vars[i] for i, coef in enumerate(coefficients) if abs(coef) > 1e-10]
                selected_features.append(target_var)  # Include target
                
                removed_features = [f for f in predictor_vars if f not in selected_features]
                
                # Store results
                st.session_state.reduction_results['regularization'] = {
                    'method': method,
                    'target': target_var,
                    'alpha': best_alpha,
                    'coefficients': dict(zip(predictor_vars, coefficients)),
                    'selected': selected_features,
                    'removed': removed_features,
                    'score': model.score(X_scaled, y_scaled),
                    'timestamp': datetime.now()
                }
                
                st.success(f"✅ Selected {len(selected_features)-1} predictors (+ target) using {method}")
                st.rerun()
                
            except Exception as e:
                display_error(f"Error in regularization: {str(e)}")
                st.exception(e)
    
    # Display results
    if st.session_state.get('reduction_results', {}).get('regularization'):
        st.markdown("---")
        st.subheader("📊 Regularization Results")
        
        results = st.session_state.reduction_results['regularization']
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Selected Features", len(results['selected'])-1)  # Exclude target
        col2.metric("Removed Features", len(results['removed']))
        col3.metric("Alpha Used", f"{results['alpha']:.4f}")
        col4.metric("R² Score", f"{results['score']:.3f}")
        
        # Feature importance plot
        st.markdown("#### Feature Importance (Coefficients)")
        
        coef_df = pd.DataFrame(list(results['coefficients'].items()), columns=['Feature', 'Coefficient'])
        coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
        coef_df = coef_df.sort_values('Abs_Coefficient', ascending=True)
        
        fig = go.Figure()
        
        colors = ['red' if c < 0 else 'steelblue' for c in coef_df['Coefficient']]
        
        fig.add_trace(go.Bar(
            y=coef_df['Feature'],
            x=coef_df['Coefficient'],
            orientation='h',
            marker_color=colors,
            text=[f"{c:.3f}" for c in coef_df['Coefficient']],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f"Feature Coefficients (predicting {results['target']})",
            xaxis_title="Coefficient",
            yaxis_title="Feature",
            height=max(400, len(coef_df) * 25),
            template='plotly_white',
            showlegend=False
        )
        
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("💾 Export Feature Importance"):
            quick_export_buttons(fig, "regularization_importance", ['png', 'pdf', 'html'])
        
        # Show selected vs removed
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**✅ Selected Features (Non-Zero):**")
            selected_coefs = {k: v for k, v in results['coefficients'].items() 
                            if k in results['selected'] and k != results['target']}
            for feat, coef in sorted(selected_coefs.items(), key=lambda x: abs(x[1]), reverse=True):
                st.text(f"  • {feat}: {coef:.4f}")
        
        with col2:
            st.markdown("**❌ Removed Features (Zero Coefficients):**")
            if results['removed']:
                for feat in results['removed']:
                    st.text(f"  • {feat}")
            else:
                st.caption("(none)")


def show_reduction_summary(df_wide, feature_cols, all_feature_cols):
    """Summary of all dimensionality reduction results (FIX #6: added all_feature_cols parameter)"""
    
    st.header("📋 Summary & Export")
    
    if not st.session_state.reduction_results:
        st.info("ℹ️ No dimensionality reduction has been performed yet")
        st.markdown("Apply one or more methods in the tabs above to see results here")
        return
    
    st.success(f"✅ {len(st.session_state.reduction_results)} method(s) applied")
    
    # Summary table
    st.subheader("Methods Applied")
    
    summary_data = []
    for method_name, results in st.session_state.reduction_results.items():
        if method_name == 'correlation':
            summary_data.append({
                "Method": "Correlation Filtering",
                "Details": f"Threshold: {results['threshold']:.2f}",
                "Features": f"{len(results['selected'])} kept, {len(results['removed'])} removed",
                "Applied": results['timestamp'].strftime("%Y-%m-%d %H:%M")
            })
        
        elif method_name == 'pca':
            accepted = "✅ Accepted" if st.session_state.pca_accepted else "⏸️ Not Accepted"
            summary_data.append({
                "Method": f"PCA ({accepted})",
                "Details": f"{results['n_components']} components",
                "Features": f"{results['cumulative_variance'][-1]*100:.1f}% variance explained",
                "Applied": results['timestamp'].strftime("%Y-%m-%d %H:%M")
            })
        
        elif method_name == 'regularization':
            summary_data.append({
                "Method": results['method'],
                "Details": f"Target: {results['target']}, α={results['alpha']:.4f}",
                "Features": f"{len(results['selected'])-1} selected, R²={results['score']:.3f}",
                "Applied": results['timestamp'].strftime("%Y-%m-%d %H:%M")
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Final recommendation
    st.subheader("💡 Recommendations for Modeling")
    
    # Determine best approach
    if st.session_state.pca_accepted and 'pca' in st.session_state.reduction_results:
        pca_results = st.session_state.reduction_results['pca']
        variance = pca_results['cumulative_variance'][-1]
        
        st.success(f"""
        ✅ **Recommended: Use PCA Components**
        
        **Why:**
        - {pca_results['n_components']} components explain {variance*100:.1f}% of variance
        - Orthogonal (uncorrelated) features
        - Reduced multicollinearity
        - User accepted for modeling
        
        **For modeling, use:** PC1 to PC{pca_results['n_components']}
        """)
    
    elif st.session_state.selected_features:
        st.info(f"""
        💡 **Recommended: Use Filtered Features**
        
        **Features selected:** {len(st.session_state.selected_features)}
        
        **Why:**
        - Reduced from {len(all_feature_cols)} original features
        - Low-information and redundant features removed
        - Preserves interpretability of original variables
        
        **For modeling, use:** Selected {len(st.session_state.selected_features)} features
        """)
    
    else:
        st.warning("""
        ⚠️ **No dimensionality reduction accepted**
        
        Will use all original features. Consider:
        - High risk of multicollinearity
        - Potentially unstable models
        - Recommend applying at least correlation filtering
        """)
    
    st.markdown("---")
    
    # Export options
    st.subheader("💾 Export Reduced Data")
    
    time_col = 'timestamp' if 'timestamp' in df_wide.columns else 'time'
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Filtered Features:**")
        
        if st.session_state.selected_features:
            # Create reduced dataset with selected features
            reduced_df = df_wide[[time_col] + st.session_state.selected_features]
            
            csv = reduced_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Features (CSV)",
                data=csv,
                file_name=f"filtered_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.caption("No filtered features available")
    
    with col2:
        st.markdown("**PCA Components:**")
        
        if st.session_state.pca_accepted and 'pca' in st.session_state.reduction_results:
            pca_results = st.session_state.reduction_results['pca']
            
            # Create PCA components dataframe
            pca_df = pd.DataFrame(
                st.session_state.pca_components,
                columns=[f"PC{i+1}" for i in range(pca_results['n_components'])]
            )
            pca_df.insert(0, time_col, df_wide[time_col].values[:len(pca_df)])
            
            csv = pca_df.to_csv(index=False)
            st.download_button(
                label="📥 Download PCA Components (CSV)",
                data=csv,
                file_name=f"pca_components_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.caption("PCA not accepted for modeling")
    
    # Analysis report
    st.markdown("---")
    st.markdown("**Analysis Report:**")
    
    report = create_analysis_report(st.session_state.reduction_results, all_feature_cols)
    
    st.download_button(
        label="📄 Download Analysis Report (TXT)",
        data=report,
        file_name=f"dimensionality_reduction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True
    )
    
    # Clear results
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col2:
        if st.button("🔄 Clear All Results", key="clear_reduction", use_container_width=True):
            st.session_state.reduction_results = {}
            st.session_state.selected_features = []
            st.session_state.pca_model = None
            st.session_state.pca_accepted = False
            st.session_state.reduced_data = None
            st.success("✅ Results cleared")
            st.rerun()


def create_analysis_report(results, original_features):
    """Create a text report of dimensionality reduction analysis"""
    
    report = []
    report.append("="*70)
    report.append("DIMENSIONALITY REDUCTION ANALYSIS REPORT")
    report.append("ExplainFutures - Stabilize Models & Improve Explainability")
    report.append("="*70)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Original features: {len(original_features)}")
    report.append("\n")
    
    # Correlation Filtering
    if 'correlation' in results:
        corr = results['correlation']
        report.append("-"*70)
        report.append("1. CORRELATION-BASED FILTERING")
        report.append("-"*70)
        report.append(f"Threshold: {corr['threshold']:.2f}")
        report.append(f"Selected features: {len(corr['selected'])}")
        report.append(f"Removed features: {len(corr['removed'])}")
        report.append(f"Reduction: {len(corr['removed'])/len(original_features)*100:.1f}%")
        report.append(f"\nKept: {', '.join(corr['selected'])}")
        if corr['removed']:
            report.append(f"\nRemoved (redundant): {', '.join(corr['removed'])}")
        report.append("\n")
    
    # PCA
    if 'pca' in results:
        pca = results['pca']
        report.append("-"*70)
        report.append("2. PRINCIPAL COMPONENT ANALYSIS (PCA)")
        report.append("-"*70)
        report.append(f"Components extracted: {pca['n_components']}")
        report.append(f"Total variance explained: {pca['cumulative_variance'][-1]*100:.2f}%")
        report.append(f"Status: {'ACCEPTED for modeling' if st.session_state.pca_accepted else 'Not accepted'}")
        report.append("\nVariance by component:")
        for i, var in enumerate(pca['explained_variance']):
            cum_var = pca['cumulative_variance'][i]
            report.append(f"  PC{i+1}: {var*100:.2f}% (cumulative: {cum_var*100:.2f}%)")
        report.append("\n")
    
    # Regularization
    if 'regularization' in results:
        reg = results['regularization']
        report.append("-"*70)
        report.append("3. REGULARIZATION-BASED SELECTION")
        report.append("-"*70)
        report.append(f"Method: {reg['method']}")
        report.append(f"Target variable: {reg['target']}")
        report.append(f"Alpha: {reg['alpha']:.4f}")
        report.append(f"R² Score: {reg['score']:.3f}")
        report.append(f"Selected features: {len(reg['selected'])-1} (+ target)")
        report.append(f"Removed features: {len(reg['removed'])}")
        report.append(f"\nSelected: {', '.join([f for f in reg['selected'] if f != reg['target']])}")
        if reg['removed']:
            report.append(f"\nRemoved: {', '.join(reg['removed'])}")
        report.append("\n")
    
    # Final Recommendation
    report.append("="*70)
    report.append("RECOMMENDATION FOR MODELING")
    report.append("="*70)
    
    if st.session_state.pca_accepted and 'pca' in results:
        pca = results['pca']
        report.append(f"\n✅ USE PCA COMPONENTS")
        report.append(f"   - {pca['n_components']} components")
        report.append(f"   - {pca['cumulative_variance'][-1]*100:.1f}% variance explained")
        report.append(f"   - Orthogonal features, reduced multicollinearity")
    elif st.session_state.selected_features:
        report.append(f"\n💡 USE FILTERED FEATURES")
        report.append(f"   - {len(st.session_state.selected_features)} features")
        report.append(f"   - Redundant and low-information features removed")
        report.append(f"   - Preserves interpretability")
    else:
        report.append(f"\n⚠️  NO REDUCTION APPLIED")
        report.append(f"   - Using all {len(original_features)} original features")
        report.append(f"   - Risk of multicollinearity and overfitting")
        report.append(f"   - Consider applying filtering")
    
    report.append("\n")
    report.append("="*70)
    report.append("END OF REPORT")
    report.append("="*70)
    
    return "\n".join(report)


if __name__ == "__main__":
    main()
