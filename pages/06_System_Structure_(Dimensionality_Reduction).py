"""
Page 5: Understand The System (Dimensionality Reduction)
Reduce complexity, stabilize models, and improve explainability
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import plotly.graph_objects as go
from datetime import datetime

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Understand The System",
    page_icon="🔮",
    layout="wide"  # CRITICAL: Use full page width
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import get_config, initialize_session_state
from core.utils import display_error, display_success, display_warning, display_info
from core.shared_sidebar import render_app_sidebar
from core.viz.export import quick_export_buttons

# Import dimensionality reduction modules
from core.dim_reduction.pca_module import run_pca_analysis
from core.dim_reduction.factor_analysis_module import run_factor_analysis
from core.dim_reduction.ica_module import run_ica_analysis
from core.dim_reduction.clustering_module import run_hierarchical_clustering

# Initialize
initialize_session_state()
config = get_config()


# Render shared sidebar
render_app_sidebar()

st.title("🔬 Understand The System")
st.markdown("*Dimensionality Reduction for System Understanding & Model Stabilization*")
st.markdown("---")

# Copy these 6 lines to the TOP of each page (02-13)
if not st.session_state.get('authenticated', False):
    st.warning("⚠️ Please log in to continue")
    time.sleep(1)
    st.switch_page("App.py")
    st.stop()

# Then your existing code continues...


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
    
    # === STEP 1: VARIABLE SELECTION ===
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
        
        ### Available Methods
        
        **🔗 Correlation Filtering:** Remove redundant variables (simplest)
        **🧮 PCA:** Find orthogonal components (most common)
        **🎯 Factor Analysis:** Discover latent factors (most interpretable)
        **🔬 ICA:** Find independent sources (most sophisticated)
        **🌳 Clustering:** Group similar variables (most visual)
        
        ### Philosophy
        This is an **optional analytical tool**, not a required step. You decide whether to apply it.
        """)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔗 Correlation Filtering",
        "🧮 PCA Analysis",
        "🎯 Factor Analysis",
        "🔬 Independent Component Analysis",
        "🌳 Hierarchical Clustering",
        "📋 Summary & Export"
    ])
    
    with tab1:
        handle_correlation_filtering(df_wide, feature_cols)
    
    with tab2:
        run_pca_analysis(df_wide, feature_cols)
    
    with tab3:
        run_factor_analysis(df_wide, feature_cols)
    
    with tab4:
        run_ica_analysis(df_wide, feature_cols)
    
    with tab5:
        run_hierarchical_clustering(df_wide, feature_cols)
    
    with tab6:
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


def show_reduction_summary(df_wide, feature_cols, all_feature_cols):
    """Comprehensive summary of all dimensionality reduction analyses"""
    
    st.header("📋 Summary & Comprehensive Analysis")
    
    if not st.session_state.reduction_results:
        st.info("ℹ️ No dimensionality reduction has been performed yet")
        st.markdown("Apply one or more methods in the tabs above to see comprehensive analysis here")
        return
    
    st.success(f"✅ {len(st.session_state.reduction_results)} method(s) applied")
    
    # === COMPREHENSIVE ANALYSIS SECTION ===
    st.markdown("---")
    st.subheader("🔍 Comprehensive Analysis")
    
    # Analysis tabs
    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
        "📊 Method Comparison",
        "💡 Key Insights",
        "🎯 Recommendations"
    ])
    
    with analysis_tab1:
        display_method_comparison(all_feature_cols)
    
    with analysis_tab2:
        display_key_insights(all_feature_cols)
    
    with analysis_tab3:
        display_recommendations()
    
    st.markdown("---")
    
    # Export options
    st.subheader("💾 Export Reduced Data & Reports")
    
    display_export_options(df_wide, all_feature_cols)
    
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


def display_method_comparison(all_feature_cols):
    """Display method comparison table"""
    
    st.markdown("### Method Comparison")
    
    # Create comprehensive summary table
    summary_data = []
    
    if 'correlation' in st.session_state.reduction_results:
        corr = st.session_state.reduction_results['correlation']
        reduction_pct = len(corr['removed']) / len(all_feature_cols) * 100
        summary_data.append({
            "Method": "🔗 Correlation Filtering",
            "Purpose": "Remove redundant variables",
            "Input Features": len(all_feature_cols),
            "Output": f"{len(corr['selected'])} features",
            "Reduction": f"{reduction_pct:.1f}%",
            "Key Result": f"Removed {len(corr['removed'])} correlated vars",
            "Status": "✅ Complete"
        })
    
    if 'pca' in st.session_state.reduction_results:
        pca = st.session_state.reduction_results['pca']
        status = "✅ Accepted" if st.session_state.pca_accepted else "⏸️ Not Accepted"
        summary_data.append({
            "Method": "🧮 PCA",
            "Purpose": "Find orthogonal components",
            "Input Features": len(pca['features']),
            "Output": f"{pca['n_components']} components",
            "Reduction": f"{pca['cumulative_variance'][-1]*100:.1f}% var explained",
            "Key Result": f"PC1 explains {pca['explained_variance'][0]*100:.1f}%",
            "Status": status
        })
    
    if 'factor_analysis' in st.session_state.reduction_results:
        fa = st.session_state.reduction_results['factor_analysis']
        avg_comm = np.mean(fa['communalities'])
        summary_data.append({
            "Method": "🎯 Factor Analysis",
            "Purpose": "Discover latent factors",
            "Input Features": len(fa['features']),
            "Output": f"{fa['n_factors']} factors",
            "Reduction": f"Avg communality: {avg_comm:.3f}",
            "Key Result": f"Rotation: {fa['rotation']}",
            "Status": "✅ Complete"
        })
    
    if 'ica' in st.session_state.reduction_results:
        ica = st.session_state.reduction_results['ica']
        summary_data.append({
            "Method": "🔬 ICA",
            "Purpose": "Find independent sources",
            "Input Features": len(ica['features']),
            "Output": f"{ica['n_components']} components",
            "Reduction": f"{ica['n_components']}/{len(ica['features'])} components",
            "Key Result": "Independent drivers identified",
            "Status": "✅ Complete"
        })
    
    if 'clustering' in st.session_state.reduction_results:
        clust = st.session_state.reduction_results['clustering']
        summary_data.append({
            "Method": "🌳 Hierarchical Clustering",
            "Purpose": "Group similar variables",
            "Input Features": len(clust['features']),
            "Output": f"{clust['n_clusters']} clusters",
            "Reduction": f"Method: {clust['linkage_method']}",
            "Key Result": f"Metric: {clust['distance_metric']}",
            "Status": "✅ Complete"
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


def display_key_insights(all_feature_cols):
    """Display key insights across methods"""
    
    st.markdown("### Key Insights Across Methods")
    
    # Variable importance analysis
    if 'pca' in st.session_state.reduction_results or 'factor_analysis' in st.session_state.reduction_results:
        st.markdown("#### 🎯 Most Important Variables")
        
        important_vars = {}
        
        # From PCA
        if 'pca' in st.session_state.reduction_results:
            pca = st.session_state.reduction_results['pca']
            loadings = pca['loadings'][0]  # First component
            for i, feat in enumerate(pca['features']):
                important_vars[feat] = important_vars.get(feat, 0) + abs(loadings[i])
        
        # From Factor Analysis
        if 'factor_analysis' in st.session_state.reduction_results:
            fa = st.session_state.reduction_results['factor_analysis']
            for i, feat in enumerate(fa['features']):
                important_vars[feat] = important_vars.get(feat, 0) + fa['communalities'][i]
        
        if important_vars:
            sorted_vars = sorted(important_vars.items(), key=lambda x: x[1], reverse=True)
            
            st.markdown("**Top 5 Most Influential Variables:**")
            for i, (var, score) in enumerate(sorted_vars[:5], 1):
                st.text(f"{i}. {var} (importance: {score:.3f})")
    
    # System complexity analysis
    st.markdown("---")
    st.markdown("#### 📊 System Complexity Assessment")
    
    if 'pca' in st.session_state.reduction_results:
        pca = st.session_state.reduction_results['pca']
        variance = pca['cumulative_variance'][-1]
        
        if variance >= 0.85:
            st.success(f"""
            **Low Complexity System ✅**
            - {pca['n_components']} components explain {variance*100:.1f}% of variance
            - System has clear dominant patterns
            - Good candidate for dimensionality reduction
            - Forecasting should be stable
            """)
        elif variance >= 0.70:
            st.info(f"""
            **Moderate Complexity System 💡**
            - {pca['n_components']} components explain {variance*100:.1f}% of variance
            - System has multiple important patterns
            - Some dimensionality reduction possible
            - Forecasting moderately stable
            """)
        else:
            st.warning(f"""
            **High Complexity System ⚠️**
            - Only {variance*100:.1f}% variance explained by {pca['n_components']} components
            - System is genuinely high-dimensional
            - Many independent patterns present
            - Forecasting may be challenging
            - Consider domain expertise for variable selection
            """)
    
    # Correlation structure
    if 'correlation' in st.session_state.reduction_results:
        st.markdown("---")
        st.markdown("#### 🔗 Correlation Structure")
        
        corr = st.session_state.reduction_results['correlation']
        if len(corr['removed']) > 0:
            removal_pct = len(corr['removed']) / len(all_feature_cols) * 100
            
            if removal_pct > 30:
                st.warning(f"""
                **High Redundancy Detected ⚠️**
                - {removal_pct:.0f}% of variables are highly correlated
                - Significant multicollinearity present
                - Dimensionality reduction highly recommended
                """)
            elif removal_pct > 15:
                st.info(f"""
                **Moderate Redundancy 💡**
                - {removal_pct:.0f}% of variables are highly correlated
                - Some multicollinearity present
                - Dimensionality reduction beneficial
                """)
            else:
                st.success(f"""
                **Low Redundancy ✅**
                - Only {removal_pct:.0f}% of variables highly correlated
                - Variables are relatively independent
                - Limited multicollinearity
                """)
    
    # Clustering insights
    if 'clustering' in st.session_state.reduction_results:
        st.markdown("---")
        st.markdown("#### 🌳 Variable Groupings")
        
        clust = st.session_state.reduction_results['clustering']
        
        st.markdown("**Natural Variable Groups:**")
        for cluster_name, variables in clust['clusters'].items():
            if len(variables) > 1:
                st.text(f"• {cluster_name}: {len(variables)} similar variables")
        
        st.info("💡 Variables in the same cluster show similar behavior and could potentially be represented by a single feature")


def display_recommendations():
    """Display recommendations for modeling"""
    
    st.markdown("### Recommendations for ExplainFutures")
    
    # Determine best approach
    recommendations = []
    
    # Check PCA acceptance
    if st.session_state.pca_accepted and 'pca' in st.session_state.reduction_results:
        pca = st.session_state.reduction_results['pca']
        variance = pca['cumulative_variance'][-1]
        
        recommendations.append({
            'priority': 1,
            'method': 'PCA Components',
            'action': f"Use {pca['n_components']} principal components",
            'reason': f"{variance*100:.1f}% variance explained, orthogonal features, user accepted",
            'benefit': "Reduced multicollinearity, stable models, simplified interpretation"
        })
    
    # Check correlation filtering
    if 'correlation' in st.session_state.reduction_results:
        corr = st.session_state.reduction_results['correlation']
        if len(corr['removed']) > 0:
            recommendations.append({
                'priority': 2,
                'method': 'Filtered Features',
                'action': f"Use {len(corr['selected'])} correlation-filtered features",
                'reason': f"Removed {len(corr['removed'])} redundant variables",
                'benefit': "Maintains interpretability while reducing multicollinearity"
            })
    
    # Check factor analysis
    if 'factor_analysis' in st.session_state.reduction_results:
        fa = st.session_state.reduction_results['factor_analysis']
        recommendations.append({
            'priority': 3,
            'method': 'Factor Analysis',
            'action': f"Consider {fa['n_factors']} latent factors for scenario analysis",
            'reason': "Factors represent meaningful underlying drivers",
            'benefit': "Excellent for scenario planning and what-if analysis"
        })
    
    # Check ICA
    if 'ica' in st.session_state.reduction_results:
        ica = st.session_state.reduction_results['ica']
        recommendations.append({
            'priority': 4,
            'method': 'Independent Components',
            'action': f"Use {ica['n_components']} independent components for multi-driver scenarios",
            'reason': "Identifies statistically independent driving forces",
            'benefit': "Best for modeling systems with multiple independent drivers"
        })
    
    # Check clustering
    if 'clustering' in st.session_state.reduction_results:
        clust = st.session_state.reduction_results['clustering']
        recommendations.append({
            'priority': 5,
            'method': 'Clustered Representatives',
            'action': f"Select representative variables from each of {clust['n_clusters']} clusters",
            'reason': "Natural groupings identified",
            'benefit': "Maintains domain interpretability, one feature per cluster"
        })
    
    # Display recommendations
    if recommendations:
        st.markdown("#### 🎯 Prioritized Recommendations")
        
        for rec in sorted(recommendations, key=lambda x: x['priority']):
            with st.expander(f"**{rec['priority']}. {rec['method']}**", expanded=(rec['priority']==1)):
                st.markdown(f"**Action:** {rec['action']}")
                st.markdown(f"**Reason:** {rec['reason']}")
                st.markdown(f"**Benefit:** {rec['benefit']}")
    
    # Overall strategy
    st.markdown("---")
    st.markdown("#### 📋 Recommended Strategy")
    
    if st.session_state.pca_accepted and 'pca' in st.session_state.reduction_results:
        pca = st.session_state.reduction_results['pca']
        st.success(f"""
        **PRIMARY STRATEGY: Use PCA Components**
        
        **For Forecasting Models:**
        1. Use PCA components as input features
        2. Build models on orthogonal components
        3. Backtransform predictions if needed
        
        **For Scenario Analysis:**
        1. Each PC represents a scenario dimension
        2. Create scenarios by varying component values
        3. Interpret using component loadings
        
        **Benefits:**
        - Maximum variance retention ({pca['cumulative_variance'][-1]*100:.1f}%)
        - No multicollinearity
        - Stable coefficient estimation
        - User-accepted approach
        """)
    
    elif st.session_state.selected_features:
        st.info(f"""
        **ALTERNATIVE STRATEGY: Use Filtered Features**
        
        **For Forecasting Models:**
        1. Use correlation-filtered features directly
        2. Monitor VIF (Variance Inflation Factor)
        3. Consider further reduction if needed
        
        **For Scenario Analysis:**
        1. Each feature represents a scenario lever
        2. Easier to explain to stakeholders
        3. Maintain original variable meanings
        
        **Benefits:**
        - Direct interpretability
        - No transformation needed
        - Stakeholder-friendly
        - {len(st.session_state.selected_features)} features selected
        """)
    
    else:
        st.warning("""
        **FALLBACK: Use All Variables with Caution**
        
        ⚠️ **Risks:**
        - High multicollinearity possible
        - Unstable model coefficients
        - Difficult to validate
        
        **Recommendations:**
        1. Apply at least correlation filtering
        2. Monitor model diagnostics closely
        3. Use regularization techniques
        4. Cross-validate extensively
        
        **Next Steps:**
        - Go back to Correlation Filtering tab
        - Or accept PCA results
        """)


def display_export_options(df_wide, all_feature_cols):
    """Display export options for reduced data"""
    
    time_col = 'timestamp' if 'timestamp' in df_wide.columns else 'time'
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Filtered Features:**")
        
        if st.session_state.selected_features:
            # Create reduced dataset with selected features
            reduced_df = df_wide[[time_col] + st.session_state.selected_features]
            
            csv = reduced_df.to_csv(index=False)
            st.download_button(
                label="📥 Filtered Features CSV",
                data=csv,
                file_name=f"filtered_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.caption("No filtered features")
    
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
                label="📥 PCA Components CSV",
                data=csv,
                file_name=f"pca_components_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.caption("PCA not accepted")
    
    with col3:
        st.markdown("**Comprehensive Report:**")
        
        report = create_comprehensive_report(st.session_state.reduction_results, all_feature_cols)
        
        st.download_button(
            label="📄 Full Analysis Report",
            data=report,
            file_name=f"dimension_reduction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )


def create_comprehensive_report(results, original_features):
    """Create comprehensive analysis report"""
    
    report = []
    report.append("="*80)
    report.append("COMPREHENSIVE DIMENSIONALITY REDUCTION ANALYSIS REPORT")
    report.append("ExplainFutures - System Understanding & Model Stabilization")
    report.append("="*80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Original features analyzed: {len(original_features)}")
    report.append("\n")
    
    # Executive Summary
    report.append("="*80)
    report.append("EXECUTIVE SUMMARY")
    report.append("="*80)
    
    methods_applied = list(results.keys())
    report.append(f"\nMethods Applied: {', '.join(methods_applied)}")
    
    if st.session_state.pca_accepted and 'pca' in results:
        pca = results['pca']
        report.append(f"\nRECOMMENDATION: Use {pca['n_components']} PCA components")
        report.append(f"Variance Explained: {pca['cumulative_variance'][-1]*100:.2f}%")
        report.append("Status: Accepted for modeling")
    elif st.session_state.selected_features:
        report.append(f"\nRECOMMENDATION: Use {len(st.session_state.selected_features)} filtered features")
        report.append("Approach: Correlation-based filtering")
    else:
        report.append("\nRECOMMENDATION: Apply dimensionality reduction before modeling")
        report.append("Reason: High risk of multicollinearity")
    
    report.append("\n")
    
    # Add details for each method
    if 'correlation' in results:
        corr = results['correlation']
        report.append("\n" + "="*80)
        report.append("CORRELATION-BASED FILTERING")
        report.append("="*80)
        report.append(f"Threshold: {corr['threshold']:.2f}")
        report.append(f"Features kept: {len(corr['selected'])}")
        report.append(f"Features removed: {len(corr['removed'])}")
        report.append(f"\nKept: {', '.join(corr['selected'])}")
        if corr['removed']:
            report.append(f"Removed: {', '.join(corr['removed'])}")
    
    if 'pca' in results:
        pca = results['pca']
        report.append("\n" + "="*80)
        report.append("PRINCIPAL COMPONENT ANALYSIS")
        report.append("="*80)
        report.append(f"Components: {pca['n_components']}")
        report.append(f"Total variance: {pca['cumulative_variance'][-1]*100:.2f}%")
        report.append(f"Status: {'Accepted' if st.session_state.pca_accepted else 'Not accepted'}")
    
    if 'factor_analysis' in results:
        fa = results['factor_analysis']
        report.append("\n" + "="*80)
        report.append("FACTOR ANALYSIS")
        report.append("="*80)
        report.append(f"Factors: {fa['n_factors']}")
        report.append(f"Rotation: {fa['rotation']}")
        report.append(f"Avg communality: {np.mean(fa['communalities']):.3f}")
    
    if 'ica' in results:
        ica = results['ica']
        report.append("\n" + "="*80)
        report.append("INDEPENDENT COMPONENT ANALYSIS")
        report.append("="*80)
        report.append(f"Components: {ica['n_components']}")
        report.append(f"Features analyzed: {len(ica['features'])}")
    
    if 'clustering' in results:
        clust = results['clustering']
        report.append("\n" + "="*80)
        report.append("HIERARCHICAL CLUSTERING")
        report.append("="*80)
        report.append(f"Clusters: {clust['n_clusters']}")
        report.append(f"Method: {clust['linkage_method']}")
        report.append(f"Metric: {clust['distance_metric']}")
    
    report.append("\n" + "="*80)
    report.append("END OF REPORT")
    report.append("="*80)
    
    return "\n".join(report)


if __name__ == "__main__":
    main()
