"""
Page 4: Variable Relationships
Explore relationships between variables with correlation, regression, and ML analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import get_config, initialize_session_state
from core.utils import display_error, display_success, display_warning, display_info
from core.shared_sidebar import render_app_sidebar

# Initialize
initialize_session_state()
config = get_config()

# Page configuration
st.set_page_config(page_title="Variable Relationships", page_icon="🔗", layout="wide")

# Render shared sidebar
render_app_sidebar()

st.title("🔗 Variable Relationships")
st.markdown("*Explore correlations, patterns, and relationships between variables*")
st.markdown("---")


def initialize_plot_configs():
    """Initialize session state for plot configurations"""
    if "num_plots" not in st.session_state:
        st.session_state.num_plots = 2
    if "plot_configs" not in st.session_state:
        st.session_state.plot_configs = {}


def create_combined_variable(df, variables, operation):
    """
    Create a combined variable from multiple variables using an operation
    Returns: array of combined values
    """
    data_arrays = []
    for var in variables:
        var_data = df[df['variable'] == var]['value'].values
        data_arrays.append(var_data)
    
    # Ensure all arrays have same length (use shortest)
    min_length = min(len(arr) for arr in data_arrays)
    data_arrays = [arr[:min_length] for arr in data_arrays]
    
    if operation == "Sum":
        return np.sum(data_arrays, axis=0)
    elif operation == "Multiply":
        return np.prod(data_arrays, axis=0)
    elif operation == "Average":
        return np.mean(data_arrays, axis=0)
    elif operation == "Divide" and len(variables) == 2:
        # Avoid division by zero
        denominator = data_arrays[1]
        denominator[denominator == 0] = np.nan
        return data_arrays[0] / denominator
    else:
        return data_arrays[0]  # Default to first variable


def calculate_correlation(x, y, method='pearson'):
    """Calculate correlation coefficient"""
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 3:
        return None, None
    
    if method == 'pearson':
        coef, p_value = stats.pearsonr(x_clean, y_clean)
    elif method == 'spearman':
        coef, p_value = stats.spearmanr(x_clean, y_clean)
    elif method == 'kendall':
        coef, p_value = stats.kendalltau(x_clean, y_clean)
    else:
        coef, p_value = stats.pearsonr(x_clean, y_clean)
    
    return coef, p_value


def fit_regression_model(x, y, model_type='linear'):
    """
    Fit various regression models
    Returns: predictions, equation, R², metrics
    """
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask].reshape(-1, 1)
    y_clean = y[mask]
    
    if len(x_clean) < 3:
        return None, None, None, None
    
    try:
        if model_type == 'linear':
            # Linear regression: y = mx + b
            model = LinearRegression()
            model.fit(x_clean, y_clean)
            y_pred = model.predict(x_clean)
            
            equation = f"y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}"
        
        elif model_type == 'polynomial_2':
            # Polynomial degree 2
            poly = PolynomialFeatures(degree=2)
            x_poly = poly.fit_transform(x_clean)
            model = LinearRegression()
            model.fit(x_poly, y_clean)
            y_pred = model.predict(x_poly)
            
            coefs = model.coef_
            equation = f"y = {coefs[2]:.4f}x² + {coefs[1]:.4f}x + {model.intercept_:.4f}"
        
        elif model_type == 'polynomial_3':
            # Polynomial degree 3
            poly = PolynomialFeatures(degree=3)
            x_poly = poly.fit_transform(x_clean)
            model = LinearRegression()
            model.fit(x_poly, y_clean)
            y_pred = model.predict(x_poly)
            
            coefs = model.coef_
            equation = f"y = {coefs[3]:.4f}x³ + {coefs[2]:.4f}x² + {coefs[1]:.4f}x + {model.intercept_:.4f}"
        
        elif model_type == 'logarithmic':
            # Logarithmic: y = a + b*log(x)
            x_log = np.log(x_clean + 1)  # Add 1 to avoid log(0)
            model = LinearRegression()
            model.fit(x_log, y_clean)
            y_pred = model.predict(x_log)
            
            equation = f"y = {model.intercept_:.4f} + {model.coef_[0]:.4f}*log(x)"
        
        elif model_type == 'exponential':
            # Exponential: y = a * e^(bx)
            # Transform: log(y) = log(a) + bx
            y_positive = y_clean - y_clean.min() + 1
            y_log = np.log(y_positive)
            model = LinearRegression()
            model.fit(x_clean, y_log)
            y_pred_log = model.predict(x_clean)
            y_pred = np.exp(y_pred_log) + y_clean.min() - 1
            
            a = np.exp(model.intercept_)
            b = model.coef_[0]
            equation = f"y = {a:.4f} * e^({b:.4f}x)"
        
        elif model_type == 'power':
            # Power: y = a * x^b
            # Transform: log(y) = log(a) + b*log(x)
            x_positive = x_clean - x_clean.min() + 1
            y_positive = y_clean - y_clean.min() + 1
            x_log = np.log(x_positive)
            y_log = np.log(y_positive)
            model = LinearRegression()
            model.fit(x_log, y_log)
            y_pred_log = model.predict(x_log)
            y_pred = np.exp(y_pred_log) + y_clean.min() - 1
            
            a = np.exp(model.intercept_)
            b = model.coef_[0]
            equation = f"y = {a:.4f} * x^{b:.4f}"
        
        else:
            return None, None, None, None
        
        # Calculate metrics
        r2 = r2_score(y_clean, y_pred)
        rmse = np.sqrt(mean_squared_error(y_clean, y_pred))
        mae = mean_absolute_error(y_clean, y_pred)
        
        metrics = {
            'R²': r2,
            'RMSE': rmse,
            'MAE': mae
        }
        
        # Sort for plotting
        sort_idx = np.argsort(x_clean.flatten())
        x_sorted = x_clean[sort_idx].flatten()
        y_pred_sorted = y_pred[sort_idx]
        
        return x_sorted, y_pred_sorted, equation, metrics
    
    except Exception as e:
        return None, None, None, None


def fit_ml_model(x, y, model_type='random_forest'):
    """
    Fit machine learning models
    Returns: predictions, feature importance, metrics
    """
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask].reshape(-1, 1)
    y_clean = y[mask]
    
    if len(x_clean) < 10:
        return None, None, None
    
    try:
        if model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            model.fit(x_clean, y_clean)
            y_pred = model.predict(x_clean)
            
            # Feature importance (for single feature, it's always 1.0)
            importance = model.feature_importances_[0]
        
        else:
            return None, None, None
        
        # Calculate metrics
        r2 = r2_score(y_clean, y_pred)
        rmse = np.sqrt(mean_squared_error(y_clean, y_pred))
        mae = mean_absolute_error(y_clean, y_pred)
        
        metrics = {
            'R²': r2,
            'RMSE': rmse,
            'MAE': mae
        }
        
        # Sort for plotting
        sort_idx = np.argsort(x_clean.flatten())
        x_sorted = x_clean[sort_idx].flatten()
        y_pred_sorted = y_pred[sort_idx]
        
        return x_sorted, y_pred_sorted, metrics
    
    except Exception as e:
        return None, None, None


def create_scatter_plot(x_data, y_data, x_label, y_label, config):
    """Create an interactive scatter plot with customization options"""
    
    fig = go.Figure()
    
    # Scatter points
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='markers',
        name='Data',
        marker=dict(
            color=config.get('color', 'blue'),
            size=config.get('marker_size', 6),
            opacity=0.6
        ),
        text=[f"x: {x:.2f}<br>y: {y:.2f}" for x, y in zip(x_data, y_data)],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    # Add regression line if requested
    if config.get('show_regression'):
        model_type = config.get('regression_type', 'linear')
        x_sorted, y_pred, equation, metrics = fit_regression_model(x_data, y_data, model_type)
        
        if x_sorted is not None:
            fig.add_trace(go.Scatter(
                x=x_sorted,
                y=y_pred,
                mode='lines',
                name=f'Fit: {model_type}',
                line=dict(color='red', width=2)
            ))
            
            # Add equation as annotation
            if equation:
                fig.add_annotation(
                    x=0.05,
                    y=0.95,
                    xref='paper',
                    yref='paper',
                    text=f"{equation}<br>R² = {metrics['R²']:.4f}",
                    showarrow=False,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='red',
                    borderwidth=1
                )
    
    # Add ML prediction if requested
    if config.get('show_ml'):
        x_sorted, y_pred, metrics = fit_ml_model(x_data, y_data, 'random_forest')
        
        if x_sorted is not None:
            fig.add_trace(go.Scatter(
                x=x_sorted,
                y=y_pred,
                mode='lines',
                name='ML: Random Forest',
                line=dict(color='green', width=2, dash='dash')
            ))
    
    # Update layout
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis_type=config.get('x_scale', 'linear'),
        yaxis_type=config.get('y_scale', 'linear'),
        height=500,
        hovermode='closest',
        showlegend=True
    )
    
    # Set axis ranges if specified
    if config.get('x_min') is not None and config.get('x_max') is not None:
        fig.update_xaxes(range=[config['x_min'], config['x_max']])
    if config.get('y_min') is not None and config.get('y_max') is not None:
        fig.update_yaxes(range=[config['y_min'], config['y_max']])
    
    return fig


def main():
    """Main page function"""
    
    initialize_plot_configs()
    
    # Check if data is loaded
    if not st.session_state.data_loaded or st.session_state.df_long is None:
        st.warning("⚠️ No data loaded yet!")
        st.info("👈 Please go to **Upload & Data Diagnostics** to load your data first")
        
        if st.button("📁 Go to Upload Page"):
            st.switch_page("pages/1_Upload_and_Data_Health.py")
        return
    
    df_long = st.session_state.get("df_clean", st.session_state.df_long)
    variables = sorted(df_long['variable'].unique().tolist())
    
    st.success("✅ Data loaded and ready for relationship analysis")
    
    # Show data info
    col1, col2, col3 = st.columns(3)
    col1.metric("Available Variables", len(variables))
    col2.metric("Data Points", len(df_long))
    col3.metric("Possible Pairs", len(variables) * (len(variables) - 1))
    
    st.markdown("---")
    
    # Number of plots selector
    st.subheader("📊 Configure Analysis Plots")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        num_plots = st.number_input(
            "Number of plots",
            min_value=1,
            max_value=6,  # Recommended maximum for usability
            value=st.session_state.num_plots,
            key="num_plots_input",
            help="Maximum 6 plots recommended for optimal visualization"
        )
        st.session_state.num_plots = num_plots
    
    with col2:
        st.info("""
        **💡 Tips:**
        - Start with 2-3 plots for focused analysis
        - Use combined variables for complex relationships
        - Enable regression/ML for quantitative insights
        """)
    
    st.markdown("---")
    
    # Create tabs for different plot configurations
    if num_plots == 1:
        render_single_plot(df_long, variables, 1)
    else:
        tabs = st.tabs([f"Plot {i+1}" for i in range(num_plots)])
        
        for i, tab in enumerate(tabs):
            with tab:
                render_single_plot(df_long, variables, i+1)
    
    st.markdown("---")
    
    # Correlation matrix section
    render_correlation_matrix(df_long, variables)


def render_single_plot(df_long, variables, plot_id):
    """Render a single relationship plot with all configuration options"""
    
    st.subheader(f"Plot {plot_id} Configuration")
    
    # Create columns for X and Y axis configuration
    col_x, col_y = st.columns(2)
    
    # X-axis configuration
    with col_x:
        st.markdown("#### 📈 X-Axis")
        
        x_mode = st.radio(
            "X-axis mode",
            ["Single Variable", "Combined Variables"],
            key=f"x_mode_{plot_id}"
        )
        
        if x_mode == "Single Variable":
            x_vars = st.selectbox(
                "Select X variable",
                variables,
                key=f"x_var_{plot_id}"
            )
            x_vars_list = [x_vars]
            x_operation = None
        else:
            x_vars_list = st.multiselect(
                "Select X variables",
                variables,
                key=f"x_vars_{plot_id}"
            )
            x_operation = st.selectbox(
                "Operation",
                ["Sum", "Average", "Multiply", "Divide"],
                key=f"x_op_{plot_id}",
                help="Divide only works with exactly 2 variables"
            )
        
        x_scale = st.selectbox(
            "X-axis scale",
            ["linear", "log"],
            key=f"x_scale_{plot_id}"
        )
    
    # Y-axis configuration
    with col_y:
        st.markdown("#### 📉 Y-Axis")
        
        y_mode = st.radio(
            "Y-axis mode",
            ["Single Variable", "Combined Variables"],
            key=f"y_mode_{plot_id}"
        )
        
        if y_mode == "Single Variable":
            y_vars = st.selectbox(
                "Select Y variable",
                variables,
                key=f"y_var_{plot_id}"
            )
            y_vars_list = [y_vars]
            y_operation = None
        else:
            y_vars_list = st.multiselect(
                "Select Y variables",
                variables,
                key=f"y_vars_{plot_id}"
            )
            y_operation = st.selectbox(
                "Operation",
                ["Sum", "Average", "Multiply", "Divide"],
                key=f"y_op_{plot_id}",
                help="Divide only works with exactly 2 variables"
            )
        
        y_scale = st.selectbox(
            "Y-axis scale",
            ["linear", "log"],
            key=f"y_scale_{plot_id}"
        )
    
    # Plot customization options
    st.markdown("#### 🎨 Plot Customization")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        marker_color = st.color_picker(
            "Marker color",
            "#1f77b4",
            key=f"color_{plot_id}"
        )
    
    with col2:
        marker_size = st.slider(
            "Marker size",
            3, 15, 6,
            key=f"size_{plot_id}"
        )
    
    with col3:
        custom_x_range = st.checkbox(
            "Custom X range",
            key=f"custom_x_{plot_id}"
        )
    
    with col4:
        custom_y_range = st.checkbox(
            "Custom Y range",
            key=f"custom_y_{plot_id}"
        )
    
    # Custom ranges
    x_min, x_max, y_min, y_max = None, None, None, None
    
    if custom_x_range or custom_y_range:
        col1, col2, col3, col4 = st.columns(4)
        
        if custom_x_range:
            with col1:
                x_min = st.number_input("X min", key=f"x_min_{plot_id}", value=0.0)
            with col2:
                x_max = st.number_input("X max", key=f"x_max_{plot_id}", value=100.0)
        
        if custom_y_range:
            with col3:
                y_min = st.number_input("Y min", key=f"y_min_{plot_id}", value=0.0)
            with col4:
                y_max = st.number_input("Y max", key=f"y_max_{plot_id}", value=100.0)
    
    # Analysis options
    st.markdown("#### 🔬 Analysis Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_correlation = st.checkbox(
            "Show correlation",
            value=True,
            key=f"show_corr_{plot_id}"
        )
        
        if show_correlation:
            corr_method = st.selectbox(
                "Correlation method",
                ["pearson", "spearman", "kendall"],
                key=f"corr_method_{plot_id}",
                help="Pearson: linear, Spearman: monotonic, Kendall: rank-based"
            )
    
    with col2:
        show_regression = st.checkbox(
            "Show regression fit",
            key=f"show_reg_{plot_id}"
        )
        
        if show_regression:
            regression_type = st.selectbox(
                "Regression type",
                ["linear", "polynomial_2", "polynomial_3", "logarithmic", 
                 "exponential", "power"],
                key=f"reg_type_{plot_id}",
                help="Select the mathematical relationship to fit"
            )
    
    with col3:
        show_ml = st.checkbox(
            "Show ML prediction",
            key=f"show_ml_{plot_id}",
            help="Use Random Forest for non-linear pattern detection"
        )
    
    st.markdown("---")
    
    # Generate plot
    if st.button(f"📊 Generate Plot {plot_id}", type="primary", key=f"generate_{plot_id}"):
        
        # Validate selections
        if (x_mode == "Combined Variables" and not x_vars_list) or \
           (y_mode == "Combined Variables" and not y_vars_list):
            st.error("❌ Please select variables for combined mode")
            return
        
        with st.spinner("Generating analysis..."):
            
            # Get X data
            if x_mode == "Single Variable":
                x_data = df_long[df_long['variable'] == x_vars_list[0]]['value'].values
                x_label = x_vars_list[0]
            else:
                x_data = create_combined_variable(df_long, x_vars_list, x_operation)
                x_label = f"{x_operation}({', '.join(x_vars_list)})"
            
            # Get Y data
            if y_mode == "Single Variable":
                y_data = df_long[df_long['variable'] == y_vars_list[0]]['value'].values
                y_label = y_vars_list[0]
            else:
                y_data = create_combined_variable(df_long, y_vars_list, y_operation)
                y_label = f"{y_operation}({', '.join(y_vars_list)})"
            
            # Ensure same length
            min_len = min(len(x_data), len(y_data))
            x_data = x_data[:min_len]
            y_data = y_data[:min_len]
            
            # Remove NaN values
            mask = ~(np.isnan(x_data) | np.isnan(y_data))
            x_data = x_data[mask]
            y_data = y_data[mask]
            
            if len(x_data) < 3:
                st.error("❌ Not enough valid data points for analysis")
                return
            
            # Create plot configuration
            plot_config = {
                'color': marker_color,
                'marker_size': marker_size,
                'x_scale': x_scale,
                'y_scale': y_scale,
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max,
                'show_regression': show_regression,
                'regression_type': regression_type if show_regression else None,
                'show_ml': show_ml
            }
            
            # Create and display plot
            fig = create_scatter_plot(x_data, y_data, x_label, y_label, plot_config)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display statistics
            st.markdown("#### 📊 Statistical Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            # Basic statistics
            with col1:
                st.markdown("**Descriptive Statistics**")
                st.metric("Data Points", len(x_data))
                st.metric("X Mean", f"{np.mean(x_data):.4f}")
                st.metric("Y Mean", f"{np.mean(y_data):.4f}")
            
            # Correlation
            with col2:
                st.markdown("**Correlation Analysis**")
                if show_correlation:
                    coef, p_value = calculate_correlation(x_data, y_data, corr_method)
                    if coef is not None:
                        st.metric(f"{corr_method.capitalize()} ρ", f"{coef:.4f}")
                        st.metric("p-value", f"{p_value:.4e}")
                        
                        # Interpretation
                        if abs(coef) > 0.7:
                            strength = "Strong"
                        elif abs(coef) > 0.4:
                            strength = "Moderate"
                        else:
                            strength = "Weak"
                        
                        direction = "positive" if coef > 0 else "negative"
                        st.caption(f"{strength} {direction} correlation")
            
            # Regression metrics
            with col3:
                st.markdown("**Regression Metrics**")
                if show_regression:
                    _, _, equation, metrics = fit_regression_model(
                        x_data, y_data, regression_type
                    )
                    if metrics:
                        st.metric("R²", f"{metrics['R²']:.4f}")
                        st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                        st.metric("MAE", f"{metrics['MAE']:.4f}")
                
                if show_ml:
                    _, _, ml_metrics = fit_ml_model(x_data, y_data)
                    if ml_metrics:
                        st.markdown("**ML Metrics (RF)**")
                        st.metric("R²", f"{ml_metrics['R²']:.4f}")
                        st.metric("RMSE", f"{ml_metrics['RMSE']:.4f}")


def render_correlation_matrix(df_long, variables):
    """Render correlation matrix for all variables"""
    
    st.header("🔢 Correlation Matrix")
    
    st.markdown("""
    View correlations between all variables at once. This helps identify:
    - Strong positive correlations (close to +1)
    - Strong negative correlations (close to -1)
    - Independent variables (close to 0)
    """)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        corr_method = st.selectbox(
            "Correlation method",
            ["pearson", "spearman", "kendall"],
            key="matrix_corr_method"
        )
        
        show_values = st.checkbox("Show values", value=True, key="show_matrix_values")
    
    with col2:
        if st.button("📊 Generate Correlation Matrix", type="primary"):
            with st.spinner("Calculating correlations..."):
                
                # Create wide format dataframe
                df_wide = df_long.pivot_table(
                    index='time',
                    columns='variable',
                    values='value'
                )
                
                # Calculate correlation matrix
                if corr_method == 'pearson':
                    corr_matrix = df_wide.corr(method='pearson')
                elif corr_method == 'spearman':
                    corr_matrix = df_wide.corr(method='spearman')
                else:
                    corr_matrix = df_wide.corr(method='kendall')
                
                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.values if show_values else None,
                    texttemplate='%{text:.2f}' if show_values else None,
                    textfont={"size": 10},
                    colorbar=dict(title="Correlation")
                ))
                
                fig.update_layout(
                    title=f"Correlation Matrix ({corr_method.capitalize()})",
                    xaxis_title="Variables",
                    yaxis_title="Variables",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Find strongest correlations
                st.markdown("#### 🔝 Strongest Correlations")
                
                # Get upper triangle (avoid duplicates and diagonal)
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                corr_pairs = corr_matrix.where(mask).stack().reset_index()
                corr_pairs.columns = ['Variable 1', 'Variable 2', 'Correlation']
                corr_pairs = corr_pairs.sort_values('Correlation', key=abs, ascending=False)
                
                # Show top 10
                st.dataframe(
                    corr_pairs.head(10),
                    use_container_width=True,
                    hide_index=True
                )


if __name__ == "__main__":
    main()
