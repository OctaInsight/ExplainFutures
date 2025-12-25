"""
Page 10: Scenario Matrix
Compare scenarios using normalized scores and visual matrix plots
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Scenario Matrix",
    page_icon="📊",
    layout="wide"
)

# Import sidebar (after page config!)
try:
    from shared_sidebar import render_app_sidebar
    render_app_sidebar()
except:
    # Fallback if import fails
    st.sidebar.title("📊 Scenario Matrix")


# ====== UTILITY FUNCTIONS ======

def infer_polarity(param_name: str, category: str = "") -> bool:
    """
    Infer if higher is better based on parameter name and category
    
    Returns:
        True if higher is better, False if lower is better
    """
    
    param_lower = param_name.lower()
    category_lower = category.lower()
    
    # Keywords where LOWER is better (negative indicators)
    negative_keywords = [
        'emission', 'co2', 'carbon', 'pollution', 'waste',
        'fossil', 'coal', 'oil consumption', 'gas consumption',
        'deforestation', 'unemployment', 'poverty', 'inequality',
        'debt', 'cost', 'mortality', 'inflation', 'deficit',
        'risk', 'loss', 'damage', 'accident', 'crime'
    ]
    
    # Keywords where HIGHER is better (positive indicators)
    positive_keywords = [
        'gdp', 'income', 'growth', 'renewable', 'efficiency',
        'employment', 'education', 'health', 'life expectancy',
        'productivity', 'innovation', 'clean energy', 'recycling',
        'solar', 'wind', 'hydro', 'revenue', 'profit', 'surplus',
        'safety', 'literacy', 'coverage', 'access'
    ]
    
    # Check negative keywords first
    for keyword in negative_keywords:
        if keyword in param_lower:
            return False  # Lower is better
    
    # Check positive keywords
    for keyword in positive_keywords:
        if keyword in param_lower:
            return True  # Higher is better
    
    # Default based on category
    if 'environmental' in category_lower:
        return False  # Assume reduction is good
    else:
        return True  # Assume growth is good


def normalize_parameter(values: list, better_when_higher: bool) -> list:
    """
    Normalize parameter values to 0-100 scale
    
    Parameters:
    -----------
    values : list of float
        Raw values from different scenarios
    better_when_higher : bool
        True if higher values are better, False if lower is better
    
    Returns:
    --------
    list of float
        Normalized scores (0-100)
    """
    
    if not values or len(values) == 0:
        return []
    
    # Filter out None values
    valid_values = [v for v in values if v is not None]
    
    if len(valid_values) == 0:
        return [50] * len(values)  # All None, return neutral
    
    min_val = min(valid_values)
    max_val = max(valid_values)
    
    # All values equal
    if max_val == min_val:
        return [50] * len(values)
    
    normalized = []
    
    for v in values:
        if v is None:
            normalized.append(50)  # Neutral for None
        elif better_when_higher:
            # Higher is better: max gets 100, min gets 0
            score = 100 * (v - min_val) / (max_val - min_val)
            normalized.append(score)
        else:
            # Lower is better: min gets 100, max gets 0
            score = 100 * (max_val - v) / (max_val - min_val)
            normalized.append(score)
    
    return normalized


def calculate_category_scores(scenarios: list, polarity_dict: dict) -> dict:
    """
    Calculate 0-100 scores for each category in each scenario
    
    Returns:
    {
        'Scenario 1': {'Economic': 85.5, 'Environmental': 60.2, ...},
        'Scenario 2': {'Economic': 45.2, 'Environmental': 90.7, ...},
    }
    """
    
    # Group parameters by category
    params_by_category = {}
    
    for scenario in scenarios:
        for item in scenario['items']:
            category = item.get('category', 'Other')
            param = item.get('parameter', '')
            
            if category not in params_by_category:
                params_by_category[category] = {}
            
            if param not in params_by_category[category]:
                params_by_category[category][param] = []
            
            params_by_category[category][param].append({
                'scenario': scenario['title'],
                'value': item.get('value'),
                'direction': item.get('direction', 'target')
            })
    
    # Normalize each parameter
    normalized_scores = {}
    
    for category, params in params_by_category.items():
        for param, data_points in params.items():
            values = [d['value'] for d in data_points]
            
            # Get polarity
            better_when_higher = polarity_dict.get(param, True)
            
            # Normalize
            scores = normalize_parameter(values, better_when_higher)
            
            # Store scores
            for i, d in enumerate(data_points):
                scenario_name = d['scenario']
                
                if scenario_name not in normalized_scores:
                    normalized_scores[scenario_name] = {}
                
                if category not in normalized_scores[scenario_name]:
                    normalized_scores[scenario_name][category] = []
                
                normalized_scores[scenario_name][category].append(scores[i])
    
    # Average scores within each category
    category_scores = {}
    
    for scenario_name, categories in normalized_scores.items():
        category_scores[scenario_name] = {}
        
        for category, scores in categories.items():
            # Filter out None values
            valid_scores = [s for s in scores if s is not None]
            if valid_scores:
                category_scores[scenario_name][category] = np.mean(valid_scores)
            else:
                category_scores[scenario_name][category] = 50  # Neutral
    
    return category_scores


# ====== MAIN PAGE ======

st.title("📊 Scenario Matrix")
st.markdown("*Compare scenarios using normalized scores across categories*")
st.markdown("---")

# Check if data exists from Page 9
if 'scenario_parameters' not in st.session_state or not st.session_state.scenario_parameters:
    st.warning("⚠️ No scenario data found. Please complete **Page 9: Scenario Parameters Extraction** first.")
    st.info("Go to Page 9 to extract and define scenario parameters.")
    st.stop()

scenarios = st.session_state.scenario_parameters

# === STEP 1: GUIDE ===
with st.expander("📖 **How to Use This Page**", expanded=True):
    st.markdown("""
    ### Understanding Parameter Polarity
    
    Each parameter can be **positive** (higher is better) or **negative** (lower is better):
    
    **Examples of POSITIVE parameters (higher is better):**
    - GDP Growth, Renewable Energy, Employment, Income, Efficiency
    - Innovation, Education, Health Coverage, Life Expectancy
    
    **Examples of NEGATIVE parameters (lower is better):**
    - CO2 Emissions, Pollution, Unemployment, Poverty, Debt
    - Fossil Fuel Use, Waste, Mortality, Crime, Costs
    
    ### Workflow:
    1. **Review Parameters** - Check auto-detected polarity for each parameter
    2. **Confirm or Adjust** - Change any incorrect assignments
    3. **Calculate Scores** - Normalize all parameters to 0-100 scale
    4. **Visualize Matrix** - Compare scenarios on X-Y plot with custom axes
    5. **Export Results** - Download plots and data
    
    ### Scoring:
    - **0-100 scale** for each category
    - **100** = Best performance in that category
    - **0** = Worst performance in that category
    - Scores are **relative** between scenarios (not absolute)
    """)

st.markdown("---")

# === STEP 2: PARAMETER POLARITY DETECTION ===
st.subheader("🎯 Step 1: Parameter Polarity Detection")
st.caption("Review and confirm whether higher values are better for each parameter")

# Collect all unique parameters
all_parameters = {}
for scenario in scenarios:
    for item in scenario['items']:
        param = item.get('parameter', '')
        category = item.get('category', 'Other')
        
        if param not in all_parameters:
            # Auto-detect polarity
            auto_polarity = infer_polarity(param, category)
            
            all_parameters[param] = {
                'category': category,
                'auto_polarity': auto_polarity,
                'user_polarity': auto_polarity  # Initialize with auto-detected
            }

# Create polarity assignment table
st.markdown("**Review parameter classifications:**")

polarity_data = []
for param, info in all_parameters.items():
    polarity_data.append({
        'Parameter': param,
        'Category': info['category'],
        'Higher is Better?': 'Yes ✅' if info['auto_polarity'] else 'No ❌',
        'Suggested': 'Positive (↑)' if info['auto_polarity'] else 'Negative (↓)'
    })

df_polarity = pd.DataFrame(polarity_data)

# Display in expandable table
with st.expander("📋 **Parameter Polarity Table**", expanded=True):
    st.dataframe(df_polarity, use_container_width=True, hide_index=True)
    
    st.caption(f"**Total parameters:** {len(all_parameters)}")
    st.caption(f"**Positive (higher is better):** {sum([1 for p in all_parameters.values() if p['auto_polarity']])}")
    st.caption(f"**Negative (lower is better):** {sum([1 for p in all_parameters.values() if not p['auto_polarity']])}")

# Manual adjustment interface
st.markdown("**Adjust any incorrect classifications:**")

col1, col2 = st.columns([3, 1])

with col1:
    param_to_change = st.selectbox(
        "Select parameter to change",
        options=['-- Select --'] + list(all_parameters.keys()),
        key='param_selector'
    )

with col2:
    if param_to_change != '-- Select --':
        current = all_parameters[param_to_change]['user_polarity']
        new_polarity = st.selectbox(
            "Set polarity",
            options=['Positive (↑ better)', 'Negative (↓ better)'],
            index=0 if current else 1,
            key='polarity_selector'
        )
        
        if st.button("✅ Update", key='update_polarity'):
            all_parameters[param_to_change]['user_polarity'] = (new_polarity == 'Positive (↑ better)')
            st.success(f"Updated {param_to_change}")
            st.rerun()

# Store polarity in session state
polarity_dict = {param: info['user_polarity'] for param, info in all_parameters.items()}

if 'polarity_confirmed' not in st.session_state:
    st.session_state.polarity_confirmed = False

if st.button("✅ Confirm All Polarities & Calculate Scores", type="primary", use_container_width=True):
    st.session_state.polarity_confirmed = True
    st.session_state.polarity_dict = polarity_dict
    st.success("✅ Polarities confirmed! Calculating scores...")
    st.rerun()

# === STEP 3: CALCULATE SCORES ===
if st.session_state.get('polarity_confirmed', False):
    st.markdown("---")
    st.subheader("📊 Step 2: Category Scores (0-100)")
    st.caption("Normalized scores for each category across scenarios")
    
    # Calculate scores
    category_scores = calculate_category_scores(scenarios, st.session_state.polarity_dict)
    
    # Store in session state
    st.session_state.category_scores = category_scores
    
    # Create summary table
    summary_data = []
    for scenario_name, scores in category_scores.items():
        row = {'Scenario': scenario_name}
        row.update(scores)
        summary_data.append(row)
    
    df_scores = pd.DataFrame(summary_data)
    
    # Display scores table
    with st.expander("📊 **Category Scores Table**", expanded=True):
        # Format numbers
        display_df = df_scores.copy()
        for col in display_df.columns:
            if col != 'Scenario':
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Download scores
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            csv_data = df_scores.to_csv(index=False)
            st.download_button(
                label="📥 Download Scores (CSV)",
                data=csv_data,
                file_name=f"scenario_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_d2:
            excel_buffer = io.BytesIO()
            df_scores.to_excel(excel_buffer, index=False, engine='openpyxl')
            st.download_button(
                label="📥 Download Scores (Excel)",
                data=excel_buffer.getvalue(),
                file_name=f"scenario_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    # === STEP 4: X-Y COMPARISON PLOT ===
    st.markdown("---")
    st.subheader("📈 Step 3: Scenario Comparison Plot")
    st.caption("Compare scenarios on a cross-axis plot with custom category selection")
    
    # Get available categories
    all_categories = list(df_scores.columns)
    all_categories.remove('Scenario')
    
    if len(all_categories) < 2:
        st.warning("⚠️ Need at least 2 categories to create comparison plot")
    else:
        # Category selection
        col_x, col_y = st.columns(2)
        
        with col_x:
            x_category = st.selectbox(
                "Select X-axis category",
                options=all_categories,
                index=0,
                key='x_axis_cat'
            )
        
        with col_y:
            y_category = st.selectbox(
                "Select Y-axis category",
                options=all_categories,
                index=min(1, len(all_categories)-1),
                key='y_axis_cat'
            )
        
        # Plot customization
        with st.expander("🎨 **Plot Customization**"):
            col_c1, col_c2, col_c3 = st.columns(3)
            
            with col_c1:
                box_opacity = st.slider("Box Transparency", 0.0, 1.0, 0.6, 0.1, key='box_opacity')
            
            with col_c2:
                error_margin = st.slider("Error Margin (%)", 0, 50, 10, 5, key='error_margin')
            
            with col_c3:
                show_legend = st.checkbox("Show Legend", value=True, key='show_legend')
            
            # Color selection for each scenario
            st.markdown("**Scenario Colors:**")
            
            default_colors = px.colors.qualitative.Set2
            scenario_colors = {}
            
            cols = st.columns(min(len(category_scores), 4))
            for idx, scenario_name in enumerate(category_scores.keys()):
                with cols[idx % 4]:
                    color = st.color_picker(
                        scenario_name[:20],
                        default_colors[idx % len(default_colors)],
                        key=f'color_{idx}'
                    )
                    scenario_colors[scenario_name] = color
        
        # Create plot
        fig = go.Figure()
        
        # Add cross at origin (0, 0) centered at (50, 50)
        fig.add_shape(
            type="line",
            x0=0, y0=50, x1=100, y1=50,
            line=dict(color="gray", width=1, dash="dash"),
            layer="below"
        )
        fig.add_shape(
            type="line",
            x0=50, y0=0, x1=50, y1=100,
            line=dict(color="gray", width=1, dash="dash"),
            layer="below"
        )
        
        # Add each scenario as a box
        for scenario_name, scores in category_scores.items():
            x_val = scores.get(x_category, 50)
            y_val = scores.get(y_category, 50)
            
            # Calculate error ranges
            x_error = (error_margin / 100) * x_val
            y_error = (error_margin / 100) * y_val
            
            color = scenario_colors.get(scenario_name, '#636EFA')
            
            # Add box as a rectangle
            fig.add_shape(
                type="rect",
                x0=x_val - x_error, y0=y_val - y_error,
                x1=x_val + x_error, y1=y_val + y_error,
                fillcolor=color,
                opacity=box_opacity,
                line=dict(color=color, width=2),
                layer="above",
                name=scenario_name
            )
            
            # Add center point
            fig.add_trace(go.Scatter(
                x=[x_val],
                y=[y_val],
                mode='markers+text',
                marker=dict(size=10, color=color, symbol='circle'),
                text=[scenario_name],
                textposition="top center",
                name=scenario_name,
                showlegend=show_legend,
                hovertemplate=(
                    f"<b>{scenario_name}</b><br>" +
                    f"{x_category}: {x_val:.1f}<br>" +
                    f"{y_category}: {y_val:.1f}<br>" +
                    "<extra></extra>"
                )
            ))
        
        # Update layout
        fig.update_layout(
            title=f"Scenario Comparison: {x_category} vs {y_category}",
            xaxis_title=f"{x_category} Score (0-100)",
            yaxis_title=f"{y_category} Score (0-100)",
            xaxis=dict(range=[0, 100], showgrid=True, gridwidth=0.5, gridcolor='lightgray'),
            yaxis=dict(range=[0, 100], showgrid=True, gridwidth=0.5, gridcolor='lightgray'),
            plot_bgcolor='white',
            hovermode='closest',
            width=800,
            height=800,
            showlegend=show_legend
        )
        
        # Add quadrant labels
        fig.add_annotation(x=75, y=75, text="High-High", showarrow=False, font=dict(size=12, color="gray"))
        fig.add_annotation(x=25, y=75, text="Low-High", showarrow=False, font=dict(size=12, color="gray"))
        fig.add_annotation(x=75, y=25, text="High-Low", showarrow=False, font=dict(size=12, color="gray"))
        fig.add_annotation(x=25, y=25, text="Low-Low", showarrow=False, font=dict(size=12, color="gray"))
        
        # Display plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Export buttons
        st.markdown("**📥 Export Plot:**")
        
        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        
        with col_e1:
            try:
                img_bytes = fig.to_image(format="png", width=1200, height=1200)
                st.download_button(
                    label="📥 PNG",
                    data=img_bytes,
                    file_name=f"scenario_matrix_{x_category}_vs_{y_category}.png",
                    mime="image/png",
                    use_container_width=True
                )
            except:
                st.caption("⚠️ Install kaleido for PNG export")
        
        with col_e2:
            try:
                pdf_bytes = fig.to_image(format="pdf", width=1200, height=1200)
                st.download_button(
                    label="📥 PDF",
                    data=pdf_bytes,
                    file_name=f"scenario_matrix_{x_category}_vs_{y_category}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except:
                st.caption("⚠️ Install kaleido for PDF export")
        
        with col_e3:
            html_buffer = io.StringIO()
            fig.write_html(html_buffer)
            st.download_button(
                label="📥 HTML",
                data=html_buffer.getvalue(),
                file_name=f"scenario_matrix_{x_category}_vs_{y_category}.html",
                mime="text/html",
                use_container_width=True
            )
        
        with col_e4:
            # Export plot data
            plot_data = []
            for scenario_name, scores in category_scores.items():
                plot_data.append({
                    'Scenario': scenario_name,
                    x_category: scores.get(x_category, 50),
                    y_category: scores.get(y_category, 50)
                })
            df_plot = pd.DataFrame(plot_data)
            csv_plot = df_plot.to_csv(index=False)
            st.download_button(
                label="📥 Data",
                data=csv_plot,
                file_name=f"scenario_matrix_data_{x_category}_vs_{y_category}.csv",
                mime="text/csv",
                use_container_width=True
            )

st.markdown("---")
st.caption("💡 Tip: Use different category combinations to explore various scenario comparisons!")
