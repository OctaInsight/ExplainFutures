"""
Page 10: Scenario Matrix
Compare scenarios using normalized scores and visual matrix plots
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import io
import re

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Scenario Matrix",
    page_icon="📊",
    layout="wide"
)

# Import and render sidebar (after page config!)
try:
    from core.shared_sidebar import render_app_sidebar
    render_app_sidebar()
except Exception:
    # Fallback - minimal sidebar
    st.sidebar.title("📊 Scenario Matrix")
    st.sidebar.markdown("---")

# Try to import export functions
try:
    from core.viz.export import quick_export_buttons
    EXPORT_AVAILABLE = True
except ImportError:
    EXPORT_AVAILABLE = False

    def quick_export_buttons(fig, filename_prefix="figure", show_formats=None):
        """Fallback export function"""
        st.info("💡 Export functionality requires the core.viz.export module")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📥 PNG", key=f"png_{filename_prefix}"):
                st.info("Export as PNG - functionality will be available when export module is loaded")
        with col2:
            if st.button("📥 PDF", key=f"pdf_{filename_prefix}"):
                st.info("Export as PDF - functionality will be available when export module is loaded")
        with col3:
            if st.button("📥 HTML", key=f"html_{filename_prefix}"):
                html_string = fig.to_html(include_plotlyjs='cdn')
                st.download_button(
                    label="Download HTML",
                    data=html_string,
                    file_name=f"{filename_prefix}.html",
                    mime="text/html"
                )


# =========================
# UTILITY FUNCTIONS
# =========================

def infer_polarity(param_name: str, category: str = "") -> bool:
    """
    Infer if higher is better based on parameter name and category.
    Returns:
        True if higher is better, False if lower is better
    """

    param_lower = (param_name or "").lower()
    category_lower = (category or "").lower()

    negative_keywords = [
        'emission', 'co2', 'carbon', 'pollution', 'waste',
        'fossil', 'coal', 'oil consumption', 'gas consumption',
        'deforestation', 'unemployment', 'poverty', 'inequality',
        'debt', 'cost', 'mortality', 'inflation', 'deficit',
        'risk', 'loss', 'damage', 'accident', 'crime'
    ]

    positive_keywords = [
        'gdp', 'income', 'growth', 'renewable', 'efficiency',
        'employment', 'education', 'health', 'life expectancy',
        'productivity', 'innovation', 'clean energy', 'recycling',
        'solar', 'wind', 'hydro', 'revenue', 'profit', 'surplus',
        'safety', 'literacy', 'coverage', 'access'
    ]

    for keyword in negative_keywords:
        if keyword in param_lower:
            return False

    for keyword in positive_keywords:
        if keyword in param_lower:
            return True

    # Default based on category
    if 'environmental' in category_lower:
        return False
    return True


def _safe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def detect_value_type_from_item(item: dict) -> str:
    """
    Detect whether an extracted scenario value is already a percent change.

    Returns: 'percent' or 'absolute_or_unknown'

    Notes:
    - If your extraction step already sets item['value_type'] or item['unit'],
      this function will respect it.
    - Otherwise, we try a conservative heuristic based on common metadata fields.
    """
    # Trust explicit metadata if present
    vt = (item.get("value_type") or item.get("value_kind") or "").strip().lower()
    if vt in {"percent", "percentage", "%", "pct"}:
        return "percent"
    if vt in {"absolute", "number", "value", "level", "units"}:
        return "absolute_or_unknown"

    unit = (item.get("unit") or "").strip().lower()
    if unit in {"%", "percent", "percentage", "pct"}:
        return "percent"

    # Common flag from extraction pipelines
    if item.get("is_percent") is True:
        return "percent"

    # If original text snippet exists and contains %, treat as percent
    snippet = (item.get("text") or item.get("raw_text") or item.get("source_text") or "")
    if isinstance(snippet, str) and "%" in snippet:
        return "percent"

    return "absolute_or_unknown"


def convert_absolute_to_percent(value_abs: float, baseline: float) -> float | None:
    """
    Convert absolute target level to percent change relative to baseline:
        pct = ((value_abs - baseline) / baseline) * 100

    Returns None if baseline invalid.
    """
    if value_abs is None:
        return None
    if baseline is None or baseline == 0:
        return None
    return ((value_abs - baseline) / baseline) * 100.0


def normalize_parameter(values: list, better_when_higher: bool) -> list:
    """
    Normalize parameter values to -100 to +100 scale across scenarios.

    IMPORTANT:
    - None values are kept as None (not forced to 0).
    - This prevents missingness from biasing category means.
    """

    if not values:
        return []

    valid_values = [v for v in values if v is not None and not np.isnan(v)]
    if len(valid_values) == 0:
        return [None] * len(values)

    min_val = min(valid_values)
    max_val = max(valid_values)

    # If all values equal, map to a constant (still relative)
    if max_val == min_val:
        # Keep interpretability: direction of the constant itself
        if min_val > 0:
            const = 50.0
        elif min_val < 0:
            const = -50.0
        else:
            const = 0.0
        return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else const for v in values]

    normalized = []
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            normalized.append(None)
            continue

        if better_when_higher:
            score = 200 * (v - min_val) / (max_val - min_val) - 100
        else:
            score = 200 * (max_val - v) / (max_val - min_val) - 100
        normalized.append(float(score))

    return normalized


def calculate_category_scores_equal_weights(scenarios: list, polarity_dict: dict) -> dict:
    """
    Calculate -100 to +100 scores for each category in each scenario.
    EQUAL WEIGHT VERSION - All parameters contribute equally.

    Missing values:
    - Excluded from the category mean (no forced zeros).
    - If a scenario has no valid values in a category, category score is NaN.
    """

    # Group parameter values by category and parameter
    params_by_category = {}

    for scenario in scenarios:
        scenario_name = scenario.get("title", "Untitled Scenario")
        for item in scenario.get("items", []):
            category = item.get("category", "Other")
            param = item.get("parameter_canonical", item.get("parameter", ""))

            if not param:
                continue

            params_by_category.setdefault(category, {}).setdefault(param, [])

            value = _safe_float(item.get("value"))
            params_by_category[category][param].append({
                "scenario": scenario_name,
                "value": value
            })

    # Normalize parameter across scenarios, then collect by scenario/category
    normalized_scores = {}

    for category, params in params_by_category.items():
        for param, data_points in params.items():
            values = [d["value"] for d in data_points]
            better_when_higher = polarity_dict.get(param, True)

            scores = normalize_parameter(values, better_when_higher)

            for i, d in enumerate(data_points):
                sname = d["scenario"]
                normalized_scores.setdefault(sname, {}).setdefault(category, [])
                normalized_scores[sname][category].append(scores[i])

    # Category mean (exclude missing)
    category_scores = {}
    all_categories = set(params_by_category.keys())
    all_scenarios = {s.get("title", "Untitled Scenario") for s in scenarios}

    for sname in all_scenarios:
        category_scores[sname] = {}
        for category in all_categories:
            scores = normalized_scores.get(sname, {}).get(category, [])
            valid = [x for x in scores if x is not None and not np.isnan(x)]
            category_scores[sname][category] = float(np.mean(valid)) if valid else np.nan

    return category_scores


def get_parameter_normalized_scores(scenarios: list, polarity_dict: dict) -> dict:
    """
    Get normalized scores for each parameter across scenarios.
    Missing remains None (not 0).

    Returns:
    {
        'category': {
            'parameter': {
                'Scenario 1': normalized_score or None,
                'Scenario 2': normalized_score or None,
            }
        }
    }
    """

    params_by_category = {}

    for scenario in scenarios:
        scenario_name = scenario.get("title", "Untitled Scenario")
        for item in scenario.get("items", []):
            category = item.get("category", "Other")
            param = item.get("parameter_canonical", item.get("parameter", ""))

            if not param:
                continue

            params_by_category.setdefault(category, {}).setdefault(param, [])

            value = _safe_float(item.get("value"))
            params_by_category[category][param].append({
                "scenario": scenario_name,
                "value": value
            })

    result = {}
    for category, params in params_by_category.items():
        result[category] = {}
        for param, data_points in params.items():
            values = [d["value"] for d in data_points]
            better_when_higher = polarity_dict.get(param, True)
            scores = normalize_parameter(values, better_when_higher)

            result[category][param] = {}
            for i, d in enumerate(data_points):
                result[category][param][d["scenario"]] = scores[i]

    return result


def calculate_category_scores_weighted(scenarios: list, polarity_dict: dict, weight_dict: dict) -> dict:
    """
    Weighted category scores.

    NOTE (per your instruction):
    - We do NOT change your weight sign logic.
    - We only fix missing handling: missing values are excluded from weighted sum and weight total.
      (No missing -> category score is NaN.)
    """

    param_scores = get_parameter_normalized_scores(scenarios, polarity_dict)

    all_scenarios = {s.get("title", "Untitled Scenario") for s in scenarios}
    category_scores = {}

    for scenario_name in all_scenarios:
        category_scores[scenario_name] = {}

        for category, params in param_scores.items():
            if category not in weight_dict or not weight_dict[category]:
                category_scores[scenario_name][category] = np.nan
                continue

            weighted_sum = 0.0
            total_weight = 0.0

            for param, scores_by_scenario in params.items():
                if param in weight_dict[category]:
                    weight = float(weight_dict[category][param])
                    score = scores_by_scenario.get(scenario_name, None)

                    # Missing: skip
                    if score is None or (isinstance(score, float) and np.isnan(score)):
                        continue

                    weighted_sum += weight * score
                    total_weight += abs(weight)

            category_scores[scenario_name][category] = (weighted_sum / total_weight) if total_weight > 0 else np.nan

    return category_scores


def extract_scenario_year(scenarios: list) -> str | None:
    if not scenarios:
        return None
    first = scenarios[0]
    if "year" in first:
        return str(first["year"])
    if "target_year" in first:
        return str(first["target_year"])
    title = first.get("title", "")
    m = re.search(r"\b(20\d{2})\b", title)
    return m.group(1) if m else None


# =========================
# MAIN PAGE
# =========================

st.title("📊 Scenario Matrix")
st.markdown("*Compare scenarios using normalized scores across categories*")
st.markdown("---")

# Check if data exists from Page 9
if 'scenario_parameters' not in st.session_state or not st.session_state.scenario_parameters:
    st.warning("⚠️ No scenario data found. Please complete **Page 9: Scenario Parameters Extraction** first.")
    st.info("Go to Page 9 to extract and define scenario parameters.")
    st.stop()

scenarios = st.session_state.scenario_parameters

# === GUIDE ===
with st.expander("📖 **How to Use This Page**", expanded=False):
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
1. **Review Parameters** — Check auto-detected polarity for each parameter
2. **Confirm or Adjust** — Change any incorrect assignments
3. **Validate Value Type** — This page scores **percent changes**. Non-percent values must be converted.
4. **Choose Weighting Method** — Equal weights or custom weighted formula
5. **Calculate Scores** — Normalize all parameters to **-100 to +100** scale
6. **Visualize Matrix** — Compare scenarios on X–Y plot with custom axes
7. **Export Results** — Download plots and data

### Scoring (Interpretation):
- **-100 to +100 scale** for each category
- **+100** = highest relative performance among included scenarios (for that category)
- **-100** = lowest relative performance among included scenarios (for that category)
- **0** = midpoint of the relative scale among included scenarios (not a baseline vs today)
- Scores are **relative between scenarios** (not absolute).
""")

st.markdown("---")

# === STEP 1: PARAMETER POLARITY DETECTION ===
st.subheader("🎯 Step 1: Parameter Polarity Detection")
st.caption("Review and confirm whether higher values are better for each parameter")

all_parameters = {}
for scenario in scenarios:
    for item in scenario.get("items", []):
        param = item.get('parameter_canonical', item.get('parameter', ''))
        if not param:
            continue
        category = item.get('category', 'Other')

        if param not in all_parameters:
            auto_polarity = infer_polarity(param, category)
            all_parameters[param] = {
                'category': category,
                'auto_polarity': auto_polarity,
                'user_polarity': auto_polarity
            }

edit_data = []
for param, info in all_parameters.items():
    edit_data.append({
        'Parameter': param,
        'Category': info['category'],
        'Polarity': 'Positive (↑)' if info['user_polarity'] else 'Negative (↓)'
    })

df_edit = pd.DataFrame(edit_data)

column_config = {
    "Parameter": st.column_config.TextColumn("Parameter Name", width="medium"),
    "Category": st.column_config.TextColumn("Category", width="medium"),
    "Polarity": st.column_config.SelectboxColumn(
        "Polarity (Higher is Better?)",
        options=["Positive (↑)", "Negative (↓)"],
        required=True,
        width="medium"
    )
}

edited_df = st.data_editor(
    df_edit,
    column_config=column_config,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    key="polarity_editor"
)

for _, row in edited_df.iterrows():
    param = row['Parameter']
    if param in all_parameters:
        all_parameters[param]['user_polarity'] = (row['Polarity'] == 'Positive (↑)')

polarity_dict = {param: info['user_polarity'] for param, info in all_parameters.items()}

st.caption(f"**Total parameters:** {len(all_parameters)}")
st.caption(f"**Positive (higher is better):** {sum(1 for p in all_parameters.values() if p['user_polarity'])}")
st.caption(f"**Negative (lower is better):** {sum(1 for p in all_parameters.values() if not p['user_polarity'])}")

if 'polarity_confirmed' not in st.session_state:
    st.session_state.polarity_confirmed = False

if st.button("✅ Confirm All Polarities & Continue", type="primary", use_container_width=True):
    st.session_state.polarity_confirmed = True
    st.session_state.polarity_dict = polarity_dict
    st.success("✅ Polarities confirmed! Proceeding...")
    st.rerun()


# === STEP 1B: ENFORCE PERCENT CHANGE CONSISTENCY (Issue 2 fix) ===
if st.session_state.get('polarity_confirmed', False):
    st.markdown("---")
    st.subheader("🧮 Step 2: Ensure Values Are Percent Changes")
    st.caption("This page scores scenarios using percent changes. If any parameter values are not in percent, convert them using a baseline.")

    # Build parameter value type table (per parameter)
    param_meta = {}
    for scenario in scenarios:
        for item in scenario.get("items", []):
            param = item.get('parameter_canonical', item.get('parameter', ''))
            if not param:
                continue
            category = item.get('category', 'Other')
            vt = detect_value_type_from_item(item)
            param_meta.setdefault(param, {"Category": category, "Detected": set()})
            param_meta[param]["Detected"].add(vt)

    rows = []
    for param, info in param_meta.items():
        detected = "percent" if info["Detected"] == {"percent"} else "mixed_or_non_percent"
        rows.append({
            "Parameter": param,
            "Category": info["Category"],
            "Detected": "Percent (%)" if detected == "percent" else "Non-percent / mixed",
            "Use as": "Percent (%)" if detected == "percent" else "Absolute → Convert to Percent"
        })

    df_vt = pd.DataFrame(rows).sort_values(["Category", "Parameter"])

    vt_editor = st.data_editor(
        df_vt,
        column_config={
            "Parameter": st.column_config.TextColumn(width="large"),
            "Category": st.column_config.TextColumn(width="medium"),
            "Detected": st.column_config.TextColumn(width="medium"),
            "Use as": st.column_config.SelectboxColumn(
                "Use as",
                options=["Percent (%)", "Absolute → Convert to Percent"],
                required=True,
                width="medium"
            )
        },
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="value_type_editor"
    )

    # Baselines input for parameters that need conversion
    needs_baseline = vt_editor[vt_editor["Use as"] == "Absolute → Convert to Percent"]["Parameter"].tolist()
    baseline_dict = st.session_state.get("baseline_dict", {})

    if needs_baseline:
        st.warning("Some parameters are not in percent. Please provide a baseline value for conversion to percent change.")
        st.markdown("**Baseline definition:** percent change is computed as:  \n`((scenario_value - baseline) / baseline) * 100`")

        with st.expander("📌 Provide Baselines (only for parameters requiring conversion)", expanded=True):
            for p in needs_baseline:
                baseline_dict[p] = st.number_input(
                    label=f"Baseline for: {p}",
                    value=float(baseline_dict.get(p, 0.0)),
                    help="Baseline must be non-zero to convert absolute values to percent change.",
                    key=f"baseline_{p}"
                )

        st.session_state.baseline_dict = baseline_dict

    # Apply conversion to scenarios (in-place copy into a new structure, to avoid mutating upstream)
    def build_percent_scenarios(original_scenarios, vt_editor_df, baselines):
        use_map = dict(zip(vt_editor_df["Parameter"], vt_editor_df["Use as"]))
        out = []

        for sc in original_scenarios:
            sc_out = dict(sc)
            sc_out["items"] = []
            for item in sc.get("items", []):
                new_item = dict(item)
                param = item.get('parameter_canonical', item.get('parameter', ''))
                if not param:
                    sc_out["items"].append(new_item)
                    continue

                use_as = use_map.get(param, "Percent (%)")
                v = _safe_float(item.get("value"))

                if use_as == "Percent (%)":
                    # Keep as-is (assume already percent change)
                    new_item["value"] = v
                    new_item["value_type"] = "percent"
                else:
                    # Convert absolute -> percent
                    baseline = _safe_float(baselines.get(param))
                    pct = convert_absolute_to_percent(v, baseline)
                    new_item["value"] = pct
                    new_item["value_type"] = "percent_converted"

                sc_out["items"].append(new_item)

            out.append(sc_out)
        return out

    # Validation: any conversion requested but baseline missing/invalid
    conversion_blocked = False
    if needs_baseline:
        for p in needs_baseline:
            b = _safe_float(baseline_dict.get(p))
            if b is None or b == 0:
                conversion_blocked = True
                break

    if conversion_blocked:
        st.error("Conversion is blocked: at least one required baseline is missing or zero. Please provide valid baselines to continue.")
        st.stop()

    # Use percent-ready scenarios downstream
    scenarios_percent = build_percent_scenarios(scenarios, vt_editor, baseline_dict)
    st.session_state.scenarios_percent = scenarios_percent

    st.success("✅ Values validated. Percent-change dataset is ready for scoring.")


# === STEP 3: WEIGHTING METHOD SELECTION ===
if st.session_state.get('polarity_confirmed', False):
    st.markdown("---")
    st.subheader("⚖️ Step 3: Parameter Weighting Method")
    st.caption("Choose how parameters contribute to each category score")

    st.info("""
**Important Note:** You have two options:
1. **Equal Weights** — All parameters contribute equally (simple average)
2. **Custom Weights** — Define specific weights for each parameter in each category (weighted formula)
""")

    weighting_method = st.radio(
        "Select weighting method:",
        options=["Equal Weights (Simple Average)", "Custom Weights (Weighted Formula)"],
        index=0,
        key="weighting_method",
        help="Choose how parameters should contribute to category scores"
    )

    st.session_state.weighting_method = weighting_method

    scenarios_for_scoring = st.session_state.get("scenarios_percent", scenarios)

    if weighting_method == "Equal Weights (Simple Average)":
        st.markdown("---")
        st.markdown("**📊 Equal Weight Calculation**")
        st.latex(r"\text{Category Score} = \frac{1}{n} \sum_{i=1}^{n} \text{Parameter}_i")
        st.caption("Where each parameter contributes equally to the category score (missing values excluded).")

        if st.button("Calculate Category Scores (Equal Weights)", type="primary", use_container_width=True):
            with st.spinner("Calculating scores with equal weights..."):
                category_scores = calculate_category_scores_equal_weights(
                    scenarios_for_scoring,
                    st.session_state.polarity_dict
                )
                st.session_state.category_scores = category_scores
                st.session_state.calculation_method = "equal_weights"
                st.success("✅ Scores calculated successfully!")
                st.rerun()

    else:
        st.markdown("---")
        st.markdown("**⚖️ Weighted Formula Calculation**")
        st.latex(r"\text{Category Score} = \frac{\sum_{i=1}^{n} (w_i \times \text{Parameter}_i)}{\sum_{i=1}^{n} |w_i|}")
        st.caption("Where wᵢ is the weight for parameter i (missing values excluded from both numerator and denominator).")

        param_scores = get_parameter_normalized_scores(
            scenarios_for_scoring,
            st.session_state.polarity_dict
        )

        all_categories = sorted(param_scores.keys())

        if 'weight_dict' not in st.session_state:
            st.session_state.weight_dict = {}
            for category in all_categories:
                st.session_state.weight_dict[category] = {}
                for param in param_scores[category].keys():
                    st.session_state.weight_dict[category][param] = 1.0

        category_tabs = st.tabs(all_categories)

        for idx, category in enumerate(all_categories):
            with category_tabs[idx]:
                st.markdown(f"**Category: {category}**")

                params_in_category = sorted(param_scores[category].keys())

                weight_data = []
                for param in params_in_category:
                    current_weight = st.session_state.weight_dict[category].get(param, 1.0)
                    direction = "↑ Positive" if st.session_state.polarity_dict.get(param, True) else "↓ Negative"
                    weight_data.append({
                        'Parameter': param,
                        'Direction': direction,
                        'Weight': current_weight
                    })

                df_weights = pd.DataFrame(weight_data)

                weight_column_config = {
                    "Parameter": st.column_config.TextColumn("Parameter Name", width="large"),
                    "Direction": st.column_config.TextColumn("Direction", width="small"),
                    "Weight": st.column_config.NumberColumn(
                        "Weight (wᵢ)",
                        min_value=-10.0,
                        max_value=10.0,
                        step=0.1,
                        format="%.2f",
                        width="small"
                    )
                }

                edited_weights = st.data_editor(
                    df_weights,
                    column_config=weight_column_config,
                    use_container_width=True,
                    hide_index=True,
                    num_rows="fixed",
                    key=f"weights_editor_{category}"
                )

                for _, row in edited_weights.iterrows():
                    p = row['Parameter']
                    w = row['Weight']
                    st.session_state.weight_dict[category][p] = w

                st.markdown("**Formula preview (this category):**")
                formula_parts = []
                for _, row in edited_weights.iterrows():
                    pname = row['Parameter'][:20] + "..." if len(row['Parameter']) > 20 else row['Parameter']
                    w = row['Weight']
                    formula_parts.append(f"{w:.2f} × {pname}" if w >= 0 else f"({w:.2f}) × {pname}")
                st.code(f"{category} Score = ({' + '.join(formula_parts)}) / Σ|wᵢ|", language="text")

        st.markdown("---")
        if st.button("Calculate Category Scores (Weighted Formula)", type="primary", use_container_width=True):
            with st.spinner("Calculating scores with custom weights..."):
                category_scores = calculate_category_scores_weighted(
                    scenarios_for_scoring,
                    st.session_state.polarity_dict,
                    st.session_state.weight_dict
                )
                st.session_state.category_scores = category_scores
                st.session_state.calculation_method = "weighted"
                st.success("✅ Weighted scores calculated successfully!")
                st.rerun()


# === STEP 4: DISPLAY RESULTS AND VISUALIZATION ===
if st.session_state.get('category_scores') is not None:
    st.markdown("---")
    st.subheader("📊 Step 4: Category Scores (-100 to +100)")

    if st.session_state.get('calculation_method') == 'equal_weights':
        st.caption("✅ Calculated using **Equal Weights** (missing values excluded).")
    else:
        st.caption("✅ Calculated using **Custom Weighted Formula** (missing values excluded).")

    category_scores = st.session_state.category_scores

    summary_data = []
    for scenario_name, scores in category_scores.items():
        row = {'Scenario': scenario_name}
        row.update(scores)
        summary_data.append(row)

    df_scores = pd.DataFrame(summary_data)

    with st.expander("📊 **Category Scores Table**", expanded=True):
        display_df = df_scores.copy()

        # Format: keep NaN visible as blank (so missingness is clear)
        for col in display_df.columns:
            if col != 'Scenario':
                display_df[col] = display_df[col].apply(lambda x: "" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.1f}")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

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

    # === X-Y COMPARISON PLOT ===
    st.markdown("---")
    st.subheader("📈 Step 5: Scenario Comparison Plot")
    st.caption("Compare scenarios on a cross-axis plot with custom category selection")

    all_categories = [c for c in df_scores.columns if c != 'Scenario']

    if len(all_categories) < 2:
        st.warning("⚠️ Need at least 2 categories to create comparison plot")
    else:
        col_x, col_y = st.columns(2)
        with col_x:
            x_category = st.selectbox("Select X-axis category", options=all_categories, index=0, key='x_axis_cat')
        with col_y:
            y_category = st.selectbox("Select Y-axis category", options=all_categories,
                                      index=min(1, len(all_categories) - 1), key='y_axis_cat')

        with st.expander("🎨 **Plot Customization**"):
            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1:
                box_opacity = st.slider("Box Transparency", 0.0, 1.0, 0.6, 0.1, key='box_opacity')
            with col_c2:
                error_margin = st.slider("Error Margin (%)", 0, 50, 10, 5, key='error_margin')
            with col_c3:
                show_legend = st.checkbox("Show Legend", value=True, key='show_legend')

            st.markdown("**Scenario Colors:**")
            default_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']

            scenario_colors = {}
            scenario_list = list(category_scores.keys())
            for idx, scenario_name in enumerate(scenario_list):
                scenario_colors[scenario_name] = default_colors[idx % len(default_colors)]

            cols_per_row = 4
            for row_start in range(0, len(scenario_list), cols_per_row):
                cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    sidx = row_start + col_idx
                    if sidx < len(scenario_list):
                        sname = scenario_list[sidx]
                        with cols[col_idx]:
                            scenario_colors[sname] = st.color_picker(
                                f"{sname[:15]}...",
                                value=scenario_colors[sname],
                                key=f'color_picker_{sidx}'
                            )

        # Create plot
        fig = go.Figure()

        # Cross at origin
        fig.add_shape(type="line", x0=-110, y0=0, x1=110, y1=0,
                      line=dict(color="rgba(255, 255, 255, 0.5)", width=2, dash="dash"),
                      layer="below")
        fig.add_shape(type="line", x0=0, y0=-110, x1=0, y1=110,
                      line=dict(color="rgba(255, 255, 255, 0.5)", width=2, dash="dash"),
                      layer="below")

        # Add scenarios
        for scenario_name, scores in category_scores.items():
            x_raw = scores.get(x_category, np.nan)
            y_raw = scores.get(y_category, np.nan)

            # If missing category score, plot at 0 but label as missing in hover
            x_missing = (x_raw is None) or (isinstance(x_raw, float) and np.isnan(x_raw))
            y_missing = (y_raw is None) or (isinstance(y_raw, float) and np.isnan(y_raw))

            x_val = 0.0 if x_missing else float(x_raw)
            y_val = 0.0 if y_missing else float(y_raw)

            x_error = (error_margin / 100) * abs(x_val) if x_val != 0 else 5
            y_error = (error_margin / 100) * abs(y_val) if y_val != 0 else 5

            color = scenario_colors.get(scenario_name, '#636EFA')

            x0 = max(-110, x_val - x_error)
            y0 = max(-110, y_val - y_error)
            x1 = min(110, x_val + x_error)
            y1 = min(110, y_val + y_error)

            fig.add_shape(
                type="rect",
                x0=x0, y0=y0,
                x1=x1, y1=y1,
                fillcolor=color,
                opacity=box_opacity,
                line=dict(color=color, width=2),
                layer="above",
                name=scenario_name
            )

            x_text = "MISSING" if x_missing else f"{x_val:.1f}"
            y_text = "MISSING" if y_missing else f"{y_val:.1f}"

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
                    f"<b>{scenario_name}</b><br>"
                    f"{x_category}: {x_text}<br>"
                    f"{y_category}: {y_text}<br>"
                    "<extra></extra>"
                )
            ))

        scenario_year = extract_scenario_year(scenarios)
        plot_title = f"Scenario Comparison: {x_category} vs {y_category}" + (f" (Projected {scenario_year})" if scenario_year else "")

        fig.update_layout(
            title=dict(text=plot_title, font=dict(color='white', size=18), x=0.5, xanchor='center'),
            xaxis_title=dict(text=f"{x_category} Score", font=dict(color='white', size=14), standoff=20),
            yaxis_title=dict(text=f"{y_category} Score", font=dict(color='white', size=14), standoff=20),
            xaxis=dict(
                range=[-110, 110],
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(255, 255, 255, 0.2)',
                showticklabels=False,
                tickvals=[],
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='rgba(255, 255, 255, 0.7)',
                color='white',
                showline=True,
                linewidth=2,
                linecolor='rgba(255, 255, 255, 0.5)',
                mirror=True
            ),
            yaxis=dict(
                range=[-110, 110],
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(255, 255, 255, 0.2)',
                showticklabels=False,
                tickvals=[],
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='rgba(255, 255, 255, 0.7)',
                color='white',
                showline=True,
                linewidth=2,
                linecolor='rgba(255, 255, 255, 0.5)',
                mirror=True
            ),
            plot_bgcolor='#0E1117',
            paper_bgcolor='#0E1117',
            font=dict(color='white'),
            hovermode='closest',
            width=800,
            height=800,
            showlegend=show_legend,
            legend=dict(
                font=dict(color='white'),
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='rgba(255,255,255,0.3)',
                borderwidth=1
            ),
            margin=dict(l=80, r=40, t=100, b=80)
        )

        # Quadrant labels (kept exactly as your design)
        fig.add_annotation(x=55, y=55, text="+ / +", showarrow=False,
                           font=dict(size=14, color="white", family="monospace"),
                           bgcolor='rgba(0,100,0,0.3)', bordercolor='rgba(255,255,255,0.5)',
                           borderwidth=1, borderpad=6)
        fig.add_annotation(x=-55, y=55, text="- / +", showarrow=False,
                           font=dict(size=14, color="white", family="monospace"),
                           bgcolor='rgba(128,128,0,0.3)', bordercolor='rgba(255,255,255,0.5)',
                           borderwidth=1, borderpad=6)
        fig.add_annotation(x=-55, y=-55, text="- / -", showarrow=False,
                           font=dict(size=14, color="white", family="monospace"),
                           bgcolor='rgba(100,0,0,0.3)', bordercolor='rgba(255,255,255,0.5)',
                           borderwidth=1, borderpad=6)
        fig.add_annotation(x=55, y=-55, text="+ / -", showarrow=False,
                           font=dict(size=14, color="white", family="monospace"),
                           bgcolor='rgba(128,128,0,0.3)', bordercolor='rgba(255,255,255,0.5)',
                           borderwidth=1, borderpad=6)

        # Export version with white background (kept exactly as your design)
        export_fig = go.Figure(fig)
        export_fig.update_layout(
            title=dict(font=dict(color='black', size=18)),
            xaxis_title=dict(font=dict(color='black', size=14)),
            yaxis_title=dict(font=dict(color='black', size=14)),
            xaxis=dict(range=[-110, 110], gridcolor='rgba(0, 0, 0, 0.1)', color='black',
                       linecolor='black', zeroline=True, zerolinewidth=2, zerolinecolor='rgba(0, 0, 0, 0.5)'),
            yaxis=dict(range=[-110, 110], gridcolor='rgba(0, 0, 0, 0.1)', color='black',
                       linecolor='black', zeroline=True, zerolinewidth=2, zerolinecolor='rgba(0, 0, 0, 0.5)'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
            legend=dict(font=dict(color='black'), bgcolor='rgba(255,255,255,0.9)', bordercolor='black', borderwidth=1)
        )

        # Update cross lines for export
        for shape in export_fig.layout.shapes[:2]:
            shape.line.color = 'rgba(0, 0, 0, 0.3)'
            shape.line.width = 2

        # Update quadrant annotations for export
        for annotation in export_fig.layout.annotations:
            annotation.font.color = 'black'
            if annotation.text == "+ / +":
                annotation.bgcolor = 'rgba(0,150,0,0.15)'
            elif annotation.text == "- / -":
                annotation.bgcolor = 'rgba(150,0,0,0.15)'
            else:
                annotation.bgcolor = 'rgba(150,150,0,0.15)'
            annotation.bordercolor = 'black'

        if 'export_figures' not in st.session_state:
            st.session_state.export_figures = {}
        st.session_state.export_figures['scenario_matrix'] = export_fig

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        with st.expander("💾 Export Plot", expanded=False):
            quick_export_buttons(
                export_fig,
                filename_prefix=f"scenario_matrix_{x_category}_vs_{y_category}",
                show_formats=['png', 'pdf', 'html']
            )

        st.markdown("**📥 Export Data:**")
        plot_data = []
        for scenario_name, scores in category_scores.items():
            x_raw = scores.get(x_category, np.nan)
            y_raw = scores.get(y_category, np.nan)
            plot_data.append({
                'Scenario': scenario_name,
                x_category: x_raw,
                y_category: y_raw
            })
        df_plot = pd.DataFrame(plot_data)
        csv_plot = df_plot.to_csv(index=False)
        st.download_button(
            label="📥 Download Plot Data (CSV)",
            data=csv_plot,
            file_name=f"scenario_matrix_data_{x_category}_vs_{y_category}.csv",
            mime="text/csv",
            use_container_width=False
        )

st.markdown("---")
st.caption("💡 Tip: Use different category combinations to explore various scenario comparisons!")
