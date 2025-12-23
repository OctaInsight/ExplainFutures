# ExplainFutures - Phase 1 MVP

## Overview

ExplainFutures is a modular Python/Streamlit application for time-series data analysis, visualization, and future exploration. This is **Phase 1**, which provides the foundational capabilities for data ingestion, health checking, and interactive visualization.

## Phase 1 Features

### ✅ Implemented
1. **Data Upload & Parsing**
   - CSV, TXT, Excel file support
   - Automatic delimiter detection
   - Time column identification and parsing
   - Variable selection and type assignment
   - Conversion to standardized long format

2. **Data Health Report**
   - Missing values analysis
   - Duplicate detection
   - Time coverage assessment
   - Sampling frequency estimation
   - Basic outlier detection

3. **Interactive Visualization**
   - Single variable time plots
   - Multi-variable plots with independent Y-axes
   - Customizable colors and scales
   - Interactive Plotly charts

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup Steps

1. **Clone or extract the project**
   ```bash
   cd explainfutures
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access in browser**
   - The app should automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in the terminal

## Project Structure

```
explainfutures/
├── app.py                          # Main application entry point
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── core/                           # Core functionality modules
│   ├── __init__.py
│   ├── config.py                   # Configuration and session state
│   ├── utils.py                    # Utility functions
│   │
│   ├── io/                         # Input/Output modules
│   │   ├── __init__.py
│   │   ├── loaders.py              # File loading functions
│   │   └── validators.py           # Data validation functions
│   │
│   ├── preprocess/                 # Preprocessing modules (Phase 1: minimal)
│   │   └── __init__.py
│   │
│   └── viz/                        # Visualization modules
│       └── __init__.py
│
├── pages/                          # Streamlit pages (multi-page app)
│   ├── __init__.py
│   ├── 1_Upload_and_Data_Health.py # Data upload and health check
│   └── 2_Explore_and_Visualize.py  # Visualization interface
│
├── db/                             # Database modules (for Phase 4)
│   └── __init__.py
│
├── temp/                           # Temporary files (created at runtime)
└── exports/                        # Exported files (created at runtime)
```

## Usage Guide

### Step 1: Upload Data

1. Navigate to **"Upload & Data Health"** page in the sidebar
2. Click "Browse files" and select your data file (CSV, TXT, or Excel)
3. For Excel files, select the sheet to load
4. For CSV/TXT, confirm or adjust the delimiter

### Step 2: Configure Data

1. **Select Time Column**: Choose which column contains timestamps
2. **Parse Time Format**: The app will try to auto-detect the format
   - If parsing fails, manually select the format from the dropdown
3. **Select Variables**: Choose which numeric columns to analyze
4. Click **"Process Data"** to convert to internal format

### Step 3: Review Data Health

The Data Health Report shows:
- **Missing Values**: Count and percentage per variable
- **Time Coverage**: Start/end dates and data points
- **Sampling Frequency**: Estimated time interval
- **Outliers**: Detected extreme values (using IQR method)
- **Duplicates**: Duplicate timestamps

Review these metrics to understand data quality.

### Step 4: Visualize

1. Navigate to **"Explore & Visualize"** page
2. Choose visualization type:
   - **Single Variable**: Plot one variable over time
   - **Multi-Variable**: Compare multiple variables with independent Y-axes

#### Single Variable Plot
- Select variable from dropdown
- Customize line color
- Toggle markers on/off
- Adjust line width

#### Multi-Variable Plot
- Select 2 or more variables
- Each variable can have:
  - Custom color
  - Independent Y-axis (left or right)
  - Linear or log scale
  - Custom axis range
- Plots are fully interactive (zoom, pan, hover)

## Data Format Requirements

### Input Data
Your data file should have:
- One column with dates/times
- One or more columns with numeric values
- Consistent time indexing

### Example CSV:
```csv
Date,Temperature,Humidity,Pressure
2023-01-01,22.5,65,1013
2023-01-02,23.1,68,1012
2023-01-03,21.8,70,1014
```

### Internal Format (Long Format)
The app converts your data to:
```
timestamp    variable      value
2023-01-01   Temperature   22.5
2023-01-01   Humidity      65.0
2023-01-01   Pressure      1013.0
2023-01-02   Temperature   23.1
...
```

This standardization enables consistent processing across all modules.

## Configuration

### Modifying Settings

Edit `core/config.py` to customize:

```python
config = {
    "max_file_size_mb": 200,  # Maximum upload size
    "accepted_file_types": ["csv", "txt", "xlsx", "xls"],
    "default_plot_height": 500,  # Plot height in pixels
    "default_color_palette": [...],  # Color scheme
    ...
}
```

### Session State

The app uses Streamlit session state to maintain:
- Loaded data (`df_raw`, `df_long`)
- Metadata (`time_column`, `value_columns`)
- Health report
- Preprocessing status

Session state persists within a browser session but resets on page refresh.

## Troubleshooting

### Common Issues

**Issue**: "Module not found" error
**Solution**: Ensure you've installed all requirements: `pip install -r requirements.txt`

**Issue**: "Could not parse time column"
**Solution**: Manually select the datetime format from the dropdown

**Issue**: "File is empty" after upload
**Solution**: Check that your file contains data and the delimiter is correct

**Issue**: Charts not displaying
**Solution**: Ensure you have Plotly installed: `pip install plotly`

### Data Quality Issues

**Missing Values**: The app will show warnings if >5% of data is missing
- Consider removing variables with >20% missing data
- Or implement imputation in Phase 2

**Outliers**: Extreme values may skew visualizations
- Use log scale for variables with wide ranges
- Or implement outlier removal in Phase 2

**Irregular Sampling**: Data with inconsistent time intervals
- Still visualizes correctly
- May need resampling in Phase 2 for modeling

## Development Roadmap

### ✅ Phase 1 (Current): MVP
- Data ingestion
- Data health reporting  
- Visualization

### 🔜 Phase 2: Core Modeling
- Interpretable time models (linear, polynomial)
- Forecasting
- Reliability scoring
- Equation generation

### 🔜 Phase 3: Future Lab
- Baseline future state
- What-if interactions
- Propagation engine

### 🔜 Phase 4: Database Integration
- Supabase integration
- Project persistence
- Model registry

### 🔜 Phase 5 & 6: NLP Scenarios
- Scenario text parsing
- Variable mapping
- Scenario quantification

## Contributing

This is Phase 1 of the ExplainFutures project. Future phases will add:
- Time-series modeling
- Pairwise relationship detection
- Multivariate target models
- Interactive future exploration
- NLP scenario analysis

## Technical Notes

### Why Long Format?

The app converts all data to long format (`timestamp`, `variable`, `value`) because:
1. **Consistency**: All downstream functions expect this format
2. **Scalability**: Easy to add/remove variables
3. **Database-ready**: Matches typical time-series database schemas
4. **Modeling-friendly**: Standard format for most ML libraries

### Performance Considerations

- Files up to 200MB are supported
- Large datasets (>100k rows) may take a few seconds to process
- Visualizations with >10k points may be slow to render
- Consider downsampling for very large datasets

## License

[To be determined]

## Contact

[To be determined]

## Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) - Web framework
- [Plotly](https://plotly.com/python/) - Interactive visualizations
- [Pandas](https://pandas.pydata.org/) - Data manipulation
- [NumPy](https://numpy.org/) - Numerical computing

