# 🌽 Iowa Motor Fuel Sales Dashboard

A comprehensive Python dashboard application for analyzing Iowa Motor Fuel Sales data with both descriptive and predictive analytics capabilities.
## Made by Fortune Analytics:
- Aldave, Adrian
- Cortes, Karmelo
- Mangubat, John Erick
- Villa, Daniel

![Dashboard Preview](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)

## 🚀 New Features & Updates

### ✨ **Latest Enhancements**
- **🎯 Advanced Year Filtering**: All Years, Year Range, or Single Year selection
- **🏪 Enhanced Retail Analysis**: Multi-dimensional correlation analysis with visual metrics
- **📊 Interactive Visualizations**: Bubble charts, dual-axis plots, and trend lines
- **🔍 Sales Efficiency Metrics**: Sales per retail location analysis
- **📈 Correlation Insights**: Automated strength interpretation and color coding
- **🎨 Professional Styling**: Gradient backgrounds and improved visual design

## 📊 Features

### 🔍 Descriptive Analytics
- **📈 Trend Analysis**: Interactive line charts showing fuel sales trends and biofuel distribution over time
- **🗺️ County Insights**: Top 10 counties by fuel sales and biofuel distribution percentage
- **⛽ Fuel Comparison**: Stacked area charts and pie charts showing fuel mix composition
- **🏪 Advanced Retail Analysis**: 
  - Multi-dimensional correlation analysis
  - Sales efficiency metrics (sales per location)
  - Distribution analysis across counties
  - Visual correlation strength indicators
  - Dual-axis visualizations combining multiple metrics

### 🔮 Predictive Analytics
- **Machine Learning Models**: Random Forest and Linear Regression algorithms
- **Future Predictions**: Biofuel distribution percentage forecasts for 2025-2027
- **Model Performance**: MAE and R² scores comparison
- **Feature Importance**: Analysis of key predictive factors

### 🎨 Dashboard Features
- **📅 Flexible Year Filtering**: 
  - All Years: Complete dataset analysis
  - Year Range: Custom date range selection
  - Single Year: Focus on specific year
- **Interactive Visualizations**: Built with Plotly for rich, interactive charts
- **Sidebar Navigation**: Easy switching between analysis types
- **Real-time Data Loading**: Automatically processes CSV data
- **Responsive Design**: Multi-column layouts optimized for different screen sizes
- **Professional Styling**: Clean, modern interface with custom CSS and gradients

## 📋 Dataset Information

- **Records**: 1,273+ fuel sales records
- **Coverage**: All Iowa counties (99 counties)
- **Time Period**: Multi-year historical data (2010-2023)
- **Fuel Types**: 5 categories including ethanol, biodiesel, and traditional fuels
- **Data Source**: Iowa Motor Fuel Sales (Cleaned dataset)

### 📊 Data Schema
- **Calendar Year**: Year of record
- **County**: Iowa county name
- **Number of Retail Locations**: Count of fuel retail locations
- **Fuel Sales Data**: 
  - Non-Ethanol Gasoline Sales (gallons)
  - Ethanol Gasoline Sales (gallons)
  - Clear and Dyed Diesel Sales (gallons)
  - Clear and Dyed Biodiesel Sales (gallons)
  - Pure Biodiesel Sales (gallons)
- **Biofuel Distribution Percentage**: Share of biofuels in total fuel mix

## 🚀 Quick Start

### Option 1: One-Click Setup (Windows)
1. Download all files to a folder
2. Double-click `run_dashboard.bat`
3. Dashboard opens automatically in your browser

### Option 2: Manual Setup
\`\`\`bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard.py
\`\`\`

### Option 3: Step-by-Step
1. **Create project folder**
   \`\`\`bash
   mkdir iowa-fuel-dashboard
   cd iowa-fuel-dashboard
   \`\`\`

2. **Save the files**
   - Copy `dashboard.py` to the folder
   - Copy `requirements.txt` to the folder
   - Copy your CSV data file as `Cleaned_Iowa_Motor_Fuel_Sales.csv`

3. **Install and run**
   \`\`\`bash
   pip install -r requirements.txt
   streamlit run dashboard.py
   \`\`\`

## 📦 Requirements

### Python Packages
- **streamlit** ≥1.28.0 - Web app framework
- **pandas** ≥1.5.0 - Data manipulation and analysis
- **numpy** ≥1.24.0 - Numerical computing
- **plotly** ≥5.15.0 - Interactive visualizations
- **scikit-learn** ≥1.3.0 - Machine learning algorithms
- **requests** ≥2.31.0 - HTTP requests for data loading

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 100MB free space
- **Internet**: Optional (for enhanced features)

## 🎯 How to Use

### 1. Year Filtering (New!)
- **All Years**: Analyze complete dataset
- **Year Range**: Select custom date range with slider
- **Single Year**: Focus on specific year analysis

### 2. Navigation
- Use the **sidebar** to switch between analysis types
- Select **"📊 Descriptive Analytics"** for historical insights
- Choose **"🔮 Predictive Analytics"** for future forecasts

### 3. Descriptive Analytics Options
- **📈 Trend Analysis**: View fuel sales trends over time
- **🗺️ County Insights**: Explore county-level performance
- **⛽ Fuel Comparison**: Compare different fuel types
- **🏪 Retail Analysis**: Advanced retail location analysis with correlations
- **📋 All Descriptive**: View all descriptive analytics at once

### 4. Enhanced Retail Analysis Features
- **Correlation Metrics**: Visual cards showing relationship strength
- **Bubble Charts**: Multi-dimensional scatter plots
- **Efficiency Analysis**: Sales per retail location metrics
- **Distribution Analysis**: Histogram of retail locations
- **Dual-Axis Charts**: Combined location and biofuel percentage views

### 5. Interactive Features
- **Hover** over charts for detailed information
- **Zoom and pan** on visualizations
- **Legend clicking** to show/hide data series
- **Responsive design** adapts to your screen size
- **Year filtering** updates all visualizations dynamically

## 📈 Key Insights Available

### Trend Analysis
- Annual fuel sales by type (millions of gallons)
- Biofuel distribution percentage growth over time
- Seasonal patterns and long-term trends
- Year-over-year comparisons

### County Performance
- Top performing counties by total fuel sales
- Counties with highest biofuel adoption rates
- Geographic distribution patterns
- County-specific year filtering

### Fuel Mix Analysis
- Composition of fuel types over time
- Market share evolution
- Biofuel vs traditional fuel trends
- Temporal fuel mix changes

### Advanced Retail Insights
- **Correlation Analysis**: 
  - Retail locations vs total fuel sales
  - Retail locations vs biofuel distribution percentage
  - Automated correlation strength interpretation
- **Efficiency Metrics**: Sales per retail location by county
- **Distribution Patterns**: Retail location spread across counties
- **Performance Benchmarking**: Top performing counties by efficiency

### Predictive Insights
- Future biofuel distribution forecasts (2025-2027)
- Model accuracy metrics and comparisons
- Key factors influencing predictions
- Feature importance analysis

## 🔧 Technical Enhancements

### New Correlation Analysis System
```python
def calculate_correlation(x, y):
    """Calculate correlation coefficient manually"""
    # Handles missing data and edge cases
    # Returns robust correlation values

def get_correlation_interpretation(correlation):
    """Interpret correlation strength with color coding"""
    # Very Strong (≥0.8), Strong (≥0.6), Moderate (≥0.4), etc.
    # Returns interpretation and color for visual display
