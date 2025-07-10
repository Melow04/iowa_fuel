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

## 📊 Features

### 🔍 Descriptive Analytics
- **📈 Trend Analysis**: Interactive line charts showing fuel sales trends and biofuel distribution over time
- **🗺️ County Insights**: Top 10 counties by fuel sales and biofuel distribution percentage
- **⛽ Fuel Comparison**: Stacked area charts and pie charts showing fuel mix composition
- **🏪 Retail Analysis**: Correlation analysis between retail locations and fuel sales

### 🔮 Predictive Analytics
- **Machine Learning Models**: Random Forest and Linear Regression algorithms
- **Future Predictions**: Biofuel distribution percentage forecasts for 2025-2027
- **Model Performance**: MAE and R² scores comparison
- **Feature Importance**: Analysis of key predictive factors

### 🎨 Dashboard Features
- **Interactive Visualizations**: Built with Plotly for rich, interactive charts
- **Sidebar Navigation**: Easy switching between analysis types
- **Real-time Data Loading**: Automatically fetches data from CSV URL
- **Responsive Design**: Multi-column layouts optimized for different screen sizes
- **Professional Styling**: Clean, modern interface with custom CSS

## 📋 Dataset Information

- **Records**: 1,273 fuel sales records
- **Coverage**: All Iowa counties
- **Time Period**: Multiple years of historical data
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
2. Double-click \`run_dashboard.bat\`
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
   - Copy \`dashboard.py\` to the folder
   - Copy \`requirements.txt\` to the folder

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
- **Internet**: Required for initial data loading

## 🎯 How to Use

### 1. Navigation
- Use the **sidebar** to switch between analysis types
- Select **"📊 Descriptive Analytics"** for historical insights
- Choose **"🔮 Predictive Analytics"** for future forecasts

### 2. Descriptive Analytics Options
- **📈 Trend Analysis**: View fuel sales trends over time
- **🗺️ County Insights**: Explore county-level performance
- **⛽ Fuel Comparison**: Compare different fuel types
- **🏪 Retail Analysis**: Analyze retail location impacts
- **📋 All Descriptive**: View all descriptive analytics at once

### 3. Predictive Analytics
- **Model Comparison**: Compare Random Forest vs Linear Regression
- **Future Predictions**: View 2025-2027 biofuel distribution forecasts
- **Feature Importance**: Understand key prediction factors

### 4. Interactive Features
- **Hover** over charts for detailed information
- **Zoom and pan** on visualizations
- **Legend clicking** to show/hide data series
- **Responsive design** adapts to your screen size

## 📈 Key Insights Available

### Trend Analysis
- Annual fuel sales by type (millions of gallons)
- Biofuel distribution percentage growth over time
- Seasonal patterns and long-term trends

### County Performance
- Top performing counties by total fuel sales
- Counties with highest biofuel adoption rates
- Geographic distribution patterns

### Fuel Mix Analysis
- Composition of fuel types over time
- Market share evolution
- Biofuel vs traditional fuel trends

### Predictive Insights
- Future biofuel distribution forecasts
- Model accuracy metrics
- Key factors influencing predictions

## 🔧 Customization

### Adding New Visualizations
1. Create a new function in \`dashboard.py\`
2. Add the function call to the main navigation logic
3. Update the sidebar options if needed

### Modifying Data Source
1. Update the URL in the \`load_data()\` function
2. Adjust column mappings if schema differs
3. Update data preprocessing steps as needed

### Styling Changes
1. Modify the CSS in the \`st.markdown()\` section
2. Update Plotly chart themes and colors
3. Adjust layout and spacing parameters

## 🐛 Troubleshooting

### Common Issues

**Dashboard won't start**
- Ensure Python 3.8+ is installed
- Check that all requirements are installed: \`pip install -r requirements.txt\`
- Verify internet connection for data loading

**Charts not displaying**
- Clear browser cache and refresh
- Check browser console for JavaScript errors
- Try a different browser (Chrome, Firefox, Safari)

**Data loading errors**
- Verify internet connection
- Check if the data URL is accessible
- Look for error messages in the terminal

**Performance issues**
- Close other browser tabs
- Restart the dashboard
- Check available system memory

### Getting Help
1. Check the terminal/command prompt for error messages
2. Ensure all files are in the same directory
3. Verify Python and package versions
4. Try running individual components to isolate issues

## 📊 Technical Architecture

### Data Pipeline
1. **Data Loading**: Fetches CSV from remote URL
2. **Data Cleaning**: Handles missing values and type conversions
3. **Feature Engineering**: Calculates derived metrics
4. **Caching**: Uses Streamlit's caching for performance

### Visualization Stack
- **Frontend**: Streamlit web framework
- **Charts**: Plotly for interactive visualizations
- **Styling**: Custom CSS for professional appearance
- **Layout**: Responsive multi-column design

### Machine Learning Pipeline
- **Data Preparation**: Feature selection and preprocessing
- **Model Training**: Random Forest and Linear Regression
- **Evaluation**: MAE and R² score metrics
- **Prediction**: Future trend forecasting

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a virtual environment
3. Install development dependencies
4. Make your changes
5. Test thoroughly
6. Submit a pull request

### Code Style
- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions
- Comment complex logic sections

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Iowa Department of Transportation** for providing the motor fuel sales data
- **Streamlit** for the excellent web app framework
- **Plotly** for interactive visualization capabilities
- **Scikit-learn** for machine learning algorithms

## 📞 Support

For questions, issues, or suggestions:
1. Check the troubleshooting section above
2. Review the code comments for implementation details
3. Create an issue with detailed error information
4. Include your Python version and operating system

---

**Built with ❤️ for Iowa fuel industry analysis**
