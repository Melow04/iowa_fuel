import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Iowa Motor Fuel Sales Dashboard",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 2rem 0 1rem 0;
        color: #2c3e50;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the Iowa Motor Fuel Sales data"""
    
    
    try:
        df = pd.read_csv('Cleaned_Iowa_Motor_Fuel_Sales.csv')
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Convert data types
        df['Calendar Year'] = pd.to_numeric(df['Calendar Year'], errors='coerce')
        df['Number of Retail Locations'] = pd.to_numeric(df['Number of Retail Locations'], errors='coerce')
        
        # Convert fuel sales columns to numeric (remove commas if present)
        fuel_columns = [
            'Non-Ethanol Gasoline Sales (in gallons)',
            'Ethanol Gasoline Sales (in gallons)',
            'Clear and Dyed Diesel Sales (in gallons)',
            'Clear and Dyed Biodiesel Sales (in gallons)',
            'Pure Biodiesel Sales (in gallons)'
        ]
        
        for col in fuel_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        
        # Convert biofuel percentage
        df['Biofuel Distribution Percentage'] = pd.to_numeric(df['Biofuel Distribution Percentage'], errors='coerce')
        
        # Calculate total fuel sales
        df['Total Fuel Sales'] = df[fuel_columns].sum(axis=1, skipna=True)
        
        # Calculate total biofuel sales
        biofuel_cols = [
            'Clear and Dyed Biodiesel Sales (in gallons)',
            'Pure Biodiesel Sales (in gallons)'
        ]
        df['Total Biofuel Sales'] = df[biofuel_cols].sum(axis=1, skipna=True)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def create_trend_analysis(df):
    """Create trend analysis visualizations"""
    st.markdown('<div class="section-header">📈 Fuel Sales Trends Over Time</div>', unsafe_allow_html=True)
    
    # Annual fuel sales by type
    annual_data = df.groupby('Calendar Year').agg({
        'Non-Ethanol Gasoline Sales (in gallons)': 'sum',
        'Ethanol Gasoline Sales (in gallons)': 'sum',
        'Clear and Dyed Diesel Sales (in gallons)': 'sum',
        'Clear and Dyed Biodiesel Sales (in gallons)': 'sum',
        'Pure Biodiesel Sales (in gallons)': 'sum',
        'Biofuel Distribution Percentage': 'mean'
    }).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fuel sales trend
        fig1 = go.Figure()
        
        fuel_types = [
            ('Non-Ethanol Gasoline Sales (in gallons)', 'Non-Ethanol Gasoline'),
            ('Ethanol Gasoline Sales (in gallons)', 'Ethanol Gasoline'),
            ('Clear and Dyed Diesel Sales (in gallons)', 'Diesel'),
            ('Clear and Dyed Biodiesel Sales (in gallons)', 'Biodiesel'),
            ('Pure Biodiesel Sales (in gallons)', 'Pure Biodiesel')
        ]
        
        for col, name in fuel_types:
            fig1.add_trace(go.Scatter(
                x=annual_data['Calendar Year'],
                y=annual_data[col] / 1e6,  # Convert to millions
                mode='lines+markers',
                name=name,
                line=dict(width=3)
            ))
        
        fig1.update_layout(
            title="Annual Fuel Sales by Type (Millions of Gallons)",
            xaxis_title="Year",
            yaxis_title="Sales (Millions of Gallons)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Biofuel distribution percentage trend
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=annual_data['Calendar Year'],
            y=annual_data['Biofuel Distribution Percentage'],
            mode='lines+markers',
            name='Biofuel Distribution %',
            line=dict(color='green', width=4),
            marker=dict(size=8)
        ))
        
        fig2.update_layout(
            title="Biofuel Distribution Percentage Over Time",
            xaxis_title="Year",
            yaxis_title="Biofuel Distribution (%)",
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)

def create_county_analysis(df):
    """Create county-level analysis"""
    st.markdown('<div class="section-header">🗺️ County-Level Insights</div>', unsafe_allow_html=True)
    
    # County aggregations
    county_data = df.groupby('County').agg({
        'Total Fuel Sales': 'sum',
        'Biofuel Distribution Percentage': 'mean',
        'Number of Retail Locations': 'mean',
        'Total Biofuel Sales': 'sum'
    }).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 counties by total fuel sales
        top_counties_sales = county_data.nlargest(10, 'Total Fuel Sales')
        
        fig3 = px.bar(
            top_counties_sales,
            x='Total Fuel Sales',
            y='County',
            orientation='h',
            title="Top 10 Counties by Total Fuel Sales",
            labels={'Total Fuel Sales': 'Total Sales (Gallons)'}
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Top 10 counties by biofuel distribution percentage
        top_counties_biofuel = county_data.nlargest(10, 'Biofuel Distribution Percentage')
        
        fig4 = px.bar(
            top_counties_biofuel,
            x='Biofuel Distribution Percentage',
            y='County',
            orientation='h',
            title="Top 10 Counties by Biofuel Distribution %",
            color='Biofuel Distribution Percentage',
            color_continuous_scale='Greens'
        )
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)

def create_fuel_comparison(df):
    """Create fuel type comparison visualizations"""
    st.markdown('<div class="section-header">⛽ Fuel Type Analysis</div>', unsafe_allow_html=True)
    
    # Calculate fuel mix percentages
    fuel_totals = df.groupby('Calendar Year').agg({
        'Non-Ethanol Gasoline Sales (in gallons)': 'sum',
        'Ethanol Gasoline Sales (in gallons)': 'sum',
        'Clear and Dyed Diesel Sales (in gallons)': 'sum',
        'Clear and Dyed Biodiesel Sales (in gallons)': 'sum',
        'Pure Biodiesel Sales (in gallons)': 'sum'
    }).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Stacked area chart of fuel sales over time
        fig5 = go.Figure()
        
        fuel_types = [
            ('Non-Ethanol Gasoline Sales (in gallons)', 'Non-Ethanol Gasoline'),
            ('Ethanol Gasoline Sales (in gallons)', 'Ethanol Gasoline'),
            ('Clear and Dyed Diesel Sales (in gallons)', 'Diesel'),
            ('Clear and Dyed Biodiesel Sales (in gallons)', 'Biodiesel'),
            ('Pure Biodiesel Sales (in gallons)', 'Pure Biodiesel')
        ]
        
        for col, name in fuel_types:
            fig5.add_trace(go.Scatter(
                x=fuel_totals['Calendar Year'],
                y=fuel_totals[col] / 1e6,
                mode='lines',
                stackgroup='one',
                name=name
            ))
        
        fig5.update_layout(
            title="Fuel Sales Composition Over Time (Stacked)",
            xaxis_title="Year",
            yaxis_title="Sales (Millions of Gallons)",
            height=400
        )
        
        st.plotly_chart(fig5, use_container_width=True)
    
    with col2:
        # Latest year fuel mix pie chart
        latest_year = fuel_totals['Calendar Year'].max()
        latest_data = fuel_totals[fuel_totals['Calendar Year'] == latest_year].iloc[0]
        
        pie_data = {
            'Non-Ethanol Gasoline': latest_data['Non-Ethanol Gasoline Sales (in gallons)'],
            'Ethanol Gasoline': latest_data['Ethanol Gasoline Sales (in gallons)'],
            'Diesel': latest_data['Clear and Dyed Diesel Sales (in gallons)'],
            'Biodiesel': latest_data['Clear and Dyed Biodiesel Sales (in gallons)'],
            'Pure Biodiesel': latest_data['Pure Biodiesel Sales (in gallons)']
        }
        
        fig6 = px.pie(
            values=list(pie_data.values()),
            names=list(pie_data.keys()),
            title=f"Fuel Mix Distribution ({int(latest_year)})"
        )
        fig6.update_layout(height=400)
        st.plotly_chart(fig6, use_container_width=True)

def create_retail_analysis(df):
    """Create retail location analysis"""
    st.markdown('<div class="section-header">🏪 Retail Location Analysis</div>', unsafe_allow_html=True)
    
    # Correlation analysis
    correlation_data = df.groupby('County').agg({
        'Number of Retail Locations': 'mean',
        'Total Fuel Sales': 'sum'
    }).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Scatter plot: Retail locations vs Total sales
        fig7 = px.scatter(
            correlation_data,
            x='Number of Retail Locations',
            y='Total Fuel Sales',
            hover_data=['County'],
            title="Retail Locations vs Total Fuel Sales by County",
            trendline="ols"
        )
        fig7.update_layout(height=400)
        st.plotly_chart(fig7, use_container_width=True)
    
    with col2:
        # Average retail locations per county
        avg_locations = correlation_data['Number of Retail Locations'].mean()
        top_locations = correlation_data.nlargest(10, 'Number of Retail Locations')
        
        fig8 = px.bar(
            top_locations,
            x='County',
            y='Number of Retail Locations',
            title="Top 10 Counties by Number of Retail Locations"
        )
        fig8.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig8, use_container_width=True)

def create_predictive_model(df):
    """Create predictive models for biofuel distribution"""
    st.markdown('<div class="section-header">🔮 Predictive Analytics</div>', unsafe_allow_html=True)
    
    # Prepare data for modeling
    model_data = df.groupby(['Calendar Year', 'County']).agg({
        'Biofuel Distribution Percentage': 'mean',
        'Number of Retail Locations': 'mean',
        'Total Fuel Sales': 'sum',
        'Total Biofuel Sales': 'sum'
    }).reset_index()
    
    # Remove rows with missing values
    model_data = model_data.dropna()
    
    if len(model_data) < 10:
        st.warning("Insufficient data for predictive modeling")
        return
    
    # Features and target
    features = ['Calendar Year', 'Number of Retail Locations', 'Total Fuel Sales', 'Total Biofuel Sales']
    X = model_data[features]
    y = model_data['Biofuel Distribution Percentage']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Model Performance")
        
        # Train models
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        lr_model = LinearRegression()
        
        rf_model.fit(X_train, y_train)
        lr_model.fit(X_train, y_train)
        
        # Predictions
        rf_pred = rf_model.predict(X_test)
        lr_pred = lr_model.predict(X_test)
        
        # Model metrics
        rf_mae = mean_absolute_error(y_test, rf_pred)
        lr_mae = mean_absolute_error(y_test, lr_pred)
        rf_r2 = r2_score(y_test, rf_pred)
        lr_r2 = r2_score(y_test, lr_pred)
        
        metrics_df = pd.DataFrame({
            'Model': ['Random Forest', 'Linear Regression'],
            'MAE': [rf_mae, lr_mae],
            'R² Score': [rf_r2, lr_r2]
        })
        
        st.dataframe(metrics_df, use_container_width=True)
        
        # Feature importance (Random Forest)
        if hasattr(rf_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig9 = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance (Random Forest)"
            )
            st.plotly_chart(fig9, use_container_width=True)
    
    with col2:
        st.subheader("📊 Future Predictions")
        
        # Predict for future years (2025-2027)
        future_years = [2025, 2026, 2027]
        
        # Use average values for other features
        avg_locations = model_data['Number of Retail Locations'].mean()
        avg_fuel_sales = model_data['Total Fuel Sales'].mean()
        avg_biofuel_sales = model_data['Total Biofuel Sales'].mean()
        
        future_predictions = []
        for year in future_years:
            future_X = np.array([[year, avg_locations, avg_fuel_sales, avg_biofuel_sales]])
            rf_pred_future = rf_model.predict(future_X)[0]
            lr_pred_future = lr_model.predict(future_X)[0]
            
            future_predictions.append({
                'Year': year,
                'Random Forest Prediction': rf_pred_future,
                'Linear Regression Prediction': lr_pred_future
            })
        
        future_df = pd.DataFrame(future_predictions)
        st.dataframe(future_df, use_container_width=True)
        
        # Visualization of predictions
        historical_trend = model_data.groupby('Calendar Year')['Biofuel Distribution Percentage'].mean().reset_index()
        
        fig10 = go.Figure()
        
        # Historical data
        fig10.add_trace(go.Scatter(
            x=historical_trend['Calendar Year'],
            y=historical_trend['Biofuel Distribution Percentage'],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='blue')
        ))
        
        # Future predictions
        fig10.add_trace(go.Scatter(
            x=future_df['Year'],
            y=future_df['Random Forest Prediction'],
            mode='lines+markers',
            name='RF Predictions',
            line=dict(color='red', dash='dash')
        ))
        
        fig10.add_trace(go.Scatter(
            x=future_df['Year'],
            y=future_df['Linear Regression Prediction'],
            mode='lines+markers',
            name='LR Predictions',
            line=dict(color='green', dash='dash')
        ))
        
        fig10.update_layout(
            title="Biofuel Distribution % - Historical vs Predicted",
            xaxis_title="Year",
            yaxis_title="Biofuel Distribution (%)",
            height=400
        )
        
        st.plotly_chart(fig10, use_container_width=True)

def main():
    """Main dashboard function"""
    
    # Sidebar
    with st.sidebar:
        st.title("Iowa Fuel Dashboard")
        st.markdown("---")
        
        # Logo placeholder
        st.markdown('<div style="justify-items: center;">', unsafe_allow_html=True)
        st.image("fortune.png", width=120)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        analysis_type = st.radio(
            "Select Analysis Type:",
            ["📊 Descriptive Analytics", "🔮 Predictive Analytics"],
            index=0
        )
        
        if analysis_type == "📊 Descriptive Analytics":
            descriptive_option = st.selectbox(
                "Choose Analysis:",
                [
                    "📈 Trend Analysis",
                    "🗺️ County Insights", 
                    "⛽ Fuel Comparison",
                    "🏪 Retail Analysis",
                    "📋 All Descriptive"
                ]
            )
        
        st.markdown("---")
        
        # Data info
        st.markdown("### 📊 Dataset Info")
        st.markdown("- **Records**: 1,273")
        st.markdown("- **Counties**: Iowa counties")
        st.markdown("- **Years**: Multiple years")
        st.markdown("- **Fuel Types**: 5 categories")
    
    # Main content
    st.markdown('<div class="main-header">Iowa Motor Fuel Sales Dashboard</div>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check the data source.")
        return
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_records = len(df)
        st.metric("Total Records", f"{total_records:,}")
    
    with col2:
        unique_counties = df['County'].nunique()
        st.metric("Counties", unique_counties)
    
    with col3:
        year_range = f"{int(df['Calendar Year'].min())}-{int(df['Calendar Year'].max())}"
        st.metric("Year Range", year_range)
    
    with col4:
        avg_biofuel_pct = df['Biofuel Distribution Percentage'].mean()
        st.metric("Avg Biofuel %", f"{avg_biofuel_pct:.1f}%")
    
    st.markdown("---")
    
    # Analysis sections
    if analysis_type == "📊 Descriptive Analytics":
        if descriptive_option == "📈 Trend Analysis":
            create_trend_analysis(df)
        elif descriptive_option == "🗺️ County Insights":
            create_county_analysis(df)
        elif descriptive_option == "⛽ Fuel Comparison":
            create_fuel_comparison(df)
        elif descriptive_option == "🏪 Retail Analysis":
            create_retail_analysis(df)
        elif descriptive_option == "📋 All Descriptive":
            create_trend_analysis(df)
            st.markdown("---")
            create_county_analysis(df)
            st.markdown("---")
            create_fuel_comparison(df)
            st.markdown("---")
            create_retail_analysis(df)
    
    elif analysis_type == "🔮 Predictive Analytics":
        create_predictive_model(df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        Iowa Motor Fuel Sales Dashboard | Data Analytics & Predictions
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
