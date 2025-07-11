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
        color: white;
    }
    .insight-box {
        background-color: #3498db;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #3498db;
    }
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .logo-container img {
        max-width: 120px;
        height: auto;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .correlation-metric {
        background: linear-gradient(to right, #2c3e50 0%, black 100%);
     
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 1rem 0;
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

def filter_data_by_year(df, year_range):
    """Filter dataframe by year range"""
    if year_range[0] == year_range[1]:
        return df[df['Calendar Year'] == year_range[0]]
    else:
        return df[(df['Calendar Year'] >= year_range[0]) & (df['Calendar Year'] <= year_range[1])]

def calculate_correlation(x, y):
    """Calculate correlation coefficient manually"""
    try:
        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 2:
            return 0
        
        # Calculate correlation
        correlation = np.corrcoef(x_clean, y_clean)[0, 1]
        return correlation if not np.isnan(correlation) else 0
    except:
        return 0

def get_correlation_interpretation(correlation):
    """Get interpretation of correlation strength"""
    abs_corr = abs(correlation)
    if abs_corr >= 0.8:
        strength = "Very Strong"
        color = "#d32f2f" if correlation > 0 else "#1976d2"
    elif abs_corr >= 0.6:
        strength = "Strong"
        color = "#f57c00" if correlation > 0 else "#388e3c"
    elif abs_corr >= 0.4:
        strength = "Moderate"
        color = "#fbc02d"
    elif abs_corr >= 0.2:
        strength = "Weak"
        color = "#9e9e9e"
    else:
        strength = "Very Weak"
        color = "#757575"
    
    direction = "Positive" if correlation > 0 else "Negative"
    return f"{strength} {direction}", color

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
        if len(fuel_totals) > 0:
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
    """Create comprehensive retail location analysis"""
    st.markdown('<div class="section-header">🏪 Retail Location Analysis</div>', unsafe_allow_html=True)
    
    # Prepare correlation data
    correlation_data = df.groupby('County').agg({
        'Number of Retail Locations': 'mean',
        'Total Fuel Sales': 'sum',
        'Biofuel Distribution Percentage': 'mean',
        'Total Biofuel Sales': 'sum'
    }).reset_index()
    
    # Remove rows with missing data
    correlation_data = correlation_data.dropna()
    
    if len(correlation_data) == 0:
        st.warning("No data available for retail analysis with current filters.")
        return
    
    # Calculate correlations
    corr_locations_sales = calculate_correlation(
        correlation_data['Number of Retail Locations'].values,
        correlation_data['Total Fuel Sales'].values
    )
    
    corr_locations_biofuel = calculate_correlation(
        correlation_data['Number of Retail Locations'].values,
        correlation_data['Biofuel Distribution Percentage'].values
    )
    
    # Display correlation metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        interp1, color1 = get_correlation_interpretation(corr_locations_sales)
        st.markdown(f"""
        <div class="correlation-metric">
            <h4>🏪 ↔️ ⛽</h4>
            <h3>{corr_locations_sales:.3f}</h3>
            <p>Locations vs Sales</p>
            <small>{interp1}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        interp2, color2 = get_correlation_interpretation(corr_locations_biofuel)
        st.markdown(f"""
        <div class="correlation-metric">
            <h4>🏪 ↔️ 🌱</h4>
            <h3>{corr_locations_biofuel:.3f}</h3>
            <p>Locations vs Biofuel %</p>
            <small>{interp2}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_locations = correlation_data['Number of Retail Locations'].mean()
        st.markdown(f"""
        <div class="correlation-metric">
            <h4>📊</h4>
            <h3>{avg_locations:.1f}</h3>
            <p>Avg Locations</p>
            <small>Per County</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Main visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced scatter plot: Retail locations vs Total sales
        fig7 = px.scatter(
            correlation_data,
            x='Number of Retail Locations',
            y='Total Fuel Sales',
            size='Biofuel Distribution Percentage',
            color='Biofuel Distribution Percentage',
            hover_data=['County'],
            title="Retail Locations vs Total Fuel Sales by County",
            labels={
                'Number of Retail Locations': 'Number of Retail Locations',
                'Total Fuel Sales': 'Total Fuel Sales (Gallons)',
                'Biofuel Distribution Percentage': 'Biofuel %'
            },
            color_continuous_scale='Viridis'
        )
        
        # Add trend line manually using linear regression
        if len(correlation_data) > 1:
            x_vals = correlation_data['Number of Retail Locations'].values
            y_vals = correlation_data['Total Fuel Sales'].values
            
            # Simple linear regression
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            
            # Add trend line
            x_trend = np.linspace(x_vals.min(), x_vals.max(), 100)
            y_trend = p(x_trend)
            
            fig7.add_trace(go.Scatter(
                x=x_trend,
                y=y_trend,
                mode='lines',
                name=f'Trend Line (r={corr_locations_sales:.3f})',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        fig7.update_layout(height=450)
        st.plotly_chart(fig7, use_container_width=True)
    
    with col2:
        # Top counties by retail locations with additional metrics
        top_locations = correlation_data.nlargest(10, 'Number of Retail Locations')
        
        fig8 = go.Figure()
        
        # Add bars for retail locations
        fig8.add_trace(go.Bar(
            x=top_locations['County'],
            y=top_locations['Number of Retail Locations'],
            name='Retail Locations',
            marker_color='lightblue',
            yaxis='y'
        ))
        
        # Add line for biofuel percentage
        fig8.add_trace(go.Scatter(
            x=top_locations['County'],
            y=top_locations['Biofuel Distribution Percentage'],
            mode='lines+markers',
            name='Biofuel %',
            line=dict(color='green', width=3),
            marker=dict(size=8),
            yaxis='y2'
        ))
        
        fig8.update_layout(
            title="Top 10 Counties: Retail Locations & Biofuel Distribution",
            xaxis_title="County",
            yaxis=dict(title="Number of Retail Locations", side="left"),
            yaxis2=dict(title="Biofuel Distribution (%)", side="right", overlaying="y"),
            height=450,
            xaxis_tickangle=-45,
            legend=dict(x=0.01, y=0.99)
        )
        
        st.plotly_chart(fig8, use_container_width=True)
    
    # Additional insights section
    st.markdown("### 📊 Retail Analysis Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Retail efficiency analysis
        correlation_data['Sales_per_Location'] = (
            correlation_data['Total Fuel Sales'] / correlation_data['Number of Retail Locations']
        )
        
        top_efficient = correlation_data.nlargest(5, 'Sales_per_Location')
        
        fig9 = px.bar(
            top_efficient,
            x='Sales_per_Location',
            y='County',
            orientation='h',
            title="Top 5 Counties: Sales per Retail Location",
            labels={'Sales_per_Location': 'Sales per Location (Gallons)'},
            color='Sales_per_Location',
            color_continuous_scale='Blues'
        )
        fig9.update_layout(height=300)
        st.plotly_chart(fig9, use_container_width=True)
    
    with col2:
        # Distribution of retail locations
        fig10 = px.histogram(
            correlation_data,
            x='Number of Retail Locations',
            nbins=20,
            title="Distribution of Retail Locations Across Counties",
            labels={'Number of Retail Locations': 'Number of Retail Locations', 'count': 'Number of Counties'}
        )
        fig10.update_layout(height=300)
        st.plotly_chart(fig10, use_container_width=True)
    
    # Summary insights
    st.markdown(f"""
    <div class="insight-box">
        <strong>🔍 Key Retail Insights:</strong><br>
        • <strong>Location-Sales Correlation:</strong> {interp1.lower()} relationship ({corr_locations_sales:.3f})<br>
        • <strong>Location-Biofuel Correlation:</strong> {interp2.lower()} relationship ({corr_locations_biofuel:.3f})<br>
        • <strong>Average Locations per County:</strong> {avg_locations:.1f}<br>
        • <strong>Most Efficient County:</strong> {top_efficient.iloc[0]['County']} ({top_efficient.iloc[0]['Sales_per_Location']:,.0f} gallons/location)<br>
        • <strong>Total Counties Analyzed:</strong> {len(correlation_data)}
    </div>
    """, unsafe_allow_html=True)

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
    
    # Load data first to get year range
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check the data source.")
        return
    
    # Get year range from data
    min_year = int(df['Calendar Year'].min())
    max_year = int(df['Calendar Year'].max())
    
    # Sidebar
    with st.sidebar:
        st.title("Iowa Fuel Dashboard")
        st.markdown("---")
        
        # Logo with improved styling
        st.markdown('<div class="logo-container">', unsafe_allow_html=True)
        try:
            st.image("fortune.png", width=120)
        except:
            st.markdown("🌽", unsafe_allow_html=True)  # Fallback if image not found
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Year Filter
        st.markdown("### 📅 Year Filter")
        year_filter_type = st.radio(
            "Filter Type:",
            ["All Years", "Year Range", "Single Year"],
            index=0
        )
        
        if year_filter_type == "Year Range":
            year_range = st.slider(
                "Select Year Range:",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
                step=1
            )
        elif year_filter_type == "Single Year":
            selected_year = st.selectbox(
                "Select Year:",
                options=sorted(df['Calendar Year'].dropna().unique()),
                index=len(sorted(df['Calendar Year'].dropna().unique())) - 1  # Default to latest year
            )
            year_range = (int(selected_year), int(selected_year))
        else:
            year_range = (min_year, max_year)
        
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
        
        # Data info (updated based on filter)
        filtered_df = filter_data_by_year(df, year_range)
        st.markdown("### 📊 Dataset Info")
        st.markdown(f"- **Records**: {len(filtered_df):,}")
        st.markdown(f"- **Counties**: {filtered_df['County'].nunique()}")
        st.markdown(f"- **Year Range**: {year_range[0]}-{year_range[1]}")
        st.markdown("- **Fuel Types**: 5 categories")
    
    # Main content
    st.markdown('<div class="main-header">Iowa Motor Fuel Sales Dashboard</div>', unsafe_allow_html=True)
    
    # Apply year filter
    filtered_df = filter_data_by_year(df, year_range)
    
    # Display filter info
    if year_filter_type != "All Years":
        st.info(f"📅 Showing data for: {year_range[0]} - {year_range[1]}")
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_records = len(filtered_df)
        st.metric("Total Records", f"{total_records:,}")
    
    with col2:
        unique_counties = filtered_df['County'].nunique()
        st.metric("Counties", unique_counties)
    
    with col3:
        if len(filtered_df) > 0:
            year_range_display = f"{int(filtered_df['Calendar Year'].min())}-{int(filtered_df['Calendar Year'].max())}"
        else:
            year_range_display = "No data"
        st.metric("Year Range", year_range_display)
    
    with col4:
        if len(filtered_df) > 0:
            avg_biofuel_pct = filtered_df['Biofuel Distribution Percentage'].mean()
            st.metric("Avg Biofuel %", f"{avg_biofuel_pct:.1f}%")
        else:
            st.metric("Avg Biofuel %", "N/A")
    
    st.markdown("---")
    
    # Check if filtered data is empty
    if len(filtered_df) == 0:
        st.warning("No data available for the selected year range. Please adjust your filter.")
        return
    
    # Analysis sections
    if analysis_type == "📊 Descriptive Analytics":
        if descriptive_option == "📈 Trend Analysis":
            create_trend_analysis(filtered_df)
        elif descriptive_option == "🗺️ County Insights":
            create_county_analysis(filtered_df)
        elif descriptive_option == "⛽ Fuel Comparison":
            create_fuel_comparison(filtered_df)
        elif descriptive_option == "🏪 Retail Analysis":
            create_retail_analysis(filtered_df)
        elif descriptive_option == "📋 All Descriptive":
            create_trend_analysis(filtered_df)
            st.markdown("---")
            create_county_analysis(filtered_df)
            st.markdown("---")
            create_fuel_comparison(filtered_df)
            st.markdown("---")
            create_retail_analysis(filtered_df)
    
    elif analysis_type == "🔮 Predictive Analytics":
        create_predictive_model(filtered_df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        Iowa Motor Fuel Sales Dashboard | Data Analytics & Predictions | Fortune Analytics
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
