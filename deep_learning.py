import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Air Quality Health Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished dark theme with animations
st.markdown("""
    <style>
    /* General styling */
    .stApp {
        background-color: #0F172A;
        color: #F1F5F9;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    }
    .main .block-container {
        max-width: 1280px;
        padding: 2rem 3rem;
    }
    /* Headers */
    h1 {
        font-size: 2.8rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 1.5rem;
        text-align: center;
        animation: fadeIn 1s ease-in;
    }
    h2 {
        font-size: 1.8rem;
        font-weight: 600;
        color: #E2E8F0;
        margin: 1.5rem 0 1rem;
    }
    h3 {
        font-size: 1.4rem;
        font-weight: 500;
        color: #E2E8F0;
        margin-bottom: 0.8rem;
    }
    /* Containers with fade-in animation */
    .stContainer {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        animation: fadeInUp 0.8s ease-out;
    }
    /* Buttons with hover and click animations */
    .stButton>button {
        background-color: #10B981;
        color: #FFFFFF;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .stButton>button:hover {
        background-color: #059669;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .stButton>button:active {
        transform: translateY(0);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    /* Loading spinner for button */
    .stButton>button:disabled {
        background-color: #6B7280;
        cursor: not-allowed;
    }
    .stButton>button:disabled::after {
        content: '';
        position: absolute;
        width: 20px;
        height: 20px;
        border: 3px solid #FFFFFF;
        border-top: 3px solid transparent;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        top: 50%;
        right: 1rem;
        transform: translateY(-50%);
    }
    /* Inputs */
    .stTextInput>div>input, .stNumberInput>div>input, .stSelectbox>div>div {
        background-color: #334155;
        color: #F1F5F9;
        border: 1px solid #475569;
        border-radius: 8px;
        padding: 0.6rem;
        font-size: 1rem;
        transitionദ
        transition: border-color 0.3s ease;
    }
    .stTextInput>div>input:focus, .stNumberInput>div>input:focus {
        border-color: #10B981;
        outline: none;
    }
    .stSelectbox>div>div {
        background-color: #334155;
    }
    /* Labels */
    .stTextInput label, .stNumberInput label, .stSelectbox label {
        color: #E2E8F0;
        font-weight: 500;
        font-size: 1rem;
    }
    /* Plotly charts */
    .stPlotlyChart {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        animation: fadeInUp 1s ease-out;
    }
    /* Metrics */
    .stMetric {
        background-color: #334155;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        animation: fadeIn 0.8s ease-out;
    }
    .stMetric label {
        font-size: 1.1rem;
        font-weight: 500;
        color: #E2E8F0;
    }
    .stMetric .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #10B981;
    }
    /* Sidebar */
    .css-1d391kg {
        background-color: #1E293B;
        padding: 1.5rem;
    }
    .css-1d391kg h1, .css-1d391kg h2 {
        color: #F8FAFC;
    }
    .css-1d391kg p, .css-1d391kg li {
        color: #D1D5DB;
        font-size: 0.95rem;
    }
    /* Form */
    .stForm {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        animation: fadeInUp 0.8s ease-out;
    }
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes spin {
        0% { transform: translateY(-50%) rotate(0deg); }
        100% { transform: translateY(-50%) rotate(360deg); }
    }
    </style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("air_quality_health_dataset.csv")
        required_columns = ['AQI', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 
                           'temperature', 'humidity', 'wind_speed', 'precipitation',
                           'population_density', 'green_cover_percentage', 'region', 
                           'lockdown_status', 'hospital_visits', 'date']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Error: Missing required columns in dataset: {', '.join(missing_columns)}")
            return None
        df = df.dropna()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if df['date'].isna().any():
            st.error("Error: Invalid date format in 'date' column. Please ensure all dates are valid.")
            return None
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        return df
    except FileNotFoundError:
        st.error("Error: 'air_quality_health_dataset.csv' not found. Please ensure the file is in the same directory as this script.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Define features and target
def preprocess_data(df):
    try:
        numeric_features = ['AQI', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 
                           'temperature', 'humidity', 'wind_speed', 'precipitation',
                           'population_density', 'green_cover_percentage']
        categorical_features = ['region', 'lockdown_status']
        target = 'hospital_visits'
        X = df[numeric_features + categorical_features]
        y = df[target]
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
            ])
        return X, y, preprocessor
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return None, None, None

# Build and train model
@st.cache_resource
def train_model(X, y, _preprocessor):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        _preprocessor.fit(X_train)
        input_shape = _preprocessor.transform(X_train).shape[1]
        model = Pipeline([
            ('preprocessor', _preprocessor),
            ('regressor', tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1)
            ]))
        ])
        model.named_steps['regressor'].compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.fit(X_train, y_train, regressor__epochs=50, regressor__batch_size=32, regressor__validation_split=0.2, regressor__verbose=0)
        mse, mae = model.named_steps['regressor'].evaluate(_preprocessor.transform(X_test), y_test, verbose=0)
        return model, mse, mae
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None

# Main app
def main():
    # Sidebar for navigation and info
    with st.sidebar:
        st.header("Air Quality Health Predictor")
        st.markdown("""
            A powerful tool to predict hospital visits based on air quality and environmental factors using a deep learning model.
        """)
        st.markdown("---")
        st.subheader("Navigation")
        st.markdown("""
            - **Model Performance**: View MSE and MAE metrics.
            - **Data Exploration**: Analyze air quality and health trends.
            - **Predict**: Input parameters to forecast hospital visits.
        """)
        st.markdown("---")
        st.subheader("About")
        st.markdown("""
            Built with Streamlit, TensorFlow, and Plotly.  
            Data: Air quality health dataset  
            Model: Neural network with preprocessing  
            Contact: support@airqualitypredictor.com
        """)

    # Main content
    st.title("Air Quality Health Impact Predictor")
    st.markdown("Leverage advanced machine learning to forecast hospital visits based on air quality, weather, and demographic data.")

    # Model Performance Section
    with st.container():
        st.header("Model Performance")
        st.markdown("Evaluate the accuracy of our deep learning model with key performance metrics.")
        df = load_data()
        if df is None:
            return
        X, y, preprocessor = preprocess_data(df)
        if X is None or y is None or preprocessor is None:
            return
        model, mse, mae = train_model(X, y, preprocessor)
        if model is None:
            return
        col1, col2 = st.columns([1, 1], gap="medium")
        with col1:
            st.metric("Mean Squared Error (MSE)", f"{mse:.2f}", help="Measures the average squared difference between predicted and actual hospital visits.")
        with col2:
            st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}", help="Average absolute difference between predicted and actual hospital visits.")

    # Data Exploration Section
    with st.container():
        st.header("Data Exploration")
        st.markdown("Gain insights into air quality and health trends through interactive visualizations.")
        
        # AQI Distribution by Region
        try:
            st.subheader("AQI Distribution by Region")
            fig1 = px.box(df, x='region', y='AQI', title='', color='region')
            fig1.update_layout(
                plot_bgcolor='#1E293B',
                paper_bgcolor='#1E293B',
                font_color='#F1F5F9',
                title_font_color='#F8FAFC',
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
                height=400
            )
            st.plotly_chart(fig1, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating AQI plot: {str(e)}")
        
        # Correlation Heatmap
        try:
            st.subheader("Correlation Heatmap")
            corr = df[['AQI', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'hospital_visits']].corr()
            fig2 = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale='Viridis',
                zmin=-1, zmax=1,
                text=corr.values.round(2),
                texttemplate="%{text}",
                textfont=dict(color='#F1F5F9')
            ))
            fig2.update_layout(
                title='',
                plot_bgcolor='#1E293B',
                paper_bgcolor='#1E293B',
                font_color='#F1F5F9',
                title_font_color='#F8FAFC',
                margin=dict(l=20, r=20, t=20, b=20),
                height=500
            )
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating correlation heatmap: {str(e)}")

    # Prediction Interface Section
    with st.container():
        st.header("Predict Hospital Visits")
        st.markdown("Enter air quality and environmental parameters to forecast hospital visits.")

        with st.form(key="prediction_form"):
            st.subheader("Input Parameters")
            col1, col2 = st.columns([1, 1], gap="medium")
            
            with col1:
                st.markdown("**Air Quality Parameters**")
                aqi = st.number_input("AQI", min_value=0.0, max_value=500.0, value=100.0, step=1.0, help="Air Quality Index (0-500)")
                pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, max_value=500.0, value=50.0, step=1.0, help="Fine particulate matter (<2.5µm)")
                pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, max_value=500.0, value=70.0, step=1.0, help="Coarse particulate matter (<10µm)")
                no2 = st.number_input("NO2 (µg/m³)", min_value=0.0, max_value=200.0, value=40.0, step=1.0, help="Nitrogen Dioxide concentration")
                so2 = st.number_input("SO2 (µg/m³)", min_value=0.0, max_value=200.0, value=20.0, step=1.0, help="Sulfur Dioxide concentration")
                co = st.number_input("CO (mg/m³)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, help="Carbon Monoxide concentration")
                o3 = st.number_input("O3 (µg/m³)", min_value=0.0, max_value=200.0, value=30.0, step=1.0, help="Ozone concentration")
            
            with col2:
                st.markdown("**Environmental & Contextual Parameters**")
                region = st.selectbox("Region", df['region'].unique(), help="Geographical region of measurement")
                temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=25.0, step=1.0, help="Ambient temperature")
                humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0, help="Relative humidity")
                wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=50.0, value=5.0, step=0.1, help="Wind speed in meters per second")
                precipitation = st.number_input("Precipitation (mm)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, help="Precipitation amount")
                population_density = st.number_input("Population Density (people/km²)", min_value=0.0, max_value=10000.0, value=5000.0, step=100.0, help="Population per square kilometer")
                green_cover_percentage = st.number_input("Green Cover (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0, help="Percentage of green cover in the area")
                lockdown_status = st.selectbox("Lockdown Status", [0, 1], format_func=lambda x: "No Lockdown" if x == 0 else "Lockdown", help="Lockdown status (0 = No, 1 = Yes)")
            
            # Submit button
            submit_button = st.form_submit_button("Predict Hospital Visits", use_container_width=True)

        # Handle prediction
        if submit_button:
            try:
                input_data = pd.DataFrame({
                    'AQI': [aqi],
                    'PM2.5': [pm25],
                    'PM10': [pm10],
                    'NO2': [no2],
                    'SO2': [so2],
                    'CO': [co],
                    'O3': [o3],
                    'temperature': [temperature],
                    'humidity': [humidity],
                    'wind_speed': [wind_speed],
                    'precipitation': [precipitation],
                    'population_density': [population_density],
                    'green_cover_percentage': [green_cover_percentage],
                    'region': [region],
                    'lockdown_status': [lockdown_status]
                })
                with st.spinner("Generating prediction..."):
                    prediction = model.predict(input_data)[0]
                
                # Display prediction
                with st.container():
                    st.subheader("Prediction Result")
                    st.markdown(f"**Predicted Hospital Visits**: {float(prediction):.2f}", unsafe_allow_html=True)
                    
                    # Visualize prediction vs average
                    avg_hospital_visits = df['hospital_visits'].mean()
                    fig3 = go.Figure(data=[
                        go.Bar(name='Predicted', x=['Prediction'], y=[prediction], marker_color='#10B981'),
                        go.Bar(name='Average', x=['Average'], y=[avg_hospital_visits], marker_color='#F97316')
                    ])
                    fig3.update_layout(
                        title='',
                        plot_bgcolor='#1E293B',
                        paper_bgcolor='#1E293B',
                        font_color='#F1F5F9',
                        title_font_color='#F8FAFC',
                        height=400,
                        margin=dict(l=20, r=20, t=20, b=20),
                        yaxis_title="Hospital Visits",
                        xaxis_title="",
                        showlegend=True
                    )
                    st.plotly_chart(fig3, use_container_width=True)
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()