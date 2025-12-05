You're right! Here's **FILE 2** separately:

---

## **FILE 2: `app.py`**
Create this file and paste:

```python
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Electricity Anomaly Detection",
    page_icon="⚡",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained model and scaler"""
    try:
        with open('model/anomaly_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data['model'], model_data['scaler']
    except FileNotFoundError:
        st.error("Model file not found! Please train the model first.")
        st.stop()

def predict_anomaly(model, scaler, features):
    """Predict if the input is an anomaly"""
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    score = model.score_samples(features_scaled)
    return prediction[0], score[0]

def main():
    st.title("⚡ Electricity Anomaly Detection System")
    st.markdown("Detect unusual patterns in household electricity consumption")
    
    # Load model
    model, scaler = load_model()
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        "This system detects anomalies in electricity consumption patterns "
        "using an Isolation Forest algorithm trained on household power data."
    )
    
    # Main content
    tab1, tab2 = st.tabs(["🔍 Single Prediction", "📊 Batch Analysis"])
    
    with tab1:
        st.header("Enter Electricity Consumption Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            global_active_power = st.number_input(
                "Global Active Power (kW)",
                min_value=0.0,
                max_value=20.0,
                value=1.5,
                step=0.1
            )
            
            global_reactive_power = st.number_input(
                "Global Reactive Power (kW)",
                min_value=0.0,
                max_value=5.0,
                value=0.2,
                step=0.01
            )
            
            voltage = st.number_input(
                "Voltage (V)",
                min_value=200.0,
                max_value=260.0,
                value=240.0,
                step=1.0
            )
            
            global_intensity = st.number_input(
                "Global Intensity (A)",
                min_value=0.0,
                max_value=50.0,
                value=6.0,
                step=0.1
            )
            
            sub_metering_1 = st.number_input(
                "Sub Metering 1 (Wh)",
                min_value=0.0,
                max_value=100.0,
                value=1.0,
                step=0.1
            )
        
        with col2:
            sub_metering_2 = st.number_input(
                "Sub Metering 2 (Wh)",
                min_value=0.0,
                max_value=100.0,
                value=1.0,
                step=0.1
            )
            
            sub_metering_3 = st.number_input(
                "Sub Metering 3 (Wh)",
                min_value=0.0,
                max_value=100.0,
                value=6.0,
                step=0.1
            )
            
            hour = st.slider("Hour of Day", 0, 23, 12)
            day_of_week = st.selectbox(
                "Day of Week",
                options=[0, 1, 2, 3, 4, 5, 6],
                format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                       'Friday', 'Saturday', 'Sunday'][x]
            )
            month = st.selectbox("Month", list(range(1, 13)))
        
        if st.button("🔍 Detect Anomaly", type="primary"):
            features = [
                global_active_power,
                global_reactive_power,
                voltage,
                global_intensity,
                sub_metering_1,
                sub_metering_2,
                sub_metering_3,
                hour,
                day_of_week,
                month
            ]
            
            prediction, score = predict_anomaly(model, scaler, features)
            
            st.markdown("---")
            st.subheader("Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == -1:
                    st.error("🚨 **ANOMALY DETECTED!**")
                    st.warning("This consumption pattern is unusual and requires attention.")
                else:
                    st.success("✓ **Normal Consumption**")
                    st.info("This consumption pattern is within expected range.")
            
            with col2:
                st.metric("Anomaly Score", f"{score:.4f}")
                st.caption("Lower scores indicate higher anomaly probability")
    
    with tab2:
        st.header("Batch Analysis")
        st.info("Upload a CSV file with electricity consumption data for batch analysis")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                
                required_cols = [
                    'Global_active_power', 'Global_reactive_power', 'Voltage',
                    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
                    'Sub_metering_3', 'hour', 'day_of_week', 'month'
                ]
                
                if all(col in df.columns for col in required_cols):
                    features_scaled = scaler.transform(df[required_cols])
                    predictions = model.predict(features_scaled)
                    scores = model.score_samples(features_scaled)
                    
                    df['Prediction'] = predictions
                    df['Anomaly_Score'] = scores
                    df['Is_Anomaly'] = df['Prediction'] == -1
                    
                    st.success(f"Analysis complete! Found {df['Is_Anomaly'].sum()} anomalies out of {len(df)} records")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Records", len(df))
                    with col2:
                        st.metric("Anomalies", df['Is_Anomaly'].sum())
                    with col3:
                        st.metric("Anomaly Rate", f"{df['Is_Anomaly'].sum()/len(df)*100:.2f}%")
                    
                    st.dataframe(df.head(20))
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download Results",
                        csv,
                        "anomaly_results.csv",
                        "text/csv"
                    )
                else:
                    st.error("CSV file is missing required columns!")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()

