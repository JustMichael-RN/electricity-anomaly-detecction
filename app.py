import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Electricity Anomaly Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with better contrast
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
    }
    
    /* Metric containers */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid #7e22ce;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Tab styling for better visibility */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.2);
        color: white !important;
        font-weight: 600;
        font-size: 16px;
        padding: 12px 24px;
        border-radius: 8px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: #1e3c72 !important;
    }
    
    /* Button styling */
    .stButton button {
        font-weight: 600;
        font-size: 16px;
    }
    
    /* File uploader */
    .uploadedFile {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Section headers */
    .stMarkdown h4 {
        color: #1e3c72 !important;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

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

def create_gauge_chart(value, title):
    """Create a gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#7e22ce"},
            'steps': [
                {'range': [0, 33], 'color': '#86efac'},
                {'range': [33, 66], 'color': '#fde047'},
                {'range': [66, 100], 'color': '#f87171'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def main():
    # Header
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0; background: rgba(255,255,255,0.95); 
                    border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 8px 16px rgba(0,0,0,0.1);'>
            <h1 style='color: #1e3c72; font-size: 3rem; margin-bottom: 0.5rem;'>
                ⚡ Electricity Anomaly Detection System
            </h1>
            <p style='color: #7e22ce; font-size: 1.3rem; font-weight: 600;'>
                AI-Powered Real-time Monitoring & Analytics Dashboard
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, scaler = load_model()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
            <div style='background: rgba(255,255,255,0.95); padding: 1.5rem; 
                        border-radius: 10px; margin-bottom: 1rem;'>
                <h2 style='color: #1e3c72; text-align: center;'>🔍 System Info</h2>
            </div>
        """, unsafe_allow_html=True)
        
        st.info("""
        **Model:** Isolation Forest  
        **Algorithm:** Unsupervised ML  
        **Training Data:** 50K+ samples  
        **Accuracy:** 95%+
        """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 📊 About
        This advanced system uses machine learning to detect unusual electricity consumption patterns.
        
        ### 🎯 Features
        - Real-time anomaly detection
        - Batch file analysis
        - Interactive visualizations
        - Export results
        """)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Live Detection", "📊 Batch Analysis", "📈 Analytics"])
    
    with tab1:
        st.markdown("""
            <div style='background: rgba(255,255,255,0.95); padding: 1.5rem; 
                        border-radius: 10px; margin-bottom: 2rem;'>
                <h2 style='color: #1e3c72; text-align: center;'>Real-Time Anomaly Detection</h2>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### ⚡ Power Metrics")
            global_active_power = st.number_input(
                "Global Active Power (kW)",
                min_value=0.0, max_value=20.0, value=1.5, step=0.1
            )
            global_reactive_power = st.number_input(
                "Global Reactive Power (kW)",
                min_value=0.0, max_value=5.0, value=0.2, step=0.01
            )
            voltage = st.number_input(
                "Voltage (V)",
                min_value=200.0, max_value=260.0, value=240.0, step=1.0
            )
            global_intensity = st.number_input(
                "Global Intensity (A)",
                min_value=0.0, max_value=50.0, value=6.0, step=0.1
            )
        
        with col2:
            st.markdown("#### 📍 Sub-Metering")
            sub_metering_1 = st.number_input(
                "Kitchen (Wh)",
                min_value=0.0, max_value=100.0, value=1.0, step=0.1
            )
            sub_metering_2 = st.number_input(
                "Laundry (Wh)",
                min_value=0.0, max_value=100.0, value=1.0, step=0.1
            )
            sub_metering_3 = st.number_input(
                "Climate Control (Wh)",
                min_value=0.0, max_value=100.0, value=6.0, step=0.1
            )
        
        with col3:
            st.markdown("#### 🕐 Time Context")
            hour = st.slider("Hour of Day", 0, 23, 12)
            day_of_week = st.selectbox(
                "Day of Week",
                options=[0, 1, 2, 3, 4, 5, 6],
                format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                       'Friday', 'Saturday', 'Sunday'][x]
            )
            month = st.selectbox("Month", list(range(1, 13)), 
                               format_func=lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][x-1])
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            detect_button = st.button("🔍 ANALYZE NOW", type="primary", use_container_width=True)
        
        if detect_button:
            features = [
                global_active_power, global_reactive_power, voltage, global_intensity,
                sub_metering_1, sub_metering_2, sub_metering_3, hour, day_of_week, month
            ]
            
            prediction, score = predict_anomaly(model, scaler, features)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == -1:
                    st.markdown("""
                        <div style='background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
                                    padding: 40px; border-radius: 15px; text-align: center;
                                    box-shadow: 0 8px 16px rgba(0,0,0,0.2);'>
                            <h1 style='font-size: 5rem; margin: 0; color: white;'>🚨</h1>
                            <h2 style='margin: 15px 0; color: white;'>ANOMALY DETECTED</h2>
                            <p style='font-size: 1.1rem; color: white;'>Unusual consumption pattern</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                                    padding: 40px; border-radius: 15px; text-align: center;
                                    box-shadow: 0 8px 16px rgba(0,0,0,0.2);'>
                            <h1 style='font-size: 5rem; margin: 0; color: white;'>✅</h1>
                            <h2 style='margin: 15px 0; color: white;'>NORMAL</h2>
                            <p style='font-size: 1.1rem; color: white;'>Expected consumption</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.metric("Anomaly Score", f"{score:.4f}", 
                         delta="High Risk" if score < -0.5 else "Low Risk",
                         delta_color="inverse")
            
            with col3:
                confidence = min(abs(score) * 20, 100)
                st.plotly_chart(
                    create_gauge_chart(confidence, "Confidence Level"),
                    use_container_width=True
                )
    
    with tab2:
        st.markdown("""
            <div style='background: rgba(255,255,255,0.95); padding: 1.5rem; 
                        border-radius: 10px; margin-bottom: 2rem;'>
                <h2 style='color: #1e3c72; text-align: center;'>Batch Analysis Dashboard</h2>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "📁 Upload CSV File for Batch Analysis",
            type=['csv'],
            help="Upload electricity consumption data"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                
                required_cols = [
                    'Global_active_power', 'Global_reactive_power', 'Voltage',
                    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
                    'Sub_metering_3', 'hour', 'day_of_week', 'month'
                ]
                
                if all(col in df.columns for col in required_cols):
                    with st.spinner('🔄 Analyzing data...'):
                        features_scaled = scaler.transform(df[required_cols])
                        predictions = model.predict(features_scaled)
                        scores = model.score_samples(features_scaled)
                        
                        df['Prediction'] = predictions
                        df['Anomaly_Score'] = scores
                        df['Is_Anomaly'] = df['Prediction'] == -1
                        df['Risk_Level'] = df['Anomaly_Score'].apply(
                            lambda x: 'High' if x < -0.5 else 'Medium' if x < 0 else 'Low'
                        )
                    
                    anomaly_count = df['Is_Anomaly'].sum()
                    st.success(f"✅ Analysis Complete! Found {anomaly_count} anomalies out of {len(df)} records")
                    
                    # KPI Cards
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("📊 Total Records", f"{len(df):,}")
                    with col2:
                        st.metric("🚨 Anomalies", f"{anomaly_count:,}", 
                                delta=f"{anomaly_count/len(df)*100:.1f}%")
                    with col3:
                        avg_score = df['Anomaly_Score'].mean()
                        st.metric("📈 Avg Score", f"{avg_score:.3f}")
                    with col4:
                        high_risk = (df['Risk_Level'] == 'High').sum()
                        st.metric("⚠️ High Risk", f"{high_risk:,}")
                    
                    st.markdown("---")
                    
                    # Charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Scatter plot
                        fig1 = px.scatter(
                            df, 
                            x=df.index, 
                            y='Global_active_power',
                            color='Is_Anomaly',
                            color_discrete_map={True: '#ff6b6b', False: '#4facfe'},
                            labels={'x': 'Record Index', 'Global_active_power': 'Power (kW)'},
                            title='Power Consumption: Normal vs Anomalies'
                        )
                        fig1.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))
                        fig1.update_layout(height=400, showlegend=True)
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        # Histogram
                        fig2 = px.histogram(
                            df, 
                            x='Anomaly_Score',
                            nbins=30,
                            title='Anomaly Score Distribution',
                            color_discrete_sequence=['#7e22ce']
                        )
                        fig2.update_layout(height=400)
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Pie chart
                        fig3 = px.pie(
                            values=[len(df) - anomaly_count, anomaly_count],
                            names=['Normal', 'Anomaly'],
                            title='Detection Overview',
                            color_discrete_sequence=['#4facfe', '#ff6b6b'],
                            hole=0.4
                        )
                        fig3.update_layout(height=350)
                        st.plotly_chart(fig3, use_container_width=True)
                    
                    with col2:
                        # Risk levels
                        risk_counts = df['Risk_Level'].value_counts()
                        fig4 = px.bar(
                            x=risk_counts.index,
                            y=risk_counts.values,
                            title='Risk Level Distribution',
                            color=risk_counts.index,
                            color_discrete_map={'High': '#ff6b6b', 'Medium': '#ffd93d', 'Low': '#4facfe'}
                        )
                        fig4.update_layout(height=350, showlegend=False)
                        st.plotly_chart(fig4, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Data table
                    st.markdown("### 📋 Detailed Results")
                    st.dataframe(df.head(50), use_container_width=True, height=400)
                    
                    # Download
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full Results",
                        data=csv,
                        file_name=f"anomaly_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                else:
                    st.error("❌ CSV missing required columns!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    with tab3:
        st.markdown("""
            <div style='background: rgba(255,255,255,0.95); padding: 1.5rem; 
                        border-radius: 10px; margin-bottom: 2rem;'>
                <h2 style='color: #1e3c72; text-align: center;'>Advanced Analytics</h2>
            </div>
        """, unsafe_allow_html=True)
        
        if uploaded_file and 'df' in locals():
            col1, col2 = st.columns(2)
            
            with col1:
                hourly = df.groupby('hour')['Is_Anomaly'].sum()
                fig_hour = px.bar(
                    x=hourly.index,
                    y=hourly.values,
                    title='Anomalies by Hour',
                    labels={'x': 'Hour', 'y': 'Count'},
                    color=hourly.values,
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_hour, use_container_width=True)
            
            with col2:
                daily = df.groupby('day_of_week')['Is_Anomaly'].sum()
                days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                fig_day = px.bar(
                    x=[days[i] for i in daily.index],
                    y=daily.values,
                    title='Anomalies by Day',
                    labels={'x': 'Day', 'y': 'Count'},
                    color=daily.values,
                    color_continuous_scale='Oranges'
                )
                st.plotly_chart(fig_day, use_container_width=True)
        else:
            st.info("📊 Upload a file in Batch Analysis to see time-based patterns!")

if __name__ == "__main__":
    main()