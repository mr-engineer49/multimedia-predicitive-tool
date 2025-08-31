import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
import random
from data_generator import generate_metrics_data, generate_media_quality_data, generate_system_event_data
from anomaly_detection import detect_anomalies, train_isolation_forest, train_autoencoder
from predictive_maintenance import PredictiveMaintenanceModel, forecast_metrics, analyze_system_health_trends
from visualization import (
    create_hardware_metrics_chart, 
    create_media_quality_chart,
    create_system_health_gauge,
    create_anomaly_heatmap
)
import utils
from utils.system_monitor import SystemMonitor, PredictiveHealthAnalyzer
from utils.csv_analyzer import CSVPredictiveAnalyzer
from utils.multimedia_analyzer import MultimediaAnalyzer

# Helper function to safely concatenate DataFrames and avoid deprecation warnings
def safe_concat(df1, df2):
    """
    Safely concatenate two DataFrames while avoiding deprecation warnings.
    
    Args:
        df1 (pandas.DataFrame): First DataFrame
        df2 (pandas.DataFrame): Second DataFrame
        
    Returns:
        pandas.DataFrame: Concatenated DataFrame
    """
    if df1.empty and df2.empty:
        return pd.DataFrame()
    elif df1.empty:
        return df2.copy()
    elif df2.empty:
        return df1.copy()
    else:
        return pd.concat([df1, df2], ignore_index=True, copy=False)

# Page configuration
st.set_page_config(
    page_title="Media Processing Predictive Maintenance",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Application title and description
st.title("Multimedia Processing Predictive Maintenance")
st.markdown("""
    Monitor and analyze media processing workflows with real-time metrics and anomaly detection.
    This dashboard provides insights into hardware performance, media quality, and potential system failures.
""")

# Initialize session state for storing data
if 'hardware_metrics' not in st.session_state:
    st.session_state.hardware_metrics = pd.DataFrame()
    
if 'media_quality_metrics' not in st.session_state:
    st.session_state.media_quality_metrics = pd.DataFrame()
    
if 'anomalies' not in st.session_state:
    st.session_state.anomalies = pd.DataFrame()
    
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = pd.DataFrame()

if 'system_events' not in st.session_state:
    st.session_state.system_events = pd.DataFrame()
    
if 'hardware_forecast' not in st.session_state:
    st.session_state.hardware_forecast = pd.DataFrame()
    
if 'media_forecast' not in st.session_state:
    st.session_state.media_forecast = pd.DataFrame()

if 'health_trend_analysis' not in st.session_state:
    st.session_state.health_trend_analysis = {
        'overall_trend': 'Insufficient data',
        'metrics_trends': {},
        'recommendations': ['Collect more data for trend analysis']
    }
    
if 'predictive_model' not in st.session_state:
    st.session_state.predictive_model = PredictiveMaintenanceModel()
    st.session_state.failure_probability = 0
    st.session_state.time_to_failure = None
    st.session_state.critical_metrics = []
    st.session_state.preventive_actions = []
    
if 'refresh_rate' not in st.session_state:
    st.session_state.refresh_rate = 5  # default refresh rate in seconds

if 'use_realistic_data' not in st.session_state:
    st.session_state.use_realistic_data = True

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

if 'show_forecast' not in st.session_state:
    st.session_state.show_forecast = False

if 'anomaly_models_trained' not in st.session_state:
    st.session_state.anomaly_models_trained = False
    st.session_state.isolation_forest = None
    st.session_state.autoencoder = None
    st.session_state.predictive_model_trained = False
    
# Media content analysis related states
if 'uploaded_media' not in st.session_state:
    st.session_state.uploaded_media = {}
    
if 'media_analysis_results' not in st.session_state:
    st.session_state.media_analysis_results = {}
    
if 'processed_media_count' not in st.session_state:
    st.session_state.processed_media_count = 0

if 'thresholds' not in st.session_state:
    st.session_state.thresholds = {
        'cpu_usage': 85,
        'gpu_usage': 90,
        'memory_usage': 80,
        'latency': 150,
        'frame_drops': 5,
        'encoding_errors': 3,
        'resolution_changes': 2
    }

# Sidebar configuration
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1514861736016-646d83403e24", caption="Monitoring Dashboard")
    
    st.header("⚙️ Settings")
    
    # Refresh rate setting
    st.session_state.refresh_rate = st.slider(
        "Dashboard Refresh Rate (seconds)",
        min_value=1,
        max_value=60,
        value=st.session_state.refresh_rate
    )
    
    # Data generation options
    st.subheader("Data Generation")
    
    # Toggle for realistic data patterns
    if st.toggle("Realistic Data Patterns", st.session_state.use_realistic_data):
        st.session_state.use_realistic_data = True
    else:
        st.session_state.use_realistic_data = False
    
    # Toggle for forecast display
    if st.toggle("Show Forecast", st.session_state.show_forecast):
        st.session_state.show_forecast = True
    else:
        st.session_state.show_forecast = False
    
    # Toggle for dark/light mode
    if st.toggle("Dark Mode", st.session_state.dark_mode):
        st.session_state.dark_mode = True
    else:
        st.session_state.dark_mode = False
    
    # Threshold configuration
    st.subheader("Alert Thresholds")
    with st.expander("Hardware Metrics Thresholds"):
        st.session_state.thresholds['cpu_usage'] = st.slider("CPU Usage (%)", 50, 100, st.session_state.thresholds['cpu_usage'])
        st.session_state.thresholds['gpu_usage'] = st.slider("GPU Usage (%)", 50, 100, st.session_state.thresholds['gpu_usage'])
        st.session_state.thresholds['memory_usage'] = st.slider("Memory Usage (%)", 50, 100, st.session_state.thresholds['memory_usage'])
        st.session_state.thresholds['latency'] = st.slider("Latency (ms)", 50, 500, st.session_state.thresholds['latency'])
    
    with st.expander("Media Quality Thresholds"):
        st.session_state.thresholds['frame_drops'] = st.slider("Frame Drops (per min)", 0, 30, st.session_state.thresholds['frame_drops'])
        st.session_state.thresholds['encoding_errors'] = st.slider("Encoding Errors (per min)", 0, 30, st.session_state.thresholds['encoding_errors'])
        st.session_state.thresholds['resolution_changes'] = st.slider("Resolution Changes (per min)", 0, 10, st.session_state.thresholds['resolution_changes'])
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        # Generate test anomaly
        if st.button("Generate Test Stress Event"):
            # Set stress mode to true in data generator patterns
            from data_generator import _pattern_state
            _pattern_state['stress_mode'] = True
            _pattern_state['stress_timer'] = _pattern_state['stress_duration']
            st.success("Stress event initiated - system load will increase temporarily")
        
        # Clear all data
        if st.button("Clear All Data"):
            for key in ['hardware_metrics', 'media_quality_metrics', 'anomalies', 'alert_history', 'system_events']:
                if key in st.session_state:
                    st.session_state[key] = pd.DataFrame()
            st.session_state.anomaly_models_trained = False
            st.session_state.predictive_model_trained = False
            st.success("All data has been cleared")
    
    st.subheader("About")
    st.markdown("""
        This predictive maintenance system monitors multimedia processing workflows,
        analyzing hardware metrics and media quality to predict potential failures.
    """)
    
    # System information
    st.subheader("System Information")
    st.markdown("**Environment:** Production")
    st.markdown("**Version:** 1.1.0")
    st.markdown("**Last Update:** {}".format(datetime.now().strftime("%Y-%m-%d")))

# Initialize system monitoring and CSV analyzer
if 'system_monitor' not in st.session_state:
    st.session_state.system_monitor = SystemMonitor()
    st.session_state.health_analyzer = PredictiveHealthAnalyzer()
    st.session_state.csv_analyzer = CSVPredictiveAnalyzer()
    st.session_state.multimedia_analyzer = MultimediaAnalyzer()
    st.session_state.real_system_data = pd.DataFrame()
    st.session_state.csv_data = pd.DataFrame()
    st.session_state.csv_analysis_results = {}

# Main dashboard layout with tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Real-time Monitoring", "📈 Predictive Analysis", "🔍 Media Content Analysis", "💾 CSV Data Analysis", "🖥️ System Health Monitor"])

# Tab 1: Real-time monitoring dashboard
with tab1:
    # Create three columns for the top metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("💻 System CPU")
        cpu_gauge = st.empty()

    with col2:
        st.subheader("🎮 GPU Status")
        gpu_gauge = st.empty()

    with col3:
        st.subheader("🧠 Memory Utilization")
        memory_gauge = st.empty()

    # Hardware metrics section
    st.header("Hardware Performance Metrics")
    hardware_metrics_container = st.container()
    with hardware_metrics_container:
        hardware_chart_placeholder = st.empty()
        hardware_metrics_cols = st.columns(4)
        cpu_metric = hardware_metrics_cols[0].empty()
        gpu_metric = hardware_metrics_cols[1].empty()
        memory_metric = hardware_metrics_cols[2].empty()
        latency_metric = hardware_metrics_cols[3].empty()

    # Media quality metrics section
    st.header("Media Processing Quality")
    media_quality_container = st.container()
    with media_quality_container:
        media_chart_placeholder = st.empty()
        media_cols = st.columns(3)
        frame_drops_metric = media_cols[0].empty()
        encoding_errors_metric = media_cols[1].empty()
        resolution_changes_metric = media_cols[2].empty()

    # Anomaly detection section
    st.header("Anomaly Detection & System Health")
    anomaly_container = st.container()
    with anomaly_container:
        anomaly_cols = st.columns([2, 1])
        with anomaly_cols[0]:
            anomaly_chart_placeholder = st.empty()
        with anomaly_cols[1]:
            system_health_placeholder = st.empty()
            predicted_failures_placeholder = st.empty()

    # Alerts section
    st.header("⚠️ System Alerts")
    alerts_container = st.container()
    alerts_placeholder = st.empty()

# Tab 2: Predictive Analysis & Forecasting
with tab2:
    st.header("🔮 Predictive Maintenance & Forecasting")
    
    # Add toggle to enable forecast visualization
    st.session_state.show_forecast = True
    
    # System health trend analysis
    st.subheader("System Health Trend Analysis")
    trend_cols = st.columns([2, 1])
    
    with trend_cols[0]:
        trend_chart_placeholder = st.empty()
        # Create a trend visualization if no data exists yet
        if not st.session_state.hardware_metrics.empty:
            # Visualize system health trend
            import plotly.express as px
            
            # Create system health trend data
            health_data = []
            for i in range(max(10, len(st.session_state.hardware_metrics))):
                if i < len(st.session_state.hardware_metrics):
                    timestamp = st.session_state.hardware_metrics.iloc[i]['timestamp']
                    # Calculate a health score based on metrics
                    hw = st.session_state.hardware_metrics.iloc[i]
                    health_score = 100 - (
                        (hw['cpu_usage'] / st.session_state.thresholds['cpu_usage'] * 25) +
                        (hw['gpu_usage'] / st.session_state.thresholds['gpu_usage'] * 25) +
                        (hw['memory_usage'] / st.session_state.thresholds['memory_usage'] * 25) +
                        (hw['latency'] / st.session_state.thresholds['latency'] * 25)
                    )
                    health_score = max(0, min(100, health_score))
                else:
                    # Create forecasted health points
                    last_timestamp = st.session_state.hardware_metrics.iloc[-1]['timestamp']
                    timestamp = last_timestamp + timedelta(minutes=(i-len(st.session_state.hardware_metrics)+1)*5)
                    # Predict a slightly declining health score for forecasted points
                    last_health = health_data[-1]['health'] if health_data else 80
                    health_score = max(0, min(100, last_health - np.random.uniform(0, 2)))
                
                health_data.append({
                    'timestamp': timestamp,
                    'health': health_score,
                    'type': 'Actual' if i < len(st.session_state.hardware_metrics) else 'Forecast'
                })
            
            health_df = pd.DataFrame(health_data)
            
            # Create the trend chart
            fig = px.line(
                health_df, 
                x='timestamp', 
                y='health',
                color='type',
                title='System Health Trend',
                color_discrete_map={'Actual': '#36B37E', 'Forecast': '#00B8D9'}
            )
            
            fig.update_layout(
                xaxis_title='Time',
                yaxis_title='System Health Score',
                legend_title='Data Type',
                hovermode='x',
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            
            # Generate a random key for the graph to avoid conflicts
            import random
            trend_key = random.randint(700001, 800000)
            
            trend_chart_placeholder.plotly_chart(
                fig,
                use_container_width=True,
                key=f"trend_chart_{trend_key}"
            )
        else:
            trend_chart_placeholder.info("Collecting data for trend analysis...")
        
    with trend_cols[1]:
        trend_analysis_placeholder = st.empty()
        if 'health_trend_analysis' in st.session_state and st.session_state.health_trend_analysis:
            trend_analysis = st.session_state.health_trend_analysis
            
            if 'overall_trend' in trend_analysis:
                trend_analysis_placeholder.markdown(f"### System Health Trend: **{trend_analysis['overall_trend']}**")
                
                if 'recommendations' in trend_analysis and trend_analysis['recommendations']:
                    trend_analysis_placeholder.markdown("#### Recommendations:")
                    for rec in trend_analysis['recommendations']:
                        trend_analysis_placeholder.markdown(f"- {rec}")
            else:
                trend_analysis_placeholder.info("Trend analysis will be available once more data is collected")
        else:
            # Create some default trend analysis
            trend_analysis_placeholder.markdown("### System Health Trend: **Stable**")
            trend_analysis_placeholder.markdown("#### Recommendations:")
            trend_analysis_placeholder.markdown("- Continue monitoring system performance")
            trend_analysis_placeholder.markdown("- Consider optimizing memory usage")
            trend_analysis_placeholder.markdown("- Schedule regular system maintenance")
    
    # Forecast metrics
    st.subheader("Forecasted Metrics")
    forecast_container = st.container()
    with forecast_container:
        forecast_chart_placeholder = st.empty()
        if st.session_state.hardware_forecast.empty:
            # Generate some forecast data if not available
            current_time = datetime.now()
            forecast_data = []
            
            # Use the last point from hardware metrics if available
            last_cpu = 60
            last_gpu = 70
            last_memory = 65
            
            if not st.session_state.hardware_metrics.empty:
                last_hw = st.session_state.hardware_metrics.iloc[-1]
                last_cpu = last_hw['cpu_usage'] 
                last_gpu = last_hw['gpu_usage']
                last_memory = last_hw['memory_usage']
                current_time = last_hw['timestamp']
            
            # Generate 10 future data points with some realistic variance
            for i in range(10):
                # Gradually increase values to show potential issues
                cpu_trend = last_cpu + i * 1.5 + np.random.normal(0, 3)
                gpu_trend = last_gpu + i * 0.8 + np.random.normal(0, 4)
                memory_trend = last_memory + i * 1.2 + np.random.normal(0, 2)
                
                # Keep values in realistic range
                cpu_trend = min(100, max(0, cpu_trend))
                gpu_trend = min(100, max(0, gpu_trend))
                memory_trend = min(100, max(0, memory_trend))
                
                forecast_data.append({
                    'timestamp': current_time + timedelta(minutes=(i+1)*5),
                    'cpu_usage': cpu_trend,
                    'gpu_usage': gpu_trend,
                    'memory_usage': memory_trend
                })
            
            forecast_df = pd.DataFrame(forecast_data)
            
            # Create a forecast chart
            import plotly.graph_objects as go
            fig = go.Figure()
            
            # Add CPU forecast
            fig.add_trace(
                go.Scatter(
                    x=forecast_df['timestamp'],
                    y=forecast_df['cpu_usage'],
                    name='CPU Forecast',
                    line=dict(color='#0747A6', width=2, dash='dash')
                )
            )
            
            # Add GPU forecast
            fig.add_trace(
                go.Scatter(
                    x=forecast_df['timestamp'],
                    y=forecast_df['gpu_usage'],
                    name='GPU Forecast',
                    line=dict(color='#008DA6', width=2, dash='dash')
                )
            )
            
            # Add memory forecast
            fig.add_trace(
                go.Scatter(
                    x=forecast_df['timestamp'],
                    y=forecast_df['memory_usage'],
                    name='Memory Forecast',
                    line=dict(color='#6554C0', width=2, dash='dash')
                )
            )
            
            # Add threshold lines
            fig.add_trace(
                go.Scatter(
                    x=[forecast_df['timestamp'].min(), forecast_df['timestamp'].max()],
                    y=[st.session_state.thresholds['cpu_usage'], st.session_state.thresholds['cpu_usage']],
                    name='CPU Threshold',
                    line=dict(color='#FF5630', width=1, dash='dot')
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[forecast_df['timestamp'].min(), forecast_df['timestamp'].max()],
                    y=[st.session_state.thresholds['gpu_usage'], st.session_state.thresholds['gpu_usage']],
                    name='GPU Threshold',
                    line=dict(color='#FF5630', width=1, dash='dot')
                )
            )
            
            # Update layout
            fig.update_layout(
                title='Resource Usage Forecast',
                xaxis=dict(title='Time'),
                yaxis=dict(title='Usage (%)'),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                ),
                margin=dict(l=20, r=20, t=40, b=20),
                height=350
            )
            
            # Generate a random key for the forecast chart
            forecast_key = random.randint(800001, 900000)
            
            # Display the forecast chart
            forecast_chart_placeholder.plotly_chart(
                fig,
                use_container_width=True,
                key=f"forecast_chart_{forecast_key}"
            )
        else:
            # Display actual forecast data if available
            import plotly.graph_objects as go
            fig = go.Figure()
            
            # Add forecasted CPU data
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.hardware_forecast['timestamp'],
                    y=st.session_state.hardware_forecast['cpu_usage'],
                    name='CPU Forecast',
                    line=dict(color='#0747A6', width=2, dash='dash')
                )
            )
            
            # Add forecasted GPU data if available
            if 'gpu_usage' in st.session_state.hardware_forecast.columns:
                fig.add_trace(
                    go.Scatter(
                        x=st.session_state.hardware_forecast['timestamp'],
                        y=st.session_state.hardware_forecast['gpu_usage'],
                        name='GPU Forecast',
                        line=dict(color='#008DA6', width=2, dash='dash')
                    )
                )
            
            # Add forecasted memory data if available
            if 'memory_usage' in st.session_state.hardware_forecast.columns:
                fig.add_trace(
                    go.Scatter(
                        x=st.session_state.hardware_forecast['timestamp'],
                        y=st.session_state.hardware_forecast['memory_usage'],
                        name='Memory Forecast',
                        line=dict(color='#6554C0', width=2, dash='dash')
                    )
                )
            
            # Update layout
            fig.update_layout(
                title='Resource Usage Forecast',
                xaxis=dict(title='Time'),
                yaxis=dict(title='Usage (%)'),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                ),
                margin=dict(l=20, r=20, t=40, b=20),
                height=350
            )
            
            # Generate a random key for the forecast chart
            forecast_key = random.randint(800001, 900000)
            
            # Display the forecast chart
            forecast_chart_placeholder.plotly_chart(
                fig,
                use_container_width=True,
                key=f"forecast_chart_{forecast_key}"
            )
        
    # Preventive actions
    st.subheader("Recommended Preventive Actions")
    preventive_actions_placeholder = st.empty()
    if not st.session_state.preventive_actions:
        # Default recommendations if none available
        preventive_actions_placeholder.markdown("### Suggested Actions:")
        preventive_actions_placeholder.markdown("1. Monitor CPU usage during peak processing times")
        preventive_actions_placeholder.markdown("2. Consider increasing memory allocation for media processing")
        preventive_actions_placeholder.markdown("3. Schedule regular system optimization routines")
        preventive_actions_placeholder.markdown("4. Update encoding libraries to latest versions")
        preventive_actions_placeholder.markdown("5. Implement cache management to prevent memory buildup")
    else:
        preventive_actions_placeholder.markdown("### Recommended Actions")
        for i, action in enumerate(st.session_state.preventive_actions):
            preventive_actions_placeholder.markdown(f"{i+1}. {action}")
    
    # System events log
    st.header("🔄 System Events Log")
    events_placeholder = st.empty()
    if st.session_state.system_events.empty:
        # Add some default events if none exist
        default_events = [
            {
                'timestamp': datetime.now() - timedelta(minutes=45),
                'component': 'Media Encoder',
                'event_type': 'Configuration Change',
                'message': 'Video codec settings updated',
                'details': 'H.265 encoding parameters optimized for quality/size ratio',
                'severity': 'Normal'
            },
            {
                'timestamp': datetime.now() - timedelta(minutes=30),
                'component': 'Storage System',
                'event_type': 'Maintenance',
                'message': 'Cache cleanup completed',
                'details': 'Temporary files removed to optimize storage performance',
                'severity': 'Normal'
            },
            {
                'timestamp': datetime.now() - timedelta(minutes=15),
                'component': 'Processing Pipeline',
                'event_type': 'Performance Alert',
                'message': 'Processing queue growing',
                'details': 'Media processing queue is increasing faster than completion rate',
                'severity': 'Warning'
            }
        ]
        
        events_df = pd.DataFrame(default_events)
        
        with events_placeholder.container():
            st.write("Recent System Events:")
            
            for _, event in events_df.iterrows():
                severity = event['severity']
                event_color = utils.get_status_color(severity)
                st.markdown(
                    f"""
                    <div style="padding: 10px; border-left: 5px solid {event_color}; margin-bottom: 10px;">
                        <strong>{event['event_type']}</strong> ({event['component']}) - {event['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
                        <br/>
                        <span>{event['message']}</span>
                        <br/>
                        <small style="color: #666;">{event['details']}</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        with events_placeholder.container():
            st.write("Recent System Events:")
            
            # Display the 20 most recent events
            recent_events = st.session_state.system_events.sort_values('timestamp', ascending=False).head(20)
            
            for _, event in recent_events.iterrows():
                severity = event['severity']
                event_color = utils.get_status_color(severity)
                st.markdown(
                    f"""
                    <div style="padding: 10px; border-left: 5px solid {event_color}; margin-bottom: 10px;">
                        <strong>{event['event_type']}</strong> ({event['component']}) - {event['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
                        <br/>
                        <span>{event['message']}</span>
                        <br/>
                        <small style="color: #666;">{event['details']}</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# Function to update dashboard metrics
def update_dashboard():
    # Generate real-time data using the realistic mode setting
    new_hardware_data = generate_metrics_data(realistic_mode=st.session_state.use_realistic_data)
    new_media_data = generate_media_quality_data(realistic_mode=st.session_state.use_realistic_data)
    
    # Occasionally generate system events
    if np.random.random() < 0.05:  # 5% chance each update
        new_events = generate_system_event_data(num_points=1)
        if st.session_state.system_events.empty:
            st.session_state.system_events = new_events
        else:
            st.session_state.system_events = safe_concat(st.session_state.system_events, new_events)
            # Keep only the most recent 50 events
            if len(st.session_state.system_events) > 50:
                st.session_state.system_events = st.session_state.system_events.iloc[-50:]
    
    # Update session state with new data
    if st.session_state.hardware_metrics.empty:
        st.session_state.hardware_metrics = new_hardware_data
    else:
        st.session_state.hardware_metrics = safe_concat(st.session_state.hardware_metrics, new_hardware_data)
        # Keep only the last 100 data points for display
        if len(st.session_state.hardware_metrics) > 100:
            st.session_state.hardware_metrics = st.session_state.hardware_metrics.iloc[-100:]
    
    if st.session_state.media_quality_metrics.empty:
        st.session_state.media_quality_metrics = new_media_data
    else:
        st.session_state.media_quality_metrics = safe_concat(st.session_state.media_quality_metrics, new_media_data)
        # Keep only the last 100 data points for display
        if len(st.session_state.media_quality_metrics) > 100:
            st.session_state.media_quality_metrics = st.session_state.media_quality_metrics.iloc[-100:]
    
    # Train anomaly detection models if necessary
    if len(st.session_state.hardware_metrics) > 20 and not st.session_state.anomaly_models_trained:
        with st.spinner("Training anomaly detection models..."):
            # Combine hardware and media metrics for anomaly detection
            combined_data = pd.merge(
                st.session_state.hardware_metrics, 
                st.session_state.media_quality_metrics,
                on="timestamp"
            )
            
            if not combined_data.empty and len(combined_data) > 10:
                features = combined_data.drop("timestamp", axis=1)
                
                # Train isolation forest model
                st.session_state.isolation_forest = train_isolation_forest(features)
                
                # Train autoencoder model (simplified)
                st.session_state.autoencoder = train_autoencoder(features)
                
                st.session_state.anomaly_models_trained = True
    
    # Perform anomaly detection
    if st.session_state.anomaly_models_trained:
        # Combine latest hardware and media metrics
        if not new_hardware_data.empty and not new_media_data.empty:
            latest_data = pd.merge(
                new_hardware_data,
                new_media_data,
                on="timestamp"
            )
            
            if not latest_data.empty:
                # Detect anomalies in the latest data
                anomalies = detect_anomalies(
                    latest_data,
                    st.session_state.isolation_forest,
                    st.session_state.autoencoder
                )
                
                if not anomalies.empty:
                    if st.session_state.anomalies.empty:
                        st.session_state.anomalies = anomalies
                    else:
                        st.session_state.anomalies = safe_concat(st.session_state.anomalies, anomalies)
                        # Keep only the last 50 anomalies
                        if len(st.session_state.anomalies) > 50:
                            st.session_state.anomalies = st.session_state.anomalies.iloc[-50:]
    
    # Check for alerts based on thresholds
    if not new_hardware_data.empty:
        latest_hw = new_hardware_data.iloc[-1]
        
        # CPU alert
        if latest_hw['cpu_usage'] > st.session_state.thresholds['cpu_usage']:
            severity = "High" if latest_hw['cpu_usage'] > st.session_state.thresholds['cpu_usage'] + 10 else "Medium"
            new_alert = pd.DataFrame({
                'timestamp': [latest_hw['timestamp']],
                'metric': ['CPU Usage'],
                'value': [latest_hw['cpu_usage']],
                'threshold': [st.session_state.thresholds['cpu_usage']],
                'severity': [severity]
            })
            st.session_state.alert_history = safe_concat(st.session_state.alert_history, new_alert)
        
        # GPU alert
        if latest_hw['gpu_usage'] > st.session_state.thresholds['gpu_usage']:
            severity = "High" if latest_hw['gpu_usage'] > st.session_state.thresholds['gpu_usage'] + 10 else "Medium"
            new_alert = pd.DataFrame({
                'timestamp': [latest_hw['timestamp']],
                'metric': ['GPU Usage'],
                'value': [latest_hw['gpu_usage']],
                'threshold': [st.session_state.thresholds['gpu_usage']],
                'severity': [severity]
            })
            st.session_state.alert_history = safe_concat(st.session_state.alert_history, new_alert)
        
        # Memory alert
        if latest_hw['memory_usage'] > st.session_state.thresholds['memory_usage']:
            severity = "High" if latest_hw['memory_usage'] > st.session_state.thresholds['memory_usage'] + 10 else "Medium"
            new_alert = pd.DataFrame({
                'timestamp': [latest_hw['timestamp']],
                'metric': ['Memory Usage'],
                'value': [latest_hw['memory_usage']],
                'threshold': [st.session_state.thresholds['memory_usage']],
                'severity': [severity]
            })
            st.session_state.alert_history = safe_concat(st.session_state.alert_history, new_alert)
        
        # Latency alert
        if latest_hw['latency'] > st.session_state.thresholds['latency']:
            severity = "High" if latest_hw['latency'] > st.session_state.thresholds['latency'] + 50 else "Medium"
            new_alert = pd.DataFrame({
                'timestamp': [latest_hw['timestamp']],
                'metric': ['Latency'],
                'value': [latest_hw['latency']],
                'threshold': [st.session_state.thresholds['latency']],
                'severity': [severity]
            })
            st.session_state.alert_history = safe_concat(st.session_state.alert_history, new_alert)
    
    if not new_media_data.empty:
        latest_media = new_media_data.iloc[-1]
        
        # Frame drops alert
        if latest_media['frame_drops'] > st.session_state.thresholds['frame_drops']:
            severity = "High" if latest_media['frame_drops'] > st.session_state.thresholds['frame_drops'] + 5 else "Medium"
            new_alert = pd.DataFrame({
                'timestamp': [latest_media['timestamp']],
                'metric': ['Frame Drops'],
                'value': [latest_media['frame_drops']],
                'threshold': [st.session_state.thresholds['frame_drops']],
                'severity': [severity]
            })
            st.session_state.alert_history = safe_concat(st.session_state.alert_history, new_alert)
        
        # Encoding errors alert
        if latest_media['encoding_errors'] > st.session_state.thresholds['encoding_errors']:
            severity = "High" if latest_media['encoding_errors'] > st.session_state.thresholds['encoding_errors'] + 2 else "Medium"
            new_alert = pd.DataFrame({
                'timestamp': [latest_media['timestamp']],
                'metric': ['Encoding Errors'],
                'value': [latest_media['encoding_errors']],
                'threshold': [st.session_state.thresholds['encoding_errors']],
                'severity': [severity]
            })
            st.session_state.alert_history = safe_concat(st.session_state.alert_history, new_alert)
        
        # Resolution changes alert
        if latest_media['resolution_changes'] > st.session_state.thresholds['resolution_changes']:
            severity = "High" if latest_media['resolution_changes'] > st.session_state.thresholds['resolution_changes'] + 2 else "Medium"
            new_alert = pd.DataFrame({
                'timestamp': [latest_media['timestamp']],
                'metric': ['Resolution Changes'],
                'value': [latest_media['resolution_changes']],
                'threshold': [st.session_state.thresholds['resolution_changes']],
                'severity': [severity]
            })
            st.session_state.alert_history = safe_concat(st.session_state.alert_history, new_alert)
    
    # Keep only the most recent 100 alerts
    if len(st.session_state.alert_history) > 100:
        st.session_state.alert_history = st.session_state.alert_history.iloc[-100:]
    
    # Update dashboard visualizations
    if not st.session_state.hardware_metrics.empty:
        latest_hw = st.session_state.hardware_metrics.iloc[-1]
        
        # Make keys unique using a random suffix
        import random
        rand_key = random.randint(1, 100000)
        
        # Update gauges with completely unique keys
        cpu_gauge.plotly_chart(
            create_system_health_gauge("CPU Usage", latest_hw['cpu_usage'], st.session_state.thresholds['cpu_usage']),
            use_container_width=True,
            key=f"cpu_gauge_{rand_key}"
        )
        
        gpu_gauge.plotly_chart(
            create_system_health_gauge("GPU Usage", latest_hw['gpu_usage'], st.session_state.thresholds['gpu_usage']),
            use_container_width=True,
            key=f"gpu_gauge_{rand_key}"
        )
        
        memory_gauge.plotly_chart(
            create_system_health_gauge("Memory Usage", latest_hw['memory_usage'], st.session_state.thresholds['memory_usage']),
            use_container_width=True,
            key=f"memory_gauge_{rand_key}"
        )
        
        # Update hardware metrics with unique key
        hardware_chart_placeholder.plotly_chart(
            create_hardware_metrics_chart(st.session_state.hardware_metrics),
            use_container_width=True,
            key=f"hardware_chart_{rand_key}"
        )
        
        cpu_status = utils.get_alert_status(latest_hw['cpu_usage'], st.session_state.thresholds['cpu_usage'])
        gpu_status = utils.get_alert_status(latest_hw['gpu_usage'], st.session_state.thresholds['gpu_usage'])
        memory_status = utils.get_alert_status(latest_hw['memory_usage'], st.session_state.thresholds['memory_usage'])
        latency_status = utils.get_alert_status(latest_hw['latency'], st.session_state.thresholds['latency'])
        
        cpu_metric.metric("CPU Usage", f"{latest_hw['cpu_usage']:.1f}%", delta=f"{latest_hw['cpu_usage'] - latest_hw['cpu_usage_prev']:.1f}%", delta_color="inverse")
        gpu_metric.metric("GPU Usage", f"{latest_hw['gpu_usage']:.1f}%", delta=f"{latest_hw['gpu_usage'] - latest_hw['gpu_usage_prev']:.1f}%", delta_color="inverse")
        memory_metric.metric("Memory Usage", f"{latest_hw['memory_usage']:.1f}%", delta=f"{latest_hw['memory_usage'] - latest_hw['memory_usage_prev']:.1f}%", delta_color="inverse")
        latency_metric.metric("Latency", f"{latest_hw['latency']:.1f} ms", delta=f"{latest_hw['latency'] - latest_hw['latency_prev']:.1f} ms", delta_color="inverse")
    
    if not st.session_state.media_quality_metrics.empty:
        latest_media = st.session_state.media_quality_metrics.iloc[-1]
        
        # Make a new random key for this section
        rand_key2 = random.randint(100001, 200000)
        
        # Update media quality chart with unique key
        media_chart_placeholder.plotly_chart(
            create_media_quality_chart(st.session_state.media_quality_metrics),
            use_container_width=True,
            key=f"media_quality_chart_{rand_key2}"
        )
        
        frame_drops_status = utils.get_alert_status(latest_media['frame_drops'], st.session_state.thresholds['frame_drops'])
        encoding_errors_status = utils.get_alert_status(latest_media['encoding_errors'], st.session_state.thresholds['encoding_errors'])
        resolution_changes_status = utils.get_alert_status(latest_media['resolution_changes'], st.session_state.thresholds['resolution_changes'])
        
        frame_drops_metric.metric("Frame Drops", f"{latest_media['frame_drops']}/min", delta=f"{latest_media['frame_drops'] - latest_media['frame_drops_prev']}", delta_color="inverse")
        encoding_errors_metric.metric("Encoding Errors", f"{latest_media['encoding_errors']}/min", delta=f"{latest_media['encoding_errors'] - latest_media['encoding_errors_prev']}", delta_color="inverse")
        resolution_changes_metric.metric("Resolution Changes", f"{latest_media['resolution_changes']}/min", delta=f"{latest_media['resolution_changes'] - latest_media['resolution_changes_prev']}", delta_color="inverse")
    
    # Update anomaly visualizations
    if not st.session_state.anomalies.empty:
        # Make a new random key for anomaly section
        rand_key3 = random.randint(200001, 300000)
        
        anomaly_chart_placeholder.plotly_chart(
            create_anomaly_heatmap(st.session_state.anomalies),
            use_container_width=True,
            key=f"anomaly_heatmap_{rand_key3}"
        )
        
        # Update the predictive analysis tab
        if not st.session_state.preventive_actions:
            preventive_actions_placeholder.info("Not enough data for predictive maintenance recommendations yet. Continue collecting data.")
        else:
            preventive_actions_placeholder.markdown("### Recommended Actions")
            for i, action in enumerate(st.session_state.preventive_actions):
                preventive_actions_placeholder.markdown(f"{i+1}. {action}")
        
        # Display system events
        if not st.session_state.system_events.empty:
            with events_placeholder.container():
                st.write("Recent System Events:")
                
                # Display the 20 most recent events
                recent_events = st.session_state.system_events.sort_values('timestamp', ascending=False).head(20)
                
                for _, event in recent_events.iterrows():
                    severity = event['severity']
                    event_color = utils.get_status_color(severity)
                    st.markdown(
                        f"""
                        <div style="padding: 10px; border-left: 5px solid {event_color}; margin-bottom: 10px;">
                            <strong>{event['event_type']}</strong> ({event['component']}) - {event['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
                            <br/>
                            <span>{event['message']}</span>
                            <br/>
                            <small style="color: #666;">{event['details']}</small>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            events_placeholder.info("No system events recorded yet.")
            
        # Display forecast data if available and requested
        if st.session_state.show_forecast and not st.session_state.hardware_forecast.empty:
            import plotly.graph_objects as go
            
            # Create a figure for the forecast chart
            fig = go.Figure()
            
            # Add actual CPU data
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.hardware_metrics['timestamp'],
                    y=st.session_state.hardware_metrics['cpu_usage'],
                    name='Actual CPU',
                    line=dict(color='#0747A6', width=2)
                )
            )
            
            # Add forecasted CPU data
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.hardware_forecast['timestamp'],
                    y=st.session_state.hardware_forecast['cpu_usage'],
                    name='Forecast CPU',
                    line=dict(color='#0747A6', width=2, dash='dash')
                )
            )
            
            # Update layout
            fig.update_layout(
                title='CPU Usage Forecast',
                xaxis=dict(title='Time'),
                yaxis=dict(title=dict(text='Usage (%)')),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                ),
                margin=dict(l=20, r=20, t=40, b=20),
                height=350
            )
            
            # Make a new random key for forecast chart
            rand_key4 = random.randint(300001, 400000)
            
            # Display the forecast chart
            forecast_chart_placeholder.plotly_chart(
                fig,
                use_container_width=True,
                key=f"forecast_chart_{rand_key4}"
            )
            
            # Display trend analysis if available
            if st.session_state.health_trend_analysis and 'overall_trend' in st.session_state.health_trend_analysis:
                trend_analysis = st.session_state.health_trend_analysis
                
                trend_analysis_placeholder.markdown(f"### System Health Trend: **{trend_analysis['overall_trend']}**")
                
                if 'recommendations' in trend_analysis and trend_analysis['recommendations']:
                    trend_analysis_placeholder.markdown("#### Recommendations:")
                    for rec in trend_analysis['recommendations']:
                        trend_analysis_placeholder.markdown(f"- {rec}")
        
        elif st.session_state.show_forecast:
            forecast_chart_placeholder.info("Not enough data for forecasting yet. Continue collecting data.")
            trend_analysis_placeholder.info("Health trend analysis will be available once more data is collected.")
        
        # Calculate system health score
        system_health = utils.calculate_system_health(
            st.session_state.hardware_metrics, 
            st.session_state.media_quality_metrics,
            st.session_state.anomalies,
            st.session_state.thresholds
        )
        
        # Make a new random key for system health gauge
        rand_key5 = random.randint(400001, 500000)
        
        system_health_placeholder.plotly_chart(
            create_system_health_gauge("System Health", system_health, 60, min_value=0, max_value=100),
            use_container_width=True,
            key=f"system_health_gauge_{rand_key5}"
        )
        
        # Train and use predictive maintenance model if enough data is available
        if len(st.session_state.hardware_metrics) > 30 and not st.session_state.predictive_model_trained:
            # Train the predictive model
            training_success = st.session_state.predictive_model.train(
                st.session_state.hardware_metrics,
                st.session_state.media_quality_metrics
            )
            if training_success:
                st.session_state.predictive_model_trained = True
        
        # If the model is trained, analyze system health and predict failures
        if st.session_state.predictive_model_trained:
            # Analyze failure risk
            risk_analysis = st.session_state.predictive_model.analyze_failure_risk(
                st.session_state.hardware_metrics,
                st.session_state.media_quality_metrics,
                st.session_state.thresholds
            )
            
            # Update session state with results
            st.session_state.failure_probability = risk_analysis['failure_probability']
            st.session_state.time_to_failure = risk_analysis['time_to_failure']
            st.session_state.critical_metrics = risk_analysis['critical_metrics']
            
            # Get preventive actions
            st.session_state.preventive_actions = st.session_state.predictive_model.get_preventive_actions()
            
            # Generate forecasts if requested
            if st.session_state.show_forecast and len(st.session_state.hardware_metrics) > 30:
                st.session_state.hardware_forecast, st.session_state.media_forecast = forecast_metrics(
                    st.session_state.hardware_metrics,
                    st.session_state.media_quality_metrics,
                    forecast_periods=10
                )
                
                # Update health trend analysis
                st.session_state.health_trend_analysis = analyze_system_health_trends(
                    st.session_state.hardware_metrics,
                    st.session_state.media_quality_metrics,
                    st.session_state.thresholds,
                    window=min(30, len(st.session_state.hardware_metrics))
                )
        
        # Predicted failures
        recent_anomalies = st.session_state.anomalies.iloc[-5:] if len(st.session_state.anomalies) >= 5 else st.session_state.anomalies
        predicted_issues = []
        
        if 'cpu_anomaly' in recent_anomalies.columns and recent_anomalies['cpu_anomaly'].sum() > 2:
            predicted_issues.append("Potential CPU overload within 30 minutes")
        
        if 'gpu_anomaly' in recent_anomalies.columns and recent_anomalies['gpu_anomaly'].sum() > 2:
            predicted_issues.append("Potential GPU failure within 1 hour")
        
        if 'memory_anomaly' in recent_anomalies.columns and recent_anomalies['memory_anomaly'].sum() > 2:
            predicted_issues.append("Potential memory leak detected")
        
        if 'frame_drops_anomaly' in recent_anomalies.columns and recent_anomalies['frame_drops_anomaly'].sum() > 2:
            predicted_issues.append("Media encoder may fail within 2 hours")
        
        if not predicted_issues:
            predicted_issues.append("No imminent failures predicted")
        
        predicted_failures_placeholder.markdown("### Predicted Issues")
        for issue in predicted_issues:
            predicted_failures_placeholder.markdown(f"- {issue}")
    
    # Update alerts section
    if not st.session_state.alert_history.empty:
        with alerts_placeholder.container():
            st.write("Recent System Alerts:")
            
            # Display the 10 most recent alerts
            recent_alerts = st.session_state.alert_history.sort_values('timestamp', ascending=False).head(10)
            
            for _, alert in recent_alerts.iterrows():
                alert_color = utils.get_status_color(alert['severity'])
                st.markdown(
                    f"""
                    <div style="padding: 10px; border-left: 5px solid {alert_color}; margin-bottom: 10px;">
                        <strong>{alert['metric']}</strong> - {alert['timestamp'].strftime('%H:%M:%S')} - 
                        Value: {alert['value']:.1f}, Threshold: {alert['threshold']} - 
                        <span style="color: {alert_color};">{alert['severity']} Alert</span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
    else:
        alerts_placeholder.info("No alerts detected. System running normally.")

# Media content analysis section
with tab3:
    st.header("🎬 Media Content Analysis")
    
    st.markdown("""
    Upload your media files (images, videos, audio) for analysis and predictive maintenance checks.
    The system will analyze the content and provide insights into potential quality issues and optimization opportunities.
    """)
    
    # File upload section
    st.subheader("Upload Media Content")
    file_types = ["Image", "Video", "Audio"]
    selected_type = st.selectbox("Select Media Type", file_types)
    
    # Display appropriate file uploader based on selection
    if selected_type == "Image":
        uploaded_file = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png", "webp", "bmp"])
        file_info = "Supported formats: JPG, PNG, WebP, BMP"
    elif selected_type == "Video":
        uploaded_file = st.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
        file_info = "Supported formats: MP4, MOV, AVI, MKV"
    else:  # Audio
        uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "ogg", "flac"])
        file_info = "Supported formats: MP3, WAV, OGG, FLAC"
    
    st.caption(file_info)
    
    # Analysis options
    st.subheader("Analysis Options")
    analysis_cols = st.columns(3)
    
    with analysis_cols[0]:
        quality_check = st.checkbox("Quality Assessment", value=True)
    with analysis_cols[1]:
        content_analysis = st.checkbox("Content Analysis", value=True)
    with analysis_cols[2]:
        encoding_check = st.checkbox("Encoding Analysis", value=True)
    
    # Advanced options
    with st.expander("Advanced Analysis Options"):
        if selected_type == "Image":
            resolution_check = st.checkbox("Check Resolution Optimization", value=True)
            metadata_check = st.checkbox("Extract & Analyze Metadata", value=True)
            compression_level = st.slider("Compression Analysis Sensitivity", 1, 10, 5)
        elif selected_type == "Video":
            frame_rate_check = st.checkbox("Frame Rate Analysis", value=True)
            bitrate_check = st.checkbox("Bitrate Analysis", value=True)
            codec_check = st.checkbox("Codec Efficiency Check", value=True)
        else:  # Audio
            sample_rate_check = st.checkbox("Sample Rate Analysis", value=True)
            bitrate_check = st.checkbox("Bitrate Analysis", value=True)
            channel_check = st.checkbox("Channel Configuration Check", value=True)
    
    # Process button
    if uploaded_file is not None:
        file_id = f"{selected_type}_{len(st.session_state.uploaded_media) + 1}"
        
        if st.button("Analyze Media"):
            with st.spinner(f"Analyzing {selected_type.lower()} content..."):
                # Store the file in session state
                st.session_state.uploaded_media[file_id] = {
                    'type': selected_type,
                    'name': uploaded_file.name,
                    'timestamp': datetime.now(),
                    'content': uploaded_file.getvalue()
                }
                
                # Generate simulated analysis results based on media type
                results = {}
                
                if selected_type == "Image":
                    results = {
                        'quality_score': round(np.random.uniform(60, 95), 1),
                        'resolution': f"{np.random.randint(800, 4000)}x{np.random.randint(600, 3000)}",
                        'format_efficiency': round(np.random.uniform(50, 95), 1),
                        'compression_level': np.random.randint(1, 10),
                        'metadata_issues': np.random.randint(0, 3),
                        'optimization_potential': round(np.random.uniform(10, 40), 1)
                    }
                    
                    # Add potential issues
                    issues = []
                    if results['quality_score'] < 70:
                        issues.append("Image quality is below recommended threshold")
                    if results['format_efficiency'] < 70:
                        issues.append("Image format is not optimal for web delivery")
                    if results['compression_level'] > 7:
                        issues.append("Image appears to be over-compressed")
                    if results['metadata_issues'] > 0:
                        issues.append(f"Found {results['metadata_issues']} metadata issues")
                    
                    if not issues:
                        issues.append("No significant issues detected")
                    
                    results['issues'] = issues
                    
                elif selected_type == "Video":
                    results = {
                        'quality_score': round(np.random.uniform(60, 95), 1),
                        'resolution': f"{np.random.randint(800, 4000)}x{np.random.randint(600, 3000)}",
                        'frame_rate': np.random.choice([24, 30, 60, 120]),
                        'bitrate': f"{np.random.randint(1, 20)} Mbps",
                        'codec_efficiency': round(np.random.uniform(50, 95), 1),
                        'encoding_issues': np.random.randint(0, 5),
                        'optimization_potential': round(np.random.uniform(10, 50), 1)
                    }
                    
                    # Add potential issues
                    issues = []
                    if results['quality_score'] < 70:
                        issues.append("Video quality is below recommended threshold")
                    if results['codec_efficiency'] < 70:
                        issues.append("Video codec is not optimal for streaming")
                    if results['encoding_issues'] > 0:
                        issues.append(f"Found {results['encoding_issues']} encoding issues")
                    
                    if not issues:
                        issues.append("No significant issues detected")
                    
                    results['issues'] = issues
                    
                else:  # Audio
                    results = {
                        'quality_score': round(np.random.uniform(60, 95), 1),
                        'sample_rate': np.random.choice([8000, 16000, 44100, 48000, 96000]),
                        'bitrate': f"{np.random.choice([96, 128, 192, 256, 320])} kbps",
                        'channels': np.random.choice(["Mono", "Stereo", "5.1 Surround"]),
                        'codec_efficiency': round(np.random.uniform(50, 95), 1),
                        'encoding_issues': np.random.randint(0, 3),
                        'optimization_potential': round(np.random.uniform(10, 40), 1)
                    }
                    
                    # Add potential issues
                    issues = []
                    if results['quality_score'] < 70:
                        issues.append("Audio quality is below recommended threshold")
                    if results['codec_efficiency'] < 70:
                        issues.append("Audio codec is not optimal for streaming")
                    if results['encoding_issues'] > 0:
                        issues.append(f"Found {results['encoding_issues']} encoding issues")
                    
                    if not issues:
                        issues.append("No significant issues detected")
                    
                    results['issues'] = issues
                
                # Store analysis results
                st.session_state.media_analysis_results[file_id] = results
                st.session_state.processed_media_count += 1
                
                st.success(f"{selected_type} analyzed successfully!")
    
    # Display analysis results
    if st.session_state.processed_media_count > 0:
        st.header("Analysis Results")
        
        # Create tabs for each processed media file
        media_tabs = st.tabs([f"{info['type']}: {info['name']}" for file_id, info in st.session_state.uploaded_media.items()])
        
        for i, (file_id, media_info) in enumerate(st.session_state.uploaded_media.items()):
            with media_tabs[i]:
                if file_id in st.session_state.media_analysis_results:
                    results = st.session_state.media_analysis_results[file_id]
                    
                    # Display uploaded media
                    if media_info['type'] == "Image":
                        st.image(media_info['content'], caption=media_info['name'])
                    elif media_info['type'] == "Video":
                        st.video(media_info['content'])
                    else:  # Audio
                        st.audio(media_info['content'])
                    
                    # Create two columns for results
                    res_col1, res_col2 = st.columns([2, 1])
                    
                    with res_col1:
                        # Display results in a nice format
                        st.subheader("Technical Analysis")
                        
                        if media_info['type'] == "Image":
                            st.markdown(f"**Resolution:** {results['resolution']}")
                            st.markdown(f"**Format Efficiency:** {results['format_efficiency']}%")
                            st.markdown(f"**Compression Level:** {results['compression_level']}/10")
                            st.markdown(f"**Metadata Issues:** {results['metadata_issues']}")
                        elif media_info['type'] == "Video":
                            st.markdown(f"**Resolution:** {results['resolution']}")
                            st.markdown(f"**Frame Rate:** {results['frame_rate']} fps")
                            st.markdown(f"**Bitrate:** {results['bitrate']}")
                            st.markdown(f"**Codec Efficiency:** {results['codec_efficiency']}%")
                            st.markdown(f"**Encoding Issues:** {results['encoding_issues']}")
                        else:  # Audio
                            st.markdown(f"**Sample Rate:** {results['sample_rate']} Hz")
                            st.markdown(f"**Bitrate:** {results['bitrate']}")
                            st.markdown(f"**Channels:** {results['channels']}")
                            st.markdown(f"**Codec Efficiency:** {results['codec_efficiency']}%")
                            st.markdown(f"**Encoding Issues:** {results['encoding_issues']}")
                        
                        # Display identified issues
                        st.subheader("Identified Issues")
                        for issue in results['issues']:
                            st.markdown(f"- {issue}")
                            
                        # Display timestamp
                        st.caption(f"Analyzed on: {media_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    with res_col2:
                        # Display quality score gauge
                        st.subheader("Quality Score")
                        
                        # Generate a random key for the gauge to avoid conflicts
                        import random
                        gauge_key = random.randint(600001, 700000)
                        
                        quality_threshold = 75
                        st.plotly_chart(
                            create_system_health_gauge("Quality", results['quality_score'], quality_threshold),
                            use_container_width=True,
                            key=f"quality_gauge_{gauge_key}"
                        )
                        
                        # Optimization recommendation
                        st.subheader("Optimization Potential")
                        st.info(f"{results['optimization_potential']}% improvement possible")
                        
                        # Provide recommendations
                        st.subheader("Recommendations")
                        if results['quality_score'] < 70:
                            st.markdown("• Consider using a higher quality source")
                        
                        if media_info['type'] == "Image" and results['format_efficiency'] < 70:
                            st.markdown("• Convert to a more efficient format (WebP)")
                        elif media_info['type'] == "Video" and results['codec_efficiency'] < 70:
                            st.markdown("• Re-encode using H.265/HEVC or AV1")
                        elif media_info['type'] == "Audio" and results['codec_efficiency'] < 70:
                            st.markdown("• Consider using AAC or Opus codec")
                            
                        if results['optimization_potential'] > 30:
                            st.markdown("• Significant optimization possible")
                            st.markdown("• Consider professional re-encoding")
                else:
                    st.warning("Analysis results not available for this file.")
        
        # Summary statistics
        st.header("Media Library Health Summary")
        
        # Calculate average quality score and other metrics
        avg_quality = np.mean([results['quality_score'] for results in st.session_state.media_analysis_results.values()])
        total_issues = sum([len(results['issues']) for results in st.session_state.media_analysis_results.values()])
        avg_optimization = np.mean([results['optimization_potential'] for results in st.session_state.media_analysis_results.values()])
        
        # Display summary metrics
        summary_cols = st.columns(4)
        summary_cols[0].metric("Files Analyzed", st.session_state.processed_media_count)
        summary_cols[1].metric("Avg. Quality Score", f"{avg_quality:.1f}%")
        summary_cols[2].metric("Total Issues", total_issues)
        summary_cols[3].metric("Avg. Optimization Potential", f"{avg_optimization:.1f}%")
        
        # Display overall health assessment
        if avg_quality >= 85:
            health_status = "Excellent"
            health_color = "#36B37E"  # Green
        elif avg_quality >= 70:
            health_status = "Good"
            health_color = "#00B8D9"  # Blue
        elif avg_quality >= 50:
            health_status = "Fair"
            health_color = "#FFAB00"  # Yellow/Orange
        else:
            health_status = "Poor"
            health_color = "#FF5630"  # Red
        
        st.markdown(f"""
        <div style="padding: 15px; border-radius: 5px; background-color: {health_color}33; border-left: 5px solid {health_color}; margin: 10px 0;">
            <strong style="color: {health_color};">Media Library Health: {health_status}</strong>
            <p>Based on the analysis of {st.session_state.processed_media_count} files, your media library is in {health_status.lower()} condition.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Export option
        if st.button("Export Analysis Report"):
            st.success("Analysis report would be exported (functionality simulated)")
            # In a real implementation, this would generate and download a PDF or CSV

# Initialize application with some data to avoid empty dashboards at startup
def initialize_data_if_empty():
    # Only initialize if hardware metrics are empty (first run)
    if 'hardware_metrics' in st.session_state and st.session_state.hardware_metrics.empty:
        # Generate initial data points for a good starting state (30 data points)
        for _ in range(30):
            update_dashboard()
        
        # Force train models with initial data
        st.session_state.anomaly_models_trained = False
        
        combined_data = pd.merge(
            st.session_state.hardware_metrics, 
            st.session_state.media_quality_metrics,
            on="timestamp"
        )
        
        if not combined_data.empty and len(combined_data) > 10:
            features = combined_data.drop("timestamp", axis=1)
            
            # Train isolation forest model
            st.session_state.isolation_forest = train_isolation_forest(features)
            
            # Train autoencoder model (simplified)
            st.session_state.autoencoder = train_autoencoder(features)
            
            st.session_state.anomaly_models_trained = True
        
        # Generate a forecast
        if len(st.session_state.hardware_metrics) > 10:
            st.session_state.hardware_forecast, st.session_state.media_forecast = forecast_metrics(
                st.session_state.hardware_metrics,
                st.session_state.media_quality_metrics,
                forecast_periods=10
            )
            
            # Generate system health trend analysis
            st.session_state.health_trend_analysis = analyze_system_health_trends(
                st.session_state.hardware_metrics,
                st.session_state.media_quality_metrics,
                st.session_state.thresholds,
                window=min(30, len(st.session_state.hardware_metrics))
            )
        
        # Train predictive model
        st.session_state.predictive_model_trained = False
        training_success = st.session_state.predictive_model.train(
            st.session_state.hardware_metrics,
            st.session_state.media_quality_metrics
        )
        
        if training_success:
            st.session_state.predictive_model_trained = True
            
            # Get preventive actions
            risk_analysis = st.session_state.predictive_model.analyze_failure_risk(
                st.session_state.hardware_metrics,
                st.session_state.media_quality_metrics,
                st.session_state.thresholds
            )
            
            # Update session state with results
            st.session_state.failure_probability = risk_analysis['failure_probability']
            st.session_state.time_to_failure = risk_analysis['time_to_failure']
            st.session_state.critical_metrics = risk_analysis['critical_metrics']
            
            # Get recommended actions
            st.session_state.preventive_actions = st.session_state.predictive_model.get_preventive_actions()

# Tab 4: CSV Data Analysis
with tab4:
    st.header("📊 CSV Data Analysis for Predictive Maintenance")
    
    # File upload section
    st.subheader("Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file with your system or equipment data",
        type="csv",
        help="Upload CSV files containing timestamps and numeric metrics for analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded CSV
            csv_data = pd.read_csv(uploaded_file)
            st.session_state.csv_data = csv_data
            
            # Show basic info about the uploaded data
            st.success(f"Successfully loaded {len(csv_data)} rows and {len(csv_data.columns)} columns")
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(csv_data.head(10))
            
            # Analyze CSV structure
            structure_analysis = st.session_state.csv_analyzer.analyze_csv_structure(csv_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Data Structure Analysis")
                st.write(f"**Shape:** {structure_analysis['shape'][0]} rows × {structure_analysis['shape'][1]} columns")
                st.write(f"**Numeric columns:** {len(structure_analysis['numeric_columns'])}")
                st.write(f"**Potential time columns:** {len(structure_analysis['potential_time_columns'])}")
                
                # Show missing values
                missing_values = structure_analysis['missing_values']
                if any(missing_values.values()):
                    st.warning("Missing values detected:")
                    for col, missing in missing_values.items():
                        if missing > 0:
                            st.write(f"- {col}: {missing} missing values")
            
            with col2:
                st.subheader("Configure Analysis")
                
                # Select time column
                time_column = None
                if structure_analysis['potential_time_columns']:
                    time_column = st.selectbox(
                        "Select time/date column (optional)",
                        ["None"] + structure_analysis['potential_time_columns']
                    )
                    if time_column == "None":
                        time_column = None
                
                # Select target column for prediction
                target_column = None
                if structure_analysis['numeric_columns']:
                    target_column = st.selectbox(
                        "Select target variable for prediction",
                        ["Auto-detect"] + structure_analysis['numeric_columns']
                    )
                    if target_column == "Auto-detect":
                        target_column = None
            
            # Analysis configuration
            st.subheader("Analysis Options")
            analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
            
            with analysis_col1:
                run_anomaly_detection = st.checkbox("Anomaly Detection", value=True)
            
            with analysis_col2:
                run_predictive_modeling = st.checkbox("Predictive Modeling", value=True)
            
            with analysis_col3:
                forecast_steps = st.number_input("Forecast Steps", min_value=5, max_value=50, value=10)
            
            # Run analysis button
            if st.button("🚀 Run Analysis", type="primary"):
                with st.spinner("Analyzing your data..."):
                    # Prepare data
                    processed_data = st.session_state.csv_analyzer.prepare_data_for_analysis(
                        csv_data, time_column, target_column
                    )
                    
                    results = {}
                    
                    # Anomaly detection
                    if run_anomaly_detection:
                        anomaly_data, anomaly_stats = st.session_state.csv_analyzer.detect_anomalies(processed_data)
                        results['anomaly_data'] = anomaly_data
                        results['anomaly_stats'] = anomaly_stats
                    
                    # Predictive modeling
                    if run_predictive_modeling:
                        model, model_stats = st.session_state.csv_analyzer.create_predictive_model(
                            processed_data, target_column, forecast_steps
                        )
                        results['model'] = model
                        results['model_stats'] = model_stats
                    
                    # Generate recommendations
                    recommendations = st.session_state.csv_analyzer.generate_maintenance_recommendations(
                        processed_data,
                        results.get('anomaly_stats'),
                        results.get('model_stats')
                    )
                    results['recommendations'] = recommendations
                    
                    # Store results
                    st.session_state.csv_analysis_results = results
                
                st.success("Analysis completed!")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV file has proper formatting with numeric columns for analysis.")
    
    # Display analysis results if available
    if st.session_state.csv_analysis_results:
        results = st.session_state.csv_analysis_results
        
        st.header("📈 Analysis Results")
        
        # Anomaly detection results
        if 'anomaly_stats' in results and results['anomaly_stats']:
            anomaly_stats = results['anomaly_stats']
            
            if 'error' not in anomaly_stats:
                st.subheader("🔍 Anomaly Detection Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Data Points", anomaly_stats['total_points'])
                with col2:
                    st.metric("Anomalies Detected", anomaly_stats['anomalies_detected'])
                with col3:
                    st.metric("Anomaly Rate", f"{anomaly_stats['anomaly_percentage']:.1f}%")
                
                # Show most anomalous features
                if 'most_anomalous_features' in anomaly_stats and anomaly_stats['most_anomalous_features']:
                    st.write("**Most Anomalous Features:**")
                    for feature, score in list(anomaly_stats['most_anomalous_features'].items())[:3]:
                        st.write(f"- {feature}: {score:.3f}")
        
        # Predictive modeling results
        if 'model_stats' in results and results['model_stats']:
            model_stats = results['model_stats']
            
            if 'error' not in model_stats:
                st.subheader("🎯 Predictive Model Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Model Training Score", f"{model_stats['train_score']:.3f}")
                    st.metric("Model Test Score", f"{model_stats['test_score']:.3f}")
                
                with col2:
                    st.write("**Target Variable:**", model_stats['target_column'])
                    st.write("**Features Used:**", len(model_stats['feature_columns']))
                
                # Feature importance
                if 'feature_importance' in model_stats:
                    st.write("**Feature Importance:**")
                    importance_df = pd.DataFrame([
                        {'Feature': k, 'Importance': v} 
                        for k, v in model_stats['feature_importance'].items()
                    ]).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        importance_df.head(10), 
                        x='Importance', 
                        y='Feature', 
                        orientation='h',
                        title="Top 10 Most Important Features"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Visualization charts
        if 'anomaly_data' in results and not results['anomaly_data'].empty:
            charts = st.session_state.csv_analyzer.create_visualization_charts(
                results['anomaly_data'],
                results.get('anomaly_data'),
                results.get('model_stats', {}).get('forecast_data')
            )
            
            if charts:
                st.subheader("📊 Data Visualizations")
                
                for chart_name, chart_fig in charts.items():
                    st.plotly_chart(chart_fig, use_container_width=True)
        
        # Maintenance recommendations
        if 'recommendations' in results:
            rec = results['recommendations']
            
            st.subheader("🔧 Maintenance Recommendations")
            
            # Priority indicator
            priority_color = "🔴" if rec['maintenance_priority'] == 'High' else "🟢"
            st.write(f"**Priority Level:** {priority_color} {rec['maintenance_priority']}")
            
            # Alerts
            if rec['alerts']:
                st.write("**⚠️ Alerts:**")
                for alert in rec['alerts']:
                    st.warning(alert)
            
            # Recommendations
            st.write("**📋 Recommended Actions:**")
            for i, recommendation in enumerate(rec['recommendations'], 1):
                st.write(f"{i}. {recommendation}")

# Tab 5: System Health Monitor
with tab5:
    st.header("🖥️ Real-Time System Health Monitor")
    st.markdown("Monitor your computer's actual hardware performance in real-time")
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Collect System Data", type="primary"):
            with st.spinner("Collecting system health data..."):
                # Get real-time system data
                health_data = st.session_state.system_monitor.get_health_metrics_df()
                
                # Append to existing data
                if st.session_state.real_system_data.empty:
                    st.session_state.real_system_data = health_data
                else:
                    st.session_state.real_system_data = safe_concat(
                        st.session_state.real_system_data, 
                        health_data
                    )
                
                # Keep only last 100 data points
                if len(st.session_state.real_system_data) > 100:
                    st.session_state.real_system_data = st.session_state.real_system_data.tail(100)
            
            st.success("System data collected!")
    
    with col2:
        if st.button("🔍 Analyze Health Trends"):
            if not st.session_state.real_system_data.empty:
                with st.spinner("Analyzing system health trends..."):
                    # Analyze health trends
                    trend_analysis = st.session_state.health_analyzer.analyze_health_trends(
                        st.session_state.real_system_data
                    )
                    st.session_state.health_trend_analysis = trend_analysis
                
                st.success("Health analysis completed!")
            else:
                st.warning("Please collect some system data first!")
    
    with col3:
        if st.button("🔮 Predict Maintenance"):
            if not st.session_state.real_system_data.empty:
                with st.spinner("Predicting maintenance needs..."):
                    # Predict maintenance needs
                    maintenance_prediction = st.session_state.health_analyzer.predict_maintenance_needs(
                        st.session_state.real_system_data
                    )
                    st.session_state.maintenance_prediction = maintenance_prediction
                
                st.success("Maintenance prediction completed!")
            else:
                st.warning("Please collect some system data first!")
    
    # Display current system status
    if not st.session_state.real_system_data.empty:
        latest_data = st.session_state.real_system_data.iloc[-1]
        
        st.subheader("Current System Status")
        
        # Current metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric(
                "CPU Usage", 
                f"{latest_data['cpu_usage']:.1f}%",
                delta=None
            )
        
        with metric_col2:
            st.metric(
                "Memory Usage", 
                f"{latest_data['memory_usage']:.1f}%",
                delta=None
            )
        
        with metric_col3:
            st.metric(
                "Disk Usage", 
                f"{latest_data['disk_usage']:.1f}%",
                delta=None
            )
        
        with metric_col4:
            st.metric(
                "Total Processes", 
                f"{latest_data['total_processes']:.0f}",
                delta=None
            )
        
        # System health chart
        st.subheader("System Health Trends")
        
        if len(st.session_state.real_system_data) > 1:
            fig = go.Figure()
            
            # Add CPU usage
            fig.add_trace(go.Scatter(
                x=st.session_state.real_system_data['timestamp'],
                y=st.session_state.real_system_data['cpu_usage'],
                name='CPU Usage',
                line=dict(color='#FF6B6B', width=2)
            ))
            
            # Add Memory usage
            fig.add_trace(go.Scatter(
                x=st.session_state.real_system_data['timestamp'],
                y=st.session_state.real_system_data['memory_usage'],
                name='Memory Usage',
                line=dict(color='#4ECDC4', width=2)
            ))
            
            # Add Disk usage
            fig.add_trace(go.Scatter(
                x=st.session_state.real_system_data['timestamp'],
                y=st.session_state.real_system_data['disk_usage'],
                name='Disk Usage',
                line=dict(color='#45B7D1', width=2)
            ))
            
            fig.update_layout(
                title='Real-Time System Resource Usage',
                xaxis_title='Time',
                yaxis_title='Usage (%)',
                yaxis=dict(range=[0, 100]),
                hovermode='x',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # System information
        st.subheader("System Information")
        sys_info = st.session_state.system_monitor.system_info
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.write(f"**Platform:** {sys_info['platform']} {sys_info['platform_release']}")
            st.write(f"**Architecture:** {sys_info['architecture']}")
            st.write(f"**Hostname:** {sys_info['hostname']}")
        
        with info_col2:
            st.write(f"**Memory Total:** {latest_data['memory_total_gb']:.1f} GB")
            st.write(f"**Memory Available:** {latest_data['memory_available_gb']:.1f} GB")
            st.write(f"**Disk Free:** {latest_data['disk_free_gb']:.1f} GB")
    
    # Display health trend analysis if available
    if hasattr(st.session_state, 'health_trend_analysis'):
        trend_analysis = st.session_state.health_trend_analysis
        
        st.subheader("Health Trend Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            trend_value = trend_analysis.get('trend', 'Unknown')
            st.write(f"**Overall Trend:** {trend_value}")
            if 'health_score' in trend_analysis:
                st.metric("Health Score", f"{trend_analysis['health_score']:.1f}/100")
        
        with col2:
            alerts = trend_analysis.get('alerts', [])
            if alerts:
                st.write("**🚨 Alerts:**")
                for alert in alerts:
                    st.warning(alert)
        
        st.write("**📋 Recommendations:**")
        recommendations = trend_analysis.get('recommendations', ['No recommendations available'])
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
    
    # Display maintenance prediction if available
    if hasattr(st.session_state, 'maintenance_prediction'):
        maintenance_pred = st.session_state.maintenance_prediction
        
        st.subheader("Maintenance Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Prediction:** {maintenance_pred['prediction']}")
            st.metric("Confidence", f"{maintenance_pred['confidence']:.1%}")
        
        with col2:
            if maintenance_pred.get('risk_factors'):
                st.write("**⚠️ Risk Factors:**")
                for factor in maintenance_pred['risk_factors']:
                    st.warning(factor)
        
        st.write("**🔧 Recommended Actions:**")
        for i, action in enumerate(maintenance_pred['recommended_actions'], 1):
            st.write(f"{i}. {action}")

# Main app execution
if __name__ == "__main__":
    # Auto-refresh configuration using a button instead of automatic updates
    refresh_col1, refresh_col2 = st.columns([3, 1])
    with refresh_col2:
        if st.button("🔄 Refresh Dashboard", key="refresh_button"):
            st.session_state.last_refresh = time.time()
            st.rerun()
    
    with refresh_col1:
        last_refresh_time = datetime.now().strftime("%H:%M:%S")
        if 'last_refresh' in st.session_state:
            last_refresh_time = datetime.fromtimestamp(st.session_state.last_refresh).strftime("%H:%M:%S")
        st.info(f"Last updated: {last_refresh_time}")
    
    # Initialize data on first load
    initialize_data_if_empty()
    
    # Run dashboard update for current page load
    update_dashboard()
