import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
from data_generator import generate_metrics_data, generate_media_quality_data, generate_system_event_data
from anomaly_detection import detect_anomalies, train_isolation_forest, train_autoencoder
from predictive_maintenance import PredictiveMaintenanceModel, forecast_metrics, analyze_system_health_trends
from visualization import (
    create_hardware_metrics_chart, 
    create_media_quality_chart,
    create_system_health_gauge,
    create_anomaly_heatmap
)
from utils import get_alert_status, get_status_color, calculate_system_health

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
    st.session_state.alert_history = pd.DataFrame(columns=['timestamp', 'metric', 'value', 'threshold', 'severity'])

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

# Main dashboard layout with tabs
tab1, tab2 = st.tabs(["📊 Real-time Monitoring", "📈 Predictive Analysis"])

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
    
    # System health trend analysis
    st.subheader("System Health Trend Analysis")
    trend_cols = st.columns([2, 1])
    
    with trend_cols[0]:
        trend_chart_placeholder = st.empty()
        
    with trend_cols[1]:
        trend_analysis_placeholder = st.empty()
    
    # Forecast metrics
    st.subheader("Forecasted Metrics")
    forecast_container = st.container()
    with forecast_container:
        forecast_chart_placeholder = st.empty()
        
    # Preventive actions
    st.subheader("Recommended Preventive Actions")
    preventive_actions_placeholder = st.empty()
    
    # System events log
    st.header("🔄 System Events Log")
    events_placeholder = st.empty()

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
            st.session_state.system_events = pd.concat([st.session_state.system_events, new_events])
            # Keep only the most recent 50 events
            if len(st.session_state.system_events) > 50:
                st.session_state.system_events = st.session_state.system_events.iloc[-50:]
    
    # Update session state with new data
    if st.session_state.hardware_metrics.empty:
        st.session_state.hardware_metrics = new_hardware_data
    else:
        st.session_state.hardware_metrics = pd.concat([st.session_state.hardware_metrics, new_hardware_data])
        # Keep only the last 100 data points for display
        if len(st.session_state.hardware_metrics) > 100:
            st.session_state.hardware_metrics = st.session_state.hardware_metrics.iloc[-100:]
    
    if st.session_state.media_quality_metrics.empty:
        st.session_state.media_quality_metrics = new_media_data
    else:
        st.session_state.media_quality_metrics = pd.concat([st.session_state.media_quality_metrics, new_media_data])
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
                        st.session_state.anomalies = pd.concat([st.session_state.anomalies, anomalies])
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
            st.session_state.alert_history = pd.concat([st.session_state.alert_history, new_alert])
        
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
            st.session_state.alert_history = pd.concat([st.session_state.alert_history, new_alert])
        
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
            st.session_state.alert_history = pd.concat([st.session_state.alert_history, new_alert])
        
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
            st.session_state.alert_history = pd.concat([st.session_state.alert_history, new_alert])
    
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
            st.session_state.alert_history = pd.concat([st.session_state.alert_history, new_alert])
        
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
            st.session_state.alert_history = pd.concat([st.session_state.alert_history, new_alert])
        
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
            st.session_state.alert_history = pd.concat([st.session_state.alert_history, new_alert])
    
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
        
        cpu_status = get_alert_status(latest_hw['cpu_usage'], st.session_state.thresholds['cpu_usage'])
        gpu_status = get_alert_status(latest_hw['gpu_usage'], st.session_state.thresholds['gpu_usage'])
        memory_status = get_alert_status(latest_hw['memory_usage'], st.session_state.thresholds['memory_usage'])
        latency_status = get_alert_status(latest_hw['latency'], st.session_state.thresholds['latency'])
        
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
        
        frame_drops_status = get_alert_status(latest_media['frame_drops'], st.session_state.thresholds['frame_drops'])
        encoding_errors_status = get_alert_status(latest_media['encoding_errors'], st.session_state.thresholds['encoding_errors'])
        resolution_changes_status = get_alert_status(latest_media['resolution_changes'], st.session_state.thresholds['resolution_changes'])
        
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
                    event_color = get_status_color(severity)
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
            
            # Display the forecast chart
            forecast_chart_placeholder.plotly_chart(
                fig,
                use_container_width=True,
                key="forecast_chart"
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
        system_health = calculate_system_health(
            st.session_state.hardware_metrics, 
            st.session_state.media_quality_metrics,
            st.session_state.anomalies,
            st.session_state.thresholds
        )
        
        system_health_placeholder.plotly_chart(
            create_system_health_gauge("System Health", system_health, 60, min_value=0, max_value=100),
            use_container_width=True,
            key="system_health_gauge"
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
                alert_color = get_status_color(alert['severity'])
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
    
    # Only run the dashboard update once per page load
    update_dashboard()
