import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from data_generator import generate_metrics_data, generate_media_quality_data
from anomaly_detection import detect_anomalies, train_isolation_forest, train_autoencoder
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
    
if 'refresh_rate' not in st.session_state:
    st.session_state.refresh_rate = 5  # default refresh rate in seconds

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

if 'anomaly_models_trained' not in st.session_state:
    st.session_state.anomaly_models_trained = False
    st.session_state.isolation_forest = None
    st.session_state.autoencoder = None

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
    
    st.subheader("About")
    st.markdown("""
        This predictive maintenance system monitors multimedia processing workflows,
        analyzing hardware metrics and media quality to predict potential failures.
    """)
    
    # System information
    st.subheader("System Information")
    st.markdown("**Environment:** Production")
    st.markdown("**Version:** 1.0.0")
    st.markdown("**Last Update:** {}".format(datetime.now().strftime("%Y-%m-%d")))

# Main dashboard layout
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
st.header("Anomaly Detection & Predictive Maintenance")
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

# Function to update dashboard metrics
def update_dashboard():
    # Generate real-time data
    new_hardware_data = generate_metrics_data()
    new_media_data = generate_media_quality_data()
    
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
        
        # Update gauges
        cpu_gauge.plotly_chart(
            create_system_health_gauge("CPU Usage", latest_hw['cpu_usage'], st.session_state.thresholds['cpu_usage']),
            use_container_width=True
        )
        
        gpu_gauge.plotly_chart(
            create_system_health_gauge("GPU Usage", latest_hw['gpu_usage'], st.session_state.thresholds['gpu_usage']),
            use_container_width=True
        )
        
        memory_gauge.plotly_chart(
            create_system_health_gauge("Memory Usage", latest_hw['memory_usage'], st.session_state.thresholds['memory_usage']),
            use_container_width=True
        )
        
        # Update hardware metrics
        hardware_chart_placeholder.plotly_chart(
            create_hardware_metrics_chart(st.session_state.hardware_metrics),
            use_container_width=True
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
        
        # Update media quality chart
        media_chart_placeholder.plotly_chart(
            create_media_quality_chart(st.session_state.media_quality_metrics),
            use_container_width=True
        )
        
        frame_drops_status = get_alert_status(latest_media['frame_drops'], st.session_state.thresholds['frame_drops'])
        encoding_errors_status = get_alert_status(latest_media['encoding_errors'], st.session_state.thresholds['encoding_errors'])
        resolution_changes_status = get_alert_status(latest_media['resolution_changes'], st.session_state.thresholds['resolution_changes'])
        
        frame_drops_metric.metric("Frame Drops", f"{latest_media['frame_drops']}/min", delta=f"{latest_media['frame_drops'] - latest_media['frame_drops_prev']}", delta_color="inverse")
        encoding_errors_metric.metric("Encoding Errors", f"{latest_media['encoding_errors']}/min", delta=f"{latest_media['encoding_errors'] - latest_media['encoding_errors_prev']}", delta_color="inverse")
        resolution_changes_metric.metric("Resolution Changes", f"{latest_media['resolution_changes']}/min", delta=f"{latest_media['resolution_changes'] - latest_media['resolution_changes_prev']}", delta_color="inverse")
    
    # Update anomaly visualizations
    if not st.session_state.anomalies.empty:
        anomaly_chart_placeholder.plotly_chart(
            create_anomaly_heatmap(st.session_state.anomalies),
            use_container_width=True
        )
        
        # Calculate system health score
        system_health = calculate_system_health(
            st.session_state.hardware_metrics, 
            st.session_state.media_quality_metrics,
            st.session_state.anomalies,
            st.session_state.thresholds
        )
        
        system_health_placeholder.plotly_chart(
            create_system_health_gauge("System Health", system_health, 60, min_value=0, max_value=100),
            use_container_width=True
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
    # Initial update
    update_dashboard()
    
    # Use a placeholder for the auto-refresh message
    refresh_placeholder = st.empty()
    
    # Create auto-refresh loop
    while True:
        # Calculate time until next refresh
        countdown = st.session_state.refresh_rate
        while countdown > 0:
            refresh_placeholder.info(f"Dashboard will refresh in {countdown} seconds")
            time.sleep(1)
            countdown -= 1
        
        refresh_placeholder.info("Refreshing dashboard...")
        update_dashboard()
