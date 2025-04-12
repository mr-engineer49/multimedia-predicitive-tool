import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime

def create_hardware_metrics_chart(data):
    """
    Create a time series chart of hardware metrics.
    
    Args:
        data (pandas.DataFrame): DataFrame with hardware metrics
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create a figure with secondary y-axis
    fig = go.Figure()
    
    # Add CPU usage
    fig.add_trace(
        go.Scatter(
            x=data['timestamp'], 
            y=data['cpu_usage'],
            name='CPU Usage (%)',
            line=dict(color='#0747A6', width=2)
        )
    )
    
    # Add GPU usage
    fig.add_trace(
        go.Scatter(
            x=data['timestamp'], 
            y=data['gpu_usage'],
            name='GPU Usage (%)',
            line=dict(color='#36B37E', width=2)
        )
    )
    
    # Add Memory usage
    fig.add_trace(
        go.Scatter(
            x=data['timestamp'], 
            y=data['memory_usage'],
            name='Memory Usage (%)',
            line=dict(color='#FFAB00', width=2)
        )
    )
    
    # Add Latency on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=data['timestamp'], 
            y=data['latency'],
            name='Latency (ms)',
            line=dict(color='#FF5630', width=2, dash='dot'),
            yaxis='y2'
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Hardware Performance Metrics Over Time',
        xaxis=dict(title='Time'),
        yaxis=dict(
            title='Usage (%)',
            titlefont=dict(color='#0747A6'),
            tickfont=dict(color='#0747A6'),
            range=[0, 100]
        ),
        yaxis2=dict(
            title='Latency (ms)',
            titlefont=dict(color='#FF5630'),
            tickfont=dict(color='#FF5630'),
            anchor='x',
            overlaying='y',
            side='right'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )
    
    return fig

def create_media_quality_chart(data):
    """
    Create a time series chart of media quality metrics.
    
    Args:
        data (pandas.DataFrame): DataFrame with media quality metrics
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create a figure
    fig = go.Figure()
    
    # Add Frame Drops
    fig.add_trace(
        go.Scatter(
            x=data['timestamp'], 
            y=data['frame_drops'],
            name='Frame Drops/min',
            line=dict(color='#FF5630', width=2)
        )
    )
    
    # Add Encoding Errors
    fig.add_trace(
        go.Scatter(
            x=data['timestamp'], 
            y=data['encoding_errors'],
            name='Encoding Errors/min',
            line=dict(color='#FFAB00', width=2)
        )
    )
    
    # Add Resolution Changes
    fig.add_trace(
        go.Scatter(
            x=data['timestamp'], 
            y=data['resolution_changes'],
            name='Resolution Changes/min',
            line=dict(color='#0747A6', width=2)
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Media Quality Metrics Over Time',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Count per minute'),
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
    
    return fig

def create_system_health_gauge(title, value, threshold, min_value=0, max_value=100):
    """
    Create a gauge chart for system health visualization.
    
    Args:
        title (str): Title of the gauge
        value (float): Current value
        threshold (float): Threshold value for warning/danger
        min_value (float): Minimum value for gauge
        max_value (float): Maximum value for gauge
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Determine color based on value and threshold
    if title == "System Health":
        # For system health, higher is better
        if value >= 80:
            color = "#36B37E"  # Success green
        elif value >= 60:
            color = "#FFAB00"  # Warning yellow
        else:
            color = "#FF5630"  # Danger red
    else:
        # For other metrics, lower is better
        if value >= threshold + 10:
            color = "#FF5630"  # Danger red
        elif value >= threshold:
            color = "#FFAB00"  # Warning yellow
        else:
            color = "#36B37E"  # Success green
    
    # Create gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [min_value, max_value], 'tickwidth': 1, 'tickcolor': "#172B4D"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#172B4D",
            'steps': [
                {'range': [min_value, threshold * 0.75], 'color': 'rgba(54, 179, 126, 0.3)'},
                {'range': [threshold * 0.75, threshold], 'color': 'rgba(255, 171, 0, 0.3)'},
                {'range': [threshold, max_value], 'color': 'rgba(255, 86, 48, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "#172B4D", 'width': 2},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    
    # Update layout
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    
    return fig

def create_anomaly_heatmap(anomalies):
    """
    Create a heatmap visualization of detected anomalies.
    
    Args:
        anomalies (pandas.DataFrame): DataFrame with anomaly data
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Filter columns that are anomaly indicators
    anomaly_cols = [col for col in anomalies.columns if col.endswith('_anomaly')]
    
    if not anomaly_cols:
        # If no anomaly columns, create an empty figure
        fig = go.Figure()
        fig.update_layout(
            title='No Anomalies Detected',
            height=350
        )
        return fig
    
    # Prepare data for heatmap
    heatmap_data = []
    
    # For each timestamp, create a row with anomaly statuses
    for idx, row in anomalies.iterrows():
        time_str = row['timestamp'].strftime('%H:%M:%S')
        
        for col in anomaly_cols:
            metric_name = col.replace('_anomaly', '')
            heatmap_data.append({
                'Timestamp': time_str,
                'Metric': metric_name.replace('_', ' ').title(),
                'Anomaly': 1 if row[col] else 0,
                'Score': row['anomaly_score'] if 'anomaly_score' in row else 0.5
            })
    
    # Create DataFrame from heatmap data
    heatmap_df = pd.DataFrame(heatmap_data)
    
    # Create pivot table for heatmap
    pivot_df = heatmap_df.pivot_table(
        values='Anomaly', 
        index='Metric', 
        columns='Timestamp', 
        fill_value=0
    )
    
    # Create heatmap figure
    fig = px.imshow(
        pivot_df,
        labels=dict(x="Time", y="Metric", color="Anomaly Status"),
        x=pivot_df.columns,
        y=pivot_df.index,
        color_continuous_scale=[
            [0, '#F4F5F7'],  # Normal
            [0.5, '#FFAB00'],  # Warning
            [1, '#FF5630']   # Anomaly
        ],
        title='Anomaly Detection Heatmap',
        height=350
    )
    
    # Update layout
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(tickangle=45)
    )
    
    return fig
