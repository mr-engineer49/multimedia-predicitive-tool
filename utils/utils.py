import pandas as pd
import numpy as np

def get_alert_status(value, threshold):
    """
    Determine alert status based on value and threshold.
    
    Args:
        value (float): The current value
        threshold (float): The threshold value
        
    Returns:
        str: Status level ("Normal", "Warning", or "Critical")
    """
    if value >= threshold + (threshold * 0.2):  # 20% above threshold
        return "Critical"
    elif value >= threshold:
        return "Warning"
    else:
        return "Normal"

def get_status_color(status):
    """
    Get color code for status level.
    
    Args:
        status (str): Status level
        
    Returns:
        str: Hex color code
    """
    if status == "Critical" or status == "High":
        return "#FF5630"  # Danger red
    elif status == "Warning" or status == "Medium":
        return "#FFAB00"  # Warning yellow
    else:
        return "#36B37E"  # Success green

def calculate_system_health(hardware_metrics, media_metrics, anomalies, thresholds):
    """
    Calculate overall system health score.
    
    Args:
        hardware_metrics (pandas.DataFrame): DataFrame with hardware metrics
        media_metrics (pandas.DataFrame): DataFrame with media quality metrics
        anomalies (pandas.DataFrame): DataFrame with detected anomalies
        thresholds (dict): Dictionary of thresholds for different metrics
        
    Returns:
        float: System health score (0-100)
    """
    # Start with perfect health
    health_score = 100
    
    # If no data, return default score
    if hardware_metrics.empty or media_metrics.empty:
        return health_score
    
    # Get the most recent metrics
    latest_hw = hardware_metrics.iloc[-1]
    latest_media = media_metrics.iloc[-1]
    
    # Calculate hardware health impact
    cpu_impact = max(0, (latest_hw['cpu_usage'] - thresholds['cpu_usage']) / 2) if latest_hw['cpu_usage'] > thresholds['cpu_usage'] else 0
    gpu_impact = max(0, (latest_hw['gpu_usage'] - thresholds['gpu_usage']) / 2) if latest_hw['gpu_usage'] > thresholds['gpu_usage'] else 0
    memory_impact = max(0, (latest_hw['memory_usage'] - thresholds['memory_usage']) / 2) if latest_hw['memory_usage'] > thresholds['memory_usage'] else 0
    latency_impact = max(0, (latest_hw['latency'] - thresholds['latency']) / 10) if latest_hw['latency'] > thresholds['latency'] else 0
    
    # Calculate media quality impact
    frame_drops_impact = max(0, (latest_media['frame_drops'] - thresholds['frame_drops']) * 2) if latest_media['frame_drops'] > thresholds['frame_drops'] else 0
    encoding_errors_impact = max(0, (latest_media['encoding_errors'] - thresholds['encoding_errors']) * 3) if latest_media['encoding_errors'] > thresholds['encoding_errors'] else 0
    resolution_changes_impact = max(0, (latest_media['resolution_changes'] - thresholds['resolution_changes']) * 5) if latest_media['resolution_changes'] > thresholds['resolution_changes'] else 0
    
    # Calculate anomaly impact
    anomaly_impact = 0
    if not anomalies.empty:
        recent_anomalies = anomalies.iloc[-5:] if len(anomalies) >= 5 else anomalies
        anomaly_count = recent_anomalies['is_anomaly'].sum()
        anomaly_impact = anomaly_count * 5  # Each anomaly reduces health by 5 points
    
    # Calculate total impact
    total_impact = cpu_impact + gpu_impact + memory_impact + latency_impact + \
                   frame_drops_impact + encoding_errors_impact + resolution_changes_impact + \
                   anomaly_impact
    
    # Calculate final health score
    health_score = max(0, min(100, 100 - total_impact))
    
    return health_score
