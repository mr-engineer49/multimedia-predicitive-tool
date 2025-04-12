import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_metrics_data(num_points=1):
    """
    Generate synthetic hardware metrics data for demonstration.
    
    Args:
        num_points (int): Number of data points to generate
        
    Returns:
        pandas.DataFrame: DataFrame with hardware metrics
    """
    current_time = datetime.now()
    data = []
    
    for i in range(num_points):
        # Generate hardware metrics with some random variations
        timestamp = current_time - timedelta(seconds=i)
        
        # Base values
        cpu_base = 50 + np.random.normal(0, 5)
        gpu_base = 60 + np.random.normal(0, 7)
        memory_base = 45 + np.random.normal(0, 3)
        latency_base = 80 + np.random.normal(0, 10)
        
        # Add some cyclical patterns
        hour_factor = np.sin(timestamp.hour / 24 * 2 * np.pi)
        cpu_usage = max(0, min(100, cpu_base + hour_factor * 10))
        gpu_usage = max(0, min(100, gpu_base + hour_factor * 15))
        memory_usage = max(0, min(100, memory_base + hour_factor * 5))
        latency = max(0, latency_base + hour_factor * 20)
        
        # Add previous values for delta calculations
        cpu_usage_prev = max(0, min(100, cpu_usage - np.random.normal(0, 2)))
        gpu_usage_prev = max(0, min(100, gpu_usage - np.random.normal(0, 3)))
        memory_usage_prev = max(0, min(100, memory_usage - np.random.normal(0, 1)))
        latency_prev = max(0, latency - np.random.normal(0, 5))
        
        # Create a dictionary for this data point
        data_point = {
            'timestamp': timestamp,
            'cpu_usage': cpu_usage,
            'gpu_usage': gpu_usage,
            'memory_usage': memory_usage,
            'latency': latency,
            'cpu_usage_prev': cpu_usage_prev,
            'gpu_usage_prev': gpu_usage_prev,
            'memory_usage_prev': memory_usage_prev,
            'latency_prev': latency_prev
        }
        
        data.append(data_point)
    
    return pd.DataFrame(data)

def generate_media_quality_data(num_points=1):
    """
    Generate synthetic media quality metrics for demonstration.
    
    Args:
        num_points (int): Number of data points to generate
        
    Returns:
        pandas.DataFrame: DataFrame with media quality metrics
    """
    current_time = datetime.now()
    data = []
    
    for i in range(num_points):
        # Generate media quality metrics with some random variations
        timestamp = current_time - timedelta(seconds=i)
        
        # Base values with some random variations
        frame_drops_base = max(0, np.random.poisson(2))
        encoding_errors_base = max(0, np.random.poisson(1))
        resolution_changes_base = max(0, np.random.poisson(0.5))
        
        # Add some time-based patterns
        hour_factor = np.sin(timestamp.hour / 24 * 2 * np.pi)
        minute_factor = np.sin(timestamp.minute / 60 * 2 * np.pi)
        
        # Introduce occasional spikes for demonstration
        spike = np.random.random() > 0.95  # 5% chance of a spike
        
        frame_drops = frame_drops_base + (5 if spike else 0) + abs(hour_factor)
        encoding_errors = encoding_errors_base + (3 if spike else 0) + abs(minute_factor)
        resolution_changes = resolution_changes_base + (2 if spike and np.random.random() > 0.5 else 0)
        
        # Add previous values for delta calculations
        frame_drops_prev = max(0, frame_drops - np.random.randint(0, 2))
        encoding_errors_prev = max(0, encoding_errors - np.random.randint(0, 2))
        resolution_changes_prev = max(0, resolution_changes - np.random.randint(0, 1))
        
        # Create a dictionary for this data point
        data_point = {
            'timestamp': timestamp,
            'frame_drops': frame_drops,
            'encoding_errors': encoding_errors,
            'resolution_changes': resolution_changes,
            'frame_drops_prev': frame_drops_prev,
            'encoding_errors_prev': encoding_errors_prev,
            'resolution_changes_prev': resolution_changes_prev
        }
        
        data.append(data_point)
    
    return pd.DataFrame(data)
