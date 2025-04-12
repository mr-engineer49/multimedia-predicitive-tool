import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Global variables to maintain data consistency across calls
_trend_direction = {
    'cpu': np.random.choice([1, -1], p=[0.7, 0.3]),  # Tend toward increasing
    'gpu': np.random.choice([1, -1], p=[0.6, 0.4]),
    'memory': np.random.choice([1, -1], p=[0.8, 0.2]),  # Strong increasing trend
    'latency': np.random.choice([1, -1], p=[0.65, 0.35]),
    'frame_drops': np.random.choice([1, -1], p=[0.55, 0.45]),
    'encoding_errors': np.random.choice([1, -1], p=[0.6, 0.4]),
    'resolution_changes': np.random.choice([1, -1], p=[0.5, 0.5])
}

_last_values = {
    'cpu_usage': 50.0,
    'gpu_usage': 60.0,
    'memory_usage': 45.0,
    'latency': 80.0,
    'frame_drops': 2.0,
    'encoding_errors': 1.0,
    'resolution_changes': 0.5
}

_pattern_state = {
    'time_index': 0,
    'stress_mode': False,
    'stress_timer': 0,
    'anomaly_chance': 0.05,
    'stress_chance': 0.02,
    'stress_duration': 30
}

def generate_metrics_data(num_points=1, realistic_mode=True):
    """
    Generate synthetic hardware metrics data for demonstration.
    
    Args:
        num_points (int): Number of data points to generate
        realistic_mode (bool): Use realistic patterns with trends and anomalies
        
    Returns:
        pandas.DataFrame: DataFrame with hardware metrics
    """
    global _last_values, _trend_direction, _pattern_state
    
    current_time = datetime.now()
    data = []
    
    # Occasionally change trend direction (less frequent with realistic mode)
    if realistic_mode and np.random.random() > 0.97:
        for key in ['cpu', 'gpu', 'memory', 'latency']:
            if np.random.random() > 0.7:  # 30% chance to change any trend
                _trend_direction[key] *= -1
    
    # Occasionally trigger stress mode (high load simulation)
    if realistic_mode and not _pattern_state['stress_mode'] and np.random.random() < _pattern_state['stress_chance']:
        _pattern_state['stress_mode'] = True
        _pattern_state['stress_timer'] = _pattern_state['stress_duration']
    
    # Update stress timer
    if _pattern_state['stress_mode']:
        _pattern_state['stress_timer'] -= num_points
        if _pattern_state['stress_timer'] <= 0:
            _pattern_state['stress_mode'] = False
    
    for i in range(num_points):
        # Advance time index for patterns
        _pattern_state['time_index'] += 1
        
        # Generate hardware metrics with realistic patterns
        timestamp = current_time - timedelta(seconds=i)
        
        if realistic_mode:
            # Get last values as base
            cpu_base = _last_values['cpu_usage']
            gpu_base = _last_values['gpu_usage']
            memory_base = _last_values['memory_usage']
            latency_base = _last_values['latency']
            
            # Add trend component (slow drift in one direction)
            cpu_trend = _trend_direction['cpu'] * np.random.uniform(0, 0.4)  # Slow drift
            gpu_trend = _trend_direction['gpu'] * np.random.uniform(0, 0.5)
            memory_trend = _trend_direction['memory'] * np.random.uniform(0, 0.2)
            latency_trend = _trend_direction['latency'] * np.random.uniform(0, 0.8)
            
            # Add cyclical patterns (daily and hourly)
            hour_factor = np.sin(timestamp.hour / 24 * 2 * np.pi)
            minute_factor = np.sin(timestamp.minute / 60 * 2 * np.pi)
            
            # Add random noise component
            cpu_noise = np.random.normal(0, 1.5)
            gpu_noise = np.random.normal(0, 2.0)
            memory_noise = np.random.normal(0, 0.8)
            latency_noise = np.random.normal(0, 3.0)
            
            # Additional stress factors when in stress mode
            stress_factor = 0
            if _pattern_state['stress_mode']:
                stress_factor = min(_pattern_state['stress_duration'] - _pattern_state['stress_timer'], 10) / 2
            
            # Combine components
            cpu_usage = cpu_base + cpu_trend + hour_factor * 5 + cpu_noise + stress_factor * 2
            gpu_usage = gpu_base + gpu_trend + hour_factor * 7 + gpu_noise + stress_factor * 3
            memory_usage = memory_base + memory_trend + hour_factor * 2 + memory_noise + stress_factor * 1.5
            latency = latency_base + latency_trend + minute_factor * 10 + latency_noise + stress_factor * 5
            
            # Add occasional spikes/anomalies
            if np.random.random() < _pattern_state['anomaly_chance']:
                anomaly_factor = np.random.uniform(5, 15)
                component = np.random.choice(['cpu', 'gpu', 'memory', 'latency'])
                if component == 'cpu':
                    cpu_usage += anomaly_factor
                elif component == 'gpu':
                    gpu_usage += anomaly_factor
                elif component == 'memory':
                    memory_usage += anomaly_factor
                elif component == 'latency':
                    latency += anomaly_factor * 2
            
            # Ensure values are within sensible ranges
            cpu_usage = max(0, min(100, cpu_usage))
            gpu_usage = max(0, min(100, gpu_usage))
            memory_usage = max(0, min(100, memory_usage))
            latency = max(0, min(500, latency))
            
            # Update last values for next iteration
            _last_values['cpu_usage'] = cpu_usage
            _last_values['gpu_usage'] = gpu_usage
            _last_values['memory_usage'] = memory_usage
            _last_values['latency'] = latency
            
        else:
            # Original simple random generation
            cpu_base = 50 + np.random.normal(0, 5)
            gpu_base = 60 + np.random.normal(0, 7)
            memory_base = 45 + np.random.normal(0, 3)
            latency_base = 80 + np.random.normal(0, 10)
            
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
    
    # Return in reverse order to maintain chronological ordering
    return pd.DataFrame(data[::-1])

def generate_media_quality_data(num_points=1, realistic_mode=True):
    """
    Generate synthetic media quality metrics for demonstration.
    
    Args:
        num_points (int): Number of data points to generate
        realistic_mode (bool): Use realistic patterns with trends and anomalies
        
    Returns:
        pandas.DataFrame: DataFrame with media quality metrics
    """
    global _last_values, _trend_direction, _pattern_state
    
    current_time = datetime.now()
    data = []
    
    # Occasionally change trend direction
    if realistic_mode and np.random.random() > 0.98:
        for key in ['frame_drops', 'encoding_errors', 'resolution_changes']:
            if np.random.random() > 0.7:
                _trend_direction[key] *= -1
    
    for i in range(num_points):
        # Generate media quality metrics with realistic patterns
        timestamp = current_time - timedelta(seconds=i)
        
        if realistic_mode:
            # Get last values as base
            frame_drops_base = _last_values['frame_drops']
            encoding_errors_base = _last_values['encoding_errors']
            resolution_changes_base = _last_values['resolution_changes']
            
            # Add trend component
            frame_drops_trend = _trend_direction['frame_drops'] * np.random.uniform(0, 0.2)
            encoding_errors_trend = _trend_direction['encoding_errors'] * np.random.uniform(0, 0.15)
            resolution_changes_trend = _trend_direction['resolution_changes'] * np.random.uniform(0, 0.1)
            
            # Add cyclical patterns
            hour_factor = np.sin(timestamp.hour / 24 * 2 * np.pi)
            minute_factor = np.sin(timestamp.minute / 60 * 2 * np.pi)
            
            # Add random noise
            frame_drops_noise = max(0, np.random.poisson(0.5))
            encoding_errors_noise = max(0, np.random.poisson(0.3))
            resolution_changes_noise = max(0, np.random.poisson(0.2))
            
            # Additional factors when in stress mode
            stress_factor = 0
            if _pattern_state['stress_mode']:
                stress_factor = min(_pattern_state['stress_duration'] - _pattern_state['stress_timer'], 10) / 3
            
            # Combine components
            frame_drops = frame_drops_base + frame_drops_trend + abs(hour_factor) + frame_drops_noise + stress_factor
            encoding_errors = encoding_errors_base + encoding_errors_trend + abs(minute_factor) + encoding_errors_noise + stress_factor * 0.7
            resolution_changes = resolution_changes_base + resolution_changes_trend + resolution_changes_noise + stress_factor * 0.3
            
            # Add occasional spikes/anomalies (higher chance during stress)
            spike_chance = _pattern_state['anomaly_chance'] * (2 if _pattern_state['stress_mode'] else 1)
            if np.random.random() < spike_chance:
                component = np.random.choice(['frame_drops', 'encoding_errors', 'resolution_changes'])
                if component == 'frame_drops':
                    frame_drops += np.random.uniform(2, 6)
                elif component == 'encoding_errors':
                    encoding_errors += np.random.uniform(1, 4)
                elif component == 'resolution_changes':
                    resolution_changes += np.random.uniform(1, 2)
            
            # Update last values for next iteration
            _last_values['frame_drops'] = frame_drops
            _last_values['encoding_errors'] = encoding_errors
            _last_values['resolution_changes'] = resolution_changes
            
        else:
            # Original simple random generation
            frame_drops_base = max(0, np.random.poisson(2))
            encoding_errors_base = max(0, np.random.poisson(1))
            resolution_changes_base = max(0, np.random.poisson(0.5))
            
            hour_factor = np.sin(timestamp.hour / 24 * 2 * np.pi)
            minute_factor = np.sin(timestamp.minute / 60 * 2 * np.pi)
            
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
    
    # Return in reverse order to maintain chronological ordering
    return pd.DataFrame(data[::-1])

def generate_system_event_data(num_points=5):
    """
    Generate synthetic system event data for demonstration.
    
    Args:
        num_points (int): Number of data points to generate
        
    Returns:
        pandas.DataFrame: DataFrame with system events
    """
    current_time = datetime.now()
    data = []
    
    event_types = [
        'System Start', 'Configuration Change', 'Media Stream Start', 
        'Media Stream End', 'Encoding Reset', 'Buffer Overflow',
        'Network Connection Change', 'Resource Allocation', 'Resource Deallocation',
        'Scheduled Maintenance', 'Error Recovery', 'User Interaction'
    ]
    
    severity_levels = ['Info', 'Warning', 'Error', 'Critical']
    severity_weights = [0.7, 0.2, 0.08, 0.02]  # Probabilities for each level
    
    components = [
        'Encoder', 'Decoder', 'Network', 'Storage', 'Input', 
        'Output', 'Processor', 'Memory', 'Configuration', 'Scheduler'
    ]
    
    for i in range(num_points):
        # Generate random event data
        timestamp = current_time - timedelta(minutes=np.random.randint(1, 60 * 24))  # Events spread over last 24 hours
        
        event_type = np.random.choice(event_types)
        severity = np.random.choice(severity_levels, p=severity_weights)
        component = np.random.choice(components)
        
        # Generate an appropriate message based on the event
        message = ""
        if event_type == 'System Start':
            message = f"{component} subsystem initialized successfully"
        elif event_type == 'Configuration Change':
            message = f"{component} configuration updated"
        elif event_type == 'Media Stream Start':
            message = f"New media stream started on {component}"
        elif event_type == 'Media Stream End':
            message = f"Media stream ended on {component}"
        elif event_type == 'Encoding Reset':
            message = f"Encoding parameters reset on {component}"
        elif event_type == 'Buffer Overflow':
            message = f"Buffer overflow detected in {component}"
        elif event_type == 'Network Connection Change':
            message = f"Network connection status changed for {component}"
        elif event_type == 'Resource Allocation':
            message = f"Additional resources allocated to {component}"
        elif event_type == 'Resource Deallocation':
            message = f"Resources released from {component}"
        elif event_type == 'Scheduled Maintenance':
            message = f"Scheduled maintenance performed on {component}"
        elif event_type == 'Error Recovery':
            message = f"Error recovery procedure executed on {component}"
        elif event_type == 'User Interaction':
            message = f"User interaction with {component}"
        else:
            message = f"Unknown event related to {component}"
        
        # Add more details for non-info events
        if severity != 'Info':
            details = [
                f"Duration: {np.random.randint(1, 60)} seconds",
                f"Affected streams: {np.random.randint(1, 5)}",
                f"Recovery attempts: {np.random.randint(0, 3)}"
            ]
            detail_text = " | ".join(details)
        else:
            detail_text = "Standard operation"
        
        # Create a dictionary for this event
        event = {
            'timestamp': timestamp,
            'event_type': event_type,
            'severity': severity,
            'component': component,
            'message': message,
            'details': detail_text
        }
        
        data.append(event)
    
    # Sort by timestamp
    df = pd.DataFrame(data)
    return df.sort_values('timestamp', ascending=False).reset_index(drop=True)
