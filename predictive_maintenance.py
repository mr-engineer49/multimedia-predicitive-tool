import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler

class PredictiveMaintenanceModel:
    """
    Class for predictive maintenance model that forecasts system metrics
    and predicts potential failures based on historical data.
    """
    
    def __init__(self):
        """Initialize the predictive maintenance model"""
        self.hardware_scaler = StandardScaler()
        self.media_scaler = StandardScaler()
        self.hardware_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.media_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.anomaly_detector = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        self.is_trained = False
        self.failure_probability = 0
        self.time_to_failure = None
        self.critical_metrics = []
        
    def train(self, hardware_metrics, media_metrics):
        """
        Train the predictive models using historical data
        
        Args:
            hardware_metrics (pandas.DataFrame): Historical hardware metrics
            media_metrics (pandas.DataFrame): Historical media quality metrics
            
        Returns:
            bool: True if training was successful
        """
        if len(hardware_metrics) < 20 or len(media_metrics) < 20:
            return False
        
        # Prepare hardware metrics for training
        hardware_features = hardware_metrics.drop('timestamp', axis=1)
        hardware_X = hardware_features[['cpu_usage_prev', 'gpu_usage_prev', 'memory_usage_prev', 'latency_prev']]
        hardware_y = hardware_features[['cpu_usage', 'gpu_usage', 'memory_usage', 'latency']]
        
        # Scale hardware data
        hardware_X_scaled = self.hardware_scaler.fit_transform(hardware_X)
        
        # Train hardware prediction model
        self.hardware_model.fit(hardware_X_scaled, hardware_y)
        
        # Prepare media metrics for training
        media_features = media_metrics.drop('timestamp', axis=1)
        media_X = media_features[['frame_drops_prev', 'encoding_errors_prev', 'resolution_changes_prev']]
        media_y = media_features[['frame_drops', 'encoding_errors', 'resolution_changes']]
        
        # Scale media data
        media_X_scaled = self.media_scaler.fit_transform(media_X)
        
        # Train media prediction model
        self.media_model.fit(media_X_scaled, media_y)
        
        # Train anomaly detector on combined features
        # Merge datasets on timestamp
        if not hardware_metrics.empty and not media_metrics.empty:
            combined_data = pd.merge(
                hardware_metrics, 
                media_metrics,
                on="timestamp"
            )
            
            if not combined_data.empty:
                combined_features = combined_data.drop('timestamp', axis=1)
                self.anomaly_detector.fit(combined_features)
        
        self.is_trained = True
        return True
    
    def predict_next_values(self, latest_hardware, latest_media):
        """
        Predict next values for hardware and media metrics
        
        Args:
            latest_hardware (pandas.DataFrame): Latest hardware metrics
            latest_media (pandas.DataFrame): Latest media quality metrics
            
        Returns:
            tuple: (predicted_hardware, predicted_media) DataFrames
        """
        if not self.is_trained or latest_hardware.empty or latest_media.empty:
            return None, None
        
        # Prepare hardware input
        hw_input = latest_hardware[['cpu_usage', 'gpu_usage', 'memory_usage', 'latency']].copy()
        hw_input.columns = ['cpu_usage_prev', 'gpu_usage_prev', 'memory_usage_prev', 'latency_prev']
        hw_input_scaled = self.hardware_scaler.transform(hw_input)
        
        # Predict hardware metrics
        hw_pred = self.hardware_model.predict(hw_input_scaled)
        hw_pred_df = pd.DataFrame(
            hw_pred, 
            columns=['cpu_usage', 'gpu_usage', 'memory_usage', 'latency']
        )
        
        # Add timestamp for future time
        future_time = latest_hardware['timestamp'].iloc[-1] + timedelta(seconds=10)
        hw_pred_df['timestamp'] = future_time
        
        # Prepare media input
        media_input = latest_media[['frame_drops', 'encoding_errors', 'resolution_changes']].copy()
        media_input.columns = ['frame_drops_prev', 'encoding_errors_prev', 'resolution_changes_prev']
        media_input_scaled = self.media_scaler.transform(media_input)
        
        # Predict media metrics
        media_pred = self.media_model.predict(media_input_scaled)
        media_pred_df = pd.DataFrame(
            media_pred, 
            columns=['frame_drops', 'encoding_errors', 'resolution_changes']
        )
        
        # Add timestamp for future time
        media_pred_df['timestamp'] = future_time
        
        return hw_pred_df, media_pred_df
    
    def analyze_failure_risk(self, hardware_metrics, media_metrics, thresholds):
        """
        Analyze risk of system failure based on metrics and thresholds
        
        Args:
            hardware_metrics (pandas.DataFrame): Hardware metrics history
            media_metrics (pandas.DataFrame): Media quality metrics history
            thresholds (dict): Dictionary of thresholds for different metrics
            
        Returns:
            dict: Failure risk assessment including probability and time to failure
        """
        if not self.is_trained or hardware_metrics.empty or media_metrics.empty:
            return {
                'failure_probability': 0,
                'time_to_failure': None,
                'critical_metrics': []
            }
        
        # Get recent data and predictions
        recent_hw = hardware_metrics.iloc[-5:] if len(hardware_metrics) >= 5 else hardware_metrics
        recent_media = media_metrics.iloc[-5:] if len(media_metrics) >= 5 else media_metrics
        
        # Get predictions for future metrics
        next_hw, next_media = self.predict_next_values(recent_hw.iloc[[-1]], recent_media.iloc[[-1]])
        
        # Count metrics exceeding thresholds
        threshold_violations = 0
        critical_metrics = []
        
        # Check hardware threshold violations
        latest_hw = recent_hw.iloc[-1]
        if latest_hw['cpu_usage'] > thresholds['cpu_usage']:
            threshold_violations += 1
            critical_metrics.append('CPU Usage')
        
        if latest_hw['gpu_usage'] > thresholds['gpu_usage']:
            threshold_violations += 1
            critical_metrics.append('GPU Usage')
            
        if latest_hw['memory_usage'] > thresholds['memory_usage']:
            threshold_violations += 1
            critical_metrics.append('Memory Usage')
            
        if latest_hw['latency'] > thresholds['latency']:
            threshold_violations += 1
            critical_metrics.append('Latency')
        
        # Check media threshold violations
        latest_media = recent_media.iloc[-1]
        if latest_media['frame_drops'] > thresholds['frame_drops']:
            threshold_violations += 1
            critical_metrics.append('Frame Drops')
            
        if latest_media['encoding_errors'] > thresholds['encoding_errors']:
            threshold_violations += 1
            critical_metrics.append('Encoding Errors')
            
        if latest_media['resolution_changes'] > thresholds['resolution_changes']:
            threshold_violations += 1
            critical_metrics.append('Resolution Changes')
        
        # Analyze trends
        cpu_trend = recent_hw['cpu_usage'].diff().mean()
        gpu_trend = recent_hw['gpu_usage'].diff().mean()
        memory_trend = recent_hw['memory_usage'].diff().mean()
        
        # Calculate failure probability based on violations and trends
        violation_weight = threshold_violations / 7  # Max of 7 metrics
        trend_factor = 0
        
        if cpu_trend > 0 or gpu_trend > 0 or memory_trend > 0:
            trend_factor = (cpu_trend + gpu_trend + memory_trend) / 3
            trend_factor = min(trend_factor / 10, 0.3)  # Cap at 0.3
        
        failure_probability = min(0.95, violation_weight * 0.7 + trend_factor)
        
        # Calculate estimated time to failure
        time_to_failure = None
        if failure_probability > 0.5 and next_hw is not None:
            # Simple linear extrapolation to threshold crossing
            metrics_to_check = {
                'cpu_usage': thresholds['cpu_usage'] * 1.2,  # 20% above threshold
                'gpu_usage': thresholds['gpu_usage'] * 1.2,
                'memory_usage': thresholds['memory_usage'] * 1.2
            }
            
            time_estimates = []
            for metric, threshold in metrics_to_check.items():
                if metric in critical_metrics:
                    current_val = latest_hw[metric]
                    next_val = next_hw[metric].iloc[0]
                    if next_val > current_val:  # Increasing trend
                        rate_of_change = (next_val - current_val) / 10  # 10 seconds between points
                        if rate_of_change > 0:
                            time_to_cross = (threshold - current_val) / rate_of_change
                            time_estimates.append(time_to_cross)
            
            if time_estimates:
                # Convert seconds to minutes and round
                time_to_failure = round(min(time_estimates) / 60, 1)
        
        self.failure_probability = failure_probability
        self.time_to_failure = time_to_failure
        self.critical_metrics = critical_metrics
        
        return {
            'failure_probability': failure_probability,
            'time_to_failure': time_to_failure,
            'critical_metrics': critical_metrics
        }
    
    def get_preventive_actions(self):
        """
        Generate recommended preventive actions based on critical metrics
        
        Returns:
            list: List of recommended actions
        """
        actions = []
        
        if not self.critical_metrics:
            return ["No critical issues detected. Continue monitoring."]
        
        if 'CPU Usage' in self.critical_metrics:
            actions.append("Reduce CPU-intensive processing tasks and optimize encoding settings.")
            
        if 'GPU Usage' in self.critical_metrics:
            actions.append("Lower GPU processing load by adjusting video processing parameters.")
            
        if 'Memory Usage' in self.critical_metrics:
            actions.append("Check for memory leaks and consider increasing system memory allocation.")
            
        if 'Latency' in self.critical_metrics:
            actions.append("Optimize network configuration and review bandwidth allocation.")
            
        if 'Frame Drops' in self.critical_metrics:
            actions.append("Reduce input resolution or framerate to prevent frame drops.")
            
        if 'Encoding Errors' in self.critical_metrics:
            actions.append("Review encoding parameters and consider using a more robust codec.")
            
        if 'Resolution Changes' in self.critical_metrics:
            actions.append("Stabilize input sources to prevent resolution fluctuations.")
        
        # Add general recommendations if failure probability is high
        if self.failure_probability > 0.7:
            actions.append("URGENT: Schedule immediate maintenance to prevent system failure.")
        elif self.failure_probability > 0.5:
            actions.append("Schedule maintenance within 24 hours to address potential issues.")
        
        return actions

def forecast_metrics(hardware_metrics, media_metrics, forecast_periods=5):
    """
    Forecast future metrics based on historical data
    
    Args:
        hardware_metrics (pandas.DataFrame): Historical hardware metrics
        media_metrics (pandas.DataFrame): Historical media quality metrics
        forecast_periods (int): Number of periods to forecast
        
    Returns:
        tuple: (hardware_forecast, media_forecast) DataFrames
    """
    if hardware_metrics.empty or media_metrics.empty or len(hardware_metrics) < 10:
        return pd.DataFrame(), pd.DataFrame()
    
    # Create predictive model
    model = PredictiveMaintenanceModel()
    
    # Train the model
    model.train(hardware_metrics, media_metrics)
    
    # Generate forecast
    hw_forecast = hardware_metrics.iloc[-1:].copy()
    media_forecast = media_metrics.iloc[-1:].copy()
    
    for i in range(forecast_periods):
        next_hw, next_media = model.predict_next_values(hw_forecast.iloc[[-1]], media_forecast.iloc[[-1]])
        if next_hw is not None and next_media is not None:
            hw_forecast = pd.concat([hw_forecast, next_hw])
            media_forecast = pd.concat([media_forecast, next_media])
    
    return hw_forecast, media_forecast

def analyze_system_health_trends(hardware_metrics, media_metrics, thresholds, window=30):
    """
    Analyze trends in system health over time
    
    Args:
        hardware_metrics (pandas.DataFrame): Historical hardware metrics
        media_metrics (pandas.DataFrame): Historical media quality metrics
        thresholds (dict): Dictionary of thresholds for different metrics
        window (int): Window size for trend analysis
        
    Returns:
        dict: System health trend analysis
    """
    if hardware_metrics.empty or media_metrics.empty or len(hardware_metrics) < window:
        return {
            'overall_trend': 'Insufficient data',
            'metrics_trends': {},
            'recommendations': ['Collect more data for trend analysis']
        }
    
    # Get subset of recent data
    recent_hw = hardware_metrics.iloc[-window:]
    recent_media = media_metrics.iloc[-window:]
    
    # Calculate trends for each metric
    trends = {}
    
    # Hardware metrics trends
    for metric in ['cpu_usage', 'gpu_usage', 'memory_usage', 'latency']:
        if metric in recent_hw.columns:
            slope = np.polyfit(range(len(recent_hw)), recent_hw[metric].values, 1)[0]
            trends[metric] = {
                'slope': slope,
                'direction': 'increasing' if slope > 0.1 else ('decreasing' if slope < -0.1 else 'stable'),
                'concern': 'high' if slope > 0.5 else ('medium' if slope > 0.1 else 'low')
            }
    
    # Media metrics trends
    for metric in ['frame_drops', 'encoding_errors', 'resolution_changes']:
        if metric in recent_media.columns:
            slope = np.polyfit(range(len(recent_media)), recent_media[metric].values, 1)[0]
            trends[metric] = {
                'slope': slope,
                'direction': 'increasing' if slope > 0.05 else ('decreasing' if slope < -0.05 else 'stable'),
                'concern': 'high' if slope > 0.2 else ('medium' if slope > 0.05 else 'low')
            }
    
    # Determine overall trend
    concern_levels = [t['concern'] for t in trends.values()]
    high_concerns = concern_levels.count('high')
    medium_concerns = concern_levels.count('medium')
    
    if high_concerns > 0:
        overall_trend = 'Degrading'
    elif medium_concerns > 1:
        overall_trend = 'Gradually degrading'
    elif medium_concerns == 1:
        overall_trend = 'Slightly degrading'
    else:
        overall_trend = 'Stable'
    
    # Generate recommendations
    recommendations = []
    
    for metric, trend in trends.items():
        if trend['concern'] == 'high':
            if 'cpu' in metric:
                recommendations.append(f"Urgent: Address rapidly increasing CPU usage trends")
            elif 'gpu' in metric:
                recommendations.append(f"Urgent: Address rapidly increasing GPU usage trends")
            elif 'memory' in metric:
                recommendations.append(f"Urgent: Address rapidly increasing memory usage trends")
            elif 'latency' in metric:
                recommendations.append(f"Urgent: Address rapidly increasing latency issues")
            elif 'frame_drops' in metric:
                recommendations.append(f"Urgent: Address increasing frame drop issues")
            elif 'encoding_errors' in metric:
                recommendations.append(f"Urgent: Address increasing encoding error trends")
    
    if not recommendations and overall_trend != 'Stable':
        recommendations.append("Monitor system for continued degradation and plan preventive maintenance")
    
    return {
        'overall_trend': overall_trend,
        'metrics_trends': trends,
        'recommendations': recommendations if recommendations else ["System is stable, no action needed"]
    }