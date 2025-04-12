import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import datetime

# Simplified autoencoder implementation without requiring TensorFlow
class SimpleAutoencoder:
    def __init__(self, n_features):
        self.n_features = n_features
        self.scaler = StandardScaler()
        # Use a simple threshold-based approach
        self.threshold = None
    
    def fit(self, X):
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Calculate reconstruction error using a simpler approach
        # Just calculate the Euclidean distance from the mean
        mean_vector = np.mean(X_scaled, axis=0)
        errors = np.sqrt(np.sum(np.square(X_scaled - mean_vector), axis=1))
        
        # Set threshold as mean + 2*std of errors
        self.threshold = np.mean(errors) + 2 * np.std(errors)
        return self
    
    def predict(self, X):
        # Scale new data
        X_scaled = self.scaler.transform(X)
        
        # Calculate reconstruction error
        mean_vector = np.mean(X_scaled, axis=0)
        errors = np.sqrt(np.sum(np.square(X_scaled - mean_vector), axis=1))
        
        # Classify as anomaly if error > threshold
        return np.where(errors > self.threshold, -1, 1)
    
    def decision_function(self, X):
        # Scale new data
        X_scaled = self.scaler.transform(X)
        
        # Calculate reconstruction error
        mean_vector = np.mean(X_scaled, axis=0)
        errors = np.sqrt(np.sum(np.square(X_scaled - mean_vector), axis=1))
        
        # Return negative error (higher values = more normal)
        return -errors

def train_isolation_forest(data):
    """
    Train an Isolation Forest model for anomaly detection.
    
    Args:
        data (pandas.DataFrame): DataFrame with features for anomaly detection
        
    Returns:
        object: Trained Isolation Forest model
    """
    # Create and fit the model
    model = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=0.05,  # Assume 5% of data points are anomalies
        random_state=42
    )
    
    model.fit(data)
    return model

def train_autoencoder(data):
    """
    Train a simple autoencoder for anomaly detection.
    
    Args:
        data (pandas.DataFrame): DataFrame with features for anomaly detection
        
    Returns:
        object: Trained autoencoder model
    """
    # Create and fit the model
    model = SimpleAutoencoder(data.shape[1])
    model.fit(data)
    return model

def detect_anomalies(data, isolation_forest, autoencoder):
    """
    Detect anomalies in the data using trained models.
    
    Args:
        data (pandas.DataFrame): New data to analyze
        isolation_forest: Trained Isolation Forest model
        autoencoder: Trained autoencoder model
        
    Returns:
        pandas.DataFrame: DataFrame with anomaly scores and classifications
    """
    if isolation_forest is None or autoencoder is None:
        return pd.DataFrame()
    
    # Extract features for anomaly detection
    timestamps = data['timestamp']
    features = data.drop('timestamp', axis=1)
    
    # Get anomaly scores from both models
    if_scores = isolation_forest.decision_function(features)
    ae_scores = autoencoder.decision_function(features)
    
    # Normalize scores to [0, 1] range
    if_scores_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10)
    ae_scores_norm = (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min() + 1e-10)
    
    # Combine scores (average)
    combined_scores = (if_scores_norm + ae_scores_norm) / 2
    
    # Determine anomaly status for individual metrics
    anomalies = pd.DataFrame()
    anomalies['timestamp'] = timestamps
    anomalies['anomaly_score'] = combined_scores
    
    # Add anomaly flags for each metric
    metric_list = ['cpu_usage', 'gpu_usage', 'memory_usage', 'latency', 
                   'frame_drops', 'encoding_errors', 'resolution_changes']
    
    for metric in metric_list:
        if metric in features.columns:
            # Consider as anomaly if in the top 5% of values for that metric
            threshold = np.percentile(features[metric], 95)
            anomalies[f'{metric}_anomaly'] = features[metric] > threshold
    
    # Global anomaly flag
    anomalies['is_anomaly'] = combined_scores < 0.3  # Lower scores indicate anomalies
    
    return anomalies
