"""
CSV data analysis utilities for predictive maintenance.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go

class CSVPredictiveAnalyzer:
    """Analyzes CSV data for predictive maintenance insights."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.regression_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.feature_columns = []
        self.target_column = None
        
    def analyze_csv_structure(self, df):
        """Analyze the structure and characteristics of uploaded CSV data."""
        analysis = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'summary_stats': df.describe().to_dict(),
            'timestamp': datetime.now()
        }
        
        # Identify potential time columns
        time_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].head())
                    time_columns.append(col)
                except:
                    pass
        
        analysis['potential_time_columns'] = time_columns
        
        # Identify numeric columns for analysis
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        analysis['numeric_columns'] = numeric_columns
        
        return analysis
    
    def prepare_data_for_analysis(self, df, time_column=None, target_column=None):
        """Prepare CSV data for predictive analysis."""
        processed_df = df.copy()
        
        # Handle time column
        if time_column and time_column in df.columns:
            try:
                processed_df[time_column] = pd.to_datetime(processed_df[time_column])
                processed_df = processed_df.sort_values(time_column)
                processed_df['timestamp'] = processed_df[time_column]
            except:
                # If conversion fails, create a synthetic timestamp
                processed_df['timestamp'] = pd.date_range(
                    start=datetime.now() - timedelta(hours=len(df)),
                    periods=len(df),
                    freq='H'
                )
        else:
            # Create synthetic timestamps
            processed_df['timestamp'] = pd.date_range(
                start=datetime.now() - timedelta(hours=len(df)),
                periods=len(df),
                freq='H'
            )
        
        # Get numeric columns for analysis
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove timestamp from numeric columns if it exists
        if 'timestamp' in numeric_cols:
            numeric_cols.remove('timestamp')
        
        self.feature_columns = numeric_cols
        if target_column and target_column in numeric_cols:
            self.target_column = target_column
            self.feature_columns = [col for col in numeric_cols if col != target_column]
        
        return processed_df
    
    def detect_anomalies(self, df):
        """Detect anomalies in the CSV data."""
        if not self.feature_columns:
            return df, {}
        
        # Prepare features for anomaly detection
        features = df[self.feature_columns].fillna(df[self.feature_columns].median())
        
        if len(features) < 2:
            return df, {'error': 'Insufficient data for anomaly detection'}
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Detect anomalies
        anomaly_labels = self.isolation_forest.fit_predict(features_scaled)
        anomaly_scores = self.isolation_forest.decision_function(features_scaled)
        
        # Add results to dataframe
        result_df = df.copy()
        result_df['anomaly'] = anomaly_labels == -1
        result_df['anomaly_score'] = anomaly_scores
        
        # Calculate anomaly statistics
        anomaly_stats = {
            'total_points': len(df),
            'anomalies_detected': (anomaly_labels == -1).sum(),
            'anomaly_percentage': ((anomaly_labels == -1).sum() / len(df)) * 100,
            'most_anomalous_features': self._identify_anomalous_features(features, anomaly_labels),
            'timestamp': datetime.now()
        }
        
        return result_df, anomaly_stats
    
    def _identify_anomalous_features(self, features, anomaly_labels):
        """Identify which features contribute most to anomalies."""
        anomalous_data = features[anomaly_labels == -1]
        normal_data = features[anomaly_labels == 1]
        
        if len(anomalous_data) == 0:
            return {}
        
        feature_importance = {}
        for i, col in enumerate(self.feature_columns):
            if len(normal_data) > 0:
                normal_mean = normal_data.iloc[:, i].mean()
                anomalous_mean = anomalous_data.iloc[:, i].mean()
                difference = abs(anomalous_mean - normal_mean)
                feature_importance[col] = difference
            else:
                feature_importance[col] = 0
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:5])  # Top 5 features
    
    def create_predictive_model(self, df, target_column=None, forecast_steps=10):
        """Create a predictive model from the CSV data."""
        if not self.feature_columns:
            return None, {'error': 'No suitable features for modeling'}
        
        features = df[self.feature_columns].fillna(df[self.feature_columns].median())
        
        if target_column and target_column in df.columns:
            target = df[target_column].fillna(df[target_column].median())
        else:
            # Use the first numeric column as target
            target_column = self.feature_columns[0]
            target = features.iloc[:, 0]
            features = features.iloc[:, 1:]  # Remove target from features
        
        if len(features) < 10:
            return None, {'error': 'Insufficient data for predictive modeling'}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Train model
        self.regression_model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate model performance
        train_score = self.regression_model.score(X_train, y_train)
        test_score = self.regression_model.score(X_test, y_test)
        
        # Generate forecast
        forecast_data = self._generate_forecast(df, features, target, forecast_steps)
        
        model_stats = {
            'target_column': target_column,
            'feature_columns': list(features.columns),
            'train_score': train_score,
            'test_score': test_score,
            'forecast_data': forecast_data,
            'feature_importance': dict(zip(
                features.columns,
                self.regression_model.feature_importances_
            )),
            'timestamp': datetime.now()
        }
        
        return self.regression_model, model_stats
    
    def _generate_forecast(self, df, features, target, forecast_steps):
        """Generate forecast data points."""
        if not self.is_trained or len(features) == 0:
            return []
        
        # Use last known values as starting point
        last_features = features.iloc[-1:].values
        last_timestamp = df['timestamp'].iloc[-1] if 'timestamp' in df.columns else datetime.now()
        
        forecast_data = []
        
        for i in range(forecast_steps):
            # Predict next value
            prediction = self.regression_model.predict(last_features)[0]
            
            # Create forecast point
            forecast_timestamp = last_timestamp + timedelta(hours=i+1)
            forecast_point = {
                'timestamp': forecast_timestamp,
                'predicted_value': prediction,
                'step': i + 1
            }
            
            forecast_data.append(forecast_point)
            
            # Update features for next prediction (simple approach)
            # In practice, this would be more sophisticated
            if len(last_features[0]) > 1:
                # Shift features and add prediction as new feature
                new_features = np.roll(last_features[0], -1)
                new_features[-1] = prediction
                last_features = new_features.reshape(1, -1)
        
        return forecast_data
    
    def create_visualization_charts(self, df, anomaly_data=None, forecast_data=None):
        """Create visualization charts for the CSV data analysis."""
        charts = {}
        
        if not self.feature_columns:
            return charts
        
        # Time series chart for numeric columns
        if 'timestamp' in df.columns and self.feature_columns:
            fig = go.Figure()
            
            for col in self.feature_columns[:5]:  # Limit to first 5 columns
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df[col],
                    mode='lines',
                    name=col,
                    line=dict(width=2)
                ))
            
            # Add anomalies if available
            if anomaly_data is not None and 'anomaly' in anomaly_data.columns:
                anomalies = anomaly_data[anomaly_data['anomaly'] == True]
                if not anomalies.empty:
                    fig.add_trace(go.Scatter(
                        x=anomalies['timestamp'],
                        y=anomalies[self.feature_columns[0]],
                        mode='markers',
                        name='Anomalies',
                        marker=dict(color='red', size=8, symbol='x')
                    ))
            
            # Add forecast if available
            if forecast_data:
                forecast_df = pd.DataFrame(forecast_data)
                fig.add_trace(go.Scatter(
                    x=forecast_df['timestamp'],
                    y=forecast_df['predicted_value'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(dash='dash', color='orange', width=3)
                ))
            
            fig.update_layout(
                title='Time Series Analysis with Anomalies and Forecast',
                xaxis_title='Time',
                yaxis_title='Values',
                hovermode='x',
                height=400
            )
            
            charts['time_series'] = fig
        
        # Correlation heatmap
        if len(self.feature_columns) > 1:
            corr_matrix = df[self.feature_columns].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0
            ))
            
            fig.update_layout(
                title='Feature Correlation Matrix',
                height=400
            )
            
            charts['correlation'] = fig
        
        # Distribution plots for top features
        if self.feature_columns:
            fig = go.Figure()
            
            for i, col in enumerate(self.feature_columns[:3]):  # Top 3 features
                fig.add_trace(go.Histogram(
                    x=df[col],
                    name=col,
                    opacity=0.7,
                    nbinsx=30
                ))
            
            fig.update_layout(
                title='Feature Distributions',
                xaxis_title='Value',
                yaxis_title='Frequency',
                barmode='overlay',
                height=400
            )
            
            charts['distributions'] = fig
        
        return charts
    
    def generate_maintenance_recommendations(self, df, anomaly_stats, model_stats):
        """Generate maintenance recommendations based on CSV analysis."""
        recommendations = []
        alerts = []
        
        # Anomaly-based recommendations
        if anomaly_stats and 'anomaly_percentage' in anomaly_stats:
            anomaly_pct = anomaly_stats['anomaly_percentage']
            
            if anomaly_pct > 15:
                alerts.append(f"HIGH: {anomaly_pct:.1f}% of data points are anomalous")
                recommendations.append("Immediate investigation required - system showing significant irregularities")
            elif anomaly_pct > 5:
                alerts.append(f"MEDIUM: {anomaly_pct:.1f}% of data points are anomalous")
                recommendations.append("Monitor system closely - elevated anomaly levels detected")
            else:
                recommendations.append("System operating within normal parameters")
        
        # Model-based recommendations
        if model_stats and 'test_score' in model_stats:
            test_score = model_stats['test_score']
            
            if test_score < 0.5:
                recommendations.append("Data shows unpredictable patterns - consider additional monitoring")
            elif test_score > 0.8:
                recommendations.append("System behavior is highly predictable - good for maintenance planning")
        
        # Feature importance recommendations
        if model_stats and 'feature_importance' in model_stats:
            importance = model_stats['feature_importance']
            top_feature = max(importance, key=importance.get)
            recommendations.append(f"Monitor '{top_feature}' closely - it's the most influential factor")
        
        # Forecast-based recommendations
        if model_stats and 'forecast_data' in model_stats:
            forecast = model_stats['forecast_data']
            if forecast:
                recent_trend = [point['predicted_value'] for point in forecast[:5]]
                if len(recent_trend) > 1:
                    trend_direction = "increasing" if recent_trend[-1] > recent_trend[0] else "decreasing"
                    recommendations.append(f"Predicted trend is {trend_direction} over next 5 periods")
        
        return {
            'recommendations': recommendations or ['Unable to generate specific recommendations from current data'],
            'alerts': alerts,
            'maintenance_priority': 'High' if alerts else 'Normal',
            'timestamp': datetime.now()
        }