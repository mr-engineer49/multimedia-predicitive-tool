"""
Enhanced multimedia analysis utilities for predictive maintenance.
"""
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

class MultimediaAnalyzer:
    """Enhanced multimedia analysis for predictive maintenance."""
    
    def __init__(self):
        self.analysis_history = []
        self.quality_thresholds = {
            'blur_threshold': 50,
            'brightness_threshold': 0.3,
            'contrast_threshold': 30,
            'saturation_threshold': 50
        }
    
    def analyze_image_quality(self, image_data):
        """Analyze image quality metrics for predictive maintenance."""
        try:
            # Convert to OpenCV format
            if isinstance(image_data, bytes):
                # Convert bytes to image
                image = Image.open(io.BytesIO(image_data))
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                image_cv = image_data
            
            # Get image dimensions
            height, width = image_cv.shape[:2]
            
            # Calculate quality metrics
            metrics = {
                'timestamp': datetime.now(),
                'width': width,
                'height': height,
                'aspect_ratio': width / height if height > 0 else 0,
                'total_pixels': width * height
            }
            
            # Blur detection using Laplacian variance
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            metrics['blur_score'] = blur_score
            metrics['is_blurry'] = blur_score < self.quality_thresholds['blur_threshold']
            
            # Brightness analysis
            brightness = np.mean(gray) / 255.0
            metrics['brightness'] = brightness
            metrics['is_dark'] = brightness < self.quality_thresholds['brightness_threshold']
            metrics['is_bright'] = brightness > (1 - self.quality_thresholds['brightness_threshold'])
            
            # Contrast analysis
            contrast = gray.std()
            metrics['contrast'] = contrast
            metrics['low_contrast'] = contrast < self.quality_thresholds['contrast_threshold']
            
            # Color analysis (if color image)
            if len(image_cv.shape) == 3:
                # Convert to HSV for better color analysis
                hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
                
                # Saturation analysis
                saturation = np.mean(hsv[:, :, 1])
                metrics['saturation'] = saturation
                metrics['low_saturation'] = saturation < self.quality_thresholds['saturation_threshold']
                
                # Color distribution
                colors = ['blue', 'green', 'red']
                for i, color in enumerate(colors):
                    metrics[f'{color}_mean'] = np.mean(image_cv[:, :, i])
                    metrics[f'{color}_std'] = np.std(image_cv[:, :, i])
            
            # Noise estimation
            noise_score = self._estimate_noise(gray)
            metrics['noise_score'] = noise_score
            metrics['high_noise'] = noise_score > 15  # Threshold for high noise
            
            # Compression artifacts detection
            artifacts_score = self._detect_compression_artifacts(gray)
            metrics['artifacts_score'] = artifacts_score
            metrics['has_artifacts'] = artifacts_score > 0.1
            
            # Overall quality score (0-100)
            quality_score = self._calculate_overall_quality(metrics)
            metrics['quality_score'] = quality_score
            
            # Add to history
            self.analysis_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            return {
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def _estimate_noise(self, gray_image):
        """Estimate noise level in the image."""
        # Use Laplacian filter to estimate noise
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        noise_score = laplacian.var()
        return min(noise_score / 1000, 100)  # Normalize to 0-100 scale
    
    def _detect_compression_artifacts(self, gray_image):
        """Detect compression artifacts."""
        # Use frequency domain analysis to detect artifacts
        f_transform = np.fft.fft2(gray_image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Look for regular patterns that indicate compression
        artifacts_score = np.std(magnitude_spectrum) / np.mean(magnitude_spectrum)
        return min(artifacts_score, 1.0)  # Normalize to 0-1 scale
    
    def _calculate_overall_quality(self, metrics):
        """Calculate overall quality score based on various metrics."""
        score = 100
        
        # Deduct points for quality issues
        if metrics.get('is_blurry', False):
            score -= 25
        if metrics.get('is_dark', False) or metrics.get('is_bright', False):
            score -= 15
        if metrics.get('low_contrast', False):
            score -= 20
        if metrics.get('low_saturation', False):
            score -= 10
        if metrics.get('high_noise', False):
            score -= 15
        if metrics.get('has_artifacts', False):
            score -= 15
        
        return max(0, score)
    
    def analyze_processing_performance(self, processing_times, file_sizes, output_qualities):
        """Analyze multimedia processing performance for predictive maintenance."""
        if not processing_times or len(processing_times) < 2:
            return {
                'error': 'Insufficient data for performance analysis',
                'timestamp': datetime.now()
            }
        
        performance_data = pd.DataFrame({
            'processing_time': processing_times,
            'file_size': file_sizes[:len(processing_times)],
            'output_quality': output_qualities[:len(processing_times)],
            'timestamp': pd.date_range(
                start=datetime.now() - timedelta(hours=len(processing_times)),
                periods=len(processing_times),
                freq='H'
            )
        })
        
        # Calculate performance metrics
        analysis = {
            'avg_processing_time': np.mean(processing_times),
            'processing_time_trend': np.polyfit(range(len(processing_times)), processing_times, 1)[0],
            'avg_output_quality': np.mean(output_qualities) if output_qualities else 0,
            'quality_trend': np.polyfit(range(len(output_qualities)), output_qualities, 1)[0] if output_qualities else 0,
            'throughput': len(processing_times) / (max(processing_times) - min(processing_times)) if len(processing_times) > 1 else 0,
            'efficiency_score': self._calculate_efficiency_score(performance_data),
            'timestamp': datetime.now()
        }
        
        # Detect performance anomalies
        analysis['anomalies'] = self._detect_performance_anomalies(performance_data)
        
        # Generate maintenance recommendations
        analysis['recommendations'] = self._generate_performance_recommendations(analysis)
        
        return analysis
    
    def _calculate_efficiency_score(self, performance_data):
        """Calculate processing efficiency score."""
        if performance_data.empty:
            return 0
        
        # Normalize metrics to 0-100 scale
        time_score = 100 - min(np.mean(performance_data['processing_time']) * 10, 100)
        quality_score = np.mean(performance_data['output_quality']) if 'output_quality' in performance_data.columns else 50
        
        # Weighted average
        efficiency = (time_score * 0.6 + quality_score * 0.4)
        return max(0, min(100, efficiency))
    
    def _detect_performance_anomalies(self, performance_data):
        """Detect anomalies in processing performance."""
        anomalies = []
        
        if len(performance_data) < 5:
            return anomalies
        
        # Check for processing time spikes
        time_mean = np.mean(performance_data['processing_time'])
        time_std = np.std(performance_data['processing_time'])
        time_threshold = time_mean + 2 * time_std
        
        time_anomalies = performance_data[performance_data['processing_time'] > time_threshold]
        if not time_anomalies.empty:
            anomalies.append(f"Processing time spikes detected: {len(time_anomalies)} instances")
        
        # Check for quality drops
        if 'output_quality' in performance_data.columns:
            quality_mean = np.mean(performance_data['output_quality'])
            quality_std = np.std(performance_data['output_quality'])
            quality_threshold = quality_mean - 2 * quality_std
            
            quality_anomalies = performance_data[performance_data['output_quality'] < quality_threshold]
            if not quality_anomalies.empty:
                anomalies.append(f"Quality drops detected: {len(quality_anomalies)} instances")
        
        return anomalies
    
    def _generate_performance_recommendations(self, analysis):
        """Generate maintenance recommendations based on performance analysis."""
        recommendations = []
        
        # Processing time recommendations
        if analysis['processing_time_trend'] > 0.1:
            recommendations.append("Processing times are increasing - consider system optimization")
        
        if analysis['avg_processing_time'] > 5:
            recommendations.append("High processing times detected - check hardware performance")
        
        # Quality recommendations
        if analysis['quality_trend'] < -0.5:
            recommendations.append("Output quality is declining - review processing parameters")
        
        if analysis['avg_output_quality'] < 70:
            recommendations.append("Low output quality detected - inspect processing pipeline")
        
        # Efficiency recommendations
        if analysis['efficiency_score'] < 60:
            recommendations.append("Low processing efficiency - consider hardware upgrade")
        
        # Anomaly recommendations
        if analysis['anomalies']:
            recommendations.append("Performance anomalies detected - investigate system stability")
        
        return recommendations or ["System performance is within normal parameters"]
    
    def create_quality_trend_chart(self):
        """Create a chart showing image quality trends over time."""
        if not self.analysis_history:
            return None
        
        df = pd.DataFrame(self.analysis_history)
        
        fig = go.Figure()
        
        # Add quality score trend
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['quality_score'],
            mode='lines+markers',
            name='Quality Score',
            line=dict(color='#2E86AB', width=3)
        ))
        
        # Add blur score (inverted for better visualization)
        if 'blur_score' in df.columns:
            normalized_blur = 100 - (df['blur_score'].clip(0, 100))
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=normalized_blur,
                mode='lines',
                name='Sharpness Score',
                line=dict(color='#A23B72', width=2, dash='dash')
            ))
        
        # Add brightness score
        if 'brightness' in df.columns:
            brightness_score = df['brightness'] * 100
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=brightness_score,
                mode='lines',
                name='Brightness Score',
                line=dict(color='#F18F01', width=2, dash='dot')
            ))
        
        fig.update_layout(
            title='Image Quality Trends Over Time',
            xaxis_title='Time',
            yaxis_title='Score (0-100)',
            yaxis=dict(range=[0, 100]),
            hovermode='x',
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_quality_distribution_chart(self):
        """Create a chart showing distribution of quality metrics."""
        if not self.analysis_history:
            return None
        
        df = pd.DataFrame(self.analysis_history)
        
        # Create subplots
        fig = go.Figure()
        
        # Quality score distribution
        fig.add_trace(go.Histogram(
            x=df['quality_score'],
            name='Quality Score',
            opacity=0.7,
            nbinsx=20,
            marker_color='#2E86AB'
        ))
        
        fig.update_layout(
            title='Quality Score Distribution',
            xaxis_title='Quality Score',
            yaxis_title='Frequency',
            height=300
        )
        
        return fig
    
    def get_quality_summary(self):
        """Get a summary of image quality analysis."""
        if not self.analysis_history:
            return {
                'message': 'No image analysis data available',
                'timestamp': datetime.now()
            }
        
        df = pd.DataFrame(self.analysis_history)
        
        summary = {
            'total_images_analyzed': len(df),
            'avg_quality_score': np.mean(df['quality_score']),
            'quality_trend': 'improving' if len(df) > 1 and df['quality_score'].iloc[-1] > df['quality_score'].iloc[0] else 'stable',
            'blurry_images': df['is_blurry'].sum() if 'is_blurry' in df.columns else 0,
            'dark_images': df['is_dark'].sum() if 'is_dark' in df.columns else 0,
            'low_contrast_images': df['low_contrast'].sum() if 'low_contrast' in df.columns else 0,
            'noisy_images': df['high_noise'].sum() if 'high_noise' in df.columns else 0,
            'timestamp': datetime.now()
        }
        
        # Calculate quality category distribution
        excellent = (df['quality_score'] >= 90).sum()
        good = ((df['quality_score'] >= 70) & (df['quality_score'] < 90)).sum()
        fair = ((df['quality_score'] >= 50) & (df['quality_score'] < 70)).sum()
        poor = (df['quality_score'] < 50).sum()
        
        summary['quality_distribution'] = {
            'excellent': excellent,
            'good': good,
            'fair': fair,
            'poor': poor
        }
        
        return summary
    
    def predict_quality_degradation(self):
        """Predict potential quality degradation based on trends."""
        if len(self.analysis_history) < 10:
            return {
                'prediction': 'Insufficient data for prediction',
                'confidence': 0,
                'recommendations': ['Analyze more images to enable trend prediction']
            }
        
        df = pd.DataFrame(self.analysis_history)
        
        # Calculate trend
        recent_scores = df['quality_score'].tail(10).values
        trend_slope = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        
        # Predict future quality
        if trend_slope < -2:
            prediction = "Quality degradation likely within next 5 images"
            confidence = 0.8
            recommendations = [
                "Investigate processing pipeline issues",
                "Check input source quality",
                "Monitor system resources during processing"
            ]
        elif trend_slope < -0.5:
            prediction = "Slight quality decline detected"
            confidence = 0.6
            recommendations = [
                "Monitor quality trends closely",
                "Consider preventive maintenance"
            ]
        else:
            prediction = "Quality stable or improving"
            confidence = 0.7
            recommendations = [
                "Continue current processing parameters",
                "Maintain regular quality monitoring"
            ]
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'trend_slope': trend_slope,
            'recommendations': recommendations,
            'timestamp': datetime.now()
        }