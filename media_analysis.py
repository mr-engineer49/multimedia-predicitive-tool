import numpy as np
import pandas as pd
from PIL import Image
import io
import base64
from datetime import datetime
import random
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import streamlit as st
import cv2
import os

def analyze_image(image_file):
    """
    Analyze an image file using computer vision techniques and return quality metrics
    
    Args:
        image_file: Uploaded image file
        
    Returns:
        dict: Image analysis results
    """
    try:
        # Read the image file
        file_bytes = image_file.getvalue()
        image = Image.open(io.BytesIO(file_bytes))
        
        # Convert to numpy array for analysis
        img_array = np.array(image)
        
        # Calculate basic metrics
        width, height = image.size
        aspect_ratio = width / height
        
        # Get color channels
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            # RGB image
            r_channel = img_array[:, :, 0]
            g_channel = img_array[:, :, 1]
            b_channel = img_array[:, :, 2]
            
            # Calculate channel means and standard deviations
            r_mean, r_std = np.mean(r_channel), np.std(r_channel)
            g_mean, g_std = np.mean(g_channel), np.std(g_channel)
            b_mean, b_std = np.mean(b_channel), np.std(b_channel)
            
            # Calculate brightness and contrast
            brightness = (r_mean + g_mean + b_mean) / 3
            contrast = (r_std + g_std + b_std) / 3
            
            # Convert to grayscale for edge detection
            gray = 0.2989 * r_channel + 0.5870 * g_channel + 0.1140 * b_channel
        else:
            # Grayscale image
            gray = img_array
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            r_mean, g_mean, b_mean = brightness, brightness, brightness
            r_std, g_std, b_std = contrast, contrast, contrast
        
        # Calculate sharpness using gradient magnitude
        dx = np.diff(gray, axis=1)
        dy = np.diff(gray, axis=0)
        
        if dx.size > 0 and dy.size > 0:
            gradient_magnitude = np.mean(np.abs(dx)) + np.mean(np.abs(dy))
            sharpness = gradient_magnitude
        else:
            sharpness = 0
        
        # Calculate noise level (approximation)
        # Using the standard deviation of high-frequency components
        noise_level = np.std(gray - np.mean(gray))
        
        # Compression artifacts detection (approximation)
        # Higher values indicate more potential compression artifacts
        if width * height > 0:
            compression_artifacts = 100 - (10 * np.log10(width * height) - noise_level)
        else:
            compression_artifacts = 0
            
        # Clamp values between 0 and 100
        compression_artifacts = max(0, min(100, compression_artifacts))
        
        # Convert metrics to percentages or scores out of 100
        brightness_score = min(100, max(0, brightness / 2.55))
        contrast_score = min(100, max(0, contrast / 1.27))
        sharpness_score = min(100, max(0, sharpness * 10))
        noise_score = min(100, max(0, noise_level * 5))
        
        # Overall quality score (weighted average)
        quality_score = (
            0.3 * (100 - noise_score) + 
            0.3 * sharpness_score + 
            0.2 * (100 - abs(brightness_score - 50) * 2) + 
            0.2 * (100 - compression_artifacts)
        )
        
        # Return analysis results
        return {
            "resolution": f"{width}x{height}",
            "aspect_ratio": round(aspect_ratio, 2),
            "file_size_kb": round(len(file_bytes) / 1024, 2),
            "color_balance": {
                "red": round(r_mean, 2),
                "green": round(g_mean, 2),
                "blue": round(b_mean, 2)
            },
            "brightness": round(brightness_score, 2),
            "contrast": round(contrast_score, 2),
            "sharpness": round(sharpness_score, 2),
            "noise_level": round(noise_score, 2),
            "compression_artifacts": round(compression_artifacts, 2),
            "overall_quality": round(quality_score, 2)
        }
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return {
            "error": str(e)
        }

def detect_image_anomalies(analysis_results):
    """
    Detect anomalies in image analysis results
    
    Args:
        analysis_results (dict): Image analysis results
        
    Returns:
        list: List of detected anomalies and recommendations
    """
    anomalies = []
    
    # Check for low resolution
    dimensions = analysis_results.get("resolution", "0x0").split("x")
    if len(dimensions) == 2:
        width, height = int(dimensions[0]), int(dimensions[1])
        if width < 800 or height < 600:
            anomalies.append({
                "issue": "Low Resolution",
                "severity": "Medium",
                "description": f"Image resolution ({width}x{height}) is below recommended minimum (800x600)",
                "recommendation": "Consider using higher resolution images for better quality"
            })
    
    # Check for extreme aspect ratio
    aspect_ratio = analysis_results.get("aspect_ratio", 1)
    if aspect_ratio > 3 or aspect_ratio < 0.33:
        anomalies.append({
            "issue": "Extreme Aspect Ratio",
            "severity": "Low",
            "description": f"Image has an unusual aspect ratio ({aspect_ratio})",
            "recommendation": "Consider using standard aspect ratios for better display compatibility"
        })
    
    # Check for high noise level
    noise_level = analysis_results.get("noise_level", 0)
    if noise_level > 40:
        anomalies.append({
            "issue": "High Noise Level",
            "severity": "High" if noise_level > 60 else "Medium",
            "description": f"Image has significant noise (score: {noise_level})",
            "recommendation": "Consider applying noise reduction or using better lighting conditions"
        })
    
    # Check for low sharpness
    sharpness = analysis_results.get("sharpness", 0)
    if sharpness < 40:
        anomalies.append({
            "issue": "Low Sharpness",
            "severity": "Medium",
            "description": f"Image lacks sharpness (score: {sharpness})",
            "recommendation": "Consider applying sharpening filters or using better focus"
        })
    
    # Check for compression artifacts
    compression = analysis_results.get("compression_artifacts", 0)
    if compression > 50:
        anomalies.append({
            "issue": "Compression Artifacts",
            "severity": "High" if compression > 70 else "Medium",
            "description": f"Image shows signs of compression artifacts (score: {compression})",
            "recommendation": "Consider using less compression or higher quality formats"
        })
    
    # Check for brightness issues
    brightness = analysis_results.get("brightness", 50)
    if brightness < 30:
        anomalies.append({
            "issue": "Low Brightness",
            "severity": "Medium",
            "description": f"Image is too dark (brightness: {brightness})",
            "recommendation": "Consider increasing exposure or brightness adjustment"
        })
    elif brightness > 70:
        anomalies.append({
            "issue": "High Brightness",
            "severity": "Medium",
            "description": f"Image is too bright (brightness: {brightness})",
            "recommendation": "Consider reducing exposure or brightness adjustment"
        })
    
    # Check for contrast issues
    contrast = analysis_results.get("contrast", 50)
    if contrast < 30:
        anomalies.append({
            "issue": "Low Contrast",
            "severity": "Medium",
            "description": f"Image has low contrast (score: {contrast})",
            "recommendation": "Consider applying contrast enhancement"
        })
    
    # Check for color balance issues
    colors = analysis_results.get("color_balance", {})
    r, g, b = colors.get("red", 0), colors.get("green", 0), colors.get("blue", 0)
    if max(r, g, b) - min(r, g, b) > 50:
        anomalies.append({
            "issue": "Color Imbalance",
            "severity": "Low",
            "description": f"Image has uneven color distribution (R:{r}, G:{g}, B:{b})",
            "recommendation": "Consider applying white balance correction"
        })
    
    # Check overall quality
    quality = analysis_results.get("overall_quality", 0)
    if quality < 50:
        severity = "High" if quality < 30 else "Medium"
        anomalies.append({
            "issue": "Low Overall Quality",
            "severity": severity,
            "description": f"Image has low overall quality (score: {quality})",
            "recommendation": "Consider multiple improvements based on other detected issues"
        })
    
    return anomalies

def get_image_optimization_recommendations(analysis_results, anomalies):
    """
    Generate optimization recommendations for the image
    
    Args:
        analysis_results (dict): Image analysis results
        anomalies (list): Detected anomalies
        
    Returns:
        list: List of optimization recommendations
    """
    recommendations = []
    
    # Add recommendations based on detected anomalies
    for anomaly in anomalies:
        recommendations.append(anomaly["recommendation"])
    
    # Additional recommendations based on file size
    file_size_kb = analysis_results.get("file_size_kb", 0)
    dimensions = analysis_results.get("resolution", "0x0").split("x")
    
    if len(dimensions) == 2:
        width, height = int(dimensions[0]), int(dimensions[1])
        pixels = width * height
        
        # Check if file size is unusually large for the resolution
        if pixels > 0 and file_size_kb / pixels > 0.01:
            recommendations.append("Consider using more efficient image format or compression settings")
        
        # Recommend optimal format based on image characteristics
        has_transparency = False  # We would need actual image data to detect this
        
        if has_transparency:
            recommendations.append("For images with transparency, use PNG format for best quality or WebP for better compression")
        elif file_size_kb > 100 and "compression_artifacts" in analysis_results and analysis_results["compression_artifacts"] > 30:
            recommendations.append("Consider using WebP or AVIF formats for better compression without quality loss")
        elif "sharpness" in analysis_results and analysis_results["sharpness"] < 50:
            recommendations.append("For images with low sharpness, avoid additional compression that could further reduce quality")
    
    # Remove duplicates while preserving order
    unique_recommendations = []
    for rec in recommendations:
        if rec not in unique_recommendations:
            unique_recommendations.append(rec)
    
    return unique_recommendations

def forecast_media_quality_issues(analysis_results):
    """
    Forecast potential quality issues that might develop in similar media based on current analysis
    
    Args:
        analysis_results (dict): Current analysis results
        
    Returns:
        list: Forecasted issues and preventive recommendations
    """
    forecasts = []
    
    # Identify risk factors
    noise_level = analysis_results.get("noise_level", 0)
    sharpness = analysis_results.get("sharpness", 100)
    compression = analysis_results.get("compression_artifacts", 0)
    overall_quality = analysis_results.get("overall_quality", 100)
    
    # Forecast based on noise level trend
    if noise_level > 30:
        risk_factor = min(100, noise_level * 1.5)
        time_estimate = max(1, int(10 * (100 - risk_factor) / 100))
        forecasts.append({
            "issue": "Increasing Noise",
            "risk_percentage": round(risk_factor, 1),
            "time_estimate": f"{time_estimate} months",
            "description": "Media processed with similar settings may show increasing noise over time",
            "preventive_action": "Implement noise reduction filters in your processing pipeline"
        })
    
    # Forecast based on compression artifacts
    if compression > 40:
        risk_factor = min(100, compression * 1.3)
        time_estimate = max(1, int(12 * (100 - risk_factor) / 100))
        forecasts.append({
            "issue": "Degrading Compression Quality",
            "risk_percentage": round(risk_factor, 1),
            "time_estimate": f"{time_estimate} months",
            "description": "Repeated processing may lead to compounding compression artifacts",
            "preventive_action": "Use higher bitrates or lossless intermediate formats in your workflow"
        })
    
    # Forecast based on sharpness
    if sharpness < 60:
        risk_factor = min(100, (100 - sharpness) * 1.2)
        time_estimate = max(1, int(8 * (100 - risk_factor) / 100))
        forecasts.append({
            "issue": "Declining Sharpness",
            "risk_percentage": round(risk_factor, 1),
            "time_estimate": f"{time_estimate} months",
            "description": "Processing pipeline may cause progressive loss of detail",
            "preventive_action": "Implement detail-preserving algorithms and avoid multiple resizing operations"
        })
    
    # Overall quality trend
    if overall_quality < 70:
        risk_factor = min(100, (100 - overall_quality) * 1.4)
        time_estimate = max(1, int(6 * (100 - risk_factor) / 100))
        forecasts.append({
            "issue": "Overall Quality Degradation",
            "risk_percentage": round(risk_factor, 1),
            "time_estimate": f"{time_estimate} months",
            "description": "Similar media processing may result in progressive quality loss",
            "preventive_action": "Review entire media processing workflow and implement quality preservation measures"
        })
    
    return forecasts

def create_image_histogram(image_file):
    """
    Create RGB histogram for the image
    
    Args:
        image_file: Uploaded image file
        
    Returns:
        matplotlib.figure.Figure: Histogram figure
    """
    try:
        # Read the image file
        file_bytes = image_file.getvalue()
        image = Image.open(io.BytesIO(file_bytes))
        
        # Convert to numpy array for analysis
        img_array = np.array(image)
        
        # Create figure for the histogram
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Check if it's an RGB image
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            # RGB image
            colors = ('r', 'g', 'b')
            for i, color in enumerate(colors):
                histogram, bins = np.histogram(img_array[:, :, i].flatten(), bins=256, range=[0, 256])
                ax.plot(bins[:-1], histogram, color=color, alpha=0.7)
            
            ax.set_title('RGB Histogram')
        else:
            # Grayscale image
            histogram, bins = np.histogram(img_array.flatten(), bins=256, range=[0, 256])
            ax.plot(bins[:-1], histogram, color='gray', alpha=0.7)
            ax.set_title('Grayscale Histogram')
        
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.grid(alpha=0.2)
        
        return fig
    except Exception as e:
        st.error(f"Error creating histogram: {str(e)}")
        # Return a blank figure in case of error
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f"Error: {str(e)}", horizontalalignment='center', verticalalignment='center')
        return fig

def create_media_quality_radar_chart(metrics):
    """
    Create a radar chart visualizing media quality metrics
    
    Args:
        metrics (dict): Dictionary of quality metrics
        
    Returns:
        matplotlib.figure.Figure: Radar chart figure
    """
    # Define metrics to include in the radar chart
    radar_metrics = [
        'overall_quality',
        'sharpness',
        'brightness',
        'contrast',
        '100-noise_level',  # Inverse of noise (higher is better)
        '100-compression_artifacts'  # Inverse of compression artifacts (higher is better)
    ]
    
    # Get values for each metric, using 50 as default if not available
    values = []
    for metric in radar_metrics:
        if '-' in metric:
            # Handle inverse metrics (where lower original value is better)
            orig_metric = metric.split('-')[1]
            if orig_metric in metrics:
                values.append(100 - float(metrics.get(orig_metric, 50)))
            else:
                values.append(50)
        else:
            values.append(float(metrics.get(metric, 50)))
    
    # Labels for the radar chart
    labels = [
        'Overall Quality',
        'Sharpness',
        'Brightness',
        'Contrast',
        'Low Noise',
        'Low Compression'
    ]
    
    # Number of variables
    N = len(labels)
    
    # Create angles for each variable
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Values need to be repeated to close the loop
    values += values[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw the outline of the radar chart
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    
    # Fill the area
    ax.fill(angles, values, alpha=0.25)
    
    # Set labels and grid
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'])
    ax.set_ylim(0, 100)
    
    # Add a title
    plt.title('Media Quality Metrics', size=15, pad=20)
    
    return fig