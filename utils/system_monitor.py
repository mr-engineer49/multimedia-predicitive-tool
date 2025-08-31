"""
System monitoring utilities for real-time PC health data collection.
"""
import psutil
import platform
import subprocess
import json
import os
from datetime import datetime
import pandas as pd

class SystemMonitor:
    """Real-time system monitoring class."""
    
    def __init__(self):
        self.system_info = self._get_system_info()
    
    def _get_system_info(self):
        """Get basic system information."""
        return {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'hostname': platform.node(),
            'processor': platform.processor(),
            'python_version': platform.python_version()
        }
    
    def get_cpu_info(self):
        """Get detailed CPU information and usage."""
        cpu_info = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'cpu_per_core': psutil.cpu_percent(interval=1, percpu=True),
            'timestamp': datetime.now()
        }
        return cpu_info
    
    def get_memory_info(self):
        """Get memory usage information."""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        memory_info = {
            'total_memory': memory.total,
            'available_memory': memory.available,
            'used_memory': memory.used,
            'memory_percent': memory.percent,
            'swap_total': swap.total,
            'swap_used': swap.used,
            'swap_percent': swap.percent,
            'timestamp': datetime.now()
        }
        return memory_info
    
    def get_disk_info(self):
        """Get disk usage information."""
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        disk_info = {
            'total_disk': disk_usage.total,
            'used_disk': disk_usage.used,
            'free_disk': disk_usage.free,
            'disk_percent': (disk_usage.used / disk_usage.total) * 100,
            'disk_read_bytes': disk_io.read_bytes if disk_io else 0,
            'disk_write_bytes': disk_io.write_bytes if disk_io else 0,
            'timestamp': datetime.now()
        }
        return disk_info
    
    def get_network_info(self):
        """Get network usage information."""
        network_io = psutil.net_io_counters()
        
        network_info = {
            'bytes_sent': network_io.bytes_sent,
            'bytes_recv': network_io.bytes_recv,
            'packets_sent': network_io.packets_sent,
            'packets_recv': network_io.packets_recv,
            'timestamp': datetime.now()
        }
        return network_info
    
    def get_temperature_info(self):
        """Get system temperature information if available."""
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                temp_info = {}
                for name, entries in temps.items():
                    temp_info[name] = [
                        {'label': entry.label or 'N/A', 'current': entry.current}
                        for entry in entries
                    ]
                temp_info['timestamp'] = datetime.now()
                return temp_info
        except Exception:
            pass
        return {'timestamp': datetime.now(), 'error': 'Temperature sensors not available'}
    
    def get_process_info(self, limit=10):
        """Get information about running processes."""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        # Sort by CPU usage and get top processes
        processes.sort(key=lambda x: x['cpu_percent'] or 0, reverse=True)
        return {
            'top_processes': processes[:limit],
            'total_processes': len(psutil.pids()),
            'timestamp': datetime.now()
        }
    
    def get_comprehensive_health_data(self):
        """Get comprehensive system health data."""
        return {
            'system_info': self.system_info,
            'cpu': self.get_cpu_info(),
            'memory': self.get_memory_info(),
            'disk': self.get_disk_info(),
            'network': self.get_network_info(),
            'temperature': self.get_temperature_info(),
            'processes': self.get_process_info(),
            'boot_time': datetime.fromtimestamp(psutil.boot_time()),
            'timestamp': datetime.now()
        }
    
    def get_health_metrics_df(self):
        """Get system health metrics as a pandas DataFrame."""
        health_data = self.get_comprehensive_health_data()
        
        # Create a flattened structure for DataFrame
        metrics_data = {
            'timestamp': health_data['timestamp'],
            'cpu_usage': health_data['cpu']['cpu_percent'],
            'memory_usage': health_data['memory']['memory_percent'],
            'disk_usage': health_data['disk']['disk_percent'],
            'memory_total_gb': health_data['memory']['total_memory'] / (1024**3),
            'memory_available_gb': health_data['memory']['available_memory'] / (1024**3),
            'disk_total_gb': health_data['disk']['total_disk'] / (1024**3),
            'disk_free_gb': health_data['disk']['free_disk'] / (1024**3),
            'network_bytes_sent': health_data['network']['bytes_sent'],
            'network_bytes_recv': health_data['network']['bytes_recv'],
            'total_processes': health_data['processes']['total_processes']
        }
        
        return pd.DataFrame([metrics_data])

class PredictiveHealthAnalyzer:
    """Analyzes system health data for predictive maintenance."""
    
    def __init__(self):
        self.health_thresholds = {
            'cpu_critical': 90,
            'cpu_warning': 75,
            'memory_critical': 90,
            'memory_warning': 80,
            'disk_critical': 95,
            'disk_warning': 85,
            'temperature_critical': 80,
            'temperature_warning': 70
        }
    
    def analyze_health_trends(self, health_data_df):
        """Analyze health trends from historical data."""
        if health_data_df.empty or len(health_data_df) < 2:
            return {
                'trend': 'Insufficient data',
                'recommendations': ['Collect more data points for trend analysis'],
                'alerts': []
            }
        
        alerts = []
        recommendations = []
        
        # Check current values against thresholds
        latest = health_data_df.iloc[-1]
        
        if latest['cpu_usage'] > self.health_thresholds['cpu_critical']:
            alerts.append(f"CRITICAL: CPU usage at {latest['cpu_usage']:.1f}%")
            recommendations.append("Immediate action: Close unnecessary applications")
        elif latest['cpu_usage'] > self.health_thresholds['cpu_warning']:
            alerts.append(f"WARNING: High CPU usage at {latest['cpu_usage']:.1f}%")
            recommendations.append("Monitor CPU usage and consider optimization")
        
        if latest['memory_usage'] > self.health_thresholds['memory_critical']:
            alerts.append(f"CRITICAL: Memory usage at {latest['memory_usage']:.1f}%")
            recommendations.append("Immediate action: Free up memory or add more RAM")
        elif latest['memory_usage'] > self.health_thresholds['memory_warning']:
            alerts.append(f"WARNING: High memory usage at {latest['memory_usage']:.1f}%")
            recommendations.append("Consider closing memory-intensive applications")
        
        if latest['disk_usage'] > self.health_thresholds['disk_critical']:
            alerts.append(f"CRITICAL: Disk usage at {latest['disk_usage']:.1f}%")
            recommendations.append("Immediate action: Free up disk space")
        elif latest['disk_usage'] > self.health_thresholds['disk_warning']:
            alerts.append(f"WARNING: High disk usage at {latest['disk_usage']:.1f}%")
            recommendations.append("Clean up unnecessary files")
        
        # Analyze trends
        if len(health_data_df) >= 5:
            cpu_trend = health_data_df['cpu_usage'].tail(5).pct_change().mean()
            memory_trend = health_data_df['memory_usage'].tail(5).pct_change().mean()
            
            if cpu_trend > 0.1:
                recommendations.append("CPU usage trending upward - investigate causes")
            if memory_trend > 0.1:
                recommendations.append("Memory usage trending upward - monitor for leaks")
        
        # Overall health assessment
        avg_cpu = health_data_df['cpu_usage'].tail(10).mean()
        avg_memory = health_data_df['memory_usage'].tail(10).mean()
        avg_disk = health_data_df['disk_usage'].tail(10).mean()
        
        health_score = 100 - ((avg_cpu + avg_memory + avg_disk) / 3)
        health_score = max(0, min(100, health_score))
        
        if health_score > 80:
            trend = "Excellent"
        elif health_score > 60:
            trend = "Good"
        elif health_score > 40:
            trend = "Fair"
        else:
            trend = "Poor"
        
        return {
            'trend': trend,
            'health_score': health_score,
            'recommendations': recommendations or ['System running normally'],
            'alerts': alerts,
            'avg_cpu': avg_cpu,
            'avg_memory': avg_memory,
            'avg_disk': avg_disk
        }
    
    def predict_maintenance_needs(self, health_data_df):
        """Predict when maintenance might be needed."""
        if health_data_df.empty or len(health_data_df) < 10:
            return {
                'prediction': 'Insufficient data for prediction',
                'confidence': 0,
                'recommended_actions': ['Collect more historical data']
            }
        
        # Simple trend-based prediction
        recent_data = health_data_df.tail(10)
        
        cpu_growth = recent_data['cpu_usage'].pct_change().mean()
        memory_growth = recent_data['memory_usage'].pct_change().mean()
        disk_growth = recent_data['disk_usage'].pct_change().mean()
        
        risk_factors = []
        if cpu_growth > 0.05:
            risk_factors.append("Increasing CPU usage trend")
        if memory_growth > 0.05:
            risk_factors.append("Increasing memory usage trend")
        if disk_growth > 0.02:
            risk_factors.append("Increasing disk usage trend")
        
        if len(risk_factors) >= 2:
            prediction = "Maintenance recommended within 7 days"
            confidence = 0.7
        elif len(risk_factors) == 1:
            prediction = "Monitor closely, possible maintenance in 14 days"
            confidence = 0.4
        else:
            prediction = "System stable, routine maintenance in 30 days"
            confidence = 0.8
        
        recommended_actions = [
            "Regular system cleanup",
            "Monitor resource usage trends",
            "Update system software",
            "Check for malware"
        ]
        
        if risk_factors:
            recommended_actions.extend([
                f"Address: {factor}" for factor in risk_factors
            ])
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'risk_factors': risk_factors,
            'recommended_actions': recommended_actions
        }