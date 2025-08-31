# Overview

This is a multimedia processing predictive maintenance system that monitors and analyzes media processing workflows with real-time metrics and anomaly detection. The application provides insights into hardware performance, media quality, and potential system failures through a comprehensive dashboard built with Streamlit. The system is designed to detect anomalies in multimedia processing environments and predict potential maintenance needs before failures occur.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit-based Dashboard**: Single-page application using Streamlit for the main user interface
- **Real-time Visualization**: Plotly charts for interactive hardware metrics, media quality indicators, and system health gauges
- **Modular Components**: Separated dashboard components in `utils/dashboard_components.py` for maintainability
- **Responsive Layout**: Multi-column layouts with sidebar navigation for different analysis views

## Backend Architecture
- **Data Generation System**: Synthetic data generators for hardware metrics, media quality data, and system events
- **Anomaly Detection Engine**: Multiple detection algorithms including Isolation Forest and simplified autoencoder implementations
- **Predictive Maintenance Models**: RandomForest-based forecasting models for system metrics and failure prediction
- **API Layer**: Flask-based REST API for external integrations and alert management
- **Message Processing**: Kafka consumer for real-time log processing and event handling

## Data Processing Pipeline
- **Real-time Metrics Collection**: Continuous generation of hardware performance data (CPU, GPU, memory usage)
- **Media Quality Analysis**: Computer vision-based image analysis and media processing quality metrics
- **Anomaly Detection**: Multi-algorithm approach combining statistical and machine learning methods
- **Trend Analysis**: Time-series forecasting for predictive maintenance scheduling

## Machine Learning Components
- **Isolation Forest**: Primary anomaly detection algorithm for identifying outliers in system metrics
- **Simplified Autoencoder**: Custom implementation for reconstruction error-based anomaly detection
- **Random Forest Regression**: Predictive models for forecasting system performance trends
- **Statistical Analysis**: Threshold-based alerting and trend detection algorithms

## Architecture Patterns
- **Modular Design**: Clear separation of concerns with dedicated modules for data generation, analysis, and visualization
- **Session State Management**: Streamlit session state for maintaining application state across interactions
- **Pipeline Architecture**: Data flows through generation → processing → analysis → visualization stages
- **Event-Driven Processing**: Real-time updates and alert generation based on threshold violations

# External Dependencies

## Core Framework Dependencies
- **Streamlit**: Primary web application framework for dashboard creation
- **Plotly**: Interactive visualization library for charts and graphs
- **Flask**: REST API framework for external integrations
- **Pandas & NumPy**: Data manipulation and numerical computing

## Machine Learning Libraries
- **Scikit-learn**: Machine learning algorithms including Isolation Forest and Random Forest
- **PIL (Pillow)**: Image processing and computer vision analysis
- **OpenCV**: Advanced image and video processing capabilities

## Data Processing
- **Kafka-Python**: Message queue integration for real-time log processing
- **Requests**: HTTP client for external API integrations
- **Trafilatura**: Web scraping and data extraction utilities

## Development Tools
- **Dev Container**: Configured development environment with Python 3.11
- **Requirements Management**: Comprehensive dependency specification for reproducible environments

## Optional Integrations
- **Database Support**: Architecture prepared for database integration (SQLite, PostgreSQL via potential Drizzle ORM addition)
- **External APIs**: Framework for fetching real-time transportation and system data
- **Cloud Services**: Extensible design for cloud-based deployment and scaling

The system is designed with a microservices-ready architecture that can be easily extended with additional data sources, analysis algorithms, and visualization components.