# ETL Pipeline Dashboard

## Overview

This is a comprehensive ETL (Extract, Transform, Load) Pipeline system built with Flask that provides a web-based dashboard for monitoring and managing data processing workflows. The system supports multiple data sources and destinations, with extensive transformation capabilities, data validation, and real-time monitoring through a clean web interface.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

The application follows a modular architecture with clear separation of concerns:

### Backend Architecture
- **Flask Web Framework**: Serves as the main web application framework
- **SQLAlchemy ORM**: Handles database operations and model definitions
- **Modular ETL Components**: Separate modules for extraction, transformation, loading, and validation
- **Configuration Management**: YAML/JSON-based pipeline configuration system
- **Logging System**: Integrated logging with both file and database storage

### Frontend Architecture
- **Server-Side Rendered Templates**: Jinja2 templates for dynamic content
- **Vanilla JavaScript**: Client-side interactivity and dashboard updates
- **CSS Grid/Flexbox**: Responsive layout system
- **Real-time Updates**: Auto-refresh functionality for monitoring active pipelines

## Key Components

### Core ETL Components
1. **ETLPipeline** (`etl_pipeline.py`): Main orchestrator that coordinates the entire pipeline execution
2. **DataExtractor** (`extractors.py`): Handles data extraction from various sources (CSV, JSON, Excel, APIs, databases, Parquet)
3. **DataTransformer** (`transformers.py`): Performs data transformations including cleaning, scaling, encoding, and feature engineering
4. **DataLoader** (`loaders.py`): Loads processed data to various destinations
5. **DataValidator** (`validators.py`): Implements data quality checks and validation rules

### Configuration and Management
- **ConfigManager** (`config_manager.py`): Manages pipeline configurations with schema validation
- **PipelineLogger** (`pipeline_logger.py`): Custom logging system with database integration
- **Models** (`models.py`): SQLAlchemy models for pipeline runs and logs

### Web Interface
- **Flask App** (`app.py`): Main web application with dashboard routes
- **Templates**: HTML templates for dashboard and pipeline detail views
- **Static Assets**: CSS and JavaScript for frontend functionality

## Data Flow

1. **Configuration Loading**: Pipeline configurations are loaded from YAML/JSON files
2. **Pipeline Initialization**: ETL components are initialized with configuration
3. **Data Extraction**: Data is pulled from configured sources
4. **Data Transformation**: Applied transformations process the data
5. **Data Validation**: Quality checks ensure data integrity
6. **Data Loading**: Processed data is saved to target destinations
7. **Logging and Monitoring**: All activities are logged and displayed in the dashboard

## External Dependencies

### Python Libraries
- **Flask**: Web framework
- **SQLAlchemy**: Database ORM
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning transformations
- **PyYAML**: YAML configuration parsing
- **Requests**: HTTP client for API integrations

### Data Sources Support
- CSV files
- JSON files
- Excel spreadsheets
- REST APIs
- SQL databases (SQLite, PostgreSQL, MySQL)
- Parquet files

### Data Destinations Support
- File formats (CSV, JSON, Excel, Parquet)
- Databases
- REST APIs
- Multiple file outputs

## Deployment Strategy

### Database Configuration
- Default: SQLite for development (`etl_pipeline.db`)
- Production: Configurable via `DATABASE_URL` environment variable
- Auto-migration: Tables are created automatically on startup

### Environment Variables
- `FLASK_SECRET_KEY`: Application secret key
- `DATABASE_URL`: Database connection string
- `ETL_LOG_FILE`: Optional file logging path

### Execution Modes
1. **Web Dashboard**: Interactive monitoring and management
2. **Command Line**: Standalone script execution (`run_pipeline.py`)
3. **Programmatic**: Direct API integration

### Scalability Considerations
- Modular component design allows for easy extension
- Database logging enables distributed execution tracking
- Configuration-driven approach supports multiple pipeline definitions
- Threading support for concurrent pipeline execution

The system is designed to be both user-friendly for manual operation and robust enough for automated deployment in production environments.