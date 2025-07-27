import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for
import threading
from etl_pipeline import ETLPipeline
from config_manager import ConfigManager
from pipeline_logger import PipelineLogger
from database import db, init_database

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "etl_pipeline_secret_key")

# Initialize database
init_database(app)


@app.route('/')
def index():
    """Main dashboard showing pipeline runs and status"""
    import models
    
    # Get pipeline runs from database, handle gracefully if table doesn't exist
    try:
        pipeline_runs = models.PipelineRun.query.order_by(models.PipelineRun.start_time.desc()).limit(10).all()
    except Exception as e:
        # If there's an issue with the database, create tables and return empty list
        print(f"Database error: {e}")
        db.create_all()
        pipeline_runs = []
    
    config_manager = ConfigManager()
    available_configs = config_manager.list_available_configs()
    
    return render_template('index.html', 
                         pipeline_runs=pipeline_runs,
                         available_configs=available_configs)


@app.route('/api/pipeline/run', methods=['POST'])
def run_pipeline():
    """API endpoint to trigger pipeline execution"""
    try:
        data = request.get_json()
        config_name = data.get('config_name', 'default')
        
        # Create pipeline run record
        pipeline_run = models.PipelineRun(
            config_name=config_name,
            status='running',
            start_time=datetime.utcnow()
        )
        db.session.add(pipeline_run)
        db.session.commit()
        
        # Start pipeline in background thread
        def run_pipeline_background():
            try:
                config_manager = ConfigManager()
                config = config_manager.load_config(config_name)
                
                pipeline = ETLPipeline(config, pipeline_run.id)
                result = pipeline.run()
                
                # Update pipeline run status
                pipeline_run.status = 'completed' if result['success'] else 'failed'
                pipeline_run.end_time = datetime.utcnow()
                pipeline_run.records_processed = result.get('records_processed', 0)
                pipeline_run.error_message = result.get('error_message')
                db.session.commit()
                
            except Exception as e:
                pipeline_run.status = 'failed'
                pipeline_run.end_time = datetime.utcnow()
                pipeline_run.error_message = str(e)
                db.session.commit()
        
        thread = threading.Thread(target=run_pipeline_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'pipeline_run_id': pipeline_run.id,
            'message': 'Pipeline started successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/pipeline/status/<int:run_id>')
def get_pipeline_status(run_id):
    """Get status of a specific pipeline run"""
    pipeline_run = models.PipelineRun.query.get_or_404(run_id)
    logs = models.PipelineLog.query.filter_by(pipeline_run_id=run_id).order_by(models.PipelineLog.timestamp).all()
    
    return jsonify({
        'id': pipeline_run.id,
        'config_name': pipeline_run.config_name,
        'status': pipeline_run.status,
        'start_time': pipeline_run.start_time.isoformat() if pipeline_run.start_time else None,
        'end_time': pipeline_run.end_time.isoformat() if pipeline_run.end_time else None,
        'records_processed': pipeline_run.records_processed,
        'error_message': pipeline_run.error_message,
        'logs': [{
            'level': log.level,
            'message': log.message,
            'timestamp': log.timestamp.isoformat()
        } for log in logs]
    })


@app.route('/pipeline/<int:run_id>')
def pipeline_detail(run_id):
    """Detailed view of a pipeline run"""
    pipeline_run = models.PipelineRun.query.get_or_404(run_id)
    return render_template('pipeline_detail.html', pipeline_run=pipeline_run)


@app.route('/api/config/validate', methods=['POST'])
def validate_config():
    """Validate pipeline configuration"""
    try:
        config_data = request.get_json()
        config_manager = ConfigManager()
        is_valid, errors = config_manager.validate_config(config_data)
        
        return jsonify({
            'valid': is_valid,
            'errors': errors
        })
        
    except Exception as e:
        return jsonify({
            'valid': False,
            'errors': [str(e)]
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
