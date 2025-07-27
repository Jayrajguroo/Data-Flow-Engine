from datetime import datetime
from database import db


class PipelineRun(db.Model):
    """Model for tracking pipeline execution runs"""
    __tablename__ = 'pipeline_runs'
    
    id = db.Column(db.Integer, primary_key=True)
    config_name = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(20), nullable=False, default='pending')  # pending, running, completed, failed
    start_time = db.Column(db.DateTime, nullable=True)
    end_time = db.Column(db.DateTime, nullable=True)
    records_processed = db.Column(db.Integer, default=0)
    error_message = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to logs
    logs = db.relationship('PipelineLog', backref='pipeline_run', lazy=True, cascade='all, delete-orphan')
    
    @property
    def duration(self):
        """Calculate duration of pipeline run"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    def to_dict(self):
        return {
            'id': self.id,
            'config_name': self.config_name,
            'status': self.status,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'records_processed': self.records_processed,
            'error_message': self.error_message,
            'duration': str(self.duration) if self.duration else None
        }


class PipelineLog(db.Model):
    """Model for storing pipeline execution logs"""
    __tablename__ = 'pipeline_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    pipeline_run_id = db.Column(db.Integer, db.ForeignKey('pipeline_runs.id'), nullable=False)
    level = db.Column(db.String(10), nullable=False)  # INFO, WARNING, ERROR, DEBUG
    message = db.Column(db.Text, nullable=False)
    step = db.Column(db.String(50), nullable=True)  # extract, transform, load, validate
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'pipeline_run_id': self.pipeline_run_id,
            'level': self.level,
            'message': self.message,
            'step': self.step,
            'timestamp': self.timestamp.isoformat()
        }


class DataQuality(db.Model):
    """Model for storing data quality metrics"""
    __tablename__ = 'data_quality'
    
    id = db.Column(db.Integer, primary_key=True)
    pipeline_run_id = db.Column(db.Integer, db.ForeignKey('pipeline_runs.id'), nullable=False)
    step = db.Column(db.String(50), nullable=False)
    metric_name = db.Column(db.String(100), nullable=False)
    metric_value = db.Column(db.Float, nullable=True)
    metric_description = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'pipeline_run_id': self.pipeline_run_id,
            'step': self.step,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'metric_description': self.metric_description,
            'timestamp': self.timestamp.isoformat()
        }
