import os
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# Create the database instance
db = SQLAlchemy(model_class=Base)


def init_database(app):
    """Initialize database with Flask app"""
    # Configure the database
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///etl_pipeline.db")
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_recycle": 300,
        "pool_pre_ping": True,
    }
    
    # Initialize the app with the extension
    db.init_app(app)
    
    with app.app_context():
        # Import models to ensure tables are created
        from models import PipelineRun, PipelineLog, DataQuality  # noqa: F401
        db.create_all()