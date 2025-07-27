#!/usr/bin/env python3
"""
Standalone script for running ETL pipelines from command line
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from etl_pipeline import ETLPipeline
from config_manager import ConfigManager
from pipeline_logger import PipelineLogger, setup_pipeline_logging


def main():
    """Main function for running ETL pipelines"""
    parser = argparse.ArgumentParser(description='Run ETL Pipeline')
    parser.add_argument('config', help='Configuration name or file path')
    parser.add_argument('--dry-run', action='store_true', help='Validate configuration without running pipeline')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    parser.add_argument('--log-file', help='Log file path (if not specified, logs to console only)')
    parser.add_argument('--output-dir', help='Override output directory for pipeline results')
    parser.add_argument('--validate-only', action='store_true', help='Only validate configuration and exit')
    parser.add_argument('--list-configs', action='store_true', help='List available configurations and exit')
    parser.add_argument('--show-config', action='store_true', help='Show configuration content and exit')
    parser.add_argument('--stats', action='store_true', help='Show detailed statistics during execution')
    parser.add_argument('--no-db', action='store_true', help='Run without database logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_pipeline_logging(args.log_level, args.log_file)
    
    # Initialize config manager
    config_manager = ConfigManager()
    
    # Handle list configs
    if args.list_configs:
        print("Available configurations:")
        configs = config_manager.list_available_configs()
        if configs:
            for config in configs:
                print(f"  - {config}")
        else:
            print("  No configurations found")
        return 0
    
    # Load configuration
    try:
        if os.path.isfile(args.config):
            # Direct file path
            with open(args.config, 'r') as f:
                if args.config.endswith('.json'):
                    config = json.load(f)
                else:
                    import yaml
                    config = yaml.safe_load(f)
            config_name = Path(args.config).stem
        else:
            # Configuration name
            config = config_manager.load_config(args.config)
            config_name = args.config
            
        print(f"Loaded configuration: {config_name}")
        
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return 1
    
    # Handle show config
    if args.show_config:
        print(f"\nConfiguration '{config_name}':")
        print(json.dumps(config, indent=2, default=str))
        return 0
    
    # Validate configuration
    is_valid, errors = config_manager.validate_config(config)
    if not is_valid:
        print("Configuration validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    
    print("✅ Configuration validation passed")
    
    if args.validate_only:
        return 0
    
    # Handle output directory override
    if args.output_dir:
        # Update all file-based targets with new output directory
        for target in config.get('loading', {}).get('targets', []):
            if target.get('type') in ['csv', 'json', 'excel', 'parquet']:
                original_path = Path(target['path'])
                target['path'] = str(Path(args.output_dir) / original_path.name)
    
    # Dry run
    if args.dry_run:
        print("🔍 Dry run mode - validating pipeline without execution")
        print(f"Configuration: {config_name}")
        print(f"Sources: {len(config.get('extraction', {}).get('sources', []))}")
        print(f"Transformation steps: {len(config.get('transformation', {}).get('steps', []))}")
        print(f"Loading targets: {len(config.get('loading', {}).get('targets', []))}")
        return 0
    
    # Create logger (without database if --no-db flag is set)
    logger = PipelineLogger(pipeline_run_id=None if args.no_db else 1, log_level=args.log_level)
    
    # Create and run pipeline
    try:
        print(f"🚀 Starting ETL pipeline execution: {config_name}")
        start_time = datetime.utcnow()
        
        pipeline = ETLPipeline(config, pipeline_run_id=None if args.no_db else 1)
        
        # Add progress callback if stats requested
        if args.stats:
            pipeline.enable_stats = True
        
        result = pipeline.run()
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # Print results
        if result['success']:
            print(f"✅ Pipeline completed successfully!")
            print(f"   Duration: {duration:.2f} seconds")
            print(f"   Records processed: {result.get('records_processed', 0):,}")
            
            if args.stats:
                # Show additional statistics
                data_summary = pipeline.get_data_summary()
                if data_summary:
                    print(f"   Final dataset shape: {data_summary.get('shape', 'Unknown')}")
                    print(f"   Memory usage: {data_summary.get('memory_usage', 0) / (1024*1024):.2f} MB")
                    
                    null_counts = data_summary.get('null_counts', {})
                    total_nulls = sum(null_counts.values()) if null_counts else 0
                    print(f"   Total null values: {total_nulls:,}")
        else:
            print(f"❌ Pipeline failed!")
            print(f"   Error: {result.get('error_message', 'Unknown error')}")
            print(f"   Duration: {duration:.2f} seconds")
            print(f"   Records processed before failure: {result.get('records_processed', 0):,}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline execution interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        logger.close()
    
    return 0


def create_sample_config():
    """Create a sample configuration file"""
    config_manager = ConfigManager()
    template = config_manager.get_config_template()
    
    # Save sample configuration
    sample_path = config_manager.save_config(template, 'sample', 'yaml')
    print(f"Sample configuration created at: {sample_path}")


if __name__ == '__main__':
    # Check if user wants to create a sample config
    if len(sys.argv) > 1 and sys.argv[1] == '--create-sample':
        create_sample_config()
        sys.exit(0)
    
    # Run main function
    exit_code = main()
    sys.exit(exit_code)
