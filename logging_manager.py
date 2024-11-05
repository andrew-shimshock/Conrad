# logging_manager.py

import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
import streamlit as st
from typing import List, Dict
import json

class LogManager:
    def __init__(self, log_file: str = "app.log"):
        self.log_file = log_file
        self._setup_logging()
        
    def _setup_logging(self):
        """Initialize logging configuration"""
        log_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
        )
        
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        # Configure file handler with rotation
        file_handler = RotatingFileHandler(
            os.path.join('logs', self.log_file),
            maxBytes=1024 * 1024 * 10,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(log_formatter)
        
        # Configure logging
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        
    def get_logs(self, n_lines: int = 100) -> List[Dict]:
        """Retrieve the last n lines of logs"""
        try:
            with open(os.path.join('logs', self.log_file), 'r') as f:
                logs = f.readlines()[-n_lines:]
                
            parsed_logs = []
            for log in logs:
                try:
                    # Parse log entry
                    timestamp = log[:23]
                    level = log[26:].split('-')[0].strip()
                    component = log.split('[')[1].split(']')[0]
                    message = '-'.join(log.split('-')[2:]).strip()
                    
                    parsed_logs.append({
                        'timestamp': timestamp,
                        'level': level,
                        'component': component,
                        'message': message
                    })
                except Exception:
                    # Handle malformed log entries
                    parsed_logs.append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3],
                        'level': 'ERROR',
                        'component': 'LogManager',
                        'message': 'Malformed log entry'
                    })
                    
            return parsed_logs
        except FileNotFoundError:
            return []

    def clear_logs(self):
        """Clear all logs"""
        try:
            open(os.path.join('logs', self.log_file), 'w').close()
            logging.info("Logs cleared by admin")
        except Exception as e:
            logging.error(f"Error clearing logs: {str(e)}")
