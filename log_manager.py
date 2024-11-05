import logging
from logging.handlers import RotatingFileHandler
from typing import List
import os
from datetime import datetime, timedelta

class LogManager:
    def __init__(self, log_file: str = 'app.log'):
        self.log_file = log_file
        self._setup_logger()

    def _setup_logger(self):
        """Configure the logging system."""
        self.logger = logging.getLogger('app')
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Create rotating file handler
        handler = RotatingFileHandler(
            self.log_file,
            maxBytes=1024 * 1024 * 10,  # 10MB
            backupCount=5
        )
        handler.setFormatter(formatter)
        
        # Add handler if it doesn't exist
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def get_logs(self, hours: int = 24, log_level: str = None) -> List[dict]:
        """
        Retrieve logs from the log file.
        
        Args:
            hours: Number of hours of logs to retrieve
            log_level: Filter logs by level (INFO, ERROR, WARNING, etc.)
        """
        logs = []
        if not os.path.exists(self.log_file):
            return logs

        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        try:
            with open(self.log_file, 'r') as file:
                for line in file:
                    try:
                        # Parse log line
                        timestamp_str = line[:19]
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        
                        if timestamp >= cutoff_time:
                            # Extract log level and message
                            parts = line[20:].split(' - ', 2)
                            if len(parts) >= 2:
                                level, message = parts[0], parts[1].strip()
                                
                                if not log_level or level == log_level:
                                    logs.append({
                                        'timestamp': timestamp,
                                        'level': level,
                                        'message': message
                                    })
                    except (ValueError, IndexError):
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error reading log file: {str(e)}")
            
        return sorted(logs, key=lambda x: x['timestamp'], reverse=True)

    def clear_logs(self):
        """Clear the log file."""
        try:
            open(self.log_file, 'w').close()
            self.logger.info("Log file cleared")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing log file: {str(e)}")
            return False
