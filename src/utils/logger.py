#!/usr/bin/env python3
"""
Unified Logging System for AI Stock Prediction System
Provides consistent logging across all modules
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json

class UnifiedLogger:
    """Centralized logging system with structured output"""
    
    _instances: Dict[str, logging.Logger] = {}
    _configured = False
    
    @classmethod
    def get_logger(cls, name: str, config: Optional[Dict[str, Any]] = None) -> logging.Logger:
        """Get or create a logger instance"""
        if name not in cls._instances:
            cls._instances[name] = cls._create_logger(name, config or {})
        return cls._instances[name]
    
    @classmethod
    def _create_logger(cls, name: str, config: Dict[str, Any]) -> logging.Logger:
        """Create a new logger with unified configuration"""
        logger = logging.getLogger(name)
        
        if not cls._configured:
            cls._configure_logging(config)
            cls._configured = True
        
        return logger
    
    @classmethod
    def _configure_logging(cls, config: Dict[str, Any]):
        """Configure global logging settings"""
        # Get configuration with defaults
        log_level = config.get('level', os.getenv('LOG_LEVEL', 'INFO')).upper()
        log_dir = config.get('log_dir', 'logs')
        enable_console = config.get('console', True)
        enable_file = config.get('file', True)
        
        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # Set global log level
        logging.getLogger().setLevel(getattr(logging, log_level))
        
        # Custom formatter with structured output
        formatter = UnifiedFormatter()
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logging.getLogger().addHandler(console_handler)
        
        # File handlers
        if enable_file:
            # Application log
            app_handler = logging.handlers.RotatingFileHandler(
                log_path / 'app.log',
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            app_handler.setFormatter(formatter)
            logging.getLogger().addHandler(app_handler)
            
            # Error log
            error_handler = logging.handlers.RotatingFileHandler(
                log_path / 'error.log',
                maxBytes=5*1024*1024,   # 5MB
                backupCount=3
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            logging.getLogger().addHandler(error_handler)

class UnifiedFormatter(logging.Formatter):
    """Custom formatter for structured logging"""
    
    def format(self, record):
        # Create structured log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'module': record.name,
            'message': record.getMessage(),
            'file': f"{record.filename}:{record.lineno}",
            'function': record.funcName
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                          'pathname', 'filename', 'module', 'lineno', 
                          'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process',
                          'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                log_entry['extra'] = log_entry.get('extra', {})
                log_entry['extra'][key] = value
        
        return json.dumps(log_entry, ensure_ascii=False)

class PerformanceLogger:
    """Performance monitoring and logging"""
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = UnifiedLogger.get_logger(logger_name)
        self._start_times = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation"""
        self._start_times[operation] = datetime.now()
        self.logger.info(f"Started: {operation}")
    
    def end_timer(self, operation: str, extra_data: Optional[Dict] = None) -> float:
        """End timing an operation and log duration"""
        if operation not in self._start_times:
            self.logger.warning(f"Timer not found for operation: {operation}")
            return 0.0
        
        duration = (datetime.now() - self._start_times[operation]).total_seconds()
        del self._start_times[operation]
        
        log_data = {
            'operation': operation,
            'duration_seconds': duration,
            'performance': True
        }
        
        if extra_data:
            log_data.update(extra_data)
        
        self.logger.info(f"Completed: {operation} in {duration:.3f}s", extra=log_data)
        return duration

class ErrorHandler:
    """Centralized error handling"""
    
    def __init__(self, logger_name: str = "error"):
        self.logger = UnifiedLogger.get_logger(logger_name)
    
    def log_exception(self, operation: str, exception: Exception, 
                     context: Optional[Dict] = None) -> None:
        """Log exception with context"""
        error_data = {
            'operation': operation,
            'error_type': type(exception).__name__,
            'error_message': str(exception),
            'context': context or {}
        }
        
        self.logger.error(
            f"Exception in {operation}: {exception}",
            extra=error_data,
            exc_info=True
        )
    
    def log_api_error(self, endpoint: str, status_code: int, 
                     response: Optional[str] = None) -> None:
        """Log API-specific errors"""
        error_data = {
            'api_endpoint': endpoint,
            'status_code': status_code,
            'response': response,
            'api_error': True
        }
        
        self.logger.error(
            f"API Error: {endpoint} returned {status_code}",
            extra=error_data
        )

# Convenience functions for common use cases
def get_app_logger(name: str) -> logging.Logger:
    """Get application logger"""
    config = {
        'level': os.getenv('LOG_LEVEL', 'INFO'),
        'console': True,
        'file': True
    }
    return UnifiedLogger.get_logger(name, config)

def get_performance_logger() -> PerformanceLogger:
    """Get performance logger"""
    return PerformanceLogger()

def get_error_handler() -> ErrorHandler:
    """Get error handler"""
    return ErrorHandler()

# Example usage
if __name__ == "__main__":
    # Basic logging
    logger = get_app_logger("test")
    logger.info("Application started")
    logger.warning("This is a warning")
    
    # Performance logging
    perf = get_performance_logger()
    perf.start_timer("test_operation")
    import time
    time.sleep(0.1)
    perf.end_timer("test_operation", {"items_processed": 100})
    
    # Error handling
    error_handler = get_error_handler()
    try:
        raise ValueError("Test error")
    except Exception as e:
        error_handler.log_exception("test_operation", e, {"user_id": 123})