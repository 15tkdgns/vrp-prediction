"""
Enhanced logging system for the SPY Analysis system.
Provides structured logging with performance monitoring and error tracking.
"""

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .exceptions import SPYAnalysisError


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        
        # Add custom fields from extra
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "getMessage"
            }:
                log_obj[key] = value
        
        return json.dumps(log_obj, default=str)


class PerformanceLogger:
    """Logger for performance monitoring."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def log_execution_time(
        self, 
        function_name: str, 
        execution_time: float, 
        **extra_context
    ) -> None:
        """Log function execution time."""
        self.logger.info(
            f"🚀 Function '{function_name}' executed in {execution_time:.4f}s",
            extra={
                "metric_type": "performance",
                "function": function_name,
                "execution_time": execution_time,
                **extra_context
            }
        )
    
    def log_memory_usage(
        self, 
        component: str, 
        memory_mb: float, 
        **extra_context
    ) -> None:
        """Log memory usage."""
        self.logger.info(
            f"💾 Component '{component}' using {memory_mb:.2f} MB",
            extra={
                "metric_type": "memory",
                "component": component,
                "memory_mb": memory_mb,
                **extra_context
            }
        )


class SPYAnalysisLogger:
    """Enhanced logger class for SPY Analysis system."""
    
    def __init__(
        self,
        name: str = "spy_analysis",
        level: Union[str, int] = logging.INFO,
        log_dir: Optional[Path] = None,
        use_json_format: bool = False,  # Keep compatibility with existing system
        enable_file_logging: bool = True,
        enable_console_logging: bool = True,
        max_file_size_mb: int = 10,
        backup_count: int = 5
    ):
        self.name = name
        self.log_dir = log_dir or Path("/root/workspace/logs")
        self.use_json_format = use_json_format
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Create formatters
        if use_json_format:
            formatter = JsonFormatter()
        else:
            # Enhanced formatter with emoji support
            formatter = logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s"
            )
        
        simple_formatter = logging.Formatter("%(levelname)s | %(message)s")
        
        # Add console handler
        if enable_console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(simple_formatter)
            self.logger.addHandler(console_handler)
        
        # Add file handler
        if enable_file_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Main log file with date
            log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size_mb * 1024 * 1024,
                backupCount=backup_count
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            # Error log file
            error_log_file = self.log_dir / f"{name}_errors_{datetime.now().strftime('%Y%m%d')}.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=max_file_size_mb * 1024 * 1024,
                backupCount=backup_count
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            self.logger.addHandler(error_handler)
        
        # Initialize performance logger
        self.performance = PerformanceLogger(self.logger)
    
    def debug(self, message: str, emoji: str = "🔍", **extra) -> None:
        """Log debug message with emoji."""
        self.logger.debug(f"{emoji} {message}", extra=extra)
    
    def info(self, message: str, emoji: str = "ℹ️", **extra) -> None:
        """Log info message with emoji."""
        self.logger.info(f"{emoji} {message}", extra=extra)
    
    def success(self, message: str, **extra) -> None:
        """Log success message."""
        self.logger.info(f"✅ {message}", extra=extra)
    
    def warning(self, message: str, **extra) -> None:
        """Log warning message."""
        self.logger.warning(f"⚠️ {message}", extra=extra)
    
    def error(self, message: str, **extra) -> None:
        """Log error message."""
        self.logger.error(f"❌ {message}", extra=extra)
    
    def critical(self, message: str, **extra) -> None:
        """Log critical message."""
        self.logger.critical(f"🚨 {message}", extra=extra)
    
    def exception(self, message: str, **extra) -> None:
        """Log exception with traceback."""
        self.logger.exception(f"💥 {message}", extra=extra)
    
    def section(self, title: str, emoji: str = "🔧") -> None:
        """Log section header."""
        separator = "=" * 60
        self.logger.info(f"\n{separator}")
        self.logger.info(f"{emoji} {title}")
        self.logger.info(separator)
    
    def progress(self, current: int, total: int, message: str = "") -> None:
        """Log progress message."""
        percentage = (current / total) * 100
        self.logger.info(f"📊 Progress: {current}/{total} ({percentage:.1f}%) {message}")
    
    def log_exception(self, exc: Exception, **extra) -> None:
        """Log exception with structured format."""
        context = {"exception_type": type(exc).__name__}
        
        if isinstance(exc, SPYAnalysisError):
            context.update(exc.to_dict())
        else:
            context["message"] = str(exc)
        
        context.update(extra)
        self.logger.error(f"💥 Exception occurred: {exc}", extra=context)
    
    def log_model_performance(
        self,
        model_name: str,
        metrics: Dict[str, float],
        **extra
    ) -> None:
        """Log model performance metrics."""
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(
            f"🏆 Model '{model_name}' performance: {metric_str}",
            extra={
                "metric_type": "model_performance",
                "model_name": model_name,
                "metrics": metrics,
                **extra
            }
        )
    
    def log_data_quality(
        self,
        dataset_name: str,
        quality_metrics: Dict[str, Any],
        **extra
    ) -> None:
        """Log data quality metrics."""
        self.logger.info(
            f"📊 Data quality for '{dataset_name}': {quality_metrics}",
            extra={
                "metric_type": "data_quality",
                "dataset": dataset_name,
                "quality_metrics": quality_metrics,
                **extra
            }
        )
    
    def log_api_call(
        self,
        endpoint: str,
        status_code: int,
        response_time: float,
        **extra
    ) -> None:
        """Log API call metrics."""
        status_emoji = "✅" if 200 <= status_code < 300 else "❌"
        self.logger.info(
            f"{status_emoji} API {endpoint} - Status: {status_code}, Time: {response_time:.3f}s",
            extra={
                "metric_type": "api_call",
                "endpoint": endpoint,
                "status_code": status_code,
                "response_time": response_time,
                **extra
            }
        )


def get_logger(
    name: Optional[str] = None,
    **kwargs
) -> SPYAnalysisLogger:
    """Get logger instance with optional configuration."""
    if name is None:
        name = __name__.split('.')[0]  # Use package name
    
    return SPYAnalysisLogger(name=name, **kwargs)


# Global logger instance - maintain compatibility
logger = SPYAnalysisLogger("SPY_Analysis")

# Legacy compatibility
StockPredictionLogger = SPYAnalysisLogger


def configure_logging(
    level: Union[str, int] = logging.INFO,
    log_dir: Optional[Path] = None,
    **kwargs
) -> SPYAnalysisLogger:
    """Configure global logging settings."""
    global logger
    logger = SPYAnalysisLogger(
        name="spy_analysis",
        level=level,
        log_dir=log_dir,
        **kwargs
    )
    return logger