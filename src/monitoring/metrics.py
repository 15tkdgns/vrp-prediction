"""
Prometheus metrics collection for SPY Analysis system.
Provides performance monitoring and alerting capabilities.
"""
import time
from typing import Dict, Any, Optional
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
import psutil
import threading

from ..core.logger import get_logger

logger = get_logger("monitoring.metrics")


class MetricsCollector:
    """Centralized metrics collection system."""
    
    def __init__(self):
        self.start_time = time.time()
        
        # System metrics
        self.cpu_gauge = Gauge('system_cpu_percent', 'CPU usage percentage')
        self.memory_gauge = Gauge('system_memory_percent', 'Memory usage percentage')
        self.disk_gauge = Gauge('system_disk_percent', 'Disk usage percentage')
        
        # API metrics
        self.api_requests_total = Counter(
            'api_requests_total', 
            'Total API requests',
            ['endpoint', 'method', 'status_code']
        )
        self.api_request_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration',
            ['endpoint', 'method']
        )
        
        # Model metrics
        self.model_predictions_total = Counter(
            'model_predictions_total',
            'Total model predictions',
            ['model_name', 'symbol']
        )
        self.model_prediction_duration = Histogram(
            'model_prediction_duration_seconds',
            'Model prediction duration',
            ['model_name']
        )
        self.model_accuracy = Gauge(
            'model_accuracy',
            'Model accuracy score',
            ['model_name']
        )
        
        # Data metrics
        self.data_points_processed = Counter(
            'data_points_processed_total',
            'Total data points processed',
            ['symbol', 'operation']
        )
        self.data_freshness = Gauge(
            'data_freshness_seconds',
            'Age of latest data point',
            ['symbol']
        )
        
        # Application metrics
        self.app_uptime = Gauge('app_uptime_seconds', 'Application uptime')
        self.active_connections = Gauge('active_connections', 'Active connections')
        self.errors_total = Counter(
            'errors_total',
            'Total errors',
            ['error_type', 'component']
        )
        
        # Start background metrics collection
        self._start_background_collection()
    
    def _start_background_collection(self):
        """Start background thread for system metrics collection."""
        def collect_system_metrics():
            while True:
                try:
                    # System metrics
                    self.cpu_gauge.set(psutil.cpu_percent())
                    self.memory_gauge.set(psutil.virtual_memory().percent)
                    self.disk_gauge.set(psutil.disk_usage('/').percent)
                    
                    # Uptime
                    self.app_uptime.set(time.time() - self.start_time)
                    
                    time.sleep(30)  # Collect every 30 seconds
                except Exception as e:
                    logger.error(f"Failed to collect system metrics: {e}")
                    time.sleep(60)  # Wait longer on error
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
        logger.info("✅ Background metrics collection started")
    
    def record_api_request(
        self, 
        endpoint: str, 
        method: str, 
        status_code: int, 
        duration: float
    ):
        """Record API request metrics."""
        self.api_requests_total.labels(
            endpoint=endpoint,
            method=method, 
            status_code=status_code
        ).inc()
        
        self.api_request_duration.labels(
            endpoint=endpoint,
            method=method
        ).observe(duration)
    
    def record_model_prediction(
        self, 
        model_name: str, 
        symbol: str, 
        duration: float,
        accuracy: Optional[float] = None
    ):
        """Record model prediction metrics."""
        self.model_predictions_total.labels(
            model_name=model_name,
            symbol=symbol
        ).inc()
        
        self.model_prediction_duration.labels(
            model_name=model_name
        ).observe(duration)
        
        if accuracy is not None:
            self.model_accuracy.labels(
                model_name=model_name
            ).set(accuracy)
    
    def record_data_processing(
        self, 
        symbol: str, 
        operation: str, 
        data_points: int,
        data_age_seconds: Optional[float] = None
    ):
        """Record data processing metrics."""
        self.data_points_processed.labels(
            symbol=symbol,
            operation=operation
        ).inc(data_points)
        
        if data_age_seconds is not None:
            self.data_freshness.labels(symbol=symbol).set(data_age_seconds)
    
    def record_error(self, error_type: str, component: str):
        """Record error occurrence."""
        self.errors_total.labels(
            error_type=error_type,
            component=component
        ).inc()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        return {
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "uptime_seconds": time.time() - self.start_time
            },
            "api": {
                "total_requests": self.api_requests_total._value.sum(),
                "active_connections": self.active_connections._value.get()
            },
            "models": {
                "total_predictions": self.model_predictions_total._value.sum()
            },
            "errors": {
                "total_errors": self.errors_total._value.sum()
            }
        }


# Global metrics collector instance
metrics = MetricsCollector()


def monitor_performance(component: str = "unknown"):
    """Decorator to monitor function performance."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.performance.log_execution_time(
                    f"{component}.{func.__name__}", 
                    duration
                )
                return result
            except Exception as e:
                metrics.record_error(type(e).__name__, component)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.performance.log_execution_time(
                    f"{component}.{func.__name__}", 
                    duration
                )
                return result
            except Exception as e:
                metrics.record_error(type(e).__name__, component)
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def start_metrics_server(port: int = 8001):
    """Start Prometheus metrics server."""
    try:
        start_http_server(port)
        logger.success(f"📊 Metrics server started on port {port}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")


class HealthChecker:
    """System health monitoring."""
    
    def __init__(self):
        self.checks = {}
    
    def register_check(self, name: str, check_func):
        """Register a health check."""
        self.checks[name] = check_func
    
    async def run_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all health checks."""
        results = {}
        
        for name, check_func in self.checks.items():
            try:
                start_time = time.time()
                result = await check_func() if asyncio.iscoroutinefunction(check_func) else check_func()
                duration = time.time() - start_time
                
                results[name] = {
                    "status": "healthy" if result else "unhealthy",
                    "duration": duration,
                    "details": result if isinstance(result, dict) else None
                }
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "error": str(e),
                    "duration": 0
                }
        
        return results


# Global health checker instance
health_checker = HealthChecker()


# Register default health checks
def check_system_resources():
    """Check system resources."""
    cpu = psutil.cpu_percent()
    memory = psutil.virtual_memory().percent
    disk = psutil.disk_usage('/').percent
    
    return {
        "cpu_ok": cpu < 90,
        "memory_ok": memory < 90,
        "disk_ok": disk < 90,
        "cpu_percent": cpu,
        "memory_percent": memory,
        "disk_percent": disk
    }


def check_data_freshness():
    """Check data freshness."""
    # This would check if data is recent enough
    # For now, return True
    return {"data_fresh": True, "last_update": time.time()}


health_checker.register_check("system_resources", check_system_resources)
health_checker.register_check("data_freshness", check_data_freshness)