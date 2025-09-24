"""
FastAPI dependencies for dependency injection and middleware.
"""
import time
from typing import Dict
from fastapi import HTTPException, Depends, Request
from functools import wraps

from ..models.factory import ModelFactory
from ..data.loader import StockDataLoader
from ..core.logger import get_logger

logger = get_logger("api.dependencies")

# Rate limiting storage (in production, use Redis)
_rate_limit_storage: Dict[str, Dict] = {}

# Singleton instances
_model_factory = None
_data_loader = None


def get_model_factory() -> ModelFactory:
    """Get ModelFactory instance (singleton)."""
    global _model_factory
    if _model_factory is None:
        _model_factory = ModelFactory()
    return _model_factory


def get_data_loader() -> StockDataLoader:
    """Get StockDataLoader instance (singleton)."""
    global _data_loader
    if _data_loader is None:
        _data_loader = StockDataLoader()
    return _data_loader


def rate_limit(
    max_requests: int = 60,
    window_seconds: int = 60
):
    """
    Rate limiting dependency.
    
    Args:
        max_requests: Maximum requests allowed in the time window
        window_seconds: Time window in seconds
    """
    async def _rate_limit(request: Request):
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        current_time = time.time()
        
        # Clean up old entries
        if client_ip in _rate_limit_storage:
            _rate_limit_storage[client_ip]["requests"] = [
                req_time for req_time in _rate_limit_storage[client_ip]["requests"]
                if current_time - req_time < window_seconds
            ]
        else:
            _rate_limit_storage[client_ip] = {"requests": []}
        
        # Check if limit exceeded
        if len(_rate_limit_storage[client_ip]["requests"]) >= max_requests:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Maximum {max_requests} requests per {window_seconds} seconds."
            )
        
        # Add current request
        _rate_limit_storage[client_ip]["requests"].append(current_time)
        
        return True
    
    return _rate_limit


def require_api_key(request: Request):
    """
    API key authentication dependency.
    In production, this should check against a proper key store.
    """
    api_key = request.headers.get("X-API-Key")
    
    # For demo purposes, accept any key or no key
    # In production, implement proper key validation
    if api_key is None:
        logger.info("Request without API key - allowing for demo")
    
    return api_key


def log_request_duration():
    """Dependency to log request duration."""
    start_time = time.time()
    
    def cleanup():
        duration = time.time() - start_time
        if duration > 1.0:  # Log slow requests
            logger.warning(f"Slow request detected: {duration:.2f}s")
    
    return cleanup


class RateLimiter:
    """Class-based rate limiter for more complex scenarios."""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.storage = {}
    
    async def __call__(self, request: Request):
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Initialize storage for new IPs
        if client_ip not in self.storage:
            self.storage[client_ip] = []
        
        # Clean old requests
        self.storage[client_ip] = [
            req_time for req_time in self.storage[client_ip]
            if current_time - req_time < self.window_seconds
        ]
        
        # Check limit
        if len(self.storage[client_ip]) >= self.max_requests:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {self.max_requests} requests per {self.window_seconds}s"
            )
        
        # Add current request
        self.storage[client_ip].append(current_time)
        return True


# Pre-configured rate limiters
rate_limit_strict = RateLimiter(max_requests=10, window_seconds=60)
rate_limit_moderate = RateLimiter(max_requests=30, window_seconds=60)
rate_limit_lenient = RateLimiter(max_requests=100, window_seconds=60)