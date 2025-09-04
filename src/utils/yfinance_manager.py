#!/usr/bin/env python3
"""
Unified YFinance Manager
- Centralized YFinance API calls with caching
- Rate limiting and intelligent retry
- Transparent error handling (no mock data)
- Performance monitoring
"""

import yfinance as yf
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from threading import Lock
import json
from pathlib import Path

class YFinanceManager:
    """
    Centralized YFinance data manager with caching, rate limiting, and error transparency
    """
    
    def __init__(self, cache_dir: str = "cache/yfinance", max_requests_per_minute: int = 60):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting
        self.max_requests_per_minute = max_requests_per_minute
        self.request_times = []
        self.rate_limit_lock = Lock()
        
        # Cache settings
        self.cache = {}
        self.cache_timestamps = {}
        self.default_cache_ttl = 300  # 5 minutes
        
        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "api_failures": 0,
            "rate_limit_waits": 0
        }
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 YFinanceManager initialized with transparent error handling")
    
    def _check_rate_limit(self) -> None:
        """Enforce rate limiting with intelligent waiting"""
        with self.rate_limit_lock:
            current_time = time.time()
            
            # Remove requests older than 1 minute
            self.request_times = [t for t in self.request_times if current_time - t < 60]
            
            # Check if we need to wait
            if len(self.request_times) >= self.max_requests_per_minute:
                wait_time = 60 - (current_time - self.request_times[0])
                if wait_time > 0:
                    self.stats["rate_limit_waits"] += 1
                    self.logger.warning(f"⏱️ Rate limit reached. Waiting {wait_time:.1f} seconds")
                    time.sleep(wait_time)
            
            # Record this request
            self.request_times.append(current_time)
    
    def _get_cache_key(self, symbol: str, data_type: str, **kwargs) -> str:
        """Generate cache key for given parameters"""
        params_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return f"{symbol}_{data_type}_{params_str}"
    
    def _is_cache_valid(self, cache_key: str, ttl: int = None) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache or cache_key not in self.cache_timestamps:
            return False
        
        ttl = ttl or self.default_cache_ttl
        age = (datetime.now() - self.cache_timestamps[cache_key]).total_seconds()
        return age < ttl
    
    def _update_cache(self, cache_key: str, data: Any) -> None:
        """Update cache with new data"""
        self.cache[cache_key] = data
        self.cache_timestamps[cache_key] = datetime.now()
        
        # Optionally persist to disk for important data
        if data is not None:
            try:
                cache_file = self.cache_dir / f"{cache_key}.json"
                if hasattr(data, 'to_dict'):
                    # DataFrame or similar
                    cache_file.write_text(json.dumps(data.to_dict(), default=str))
                elif isinstance(data, dict):
                    cache_file.write_text(json.dumps(data, default=str))
            except Exception as e:
                self.logger.debug(f"Failed to persist cache for {cache_key}: {e}")
    
    def _create_error_response(self, operation: str, symbol: str, error: Exception) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "success": False,
            "error": True,
            "error_type": "yfinance_api_failure", 
            "operation": operation,
            "symbol": symbol,
            "message": f"YFinance API 호출 실패 ({operation}): {str(error)}",
            "error_details": {
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "timestamp": datetime.now().isoformat(),
                "retry_recommended": True,
                "retry_after_seconds": 30
            }
        }
    
    def get_stock_history(self, symbol: str, period: str = "1y", interval: str = "1d", 
                         cache_ttl: int = 300) -> Dict[str, Any]:
        """
        Get historical stock data with caching and transparent error handling
        
        Returns:
            Dict containing either success data or error information
        """
        cache_key = self._get_cache_key(symbol, "history", period=period, interval=interval)
        
        # Check cache first
        if self._is_cache_valid(cache_key, cache_ttl):
            self.stats["cache_hits"] += 1
            self.logger.debug(f"📊 Cache hit for {symbol} history")
            return {
                "success": True,
                "data": self.cache[cache_key],
                "source": "cache",
                "symbol": symbol
            }
        
        # Rate limit check
        self._check_rate_limit()
        self.stats["total_requests"] += 1
        
        try:
            self.logger.info(f"📈 Fetching {symbol} history (period={period}, interval={interval})")
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                error_msg = f"No historical data available for {symbol}"
                self.logger.warning(error_msg)
                return {
                    "success": False,
                    "error": True,
                    "error_type": "no_data_available",
                    "symbol": symbol,
                    "message": error_msg,
                    "operation": "get_history"
                }
            
            # Clean and validate data
            hist = hist.dropna()
            if hist.empty:
                error_msg = f"All historical data for {symbol} contains invalid values"
                self.logger.warning(error_msg)
                return {
                    "success": False,
                    "error": True,
                    "error_type": "invalid_data",
                    "symbol": symbol,
                    "message": error_msg,
                    "operation": "get_history"
                }
            
            # Reset index to make Date a column
            hist_dict = hist.reset_index().to_dict('records')
            
            # Update cache
            self._update_cache(cache_key, hist_dict)
            
            self.logger.info(f"✅ Successfully fetched {len(hist_dict)} records for {symbol}")
            
            return {
                "success": True,
                "data": hist_dict,
                "source": "yfinance_api",
                "symbol": symbol,
                "records_count": len(hist_dict),
                "period": period,
                "interval": interval,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.stats["api_failures"] += 1
            error_response = self._create_error_response("get_history", symbol, e)
            self.logger.error(f"❌ Failed to fetch history for {symbol}: {e}")
            return error_response
    
    def get_stock_info(self, symbol: str, cache_ttl: int = 600) -> Dict[str, Any]:
        """
        Get stock info/metadata with caching and transparent error handling
        """
        cache_key = self._get_cache_key(symbol, "info")
        
        # Check cache
        if self._is_cache_valid(cache_key, cache_ttl):
            self.stats["cache_hits"] += 1
            return {
                "success": True,
                "data": self.cache[cache_key],
                "source": "cache",
                "symbol": symbol
            }
        
        # Rate limit check  
        self._check_rate_limit()
        self.stats["total_requests"] += 1
        
        try:
            self.logger.info(f"ℹ️ Fetching {symbol} info")
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info or len(info) == 0:
                error_msg = f"No information available for {symbol}"
                self.logger.warning(error_msg)
                return {
                    "success": False,
                    "error": True,
                    "error_type": "no_info_available",
                    "symbol": symbol,
                    "message": error_msg,
                    "operation": "get_info"
                }
            
            # Extract key information
            cleaned_info = {
                "symbol": symbol,
                "shortName": info.get("shortName", "Unknown"),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "currentPrice": info.get("currentPrice", info.get("regularMarketPrice")),
                "marketCap": info.get("marketCap"),
                "volume": info.get("volume"),
                "averageVolume": info.get("averageVolume"),
                "peRatio": info.get("trailingPE"),
                "dividendYield": info.get("dividendYield"),
                "currency": info.get("currency", "USD"),
                "exchange": info.get("exchange", "Unknown")
            }
            
            # Update cache
            self._update_cache(cache_key, cleaned_info)
            
            self.logger.info(f"✅ Successfully fetched info for {symbol}")
            
            return {
                "success": True,
                "data": cleaned_info,
                "source": "yfinance_api",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.stats["api_failures"] += 1
            error_response = self._create_error_response("get_info", symbol, e)
            self.logger.error(f"❌ Failed to fetch info for {symbol}: {e}")
            return error_response
    
    def get_multiple_stocks(self, symbols: List[str], operation: str = "info", **kwargs) -> Dict[str, Any]:
        """
        Get data for multiple stocks efficiently
        """
        results = {}
        
        self.logger.info(f"📊 Fetching {operation} for {len(symbols)} symbols")
        
        for symbol in symbols:
            if operation == "history":
                result = self.get_stock_history(symbol, **kwargs)
            elif operation == "info":
                result = self.get_stock_info(symbol, **kwargs)
            else:
                result = {
                    "success": False,
                    "error": True,
                    "message": f"Unknown operation: {operation}"
                }
            
            results[symbol] = result
        
        # Summary statistics
        successful = sum(1 for r in results.values() if r.get("success"))
        failed = len(results) - successful
        
        return {
            "success": failed == 0,
            "results": results,
            "summary": {
                "total_symbols": len(symbols),
                "successful": successful,
                "failed": failed,
                "operation": operation
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        cache_hit_rate = (self.stats["cache_hits"] / max(1, self.stats["total_requests"])) * 100
        
        return {
            "performance": {
                "total_api_requests": self.stats["total_requests"],
                "cache_hits": self.stats["cache_hits"],
                "cache_hit_rate_percent": round(cache_hit_rate, 2),
                "api_failures": self.stats["api_failures"],
                "failure_rate_percent": round((self.stats["api_failures"] / max(1, self.stats["total_requests"])) * 100, 2),
                "rate_limit_waits": self.stats["rate_limit_waits"]
            },
            "cache": {
                "cached_entries": len(self.cache),
                "cache_directory": str(self.cache_dir)
            },
            "configuration": {
                "max_requests_per_minute": self.max_requests_per_minute,
                "default_cache_ttl_seconds": self.default_cache_ttl
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def clear_cache(self, symbol: str = None) -> Dict[str, Any]:
        """Clear cache for specific symbol or all cache"""
        if symbol:
            # Clear cache for specific symbol
            keys_to_remove = [key for key in self.cache.keys() if key.startswith(f"{symbol}_")]
            for key in keys_to_remove:
                self.cache.pop(key, None)
                self.cache_timestamps.pop(key, None)
            
            return {
                "success": True,
                "message": f"Cache cleared for {symbol}",
                "cleared_entries": len(keys_to_remove)
            }
        else:
            # Clear all cache
            entries_count = len(self.cache)
            self.cache.clear()
            self.cache_timestamps.clear()
            
            return {
                "success": True,
                "message": "All cache cleared",
                "cleared_entries": entries_count
            }

# Global instance for easy access
_yfinance_manager = None

def get_yfinance_manager() -> YFinanceManager:
    """Get global YFinanceManager instance"""
    global _yfinance_manager
    if _yfinance_manager is None:
        _yfinance_manager = YFinanceManager()
    return _yfinance_manager

# Convenience functions
def get_stock_data(symbol: str, period: str = "1y") -> Dict[str, Any]:
    """Convenience function for getting stock history"""
    return get_yfinance_manager().get_stock_history(symbol, period)

def get_stock_info(symbol: str) -> Dict[str, Any]:
    """Convenience function for getting stock info"""
    return get_yfinance_manager().get_stock_info(symbol)

def get_multiple_stocks_data(symbols: List[str], operation: str = "info") -> Dict[str, Any]:
    """Convenience function for getting multiple stocks data"""
    return get_yfinance_manager().get_multiple_stocks(symbols, operation)

if __name__ == "__main__":
    # Test the manager
    manager = YFinanceManager()
    
    # Test single stock
    print("Testing single stock (AAPL)...")
    result = manager.get_stock_history("AAPL", period="5d")
    print(f"Success: {result['success']}")
    
    if result['success']:
        print(f"Records: {result['records_count']}")
    else:
        print(f"Error: {result['message']}")
    
    # Test multiple stocks
    print("\nTesting multiple stocks...")
    symbols = ["AAPL", "GOOGL", "MSFT"]
    results = manager.get_multiple_stocks(symbols, "info")
    print(f"Total: {results['summary']['total_symbols']}, Success: {results['summary']['successful']}")
    
    # Show performance stats
    print("\nPerformance stats:")
    stats = manager.get_performance_stats()
    print(f"Cache hit rate: {stats['performance']['cache_hit_rate_percent']}%")