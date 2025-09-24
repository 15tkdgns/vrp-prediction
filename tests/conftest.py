"""Pytest configuration and fixtures."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_stock_data():
    """Create deterministic sample stock data for testing."""
    dates = pd.date_range('2020-01-01', periods=100)
    
    # Create deterministic stock price data (linear trend + sine wave)
    base_price = 100
    prices = []
    
    for i, _ in enumerate(dates):
        # Deterministic price: linear trend + sine wave pattern
        trend = base_price + (i * 0.5)  # Slow upward trend
        seasonal = 5 * np.sin(i * 0.1)  # Seasonal pattern
        price = trend + seasonal
        prices.append(price)
    
    # Deterministic OHLV data based on Close prices
    data = pd.DataFrame({
        'Date': dates,
        'Open': [p * 0.995 for p in prices],   # Open slightly below Close
        'High': [p * 1.02 for p in prices],    # High 2% above Close
        'Low': [p * 0.98 for p in prices],     # Low 2% below Close  
        'Close': prices,
        'Volume': [1000000 + (i * 10000) for i in range(100)]  # Increasing volume
    })
    
    data.set_index('Date', inplace=True)
    return data


@pytest.fixture
def sample_features():
    """Create deterministic sample feature data for testing."""
    n_samples = 100
    
    # Deterministic features with known patterns
    return pd.DataFrame({
        'feature_1': [np.sin(i * 0.1) for i in range(n_samples)],           # Sine wave [-1, 1]
        'feature_2': [5 + np.cos(i * 0.05) * 2 for i in range(n_samples)], # Cosine wave [3, 7]
        'feature_3': [2 * (i / n_samples) for i in range(n_samples)],       # Linear growth [0, 2]
        'target': [1 if i % 5 == 0 else 0 for i in range(n_samples)]        # 20% positive class
    })


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    class MockConfig:
        def __init__(self):
            self.data = type('', (), {
                'symbol': 'SPY',
                'period': '1y',
                'interval': '1d'
            })()
            
            self.model = type('', (), {
                'test_size': 0.2,
                'random_state': 42,
                'cv_folds': 5
            })()
            
            self.logging = type('', (), {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            })()
    
    return MockConfig()


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment variables."""
    os.environ['TESTING'] = '1'
    os.environ['LOG_LEVEL'] = 'ERROR'  # Reduce log noise in tests
    yield
    os.environ.pop('TESTING', None)
    os.environ.pop('LOG_LEVEL', None)