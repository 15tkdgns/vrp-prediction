"""
Performance benchmarks for SPY Analysis system.
"""
import pytest
import time
import pandas as pd
import numpy as np
from memory_profiler import profile

from src.data.loader import StockDataLoader
from src.features.engineering import FeatureEngineering
from src.models.factory import ModelFactory
from src.monitoring.metrics import monitor_performance


class TestDataLoaderPerformance:
    """Benchmark data loading operations."""
    
    @pytest.mark.benchmark(group="data_loading")
    def test_load_data_1year(self, benchmark, sample_stock_data):
        """Benchmark loading 1 year of data."""
        loader = StockDataLoader()
        
        # Mock the actual data loading with sample data
        def load_data():
            return sample_stock_data
        
        result = benchmark(load_data)
        assert not result.empty
    
    @pytest.mark.benchmark(group="data_loading", min_rounds=5)
    def test_data_processing_pipeline(self, benchmark, sample_stock_data):
        """Benchmark complete data processing pipeline."""
        loader = StockDataLoader()
        
        def process_pipeline():
            # Simulate data loading and basic processing
            data = sample_stock_data.copy()
            
            # Add technical indicators
            data['SMA_10'] = data['Close'].rolling(10).mean()
            data['SMA_20'] = data['Close'].rolling(20).mean()
            data['RSI'] = calculate_rsi(data['Close'])
            
            return data
        
        result = benchmark(process_pipeline)
        assert 'SMA_10' in result.columns


class TestFeatureEngineeringPerformance:
    """Benchmark feature engineering operations."""
    
    @pytest.mark.benchmark(group="feature_engineering")
    def test_create_features(self, benchmark, sample_stock_data):
        """Benchmark feature creation."""
        feature_eng = FeatureEngineering()
        
        result = benchmark(feature_eng.create_features, sample_stock_data)
        assert result.shape[1] > sample_stock_data.shape[1]
    
    @pytest.mark.benchmark(group="feature_engineering")
    def test_technical_indicators(self, benchmark, sample_stock_data):
        """Benchmark technical indicators calculation."""
        feature_eng = FeatureEngineering()
        
        def calculate_indicators():
            return feature_eng._create_technical_indicators(sample_stock_data)
        
        result = benchmark(calculate_indicators)
        assert not result.empty
    
    @pytest.mark.benchmark(group="feature_engineering", min_rounds=3)
    def test_large_dataset_features(self, benchmark):
        """Benchmark feature creation on large dataset."""
        # Create larger deterministic dataset
        dates = pd.date_range('2010-01-01', periods=2500)  # ~7 years
        
        # Deterministic price generation
        base_price = 100
        prices = []
        for i in range(len(dates)):
            # Deterministic price: trend + seasonal + cyclical
            trend = base_price + (i * 0.02)  # Long-term trend
            seasonal = 10 * np.sin(i * 2 * np.pi / 252)  # Annual seasonality
            cyclical = 5 * np.sin(i * 2 * np.pi / 1260)  # 5-year cycle
            price = trend + seasonal + cyclical
            prices.append(price)
        
        large_data = pd.DataFrame({
            'Date': dates,
            'Open': [p * 0.998 for p in prices],  # Deterministic Open
            'High': [p * 1.015 for p in prices],  # Deterministic High
            'Low': [p * 0.985 for p in prices],   # Deterministic Low
            'Close': prices,
            'Volume': [5000000 + (i * 1000) for i in range(len(dates))]  # Linear volume growth
        })
        large_data.set_index('Date', inplace=True)
        
        feature_eng = FeatureEngineering()
        result = benchmark(feature_eng.create_features, large_data)
        assert result.shape[0] == large_data.shape[0]


class TestModelPerformance:
    """Benchmark model operations."""
    
    @pytest.mark.benchmark(group="model_training")
    def test_random_forest_training(self, benchmark, sample_features):
        """Benchmark RandomForest training."""
        factory = ModelFactory()
        model = factory.create_model('RandomForest')
        
        X = sample_features.drop('target', axis=1)
        y = sample_features['target']
        
        def train_model():
            return model.fit(X, y)
        
        trained_model = benchmark(train_model)
        assert hasattr(trained_model, 'feature_importances_')
    
    @pytest.mark.benchmark(group="model_prediction")
    def test_model_prediction_speed(self, benchmark, sample_features):
        """Benchmark model prediction speed."""
        factory = ModelFactory()
        model = factory.create_model('RandomForest')
        
        X = sample_features.drop('target', axis=1)
        y = sample_features['target']
        
        # Train first
        model.fit(X, y)
        
        # Benchmark prediction
        result = benchmark(model.predict, X)
        assert len(result) == len(y)
    
    @pytest.mark.benchmark(group="model_prediction")
    def test_batch_predictions(self, benchmark, sample_features):
        """Benchmark batch predictions."""
        factory = ModelFactory()
        model = factory.create_model('XGBoost')
        
        X = sample_features.drop('target', axis=1)
        y = sample_features['target']
        
        # Train model
        model.fit(X, y)
        
        # Create larger prediction set
        large_X = pd.concat([X] * 10, ignore_index=True)
        
        def batch_predict():
            return model.predict(large_X)
        
        result = benchmark(batch_predict)
        assert len(result) == len(large_X)


class TestMemoryUsage:
    """Memory usage benchmarks."""
    
    @pytest.mark.slow
    def test_memory_usage_data_loading(self, sample_stock_data):
        """Test memory usage during data loading."""
        import tracemalloc
        
        tracemalloc.start()
        
        # Simulate loading multiple symbols
        data_cache = {}
        symbols = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI']
        
        for symbol in symbols:
            # Use sample data for each symbol
            data_cache[symbol] = sample_stock_data.copy()
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
        print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
        
        # Assert reasonable memory usage (< 100 MB for test data)
        assert peak < 100 * 1024 * 1024
    
    @pytest.mark.slow
    def test_memory_usage_feature_engineering(self, sample_stock_data):
        """Test memory usage during feature engineering."""
        import tracemalloc
        
        tracemalloc.start()
        
        feature_eng = FeatureEngineering()
        features = feature_eng.create_features(sample_stock_data)
        
        # Create features for multiple datasets
        feature_cache = {}
        for i in range(5):
            feature_cache[f'dataset_{i}'] = feature_eng.create_features(
                sample_stock_data.copy()
            )
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"Feature engineering memory - Current: {current / 1024 / 1024:.1f} MB")
        print(f"Feature engineering memory - Peak: {peak / 1024 / 1024:.1f} MB")
        
        assert peak < 200 * 1024 * 1024  # < 200 MB


class TestConcurrencyPerformance:
    """Test concurrent operations performance."""
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark(group="concurrency")
    async def test_concurrent_data_loading(self, benchmark):
        """Test concurrent data loading performance."""
        import asyncio
        
        async def concurrent_load():
            tasks = []
            symbols = ['SPY', 'QQQ', 'IWM', 'DIA']
            
            for symbol in symbols:
                # Simulate async data loading
                task = asyncio.create_task(simulate_data_load(symbol))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return results
        
        result = benchmark(lambda: asyncio.run(concurrent_load()))
        assert len(result) == 4
    
    @pytest.mark.benchmark(group="concurrency")
    def test_parallel_model_training(self, benchmark, sample_features):
        """Test parallel model training."""
        from concurrent.futures import ThreadPoolExecutor
        
        def train_model(model_name):
            factory = ModelFactory()
            model = factory.create_model(model_name)
            
            X = sample_features.drop('target', axis=1)
            y = sample_features['target']
            
            return model.fit(X, y)
        
        def parallel_training():
            models = ['RandomForest', 'XGBoost', 'GradientBoosting']
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                results = list(executor.map(train_model, models))
            
            return results
        
        result = benchmark(parallel_training)
        assert len(result) == 3


# Helper functions
def calculate_rsi(prices, period=14):
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


async def simulate_data_load(symbol):
    """Simulate async data loading."""
    await asyncio.sleep(0.1)  # Simulate I/O
    return pd.DataFrame({'Close': np.random.randn(100)})


@pytest.fixture
def performance_config():
    """Configuration for performance tests."""
    return {
        "max_execution_time": 5.0,  # 5 seconds
        "max_memory_usage": 500 * 1024 * 1024,  # 500 MB
        "min_throughput": 100  # operations per second
    }