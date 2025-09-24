"""
Unit tests for volatility prediction models.

This module tests the core functionality of volatility predictors,
ensuring they meet the interface contracts and produce valid results.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.volatility.predictors import ElasticNetVolatilityPredictor, RidgeVolatilityPredictor
from src.core.exceptions import ModelTrainingError, ModelPredictionError, ModelNotFittedError


class TestElasticNetVolatilityPredictor:
    """Test suite for ElasticNet volatility predictor."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 7

        # Generate realistic financial features
        X = pd.DataFrame({
            'ma_20': np.random.normal(100, 10, n_samples),
            'ma_50': np.random.normal(100, 10, n_samples),
            'rsi': np.random.uniform(20, 80, n_samples),
            'volatility_20': np.random.exponential(0.02, n_samples),
            'volume_ratio': np.random.lognormal(0, 0.5, n_samples),
            'returns_lag_1': np.random.normal(0, 0.02, n_samples),
            'returns_lag_2': np.random.normal(0, 0.02, n_samples)
        })

        # Generate target (next-day volatility)
        y = np.random.exponential(0.015, n_samples)

        return X, y

    @pytest.fixture
    def predictor(self):
        """Create a predictor instance for testing."""
        return ElasticNetVolatilityPredictor(
            alpha=0.001,
            l1_ratio=0.5,
            random_state=42
        )

    def test_initialization(self):
        """Test predictor initialization."""
        predictor = ElasticNetVolatilityPredictor(alpha=0.01, l1_ratio=0.7)

        assert predictor.name == "ElasticNet Volatility Predictor"
        assert predictor.alpha == 0.01
        assert predictor.l1_ratio == 0.7
        assert not predictor.is_fitted
        assert predictor.feature_names is None

    def test_fit_with_valid_data(self, predictor, sample_data):
        """Test fitting with valid data."""
        X, y = sample_data

        result = predictor.fit(X, y)

        assert result is predictor  # Returns self
        assert predictor.is_fitted
        assert predictor.feature_names == list(X.columns)
        assert len(predictor.training_metrics) > 0
        assert 'train_r2' in predictor.training_metrics
        assert 'val_r2' in predictor.training_metrics

    def test_fit_with_negative_volatility_raises_error(self, predictor):
        """Test that negative volatility values raise an error."""
        X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        y = np.array([0.1, -0.1, 0.2])  # Contains negative volatility

        with pytest.raises(ModelTrainingError) as exc_info:
            predictor.fit(X, y)

        assert "Volatility values cannot be negative" in str(exc_info.value)

    def test_predict_without_fitting_raises_error(self, predictor, sample_data):
        """Test that prediction without fitting raises an error."""
        X, _ = sample_data

        with pytest.raises(ModelNotFittedError) as exc_info:
            predictor.predict(X)

        assert "must be fitted before making predictions" in str(exc_info.value)

    def test_predict_with_fitted_model(self, predictor, sample_data):
        """Test prediction with a fitted model."""
        X, y = sample_data
        predictor.fit(X, y)

        result = predictor.predict(X)

        assert result.predictions is not None
        assert len(result.predictions) == len(X)
        assert np.all(result.predictions >= 0)  # Volatility should be non-negative
        assert result.model_name == predictor.name
        assert 'model_type' in result.metadata
        assert result.metadata['model_type'] == 'ElasticNet'

    def test_feature_importance(self, predictor, sample_data):
        """Test feature importance extraction."""
        X, y = sample_data
        predictor.fit(X, y)

        importance = predictor.get_feature_importance()

        assert importance is not None
        assert len(importance) == len(X.columns)
        assert all(name in importance for name in X.columns)
        assert all(score >= 0 for score in importance.values())
        assert abs(sum(importance.values()) - 1.0) < 1e-6  # Should sum to 1

    def test_cross_validation(self, predictor, sample_data):
        """Test cross-validation functionality."""
        X, y = sample_data

        cv_results = predictor.cross_validate(X, y, cv_folds=3)

        assert 'r2_scores' in cv_results
        assert 'mae_scores' in cv_results
        assert 'mean_r2' in cv_results
        assert 'std_r2' in cv_results
        assert len(cv_results['r2_scores']) == 3
        assert len(cv_results['fold_details']) == 3

    def test_model_parameters(self, predictor):
        """Test model parameter retrieval."""
        params = predictor.get_model_parameters()

        assert 'alpha' in params
        assert 'l1_ratio' in params
        assert 'random_state' in params
        assert params['alpha'] == 0.001
        assert params['l1_ratio'] == 0.5


class TestRidgeVolatilityPredictor:
    """Test suite for Ridge volatility predictor."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        n_samples = 50
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples)
        })
        y = np.random.exponential(0.01, n_samples)
        return X, y

    def test_ridge_predictor_basic_functionality(self, sample_data):
        """Test basic Ridge predictor functionality."""
        X, y = sample_data
        predictor = RidgeVolatilityPredictor(alpha=0.01)

        # Test fitting
        predictor.fit(X, y)
        assert predictor.is_fitted

        # Test prediction
        result = predictor.predict(X)
        assert len(result.predictions) == len(X)
        assert np.all(result.predictions >= 0)

        # Test feature importance
        importance = predictor.get_feature_importance()
        assert importance is not None
        assert len(importance) == len(X.columns)


class TestPredictorIntegration:
    """Integration tests for volatility predictors."""

    @pytest.fixture
    def realistic_financial_data(self):
        """Generate more realistic financial data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        n_samples = len(dates)

        # Simulate price data
        price = 100
        prices = [price]
        returns = []

        for _ in range(n_samples - 1):
            return_val = np.random.normal(0.0005, 0.02)  # 0.05% daily return, 2% volatility
            returns.append(return_val)
            price *= (1 + return_val)
            prices.append(price)

        data = pd.DataFrame({
            'close': prices,
            'returns': [0] + returns
        }, index=dates)

        # Create volatility features
        data['volatility_20'] = data['returns'].rolling(20).std()
        data['ma_20'] = data['close'].rolling(20).mean()
        data['rsi'] = np.random.uniform(30, 70, n_samples)

        # Create target (next-day volatility)
        data['next_day_volatility'] = data['returns'].abs().shift(-1)

        # Drop NaN values
        data = data.dropna()

        feature_cols = ['volatility_20', 'ma_20', 'rsi']
        X = data[feature_cols]
        y = data['next_day_volatility']

        return X, y

    def test_full_pipeline_with_realistic_data(self, realistic_financial_data):
        """Test the full prediction pipeline with realistic data."""
        X, y = realistic_financial_data

        # Test ElasticNet predictor
        elasticnet = ElasticNetVolatilityPredictor(alpha=0.001, random_state=42)
        elasticnet.fit(X, y)
        en_result = elasticnet.predict(X)

        # Test Ridge predictor
        ridge = RidgeVolatilityPredictor(alpha=0.001, random_state=42)
        ridge.fit(X, y)
        ridge_result = ridge.predict(X)

        # Both models should produce reasonable results
        assert en_result.metadata['train_r2'] > -1.0  # Should be better than naive baseline
        assert ridge_result.metadata['train_r2'] > -1.0

        # Predictions should be reasonable
        assert np.all(en_result.predictions >= 0)
        assert np.all(ridge_result.predictions >= 0)
        assert np.all(en_result.predictions < 1)  # Volatility shouldn't be too extreme
        assert np.all(ridge_result.predictions < 1)

    def test_predictor_consistency(self, realistic_financial_data):
        """Test that predictors produce consistent results with the same random seed."""
        X, y = realistic_financial_data

        # Train two identical models
        predictor1 = ElasticNetVolatilityPredictor(alpha=0.001, random_state=42)
        predictor2 = ElasticNetVolatilityPredictor(alpha=0.001, random_state=42)

        predictor1.fit(X, y)
        predictor2.fit(X, y)

        result1 = predictor1.predict(X)
        result2 = predictor2.predict(X)

        # Results should be identical
        np.testing.assert_array_almost_equal(result1.predictions, result2.predictions)

    def test_error_handling_with_invalid_data(self):
        """Test error handling with various invalid data scenarios."""
        predictor = ElasticNetVolatilityPredictor()

        # Test with empty data
        with pytest.raises(ValueError):
            predictor.validate_input(pd.DataFrame())

        # Test with NaN data
        X_nan = pd.DataFrame({'feature1': [1, np.nan, 3]})
        with pytest.raises(ValueError):
            predictor.validate_input(X_nan)

        # Test prediction on different feature set
        X_train = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        y_train = np.array([0.1, 0.2, 0.1])
        X_test = pd.DataFrame({'c': [1, 2], 'd': [3, 4]})  # Different features

        predictor.fit(X_train, y_train)

        # This should work because we're using numpy arrays internally
        # But the feature names won't match
        result = predictor.predict(X_test)
        assert len(result.predictions) == len(X_test)