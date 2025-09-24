#!/usr/bin/env python3
"""
통합 시스템 테스트 - 리팩토링된 모든 시스템의 통합 테스트
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from src.models.unified_ensemble import (
    UnifiedEnsembleSystem, EnsembleConfig, EnsembleFactory
)
from src.validation.unified_validation import (
    UnifiedCrossValidator, ValidationConfig, create_validator
)
from src.evaluation.unified_metrics import (
    UnifiedMetricsCalculator, MetricsConfig, create_metrics_calculator
)
from src.core.unified_config import UnifiedConfigManager, get_config


class TestUnifiedEnsemble(unittest.TestCase):
    """통합 앙상블 시스템 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.config = EnsembleConfig(use_gpu=False, stacking_cv=3)
        self.system = UnifiedEnsembleSystem(self.config)

        # 테스트 데이터 생성
        np.random.seed(42)
        self.X = np.random.randn(100, 10)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(int)

        # 간단한 모델들 생성
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        self.models = {
            'rf': RandomForestClassifier(n_estimators=10, random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=100)
        }

        # 모델들 훈련
        for model in self.models.values():
            model.fit(self.X, self.y)

    def test_ensemble_creation(self):
        """앙상블 생성 테스트"""
        # Voting 앙상블
        self.system.add_ensemble("voting", "voting")
        self.assertIn("voting", self.system.ensembles)

        # Stacking 앙상블
        self.system.add_ensemble("stacking", "stacking")
        self.assertIn("stacking", self.system.ensembles)

        # Neural Stacking 앙상블
        self.system.add_ensemble("neural", "neural_stacking")
        self.assertIn("neural", self.system.ensembles)

    def test_model_addition(self):
        """모델 추가 테스트"""
        self.system.add_ensemble("test_ensemble", "voting")
        self.system.add_models_to_ensemble("test_ensemble", self.models)

        ensemble = self.system.ensembles["test_ensemble"]
        self.assertEqual(len(ensemble.models), 2)
        self.assertIn('rf', ensemble.models)
        self.assertIn('lr', ensemble.models)

    def test_ensemble_training(self):
        """앙상블 훈련 테스트"""
        self.system.add_ensemble("test_ensemble", "voting")
        self.system.add_models_to_ensemble("test_ensemble", self.models)
        self.system.train_all_ensembles(self.X, self.y)

        ensemble = self.system.ensembles["test_ensemble"]
        self.assertTrue(ensemble.is_fitted)

    def test_ensemble_prediction(self):
        """앙상블 예측 테스트"""
        self.system.add_ensemble("test_ensemble", "voting")
        self.system.add_models_to_ensemble("test_ensemble", self.models)
        self.system.train_all_ensembles(self.X, self.y)

        predictions = self.system.predict(self.X[:10])
        self.assertEqual(len(predictions), 10)
        self.assertTrue(all(0 <= p <= 1 for p in predictions))

    def test_ensemble_evaluation(self):
        """앙상블 평가 테스트"""
        self.system.add_ensemble("test_ensemble", "voting")
        self.system.add_models_to_ensemble("test_ensemble", self.models)
        self.system.train_all_ensembles(self.X, self.y)

        # 테스트 데이터 생성
        X_test = np.random.randn(20, 10)
        y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)

        results = self.system.evaluate_ensembles(X_test, y_test)
        self.assertIn("test_ensemble", results)
        self.assertIn("accuracy", results["test_ensemble"])
        self.assertIn("f1_score", results["test_ensemble"])


class TestUnifiedValidation(unittest.TestCase):
    """통합 검증 시스템 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.config = ValidationConfig(
            method="time_series_split",
            n_splits=3,
            enable_leak_detection=True,
            strict_mode=False
        )
        self.validator = create_validator(self.config)

        # 테스트 데이터 생성
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(int)

        # DataFrame 버전 (데이터 누출 탐지용)
        self.X_df = pd.DataFrame(self.X, columns=[f'feature_{i}' for i in range(5)])

        # 간단한 모델
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)

    def test_data_leak_detection(self):
        """데이터 누출 탐지 테스트"""
        # 정상 데이터
        leak_results = self.validator.leak_detector.comprehensive_leak_detection(
            self.X_df, pd.Series(self.y)
        )

        self.assertIn('overall_leak_detected', leak_results)
        self.assertIn('recommendations', leak_results)

        # 누출이 있는 데이터 생성
        X_leak = self.X_df.copy()
        X_leak['target_leak'] = self.y  # 직접적인 타겟 누출

        leak_results_with_leak = self.validator.leak_detector.comprehensive_leak_detection(
            X_leak, pd.Series(self.y)
        )

        # 누출이 감지되어야 함
        self.assertTrue(leak_results_with_leak['overall_leak_detected'])

    def test_cross_validation(self):
        """교차 검증 테스트"""
        result = self.validator.validate(self.X, self.y, self.model, self.X_df)

        self.assertIsInstance(result.scores, dict)
        self.assertIn('accuracy', result.scores)
        self.assertIn('f1', result.scores)
        self.assertEqual(len(result.scores['accuracy']), self.config.n_splits)

    def test_statistical_validation(self):
        """통계적 검증 테스트"""
        scores = [0.7, 0.8, 0.75, 0.82, 0.78]
        stat_results = self.validator.stat_validator.test_statistical_significance(scores)

        self.assertIn('single_model_test', stat_results)
        self.assertIn('normality_test', stat_results)

    def test_overfitting_detection(self):
        """과적합 탐지 테스트"""
        train_scores = [0.95, 0.97, 0.94, 0.96, 0.98]  # 높은 훈련 성능
        val_scores = [0.75, 0.73, 0.76, 0.74, 0.77]    # 낮은 검증 성능

        overfitting_results = self.validator.stat_validator.test_overfitting(
            train_scores, val_scores
        )

        self.assertIn('overfitting_detected', overfitting_results)
        self.assertIn('overfitting_ratio', overfitting_results)
        self.assertTrue(overfitting_results['overfitting_detected'])  # 과적합이 감지되어야 함


class TestUnifiedMetrics(unittest.TestCase):
    """통합 성능 평가 시스템 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.config = MetricsConfig(
            classification_metrics=['accuracy', 'f1_score', 'precision', 'recall'],
            regression_metrics=['mse', 'mae', 'r2_score'],
            enable_visualization=False  # 테스트에서는 시각화 비활성화
        )
        self.calculator = create_metrics_calculator(self.config)

        # 테스트 데이터 생성
        np.random.seed(42)
        self.y_true_cls = np.random.randint(0, 2, 100)
        self.y_pred_cls = np.random.randint(0, 2, 100)
        self.y_proba_cls = np.random.rand(100)

        self.y_true_reg = np.random.randn(100)
        self.y_pred_reg = self.y_true_reg + np.random.randn(100) * 0.1

    def test_classification_metrics(self):
        """분류 지표 테스트"""
        result = self.calculator.evaluate_model(
            "test_classifier",
            self.y_true_cls,
            self.y_pred_cls,
            task_type="classification",
            y_proba=self.y_proba_cls
        )

        self.assertEqual(result.model_name, "test_classifier")
        self.assertEqual(result.task_type, "classification")
        self.assertIn('accuracy', result.metrics)
        self.assertIn('f1_score', result.metrics)
        self.assertIsInstance(result.confusion_matrix, np.ndarray)

    def test_regression_metrics(self):
        """회귀 지표 테스트"""
        result = self.calculator.evaluate_model(
            "test_regressor",
            self.y_true_reg,
            self.y_pred_reg,
            task_type="regression"
        )

        self.assertEqual(result.model_name, "test_regressor")
        self.assertEqual(result.task_type, "regression")
        self.assertIn('mse', result.metrics)
        self.assertIn('mae', result.metrics)
        self.assertIn('r2_score', result.metrics)

    def test_financial_metrics(self):
        """금융 지표 테스트"""
        returns = np.random.randn(100) * 0.01  # 1% 일일 수익률

        result = self.calculator.evaluate_model(
            "test_financial",
            self.y_true_reg,
            self.y_pred_reg,
            task_type="financial",
            returns=returns
        )

        self.assertEqual(result.task_type, "financial")
        self.assertIn('sharpe_ratio', result.metrics)
        self.assertIn('max_drawdown', result.metrics)

    def test_model_comparison(self):
        """모델 비교 테스트"""
        results = []

        # 여러 모델 결과 생성
        for i in range(3):
            result = self.calculator.evaluate_model(
                f"model_{i}",
                self.y_true_cls,
                self.y_pred_cls,
                task_type="classification"
            )
            results.append(result)

        comparison = self.calculator.compare_models(results, 'accuracy')

        self.assertIn('models', comparison)
        self.assertIn('rankings', comparison)
        self.assertIn('best_model', comparison)
        self.assertEqual(len(comparison['models']), 3)

    def test_comprehensive_report(self):
        """종합 보고서 테스트"""
        results = []

        # 분류 모델
        cls_result = self.calculator.evaluate_model(
            "classifier",
            self.y_true_cls,
            self.y_pred_cls,
            task_type="classification"
        )
        results.append(cls_result)

        # 회귀 모델
        reg_result = self.calculator.evaluate_model(
            "regressor",
            self.y_true_reg,
            self.y_pred_reg,
            task_type="regression"
        )
        results.append(reg_result)

        report = self.calculator.generate_comprehensive_report(results)

        self.assertIn('summary', report)
        self.assertIn('individual_results', report)
        self.assertIn('model_comparison', report)
        self.assertIn('recommendations', report)
        self.assertEqual(report['summary']['total_models'], 2)


class TestUnifiedConfig(unittest.TestCase):
    """통합 설정 시스템 테스트"""

    def setUp(self):
        """테스트 설정"""
        # 임시 설정 디렉토리 생성
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # 테스트용 설정 파일 생성
        default_config = {
            'app': {'name': 'Test App'},
            'data': {
                'symbol': 'TEST',
                'period': '1y',
                'directories': {
                    'raw': 'data/test/raw',
                    'models': 'data/test/models'
                }
            },
            'models': {
                'supported_types': ['test_model'],
                'default_config': {
                    'test_model': {'param1': 10, 'param2': 'value'}
                }
            }
        }

        with open(self.config_dir / "default.yaml", 'w') as f:
            import yaml
            yaml.dump(default_config, f)

    def tearDown(self):
        """테스트 정리"""
        shutil.rmtree(self.temp_dir)

    def test_config_loading(self):
        """설정 로딩 테스트"""
        config = UnifiedConfigManager('development', str(self.config_dir))

        self.assertEqual(config.data.symbol, 'TEST')
        self.assertEqual(config.data.period, '1y')
        self.assertIn('test_model', config.models.supported_types)

    def test_config_get_set(self):
        """설정 get/set 테스트"""
        config = UnifiedConfigManager('development', str(self.config_dir))

        # Get 테스트
        symbol = config.get('data.symbol')
        self.assertEqual(symbol, 'TEST')

        # Set 테스트
        config.set('data.symbol', 'NEW_TEST')
        new_symbol = config.get('data.symbol')
        self.assertEqual(new_symbol, 'NEW_TEST')

    def test_model_config_retrieval(self):
        """모델 설정 조회 테스트"""
        config = UnifiedConfigManager('development', str(self.config_dir))

        model_config = config.get_model_config('test_model')
        self.assertEqual(model_config['param1'], 10)
        self.assertEqual(model_config['param2'], 'value')

    def test_directory_retrieval(self):
        """디렉토리 조회 테스트"""
        config = UnifiedConfigManager('development', str(self.config_dir))

        raw_dir = config.get_data_directory('raw')
        self.assertEqual(raw_dir, 'data/test/raw')

        models_dir = config.get_data_directory('models')
        self.assertEqual(models_dir, 'data/test/models')


class TestSystemIntegration(unittest.TestCase):
    """시스템 통합 테스트"""

    def setUp(self):
        """테스트 설정"""
        # 테스트 데이터 생성
        np.random.seed(42)
        self.X = np.random.randn(200, 8)
        self.y = (self.X[:, 0] + self.X[:, 1] + np.random.randn(200) * 0.1 > 0).astype(int)

        # 훈련/테스트 분할
        split_idx = 150
        self.X_train, self.X_test = self.X[:split_idx], self.X[split_idx:]
        self.y_train, self.y_test = self.y[:split_idx], self.y[split_idx:]

    def test_full_pipeline(self):
        """전체 파이프라인 통합 테스트"""
        # 1. 모델 생성 및 훈련
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        models = {
            'rf': RandomForestClassifier(n_estimators=10, random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=100)
        }

        for model in models.values():
            model.fit(self.X_train, self.y_train)

        # 2. 앙상블 시스템 생성 및 훈련
        ensemble_config = EnsembleConfig(use_gpu=False, stacking_cv=3)
        ensemble_system = UnifiedEnsembleSystem(ensemble_config)

        ensemble_system.add_ensemble("test_ensemble", "voting")
        ensemble_system.add_models_to_ensemble("test_ensemble", models)
        ensemble_system.train_all_ensembles(self.X_train, self.y_train)

        # 3. 검증 수행
        validation_config = ValidationConfig(n_splits=3, strict_mode=False)
        validator = create_validator(validation_config)

        validation_result = validator.validate(
            self.X_train, self.y_train, models['rf']
        )

        # 4. 성능 평가
        metrics_config = MetricsConfig(enable_visualization=False)
        calculator = create_metrics_calculator(metrics_config)

        # 개별 모델 평가
        rf_predictions = models['rf'].predict(self.X_test)
        rf_result = calculator.evaluate_model(
            "RandomForest", self.y_test, rf_predictions, "classification"
        )

        # 앙상블 평가
        ensemble_predictions = ensemble_system.predict(self.X_test)
        ensemble_binary = (ensemble_predictions > 0.5).astype(int)
        ensemble_result = calculator.evaluate_model(
            "Ensemble", self.y_test, ensemble_binary, "classification"
        )

        # 5. 결과 검증
        self.assertTrue(validation_result.passed)
        self.assertGreater(rf_result.metrics['accuracy'], 0.3)  # 최소한의 성능
        self.assertGreater(ensemble_result.metrics['accuracy'], 0.3)

        # 6. 모델 비교
        comparison = calculator.compare_models([rf_result, ensemble_result])
        self.assertEqual(len(comparison['models']), 2)
        self.assertIn('best_model', comparison)

        print("✅ 전체 파이프라인 통합 테스트 완료")


if __name__ == '__main__':
    # 테스트 실행
    unittest.main(verbosity=2)