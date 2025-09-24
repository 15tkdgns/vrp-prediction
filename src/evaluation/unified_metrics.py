#!/usr/bin/env python3
"""
통합 성능 평가 시스템 - 12개 분산 성능 모듈의 핵심 기능 통합
분류, 회귀, 앙상블, 시계열 모델의 포괄적 성능 평가
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, mean_squared_error, mean_absolute_error,
    r2_score, explained_variance_score, median_absolute_error
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from datetime import datetime
import json
from pathlib import Path

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class MetricsConfig:
    """성능 지표 설정 클래스"""
    classification_metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'
    ])
    regression_metrics: List[str] = field(default_factory=lambda: [
        'mse', 'mae', 'r2_score', 'explained_variance', 'median_ae'
    ])
    financial_metrics: List[str] = field(default_factory=lambda: [
        'sharpe_ratio', 'max_drawdown', 'calmar_ratio', 'sortino_ratio'
    ])
    enable_visualization: bool = True
    save_plots: bool = False
    plot_directory: str = "plots"


@dataclass
class PerformanceResult:
    """성능 평가 결과 클래스"""
    model_name: str
    task_type: str  # 'classification', 'regression', 'financial'
    metrics: Dict[str, float]
    confusion_matrix: np.ndarray = None
    feature_importance: Dict[str, float] = None
    predictions: np.ndarray = None
    probabilities: np.ndarray = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseMetricsCalculator(ABC):
    """성능 지표 계산기 기본 클래스"""

    def __init__(self, config: MetricsConfig):
        self.config = config

    @abstractmethod
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Dict[str, float]:
        """성능 지표 계산"""
        pass


class ClassificationMetrics(BaseMetricsCalculator):
    """분류 성능 지표"""

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
        """분류 성능 지표 계산"""
        metrics = {}

        # 기본 분류 지표
        if 'accuracy' in self.config.classification_metrics:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)

        if 'precision' in self.config.classification_metrics:
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)

        if 'recall' in self.config.classification_metrics:
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)

        if 'f1_score' in self.config.classification_metrics:
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # ROC-AUC (확률 필요)
        if 'roc_auc' in self.config.classification_metrics and y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:  # 이진 분류
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                else:  # 다중 클래스
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            except ValueError:
                metrics['roc_auc'] = 0.5  # 기본값

        # 추가 지표
        metrics.update(self._calculate_advanced_classification_metrics(y_true, y_pred, y_proba))

        return metrics

    def _calculate_advanced_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
        """고급 분류 지표"""
        metrics = {}

        # 균형 정확도 (Balanced Accuracy)
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):  # 이진 분류
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['balanced_accuracy'] = (sensitivity + specificity) / 2
            metrics['sensitivity'] = sensitivity
            metrics['specificity'] = specificity

            # Matthews Correlation Coefficient
            mcc_num = (tp * tn) - (fp * fn)
            mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            metrics['mcc'] = mcc_num / mcc_den if mcc_den != 0 else 0

        # Cohen's Kappa
        try:
            from sklearn.metrics import cohen_kappa_score
            metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        except:
            pass

        # Log Loss (확률 필요)
        if y_proba is not None:
            try:
                from sklearn.metrics import log_loss
                metrics['log_loss'] = log_loss(y_true, y_proba)
            except:
                pass

        return metrics


class RegressionMetrics(BaseMetricsCalculator):
    """회귀 성능 지표"""

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """회귀 성능 지표 계산"""
        metrics = {}

        # 기본 회귀 지표
        if 'mse' in self.config.regression_metrics:
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])

        if 'mae' in self.config.regression_metrics:
            metrics['mae'] = mean_absolute_error(y_true, y_pred)

        if 'r2_score' in self.config.regression_metrics:
            metrics['r2_score'] = r2_score(y_true, y_pred)

        if 'explained_variance' in self.config.regression_metrics:
            metrics['explained_variance'] = explained_variance_score(y_true, y_pred)

        if 'median_ae' in self.config.regression_metrics:
            metrics['median_ae'] = median_absolute_error(y_true, y_pred)

        # 추가 지표
        metrics.update(self._calculate_advanced_regression_metrics(y_true, y_pred))

        return metrics

    def _calculate_advanced_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """고급 회귀 지표"""
        metrics = {}

        # Mean Absolute Percentage Error
        def mape(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-7, y_true))) * 100

        metrics['mape'] = mape(y_true, y_pred)

        # Symmetric MAPE
        def smape(y_true, y_pred):
            return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-7))

        metrics['smape'] = smape(y_true, y_pred)

        # Pearson 상관계수
        correlation, _ = stats.pearsonr(y_true, y_pred)
        metrics['pearson_correlation'] = correlation

        # Spearman 상관계수
        spearman_corr, _ = stats.spearmanr(y_true, y_pred)
        metrics['spearman_correlation'] = spearman_corr

        # 최대 절대 오차
        metrics['max_error'] = np.max(np.abs(y_true - y_pred))

        # 평균 편향 (Mean Bias)
        metrics['mean_bias'] = np.mean(y_pred - y_true)

        return metrics


class FinancialMetrics(BaseMetricsCalculator):
    """금융 성능 지표"""

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, returns: np.ndarray = None) -> Dict[str, float]:
        """금융 성능 지표 계산"""
        metrics = {}

        if returns is not None:
            # 샤프 비율
            if 'sharpe_ratio' in self.config.financial_metrics:
                metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)

            # 최대 손실 (Maximum Drawdown)
            if 'max_drawdown' in self.config.financial_metrics:
                metrics['max_drawdown'] = self._calculate_max_drawdown(returns)

            # 칼마 비율
            if 'calmar_ratio' in self.config.financial_metrics:
                metrics['calmar_ratio'] = self._calculate_calmar_ratio(returns)

            # 소르티노 비율
            if 'sortino_ratio' in self.config.financial_metrics:
                metrics['sortino_ratio'] = self._calculate_sortino_ratio(returns)

        # 방향성 정확도
        if len(y_true) > 1 and len(y_pred) > 1:
            direction_true = np.diff(y_true) > 0
            direction_pred = np.diff(y_pred) > 0
            metrics['directional_accuracy'] = accuracy_score(direction_true, direction_pred)

        return metrics

    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """샤프 비율 계산"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        annualized_return = np.mean(returns) * 252  # 연간화
        annualized_volatility = np.std(returns) * np.sqrt(252)

        return (annualized_return - risk_free_rate) / annualized_volatility

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """최대 손실 계산"""
        if len(returns) == 0:
            return 0.0

        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        return np.min(drawdown)

    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """칼마 비율 계산"""
        annualized_return = np.mean(returns) * 252
        max_drawdown = abs(self._calculate_max_drawdown(returns))

        return annualized_return / max_drawdown if max_drawdown != 0 else 0.0

    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """소르티노 비율 계산"""
        if len(returns) == 0:
            return 0.0

        annualized_return = np.mean(returns) * 252
        negative_returns = returns[returns < 0]

        if len(negative_returns) == 0:
            return float('inf')

        downside_deviation = np.std(negative_returns) * np.sqrt(252)

        return (annualized_return - risk_free_rate) / downside_deviation


class UnifiedMetricsCalculator:
    """통합 성능 지표 계산기"""

    def __init__(self, config: MetricsConfig = None):
        self.config = config or MetricsConfig()
        self.classification = ClassificationMetrics(self.config)
        self.regression = RegressionMetrics(self.config)
        self.financial = FinancialMetrics(self.config)

    def evaluate_model(self,
                      model_name: str,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      task_type: str = 'classification',
                      y_proba: np.ndarray = None,
                      returns: np.ndarray = None,
                      feature_names: List[str] = None,
                      model: Any = None) -> PerformanceResult:
        """모델 성능 종합 평가"""

        # 기본 지표 계산
        if task_type == 'classification':
            metrics = self.classification.calculate_metrics(y_true, y_pred, y_proba)
            cm = confusion_matrix(y_true, y_pred)
        elif task_type == 'regression':
            metrics = self.regression.calculate_metrics(y_true, y_pred)
            cm = None
        elif task_type == 'financial':
            metrics = self.financial.calculate_metrics(y_true, y_pred, returns)
            cm = None
        else:
            raise ValueError(f"지원하지 않는 태스크 타입: {task_type}")

        # 특성 중요도 추출
        feature_importance = self._extract_feature_importance(model, feature_names)

        # 결과 객체 생성
        result = PerformanceResult(
            model_name=model_name,
            task_type=task_type,
            metrics=metrics,
            confusion_matrix=cm,
            feature_importance=feature_importance,
            predictions=y_pred,
            probabilities=y_proba,
            metadata={
                'n_samples': len(y_true),
                'n_features': len(feature_names) if feature_names else None,
                'unique_labels': len(np.unique(y_true)),
                'config': self.config.__dict__
            }
        )

        # 시각화 생성
        if self.config.enable_visualization:
            self._create_visualizations(result, y_true)

        return result

    def _extract_feature_importance(self, model: Any, feature_names: List[str] = None) -> Dict[str, float]:
        """특성 중요도 추출"""
        if model is None or feature_names is None:
            return {}

        importance_dict = {}

        # sklearn 기반 모델들
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for i, importance in enumerate(importances):
                feature_name = feature_names[i] if i < len(feature_names) else f'feature_{i}'
                importance_dict[feature_name] = float(importance)

        # 선형 모델들 (계수)
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            if len(coef.shape) > 1:
                coef = coef[0]  # 이진 분류의 경우 첫 번째 클래스

            for i, coefficient in enumerate(coef):
                feature_name = feature_names[i] if i < len(feature_names) else f'feature_{i}'
                importance_dict[feature_name] = float(abs(coefficient))

        return importance_dict

    def _create_visualizations(self, result: PerformanceResult, y_true: np.ndarray) -> None:
        """시각화 생성"""
        if not self.config.enable_visualization:
            return

        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{result.model_name} 성능 분석', fontsize=16)

        # 1. 혼동 행렬 (분류)
        if result.task_type == 'classification' and result.confusion_matrix is not None:
            sns.heatmap(result.confusion_matrix, annot=True, fmt='d', ax=axes[0, 0])
            axes[0, 0].set_title('혼동 행렬')
            axes[0, 0].set_xlabel('예측값')
            axes[0, 0].set_ylabel('실제값')

        # 2. 예측 vs 실제값
        axes[0, 1].scatter(y_true, result.predictions, alpha=0.6)
        axes[0, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('실제값')
        axes[0, 1].set_ylabel('예측값')
        axes[0, 1].set_title('예측 vs 실제값')

        # 3. 특성 중요도
        if result.feature_importance:
            # 상위 10개 특성만 표시
            top_features = dict(sorted(result.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
            features = list(top_features.keys())
            importances = list(top_features.values())

            axes[1, 0].barh(features, importances)
            axes[1, 0].set_xlabel('중요도')
            axes[1, 0].set_title('특성 중요도 (상위 10개)')

        # 4. 성능 지표 요약
        metrics_text = ""
        for metric, value in result.metrics.items():
            metrics_text += f"{metric}: {value:.4f}\n"

        axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top')
        axes[1, 1].set_title('성능 지표 요약')
        axes[1, 1].axis('off')

        plt.tight_layout()

        # 저장
        if self.config.save_plots:
            save_dir = Path(self.config.plot_directory)
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / f'{result.model_name}_performance.png', dpi=300, bbox_inches='tight')
            logger.info(f"성능 차트 저장: {save_dir / f'{result.model_name}_performance.png'}")

        plt.show()

    def compare_models(self, results: List[PerformanceResult], primary_metric: str = 'f1_score') -> Dict[str, Any]:
        """모델들 성능 비교"""
        if not results:
            return {}

        comparison = {
            'models': [r.model_name for r in results],
            'primary_metric': primary_metric,
            'rankings': {},
            'metrics_summary': {},
            'best_model': None
        }

        # 각 지표별 성능 수집
        all_metrics = set()
        for result in results:
            all_metrics.update(result.metrics.keys())

        for metric in all_metrics:
            values = []
            for result in results:
                values.append(result.metrics.get(metric, 0))

            comparison['metrics_summary'][metric] = {
                'values': values,
                'mean': np.mean(values),
                'std': np.std(values),
                'best_value': max(values) if metric != 'mse' else min(values),
                'best_model': results[np.argmax(values) if metric != 'mse' else np.argmin(values)].model_name
            }

        # 주요 지표 기준 최고 모델 선정
        if primary_metric in comparison['metrics_summary']:
            best_idx = np.argmax(comparison['metrics_summary'][primary_metric]['values'])
            comparison['best_model'] = results[best_idx].model_name

        # 랭킹 생성
        if primary_metric in comparison['metrics_summary']:
            scores = comparison['metrics_summary'][primary_metric]['values']
            rankings = [(results[i].model_name, scores[i]) for i in range(len(results))]
            rankings.sort(key=lambda x: x[1], reverse=True)
            comparison['rankings'] = rankings

        return comparison

    def generate_comprehensive_report(self, results: List[PerformanceResult], save_path: str = None) -> Dict[str, Any]:
        """종합 성능 보고서 생성"""
        report = {
            'summary': {
                'total_models': len(results),
                'evaluation_timestamp': datetime.now().isoformat(),
                'task_types': list(set(r.task_type for r in results))
            },
            'individual_results': {},
            'model_comparison': {},
            'recommendations': []
        }

        # 개별 모델 결과
        for result in results:
            report['individual_results'][result.model_name] = {
                'task_type': result.task_type,
                'metrics': result.metrics,
                'timestamp': result.timestamp,
                'metadata': result.metadata
            }

        # 모델 비교
        if len(results) > 1:
            report['model_comparison'] = self.compare_models(results)

        # 권장사항 생성
        report['recommendations'] = self._generate_recommendations(results)

        # 저장
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"성능 보고서 저장: {save_path}")

        return report

    def _generate_recommendations(self, results: List[PerformanceResult]) -> List[str]:
        """권장사항 생성"""
        recommendations = []

        if not results:
            return recommendations

        # 성능 기반 권장사항
        for result in results:
            if result.task_type == 'classification':
                accuracy = result.metrics.get('accuracy', 0)
                f1 = result.metrics.get('f1_score', 0)

                if accuracy < 0.7:
                    recommendations.append(f"{result.model_name}: 정확도가 낮습니다. 특성 엔지니어링이나 하이퍼파라미터 튜닝을 고려하세요.")

                if f1 < 0.6:
                    recommendations.append(f"{result.model_name}: F1 점수가 낮습니다. 클래스 불균형 문제를 확인하세요.")

            elif result.task_type == 'regression':
                r2 = result.metrics.get('r2_score', 0)
                mape = result.metrics.get('mape', float('inf'))

                if r2 < 0.5:
                    recommendations.append(f"{result.model_name}: R² 점수가 낮습니다. 모델 복잡도를 늘리거나 특성을 추가하세요.")

                if mape > 20:
                    recommendations.append(f"{result.model_name}: MAPE가 높습니다. 이상값 처리를 검토하세요.")

        return recommendations


# 팩토리 함수들
def create_metrics_calculator(config: MetricsConfig = None) -> UnifiedMetricsCalculator:
    """통합 지표 계산기 생성"""
    return UnifiedMetricsCalculator(config)


def quick_evaluate(model_name: str, y_true: np.ndarray, y_pred: np.ndarray,
                  task_type: str = 'classification', **kwargs) -> PerformanceResult:
    """빠른 평가 수행"""
    calculator = create_metrics_calculator()
    return calculator.evaluate_model(model_name, y_true, y_pred, task_type, **kwargs)


if __name__ == "__main__":
    # 사용 예시
    config = MetricsConfig(
        classification_metrics=['accuracy', 'f1_score', 'roc_auc'],
        enable_visualization=True,
        save_plots=False
    )

    calculator = create_metrics_calculator(config)
    print("✅ 통합 성능 평가 시스템 초기화 완료")
    print(f"분류 지표: {config.classification_metrics}")
    print(f"회귀 지표: {config.regression_metrics}")
    print(f"금융 지표: {config.financial_metrics}")
    print(f"시각화 활성화: {config.enable_visualization}")