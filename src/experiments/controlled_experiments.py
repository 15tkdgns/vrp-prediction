#!/usr/bin/env python3
"""
🧪 통제 실험 시스템
학술 논문을 위한 체계적인 실험 설계 및 실행 프레임워크

주요 기능:
- 통제 변수 관리
- 실험 조건 설정
- A/B 테스트 프레임워크
- 결과 분석 및 해석
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import itertools
from sklearn.model_selection import ParameterGrid
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class ExperimentType(Enum):
    """실험 유형 열거형"""
    AB_TEST = "ab_test"
    MULTI_ARM = "multi_arm"
    FACTORIAL = "factorial"
    ABLATION = "ablation"
    SENSITIVITY = "sensitivity"

@dataclass
class ExperimentalCondition:
    """
    실험 조건 정의 클래스
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    control_variables: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: Optional[str] = None

@dataclass
class ExperimentResult:
    """
    실험 결과 클래스
    """
    condition_name: str
    metrics: Dict[str, float]
    sample_size: int
    execution_time: float
    additional_info: Dict[str, Any] = field(default_factory=dict)

class ControlledExperimentFramework:
    """
    통제 실험을 위한 포괄적 프레임워크

    변수 통제, 실험 설계, 결과 분석을 체계적으로 수행
    """

    def __init__(self, random_state=42):
        """
        초기화

        Args:
            random_state: 랜덤 시드
        """
        self.random_state = random_state
        np.random.seed(random_state)

        self.experiments: Dict[str, List[ExperimentalCondition]] = {}
        self.results: Dict[str, List[ExperimentResult]] = {}
        self.control_variables: Dict[str, Any] = {}

    def set_control_variables(self, **kwargs):
        """
        통제 변수 설정

        모든 실험에서 동일하게 유지될 변수들
        """
        self.control_variables.update(kwargs)

    def design_ab_test(self, experiment_name: str,
                      condition_a: Dict[str, Any],
                      condition_b: Dict[str, Any],
                      condition_a_name: str = "Control",
                      condition_b_name: str = "Treatment") -> str:
        """
        A/B 테스트 설계

        Args:
            experiment_name: 실험 이름
            condition_a: 조건 A 파라미터
            condition_b: 조건 B 파라미터
            condition_a_name: 조건 A 이름
            condition_b_name: 조건 B 이름

        Returns:
            실험 ID
        """
        conditions = [
            ExperimentalCondition(
                name=condition_a_name,
                description=f"A/B Test - {condition_a_name} condition",
                parameters=condition_a,
                control_variables=self.control_variables.copy()
            ),
            ExperimentalCondition(
                name=condition_b_name,
                description=f"A/B Test - {condition_b_name} condition",
                parameters=condition_b,
                control_variables=self.control_variables.copy()
            )
        ]

        self.experiments[experiment_name] = conditions
        return experiment_name

    def design_factorial_experiment(self, experiment_name: str,
                                  factors: Dict[str, List[Any]]) -> str:
        """
        요인 설계 실험 (Factorial Design)

        Args:
            experiment_name: 실험 이름
            factors: 요인별 수준 딕셔너리

        Returns:
            실험 ID
        """
        # 모든 요인 조합 생성
        factor_names = list(factors.keys())
        factor_levels = list(factors.values())

        conditions = []
        for i, combination in enumerate(itertools.product(*factor_levels)):
            condition_params = dict(zip(factor_names, combination))
            condition_name = f"Condition_{i+1}"

            # 조건 설명 생성
            desc_parts = [f"{name}={value}" for name, value in condition_params.items()]
            description = f"Factorial - {', '.join(desc_parts)}"

            conditions.append(ExperimentalCondition(
                name=condition_name,
                description=description,
                parameters=condition_params,
                control_variables=self.control_variables.copy()
            ))

        self.experiments[experiment_name] = conditions
        return experiment_name

    def design_ablation_study(self, experiment_name: str,
                            full_model_params: Dict[str, Any],
                            components_to_ablate: List[str]) -> str:
        """
        소거 연구 (Ablation Study) 설계

        Args:
            experiment_name: 실험 이름
            full_model_params: 전체 모델 파라미터
            components_to_ablate: 제거할 구성요소 목록

        Returns:
            실험 ID
        """
        conditions = []

        # 전체 모델 (베이스라인)
        conditions.append(ExperimentalCondition(
            name="Full_Model",
            description="Complete model with all components",
            parameters=full_model_params.copy(),
            control_variables=self.control_variables.copy()
        ))

        # 각 구성요소를 하나씩 제거한 모델들
        for component in components_to_ablate:
            ablated_params = full_model_params.copy()

            # 구성요소 제거 방법 (파라미터에 따라 다름)
            if component in ablated_params:
                if isinstance(ablated_params[component], bool):
                    ablated_params[component] = False
                elif isinstance(ablated_params[component], (int, float)):
                    ablated_params[component] = 0
                elif isinstance(ablated_params[component], list):
                    ablated_params[component] = []
                else:
                    del ablated_params[component]

            conditions.append(ExperimentalCondition(
                name=f"Without_{component}",
                description=f"Model without {component}",
                parameters=ablated_params,
                control_variables=self.control_variables.copy()
            ))

        self.experiments[experiment_name] = conditions
        return experiment_name

    def design_sensitivity_analysis(self, experiment_name: str,
                                  base_params: Dict[str, Any],
                                  sensitivity_params: Dict[str, List[Any]]) -> str:
        """
        민감도 분석 설계

        Args:
            experiment_name: 실험 이름
            base_params: 기본 파라미터
            sensitivity_params: 민감도 분석할 파라미터와 그 값들

        Returns:
            실험 ID
        """
        conditions = []

        # 베이스라인 조건
        conditions.append(ExperimentalCondition(
            name="Baseline",
            description="Baseline configuration",
            parameters=base_params.copy(),
            control_variables=self.control_variables.copy()
        ))

        # 각 파라미터별 민감도 테스트
        for param_name, param_values in sensitivity_params.items():
            for i, param_value in enumerate(param_values):
                test_params = base_params.copy()
                test_params[param_name] = param_value

                condition_name = f"{param_name}_{i+1}"
                description = f"Sensitivity test - {param_name}={param_value}"

                conditions.append(ExperimentalCondition(
                    name=condition_name,
                    description=description,
                    parameters=test_params,
                    control_variables=self.control_variables.copy()
                ))

        self.experiments[experiment_name] = conditions
        return experiment_name

    def run_experiment(self, experiment_name: str,
                      model_factory: Callable,
                      X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      custom_metrics: Optional[Dict[str, Callable]] = None,
                      n_runs: int = 1) -> Dict[str, List[ExperimentResult]]:
        """
        실험 실행

        Args:
            experiment_name: 실험 이름
            model_factory: 모델 생성 함수
            X_train, y_train: 훈련 데이터
            X_test, y_test: 테스트 데이터
            custom_metrics: 사용자 정의 메트릭
            n_runs: 실행 횟수 (결과 안정성 확보)

        Returns:
            실험 결과 딕셔너리
        """
        if experiment_name not in self.experiments:
            raise ValueError(f"실험 '{experiment_name}'이 정의되지 않았습니다.")

        conditions = self.experiments[experiment_name]
        results = []

        # 기본 메트릭 정의
        default_metrics = {
            'mae': lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
            'mse': lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
            'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': lambda y_true, y_pred: r2_score(y_true, y_pred)
        }

        # 사용자 메트릭 추가
        if custom_metrics:
            default_metrics.update(custom_metrics)

        for condition in conditions:
            print(f"실행 중: {condition.name}")

            condition_results = []

            for run in range(n_runs):
                try:
                    import time
                    start_time = time.time()

                    # 모델 생성 및 훈련
                    model = model_factory(**condition.parameters, **condition.control_variables)
                    model.fit(X_train, y_train)

                    # 예측
                    y_pred = model.predict(X_test)

                    # 메트릭 계산
                    metrics = {}
                    for metric_name, metric_func in default_metrics.items():
                        try:
                            metrics[metric_name] = metric_func(y_test, y_pred)
                        except Exception as e:
                            print(f"메트릭 {metric_name} 계산 오류: {str(e)}")
                            metrics[metric_name] = np.nan

                    # 방향 정확도 추가
                    metrics['direction_accuracy'] = self._calculate_direction_accuracy(y_test, y_pred)

                    execution_time = time.time() - start_time

                    # 추가 정보 수집
                    additional_info = {
                        'run_number': run + 1,
                        'model_type': type(model).__name__,
                        'parameters_used': condition.parameters.copy()
                    }

                    # 특징 중요도 (가능한 경우)
                    if hasattr(model, 'feature_importances_'):
                        additional_info['feature_importances'] = model.feature_importances_.tolist()
                    elif hasattr(model, 'coef_'):
                        additional_info['coefficients'] = model.coef_.tolist()

                    result = ExperimentResult(
                        condition_name=condition.name,
                        metrics=metrics,
                        sample_size=len(y_test),
                        execution_time=execution_time,
                        additional_info=additional_info
                    )

                    condition_results.append(result)

                except Exception as e:
                    print(f"조건 {condition.name}, 실행 {run+1} 오류: {str(e)}")
                    # 오류 결과도 기록
                    error_result = ExperimentResult(
                        condition_name=condition.name,
                        metrics={'error': str(e)},
                        sample_size=0,
                        execution_time=0,
                        additional_info={'error': True, 'run_number': run + 1}
                    )
                    condition_results.append(error_result)

            results.extend(condition_results)

        self.results[experiment_name] = results
        return {experiment_name: results}

    def _calculate_direction_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """방향 정확도 계산"""
        if len(y_true) <= 1:
            return 0.5

        true_directions = np.sign(y_true)
        pred_directions = np.sign(y_pred)

        # 0인 경우 처리
        true_directions[true_directions == 0] = 1
        pred_directions[pred_directions == 0] = 1

        accuracy = np.mean(true_directions == pred_directions)
        return accuracy * 100

    def analyze_experiment_results(self, experiment_name: str,
                                 primary_metric: str = 'mae') -> Dict[str, Any]:
        """
        실험 결과 분석

        Args:
            experiment_name: 실험 이름
            primary_metric: 주요 평가 지표

        Returns:
            분석 결과 딕셔너리
        """
        if experiment_name not in self.results:
            raise ValueError(f"실험 '{experiment_name}' 결과가 없습니다.")

        results = self.results[experiment_name]

        # 조건별 결과 집계
        condition_summaries = {}
        for result in results:
            if 'error' in result.metrics:
                continue

            condition = result.condition_name
            if condition not in condition_summaries:
                condition_summaries[condition] = {
                    'metrics': {},
                    'execution_times': [],
                    'n_runs': 0
                }

            # 메트릭별 값 수집
            for metric, value in result.metrics.items():
                if metric not in condition_summaries[condition]['metrics']:
                    condition_summaries[condition]['metrics'][metric] = []
                condition_summaries[condition]['metrics'][metric].append(value)

            condition_summaries[condition]['execution_times'].append(result.execution_time)
            condition_summaries[condition]['n_runs'] += 1

        # 통계 계산
        analysis = {
            'experiment_name': experiment_name,
            'primary_metric': primary_metric,
            'conditions': {},
            'comparison': {}
        }

        for condition, summary in condition_summaries.items():
            condition_stats = {
                'n_runs': summary['n_runs'],
                'avg_execution_time': np.mean(summary['execution_times']),
                'metrics': {}
            }

            for metric, values in summary['metrics'].items():
                condition_stats['metrics'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values, ddof=1) if len(values) > 1 else 0,
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }

            analysis['conditions'][condition] = condition_stats

        # 조건 간 비교 (주요 메트릭 기준)
        if len(condition_summaries) >= 2 and primary_metric in list(condition_summaries.values())[0]['metrics']:
            condition_means = {}
            for condition, summary in condition_summaries.items():
                condition_means[condition] = np.mean(summary['metrics'][primary_metric])

            # 최고 성능 조건
            if primary_metric in ['mae', 'mse', 'rmse']:  # 낮을수록 좋음
                best_condition = min(condition_means.items(), key=lambda x: x[1])
                worst_condition = max(condition_means.items(), key=lambda x: x[1])
            else:  # 높을수록 좋음
                best_condition = max(condition_means.items(), key=lambda x: x[1])
                worst_condition = min(condition_means.items(), key=lambda x: x[1])

            analysis['comparison'] = {
                'best_condition': best_condition[0],
                'best_value': best_condition[1],
                'worst_condition': worst_condition[0],
                'worst_value': worst_condition[1],
                'improvement': abs(best_condition[1] - worst_condition[1]),
                'relative_improvement': abs(best_condition[1] - worst_condition[1]) / abs(worst_condition[1]) * 100 if worst_condition[1] != 0 else 0
            }

        return analysis

    def generate_experiment_report(self, experiment_name: str,
                                 primary_metric: str = 'mae') -> str:
        """
        실험 보고서 생성

        Args:
            experiment_name: 실험 이름
            primary_metric: 주요 평가 지표

        Returns:
            실험 보고서 문자열
        """
        if experiment_name not in self.results:
            return f"실험 '{experiment_name}' 결과가 없습니다."

        analysis = self.analyze_experiment_results(experiment_name, primary_metric)

        report = [f"🧪 실험 보고서: {experiment_name}", "=" * 60, ""]

        # 실험 개요
        experiment_type = "Unknown"
        if experiment_name in self.experiments:
            n_conditions = len(self.experiments[experiment_name])
            experiment_type = f"{n_conditions}개 조건 비교"

        report.append(f"📋 실험 개요:")
        report.append(f"   실험 유형: {experiment_type}")
        report.append(f"   주요 지표: {primary_metric.upper()}")
        report.append(f"   분석 조건: {len(analysis['conditions'])}개")
        report.append("")

        # 조건별 성능 표
        report.append("📊 조건별 성능 결과:")
        report.append("-" * 50)

        headers = ["Condition", f"{primary_metric.upper()}", "Std", "Dir Acc(%)", "Exec Time(s)"]
        rows = []

        for condition, stats in analysis['conditions'].items():
            metrics = stats['metrics']
            primary_mean = metrics.get(primary_metric, {}).get('mean', 0)
            primary_std = metrics.get(primary_metric, {}).get('std', 0)
            dir_acc = metrics.get('direction_accuracy', {}).get('mean', 0)

            rows.append([
                condition,
                f"{primary_mean:.6f}",
                f"{primary_std:.6f}",
                f"{dir_acc:.1f}",
                f"{stats['avg_execution_time']:.3f}"
            ])

        # 표 포맷팅
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]

        header_line = " | ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers)))
        report.append(header_line)
        report.append("-" * len(header_line))

        for row in rows:
            data_line = " | ".join(f"{row[i]:<{col_widths[i]}}" for i in range(len(row)))
            report.append(data_line)

        report.append("")

        # 비교 분석
        if 'best_condition' in analysis['comparison']:
            comp = analysis['comparison']
            report.append("🏆 성능 비교:")
            report.append(f"   최고 성능: {comp['best_condition']} ({comp['best_value']:.6f})")
            report.append(f"   최저 성능: {comp['worst_condition']} ({comp['worst_value']:.6f})")
            report.append(f"   성능 차이: {comp['improvement']:.6f} ({comp['relative_improvement']:.1f}%)")
            report.append("")

        # 통계적 유의성 (조건이 2개인 경우)
        if len(analysis['conditions']) == 2:
            conditions = list(analysis['conditions'].keys())
            cond1, cond2 = conditions[0], conditions[1]

            # 원본 데이터에서 값 추출
            cond1_values = []
            cond2_values = []

            for result in self.results[experiment_name]:
                if result.condition_name == cond1 and primary_metric in result.metrics:
                    cond1_values.append(result.metrics[primary_metric])
                elif result.condition_name == cond2 and primary_metric in result.metrics:
                    cond2_values.append(result.metrics[primary_metric])

            if len(cond1_values) > 1 and len(cond2_values) > 1:
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(cond1_values, cond2_values)

                report.append("📈 통계적 유의성:")
                report.append(f"   t-통계량: {t_stat:.4f}")
                report.append(f"   p-value: {p_value:.6f}")

                if p_value < 0.05:
                    report.append("   ✅ 통계적으로 유의한 차이 (p < 0.05)")
                else:
                    report.append("   ❌ 통계적으로 유의하지 않음 (p ≥ 0.05)")

                report.append("")

        # 권고사항
        report.append("💡 권고사항:")
        if 'best_condition' in analysis['comparison']:
            best_cond = analysis['comparison']['best_condition']
            improvement = analysis['comparison']['relative_improvement']

            if improvement > 5:  # 5% 이상 개선
                report.append(f"   ✅ '{best_cond}' 조건 사용 권장 ({improvement:.1f}% 성능 향상)")
            else:
                report.append(f"   ⚠️  조건 간 성능 차이 미미 ({improvement:.1f}%)")
                report.append("   복잡성과 성능의 트레이드오프 고려 필요")
        else:
            report.append("   모든 조건에서 유사한 성능. 단순한 모델 우선 고려")

        # 실험 한계점
        report.append("")
        report.append("⚠️  실험 한계점:")
        total_runs = sum(stats['n_runs'] for stats in analysis['conditions'].values())
        report.append(f"   - 총 실행 횟수: {total_runs}회 (더 많은 반복으로 안정성 확보 가능)")
        report.append("   - 단일 데이터셋 기준 (다양한 데이터셋에서 검증 필요)")
        report.append("   - 특정 하이퍼파라미터 범위 내 결과")

        return "\n".join(report)

def main():
    """테스트 및 예제 실행"""
    print("🧪 통제 실험 시스템 테스트")
    print("=" * 60)

    # 시뮬레이션 데이터 생성
    np.random.seed(42)

    n_samples = 300
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    y = 0.1 * X[:, 0] + 0.05 * X[:, 1] + np.random.normal(0, 0.1, n_samples)

    # 훈련/테스트 분할
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 실험 프레임워크 초기화
    framework = ControlledExperimentFramework(random_state=42)

    # 통제 변수 설정
    framework.set_control_variables(
        random_state=42,
        max_iter=1000
    )

    # 간단한 모델 팩토리 (Linear Regression 변형)
    def model_factory(alpha=1.0, fit_intercept=True, **kwargs):
        from sklearn.linear_model import Ridge
        return Ridge(alpha=alpha, fit_intercept=fit_intercept, **kwargs)

    # 1. A/B 테스트 설계 및 실행
    print("1. A/B 테스트: 정규화 유무")
    ab_experiment = framework.design_ab_test(
        experiment_name="regularization_test",
        condition_a={'alpha': 0.0},  # 정규화 없음
        condition_b={'alpha': 1.0},  # 정규화 있음
        condition_a_name="No_Regularization",
        condition_b_name="With_Regularization"
    )

    ab_results = framework.run_experiment(
        ab_experiment, model_factory,
        X_train, y_train, X_test, y_test,
        n_runs=3
    )

    ab_report = framework.generate_experiment_report(ab_experiment)
    print(ab_report)
    print("\n" + "="*60 + "\n")

    # 2. 요인 설계 실험
    print("2. 요인 설계: 정규화 강도 × 절편 유무")
    factorial_experiment = framework.design_factorial_experiment(
        experiment_name="factorial_test",
        factors={
            'alpha': [0.1, 1.0, 10.0],
            'fit_intercept': [True, False]
        }
    )

    factorial_results = framework.run_experiment(
        factorial_experiment, model_factory,
        X_train, y_train, X_test, y_test,
        n_runs=2
    )

    factorial_report = framework.generate_experiment_report(factorial_experiment)
    print(factorial_report)
    print("\n" + "="*60 + "\n")

    # 3. 민감도 분석
    print("3. 민감도 분석: 정규화 강도")
    sensitivity_experiment = framework.design_sensitivity_analysis(
        experiment_name="alpha_sensitivity",
        base_params={'alpha': 1.0, 'fit_intercept': True},
        sensitivity_params={
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        }
    )

    sensitivity_results = framework.run_experiment(
        sensitivity_experiment, model_factory,
        X_train, y_train, X_test, y_test,
        n_runs=2
    )

    sensitivity_report = framework.generate_experiment_report(sensitivity_experiment)
    print(sensitivity_report)

    # 4. 전체 실험 요약
    print("\n" + "="*60)
    print("🎯 실험 세션 요약")
    print("="*60)
    print(f"실행된 실험 수: {len(framework.experiments)}")
    print(f"총 실험 조건 수: {sum(len(conditions) for conditions in framework.experiments.values())}")
    print(f"총 실행 횟수: {sum(len(results) for results in framework.results.values())}")

    # 각 실험의 최고 성능 조건
    print("\n최고 성능 조건 요약:")
    for exp_name in framework.experiments.keys():
        analysis = framework.analyze_experiment_results(exp_name)
        if 'best_condition' in analysis['comparison']:
            best = analysis['comparison']['best_condition']
            best_value = analysis['comparison']['best_value']
            print(f"  {exp_name}: {best} (MAE: {best_value:.6f})")

if __name__ == "__main__":
    main()