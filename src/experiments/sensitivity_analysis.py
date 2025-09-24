#!/usr/bin/env python3
"""
📊 민감도 분석 시스템
학술 논문을 위한 하이퍼파라미터 및 모델 민감도 분석

주요 기능:
- 하이퍼파라미터 민감도 분석
- 데이터 크기 민감도 분석
- 특징 중요도 민감도 분석
- 노이즈 영향 분석
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SensitivityResult:
    """민감도 분석 결과 클래스"""
    parameter_name: str
    parameter_values: List[Any]
    mean_scores: List[float]
    std_scores: List[float]
    optimal_value: Any
    optimal_score: float
    sensitivity_score: float
    interpretation: str

class SensitivityAnalyzer:
    """
    포괄적 민감도 분석 도구

    모델과 하이퍼파라미터의 민감도를 체계적으로 분석
    """

    def __init__(self, random_state=42):
        """
        초기화

        Args:
            random_state: 랜덤 시드
        """
        self.random_state = random_state
        np.random.seed(random_state)

        self.results: Dict[str, SensitivityResult] = {}

    def hyperparameter_sensitivity(self, model: BaseEstimator,
                                 parameter_name: str,
                                 parameter_range: List[Any],
                                 X_train: np.ndarray, y_train: np.ndarray,
                                 X_test: np.ndarray, y_test: np.ndarray,
                                 scoring: str = 'neg_mean_absolute_error',
                                 cv: int = 5) -> SensitivityResult:
        """
        하이퍼파라미터 민감도 분석

        Args:
            model: 분석할 모델
            parameter_name: 파라미터 이름
            parameter_range: 파라미터 값 범위
            X_train, y_train: 훈련 데이터
            X_test, y_test: 테스트 데이터
            scoring: 평가 지표
            cv: 교차 검증 폴드 수

        Returns:
            민감도 분석 결과
        """
        print(f"하이퍼파라미터 민감도 분석: {parameter_name}")

        # Validation curve 계산
        train_scores, test_scores = validation_curve(
            model, X_train, y_train,
            param_name=parameter_name,
            param_range=parameter_range,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        # 점수 변환 (음수 점수를 양수로)
        if scoring.startswith('neg_'):
            test_scores = -test_scores

        # 통계 계산
        mean_scores = np.mean(test_scores, axis=1)
        std_scores = np.std(test_scores, axis=1)

        # 최적값 찾기
        if scoring in ['neg_mean_absolute_error', 'neg_mean_squared_error']:
            optimal_idx = np.argmax(mean_scores)  # 음수 점수는 높을수록 좋음 (원래는 낮을수록 좋음)
        else:
            optimal_idx = np.argmax(mean_scores)  # 일반적으로 높을수록 좋음

        optimal_value = parameter_range[optimal_idx]
        optimal_score = mean_scores[optimal_idx]

        # 민감도 점수 계산 (변동 계수)
        sensitivity_score = np.std(mean_scores) / np.mean(mean_scores) if np.mean(mean_scores) != 0 else 0

        # 해석
        if sensitivity_score < 0.05:
            interpretation = f"{parameter_name}에 대해 매우 안정적 (민감도 낮음)"
        elif sensitivity_score < 0.15:
            interpretation = f"{parameter_name}에 대해 적당히 민감"
        else:
            interpretation = f"{parameter_name}에 대해 매우 민감 (신중한 튜닝 필요)"

        result = SensitivityResult(
            parameter_name=parameter_name,
            parameter_values=parameter_range,
            mean_scores=mean_scores.tolist(),
            std_scores=std_scores.tolist(),
            optimal_value=optimal_value,
            optimal_score=optimal_score,
            sensitivity_score=sensitivity_score,
            interpretation=interpretation
        )

        self.results[parameter_name] = result
        return result

    def data_size_sensitivity(self, model: BaseEstimator,
                            X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            train_sizes: Optional[List[float]] = None,
                            cv: int = 5) -> SensitivityResult:
        """
        데이터 크기 민감도 분석 (Learning Curve)

        Args:
            model: 분석할 모델
            X_train, y_train: 훈련 데이터
            X_test, y_test: 테스트 데이터
            train_sizes: 훈련 크기 비율들
            cv: 교차 검증 폴드 수

        Returns:
            민감도 분석 결과
        """
        print("데이터 크기 민감도 분석 (Learning Curve)")

        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        # Learning curve 계산
        train_sizes_abs, train_scores, test_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=train_sizes,
            cv=cv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )

        # 점수 변환
        test_scores = -test_scores
        mean_scores = np.mean(test_scores, axis=1)
        std_scores = np.std(test_scores, axis=1)

        # 최적 데이터 크기 (성능 포화점)
        # 성능이 더 이상 크게 개선되지 않는 지점 찾기
        score_improvements = np.diff(mean_scores)
        plateau_threshold = 0.001  # 개선이 0.1% 미만인 지점

        plateau_idx = None
        for i, improvement in enumerate(score_improvements):
            if improvement < plateau_threshold:
                plateau_idx = i + 1
                break

        if plateau_idx is None:
            plateau_idx = len(mean_scores) - 1

        optimal_size = train_sizes[plateau_idx]
        optimal_score = mean_scores[plateau_idx]

        # 민감도 계산 (데이터 크기 증가에 따른 성능 변화율)
        sensitivity_score = abs(score_improvements).mean() if len(score_improvements) > 0 else 0

        # 해석
        if sensitivity_score > 0.01:
            interpretation = "데이터 크기에 매우 민감. 더 많은 데이터로 성능 향상 가능"
        elif sensitivity_score > 0.005:
            interpretation = "데이터 크기에 적당히 민감. 추가 데이터 고려"
        else:
            interpretation = "데이터 크기에 둔감. 현재 데이터로 충분"

        result = SensitivityResult(
            parameter_name="training_data_size",
            parameter_values=train_sizes.tolist(),
            mean_scores=mean_scores.tolist(),
            std_scores=std_scores.tolist(),
            optimal_value=optimal_size,
            optimal_score=optimal_score,
            sensitivity_score=sensitivity_score,
            interpretation=interpretation
        )

        self.results["data_size"] = result
        return result

    def feature_importance_sensitivity(self, model: BaseEstimator,
                                     X_train: np.ndarray, y_train: np.ndarray,
                                     X_test: np.ndarray, y_test: np.ndarray,
                                     feature_names: Optional[List[str]] = None,
                                     n_permutations: int = 10) -> Dict[str, SensitivityResult]:
        """
        특징 중요도 민감도 분석 (Permutation Importance)

        Args:
            model: 분석할 모델
            X_train, y_train: 훈련 데이터
            X_test, y_test: 테스트 데이터
            feature_names: 특징 이름들
            n_permutations: 순열 반복 횟수

        Returns:
            특징별 민감도 결과 딕셔너리
        """
        print("특징 중요도 민감도 분석")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        # 모델 훈련
        model_fitted = clone(model)
        model_fitted.fit(X_train, y_train)

        # 베이스라인 성능
        baseline_pred = model_fitted.predict(X_test)
        baseline_score = mean_absolute_error(y_test, baseline_pred)

        feature_results = {}

        for i, feature_name in enumerate(feature_names):
            print(f"  분석 중: {feature_name}")

            # 해당 특징 순열하여 성능 변화 측정
            permutation_scores = []

            for _ in range(n_permutations):
                X_test_permuted = X_test.copy()
                np.random.shuffle(X_test_permuted[:, i])  # i번째 특징 순열

                permuted_pred = model_fitted.predict(X_test_permuted)
                permuted_score = mean_absolute_error(y_test, permuted_pred)
                permutation_scores.append(permuted_score)

            # 성능 저하 계산
            importance_scores = np.array(permutation_scores) - baseline_score
            mean_importance = np.mean(importance_scores)
            std_importance = np.std(importance_scores)

            # 민감도 계산 (성능 저하 정도)
            sensitivity = mean_importance / baseline_score if baseline_score != 0 else 0

            # 해석
            if sensitivity > 0.1:  # 10% 이상 성능 저하
                interpretation = f"{feature_name}은 매우 중요한 특징 (제거 시 {sensitivity*100:.1f}% 성능 저하)"
            elif sensitivity > 0.05:  # 5% 이상 성능 저하
                interpretation = f"{feature_name}은 중요한 특징 (제거 시 {sensitivity*100:.1f}% 성능 저하)"
            elif sensitivity > 0.01:  # 1% 이상 성능 저하
                interpretation = f"{feature_name}은 약간 중요한 특징"
            else:
                interpretation = f"{feature_name}은 중요도가 낮은 특징"

            result = SensitivityResult(
                parameter_name=feature_name,
                parameter_values=[0, 1],  # 특징 유무
                mean_scores=[baseline_score, np.mean(permutation_scores)],
                std_scores=[0, std_importance],
                optimal_value=1,  # 특징 사용
                optimal_score=baseline_score,
                sensitivity_score=sensitivity,
                interpretation=interpretation
            )

            feature_results[feature_name] = result

        return feature_results

    def noise_sensitivity(self, model: BaseEstimator,
                         X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         noise_levels: Optional[List[float]] = None,
                         n_trials: int = 5) -> SensitivityResult:
        """
        노이즈 민감도 분석

        Args:
            model: 분석할 모델
            X_train, y_train: 훈련 데이터
            X_test, y_test: 테스트 데이터
            noise_levels: 노이즈 수준들 (표준편차 비율)
            n_trials: 각 노이즈 수준별 시행 횟수

        Returns:
            노이즈 민감도 결과
        """
        print("노이즈 민감도 분석")

        if noise_levels is None:
            noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]

        mean_scores = []
        std_scores = []

        original_std = np.std(X_train)

        for noise_level in noise_levels:
            trial_scores = []

            for trial in range(n_trials):
                # 노이즈 추가
                noise = np.random.normal(0, noise_level * original_std, X_train.shape)
                X_train_noisy = X_train + noise

                # 테스트 데이터에도 동일한 노이즈 추가
                noise_test = np.random.normal(0, noise_level * original_std, X_test.shape)
                X_test_noisy = X_test + noise_test

                # 모델 훈련 및 평가
                model_trial = clone(model)
                model_trial.fit(X_train_noisy, y_train)

                pred = model_trial.predict(X_test_noisy)
                score = mean_absolute_error(y_test, pred)
                trial_scores.append(score)

            mean_scores.append(np.mean(trial_scores))
            std_scores.append(np.std(trial_scores))

        # 최적 노이즈 수준 (가장 좋은 성능)
        optimal_idx = np.argmin(mean_scores)
        optimal_noise = noise_levels[optimal_idx]
        optimal_score = mean_scores[optimal_idx]

        # 민감도 계산 (노이즈 증가에 따른 성능 저하율)
        clean_score = mean_scores[0]  # 노이즈 0인 경우
        max_degradation = max(mean_scores) - clean_score
        sensitivity_score = max_degradation / clean_score if clean_score != 0 else 0

        # 해석
        if sensitivity_score > 0.2:
            interpretation = "노이즈에 매우 민감. 데이터 품질 관리 중요"
        elif sensitivity_score > 0.1:
            interpretation = "노이즈에 적당히 민감. 일부 노이즈 제거 권장"
        else:
            interpretation = "노이즈에 강건함. 현재 데이터 품질로 충분"

        result = SensitivityResult(
            parameter_name="noise_level",
            parameter_values=noise_levels,
            mean_scores=mean_scores,
            std_scores=std_scores,
            optimal_value=optimal_noise,
            optimal_score=optimal_score,
            sensitivity_score=sensitivity_score,
            interpretation=interpretation
        )

        self.results["noise"] = result
        return result

    def cross_parameter_sensitivity(self, model: BaseEstimator,
                                  parameter_grid: Dict[str, List[Any]],
                                  X_train: np.ndarray, y_train: np.ndarray,
                                  X_test: np.ndarray, y_test: np.ndarray,
                                  cv: int = 3) -> Dict[str, Any]:
        """
        다중 파라미터 교호작용 분석

        Args:
            model: 분석할 모델
            parameter_grid: 파라미터 그리드
            X_train, y_train: 훈련 데이터
            X_test, y_test: 테스트 데이터
            cv: 교차 검증 폴드 수

        Returns:
            교호작용 분석 결과
        """
        print("다중 파라미터 교호작용 분석")

        from sklearn.model_selection import GridSearchCV

        # 그리드 서치로 모든 조합 평가
        grid_search = GridSearchCV(
            model, parameter_grid,
            cv=cv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        # 결과 분석
        results_df = pd.DataFrame(grid_search.cv_results_)

        # 각 파라미터별 주효과 계산
        main_effects = {}
        for param_name in parameter_grid.keys():
            param_scores = results_df.groupby(f'param_{param_name}')['mean_test_score'].mean()
            main_effects[param_name] = {
                'values': param_scores.index.tolist(),
                'scores': (-param_scores).tolist(),  # 음수 점수를 양수로 변환
                'effect_size': np.std(-param_scores)
            }

        # 교호작용 강도 계산
        interaction_strength = {}
        param_names = list(parameter_grid.keys())

        if len(param_names) >= 2:
            for i in range(len(param_names)):
                for j in range(i+1, len(param_names)):
                    param1, param2 = param_names[i], param_names[j]

                    # 2-way 교호작용 분석
                    interaction_scores = results_df.groupby([f'param_{param1}', f'param_{param2}'])['mean_test_score'].mean()

                    # 교호작용 효과 크기 (예측값과 실제값의 차이)
                    predicted_scores = []
                    actual_scores = []

                    for (val1, val2), actual_score in interaction_scores.items():
                        # 주효과로 예측한 점수
                        main1_score = main_effects[param1]['scores'][main_effects[param1]['values'].index(val1)]
                        main2_score = main_effects[param2]['scores'][main_effects[param2]['values'].index(val2)]
                        predicted_score = (main1_score + main2_score) / 2

                        predicted_scores.append(predicted_score)
                        actual_scores.append(-actual_score)

                    interaction_effect = np.std(np.array(actual_scores) - np.array(predicted_scores))
                    interaction_strength[f"{param1}_x_{param2}"] = interaction_effect

        return {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,
            'main_effects': main_effects,
            'interaction_effects': interaction_strength,
            'cv_results': results_df
        }

    def generate_sensitivity_report(self) -> str:
        """민감도 분석 종합 보고서 생성"""
        if not self.results:
            return "민감도 분석 결과가 없습니다."

        report = ["📊 민감도 분석 종합 보고서", "=" * 60, ""]

        # 각 분석별 결과 요약
        for analysis_name, result in self.results.items():
            report.append(f"## {analysis_name.upper()} 민감도 분석")
            report.append("-" * 40)

            if analysis_name == "data_size":
                report.append(f"**최적 데이터 크기**: {result.optimal_value:.1%}")
                report.append(f"**해당 성능**: {result.optimal_score:.6f}")
            else:
                report.append(f"**최적 값**: {result.optimal_value}")
                report.append(f"**최적 성능**: {result.optimal_score:.6f}")

            report.append(f"**민감도 점수**: {result.sensitivity_score:.4f}")
            report.append(f"**해석**: {result.interpretation}")
            report.append("")

            # 성능 변화 요약
            if len(result.mean_scores) > 1:
                min_score = min(result.mean_scores)
                max_score = max(result.mean_scores)
                score_range = max_score - min_score
                relative_range = score_range / min_score * 100 if min_score != 0 else 0

                report.append(f"**성능 변화 범위**: {score_range:.6f} ({relative_range:.1f}%)")
                report.append("")

        # 전체 결론
        report.append("## 🎯 종합 결론")
        report.append("-" * 30)

        # 가장 민감한 요소 찾기
        if self.results:
            most_sensitive = max(self.results.items(), key=lambda x: x[1].sensitivity_score)
            least_sensitive = min(self.results.items(), key=lambda x: x[1].sensitivity_score)

            report.append(f"**가장 민감한 요소**: {most_sensitive[0]} (민감도: {most_sensitive[1].sensitivity_score:.4f})")
            report.append(f"**가장 안정적 요소**: {least_sensitive[0]} (민감도: {least_sensitive[1].sensitivity_score:.4f})")
            report.append("")

        # 권고사항
        report.append("## 💡 권고사항")
        report.append("-" * 20)

        high_sensitivity_items = [name for name, result in self.results.items() if result.sensitivity_score > 0.1]
        low_sensitivity_items = [name for name, result in self.results.items() if result.sensitivity_score < 0.05]

        if high_sensitivity_items:
            report.append("**신중한 튜닝 필요**:")
            for item in high_sensitivity_items:
                report.append(f"  - {item}: {self.results[item].interpretation}")
            report.append("")

        if low_sensitivity_items:
            report.append("**안정적 요소들**:")
            for item in low_sensitivity_items:
                report.append(f"  - {item}: 기본값 사용 가능")
            report.append("")

        report.append("**일반적 권고**:")
        report.append("- 민감도가 높은 파라미터를 우선적으로 튜닝")
        report.append("- 안정적인 파라미터는 기본값 사용으로 복잡성 감소")
        report.append("- 데이터 크기와 노이즈 수준을 고려한 견고한 모델 구축")

        return "\n".join(report)

def main():
    """테스트 및 예제 실행"""
    print("📊 민감도 분석 시스템 테스트")
    print("=" * 60)

    # 시뮬레이션 데이터 생성
    np.random.seed(42)

    n_samples = 500
    n_features = 8
    X = np.random.randn(n_samples, n_features)
    # 약간의 비선형성과 노이즈 추가
    y = (0.3 * X[:, 0] + 0.2 * X[:, 1] ** 2 + 0.1 * X[:, 2] * X[:, 3] +
         np.random.normal(0, 0.1, n_samples))

    # 훈련/테스트 분할
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 민감도 분석기 초기화
    analyzer = SensitivityAnalyzer(random_state=42)

    # 테스트 모델 (Random Forest)
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(random_state=42)

    # 1. 하이퍼파라미터 민감도 분석
    print("1. n_estimators 민감도 분석")
    estimators_result = analyzer.hyperparameter_sensitivity(
        model=model,
        parameter_name='n_estimators',
        parameter_range=[10, 25, 50, 100, 200, 300],
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        cv=3
    )

    print(f"최적값: {estimators_result.optimal_value}")
    print(f"민감도: {estimators_result.sensitivity_score:.4f}")
    print(f"해석: {estimators_result.interpretation}")
    print()

    # 2. max_depth 민감도 분석
    print("2. max_depth 민감도 분석")
    depth_result = analyzer.hyperparameter_sensitivity(
        model=model,
        parameter_name='max_depth',
        parameter_range=[3, 5, 7, 10, 15, 20, None],
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        cv=3
    )

    print(f"최적값: {depth_result.optimal_value}")
    print(f"민감도: {depth_result.sensitivity_score:.4f}")
    print()

    # 3. 데이터 크기 민감도 분석
    print("3. 데이터 크기 민감도 분석")
    data_size_result = analyzer.data_size_sensitivity(
        model=model,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        cv=3
    )

    print(f"최적 데이터 크기: {data_size_result.optimal_value:.1%}")
    print(f"해석: {data_size_result.interpretation}")
    print()

    # 4. 특징 중요도 민감도 분석 (일부 특징만)
    print("4. 특징 중요도 민감도 분석")
    feature_names = [f"feature_{i}" for i in range(min(4, n_features))]  # 처음 4개 특징만
    X_train_subset = X_train[:, :len(feature_names)]
    X_test_subset = X_test[:, :len(feature_names)]

    feature_results = analyzer.feature_importance_sensitivity(
        model=RandomForestRegressor(n_estimators=50, random_state=42),
        X_train=X_train_subset, y_train=y_train,
        X_test=X_test_subset, y_test=y_test,
        feature_names=feature_names,
        n_permutations=5
    )

    for feature, result in feature_results.items():
        print(f"{feature}: {result.interpretation}")

    print()

    # 5. 노이즈 민감도 분석
    print("5. 노이즈 민감도 분석")
    noise_result = analyzer.noise_sensitivity(
        model=RandomForestRegressor(n_estimators=50, random_state=42),
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        noise_levels=[0.0, 0.05, 0.1, 0.2],
        n_trials=3
    )

    print(f"최적 노이즈 수준: {noise_result.optimal_value}")
    print(f"해석: {noise_result.interpretation}")
    print()

    # 6. 다중 파라미터 교호작용 분석
    print("6. 다중 파라미터 교호작용 분석")
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None]
    }

    interaction_results = analyzer.cross_parameter_sensitivity(
        model=model,
        parameter_grid=param_grid,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        cv=3
    )

    print(f"최적 파라미터: {interaction_results['best_params']}")
    print(f"최고 성능: {interaction_results['best_score']:.6f}")

    if interaction_results['interaction_effects']:
        for interaction, effect in interaction_results['interaction_effects'].items():
            print(f"교호작용 {interaction}: {effect:.6f}")

    print()

    # 종합 보고서 생성
    print("📋 민감도 분석 종합 보고서:")
    print("=" * 60)
    report = analyzer.generate_sensitivity_report()
    print(report)

if __name__ == "__main__":
    main()