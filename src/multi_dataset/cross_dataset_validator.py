"""
다중 데이터셋 검증 시스템

이 모듈은 기계학습 모델의 일반화 성능을 검증하기 위해
다양한 데이터셋에서 체계적인 평가를 수행합니다.
도메인 간 성능 분석, 전이학습 효과, 메타 분석 등을 지원합니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
from datetime import datetime
import warnings
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
warnings.filterwarnings('ignore')

@dataclass
class DatasetInfo:
    """데이터셋 정보"""
    name: str
    domain: str
    task_type: str  # classification, regression
    n_samples: int
    n_features: int
    n_classes: Optional[int] = None
    class_distribution: Optional[Dict[str, float]] = None
    feature_types: Optional[Dict[str, str]] = None
    missing_values: float = 0.0
    outlier_percentage: float = 0.0
    data_quality_score: float = 1.0

@dataclass
class ModelPerformanceRecord:
    """모델 성능 기록"""
    model_name: str
    dataset_name: str
    metrics: Dict[str, float]
    training_time: float
    prediction_time: float
    cross_val_scores: List[float]
    feature_importance: Optional[Dict[str, float]] = None
    hyperparameters: Optional[Dict[str, Any]] = None

@dataclass
class TransferLearningResult:
    """전이학습 결과"""
    source_dataset: str
    target_dataset: str
    model_name: str
    baseline_performance: Dict[str, float]
    transfer_performance: Dict[str, float]
    improvement: Dict[str, float]
    transfer_effectiveness: float
    domain_similarity: float

@dataclass
class GeneralizationAnalysis:
    """일반화 분석 결과"""
    model_name: str
    dataset_performances: Dict[str, Dict[str, float]]
    mean_performance: Dict[str, float]
    std_performance: Dict[str, float]
    cv_performance: Dict[str, float]  # Coefficient of Variation
    worst_case_performance: Dict[str, float]
    best_case_performance: Dict[str, float]
    generalization_score: float
    stability_score: float

@dataclass
class MetaAnalysisResult:
    """메타 분석 결과"""
    models_compared: List[str]
    datasets_analyzed: List[str]
    aggregate_performance: Dict[str, Dict[str, float]]
    statistical_significance: Dict[str, Dict[str, float]]
    effect_sizes: Dict[str, Dict[str, float]]
    confidence_intervals: Dict[str, Dict[str, Tuple[float, float]]]
    ranking: Dict[str, List[Tuple[str, float]]]
    heterogeneity_analysis: Dict[str, float]

@dataclass
class CrossDatasetValidationReport:
    """다중 데이터셋 검증 보고서"""
    validation_id: str
    validation_timestamp: str
    datasets_info: List[DatasetInfo]
    model_performances: List[ModelPerformanceRecord]
    transfer_learning_results: List[TransferLearningResult]
    generalization_analyses: List[GeneralizationAnalysis]
    meta_analysis: MetaAnalysisResult
    domain_analysis: Dict[str, Any]
    recommendations: List[str]
    overall_assessment: str

class CrossDatasetValidator:
    """다중 데이터셋 검증 시스템"""

    def __init__(self, output_dir: str = "results/multi_dataset_validation"):
        """
        초기화

        Args:
            output_dir: 결과 저장 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 서브 디렉토리 생성
        (self.output_dir / "performance_records").mkdir(exist_ok=True)
        (self.output_dir / "transfer_learning").mkdir(exist_ok=True)
        (self.output_dir / "meta_analysis").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)

        # 메트릭 함수 설정
        self.classification_metrics = {
            'accuracy': accuracy_score,
            'precision': lambda y_true, y_pred: self._safe_precision(y_true, y_pred),
            'recall': lambda y_true, y_pred: self._safe_recall(y_true, y_pred),
            'f1': lambda y_true, y_pred: self._safe_f1(y_true, y_pred)
        }

        self.regression_metrics = {
            'mse': mean_squared_error,
            'mae': mean_absolute_error,
            'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': lambda y_true, y_pred: self._safe_mape(y_true, y_pred)
        }

    def analyze_dataset(self, data: pd.DataFrame, target_column: str,
                       dataset_name: str, domain: str) -> DatasetInfo:
        """
        데이터셋 분석

        Args:
            data: 데이터셋
            target_column: 타겟 컬럼명
            dataset_name: 데이터셋 이름
            domain: 도메인

        Returns:
            데이터셋 정보
        """
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # 기본 정보
        n_samples, n_features = X.shape

        # 작업 유형 판단
        task_type = 'classification' if self._is_classification(y) else 'regression'

        # 클래스 분포 (분류인 경우)
        n_classes = None
        class_distribution = None
        if task_type == 'classification':
            class_counts = y.value_counts()
            n_classes = len(class_counts)
            class_distribution = (class_counts / len(y)).to_dict()

        # 특성 타입 분석
        feature_types = {}
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                feature_types[col] = 'numeric'
            elif X[col].dtype == 'object':
                feature_types[col] = 'categorical'
            else:
                feature_types[col] = 'other'

        # 결측값 비율
        missing_values = X.isnull().sum().sum() / (n_samples * n_features)

        # 이상치 비율 (간단한 IQR 방법)
        outlier_percentage = self._calculate_outlier_percentage(X)

        # 데이터 품질 점수
        data_quality_score = self._calculate_data_quality_score(
            missing_values, outlier_percentage, n_samples, n_features
        )

        return DatasetInfo(
            name=dataset_name,
            domain=domain,
            task_type=task_type,
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            class_distribution=class_distribution,
            feature_types=feature_types,
            missing_values=missing_values,
            outlier_percentage=outlier_percentage,
            data_quality_score=data_quality_score
        )

    def evaluate_model_on_dataset(self,
                                model: Any,
                                data: pd.DataFrame,
                                target_column: str,
                                dataset_info: DatasetInfo,
                                cv_folds: int = 5) -> ModelPerformanceRecord:
        """
        특정 데이터셋에서 모델 성능 평가

        Args:
            model: 평가할 모델
            data: 데이터셋
            target_column: 타겟 컬럼명
            dataset_info: 데이터셋 정보
            cv_folds: 교차 검증 폴드 수

        Returns:
            모델 성능 기록
        """
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # 데이터 전처리
        X_processed, y_processed = self._preprocess_data(X, y, dataset_info.task_type)

        # 모델 훈련 시간 측정
        start_time = datetime.now()
        model.fit(X_processed, y_processed)
        training_time = (datetime.now() - start_time).total_seconds()

        # 예측 시간 측정
        start_time = datetime.now()
        y_pred = model.predict(X_processed)
        prediction_time = (datetime.now() - start_time).total_seconds()

        # 메트릭 계산
        metrics = {}
        if dataset_info.task_type == 'classification':
            for metric_name, metric_func in self.classification_metrics.items():
                try:
                    metrics[metric_name] = metric_func(y_processed, y_pred)
                except:
                    metrics[metric_name] = 0.0
        else:
            for metric_name, metric_func in self.regression_metrics.items():
                try:
                    metrics[metric_name] = metric_func(y_processed, y_pred)
                except:
                    metrics[metric_name] = float('inf')

        # 교차 검증
        cv_scores = []
        try:
            primary_metric = 'accuracy' if dataset_info.task_type == 'classification' else 'neg_mean_squared_error'
            cv_scores = cross_val_score(model, X_processed, y_processed, cv=cv_folds, scoring=primary_metric)
            cv_scores = cv_scores.tolist()
        except:
            cv_scores = [0.0] * cv_folds

        # 특성 중요도 (가능한 경우)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            if len(model.coef_.shape) == 1:
                feature_importance = dict(zip(X.columns, np.abs(model.coef_)))
            else:
                feature_importance = dict(zip(X.columns, np.abs(model.coef_).mean(axis=0)))

        # 하이퍼파라미터
        hyperparameters = None
        if hasattr(model, 'get_params'):
            hyperparameters = model.get_params()

        return ModelPerformanceRecord(
            model_name=model.__class__.__name__,
            dataset_name=dataset_info.name,
            metrics=metrics,
            training_time=training_time,
            prediction_time=prediction_time,
            cross_val_scores=cv_scores,
            feature_importance=feature_importance,
            hyperparameters=hyperparameters
        )

    def evaluate_transfer_learning(self,
                                 model: Any,
                                 source_data: pd.DataFrame,
                                 target_data: pd.DataFrame,
                                 target_column: str,
                                 source_info: DatasetInfo,
                                 target_info: DatasetInfo) -> TransferLearningResult:
        """
        전이학습 효과 평가

        Args:
            model: 평가할 모델
            source_data: 소스 데이터셋
            target_data: 타겟 데이터셋
            target_column: 타겟 컬럼명
            source_info: 소스 데이터셋 정보
            target_info: 타겟 데이터셋 정보

        Returns:
            전이학습 결과
        """
        # 타겟 데이터셋에서 직접 훈련 (베이스라인)
        baseline_performance = self.evaluate_model_on_dataset(
            model, target_data, target_column, target_info
        )

        # 소스 데이터셋에서 사전 훈련 후 타겟 데이터셋으로 전이
        X_source = source_data.drop(columns=[target_column])
        y_source = source_data[target_column]
        X_target = target_data.drop(columns=[target_column])
        y_target = target_data[target_column]

        # 데이터 전처리
        X_source_processed, y_source_processed = self._preprocess_data(
            X_source, y_source, source_info.task_type
        )

        # 소스에서 사전 훈련
        model.fit(X_source_processed, y_source_processed)

        # 타겟에서 파인 튜닝 (일부 데이터만 사용)
        transfer_model = self._clone_model(model)
        sample_size = min(len(target_data), len(source_data) // 4)  # 타겟 데이터의 일부만 사용
        target_sample = target_data.sample(n=sample_size, random_state=42)

        transfer_performance = self.evaluate_model_on_dataset(
            transfer_model, target_sample, target_column, target_info
        )

        # 개선도 계산
        improvement = {}
        for metric in baseline_performance.metrics:
            baseline_val = baseline_performance.metrics[metric]
            transfer_val = transfer_performance.metrics[metric]

            if target_info.task_type == 'classification':
                # 높을수록 좋은 메트릭
                improvement[metric] = transfer_val - baseline_val
            else:
                # 낮을수록 좋은 메트릭 (오차)
                if baseline_val == 0:
                    improvement[metric] = 0
                else:
                    improvement[metric] = (baseline_val - transfer_val) / baseline_val

        # 전이 효과성
        primary_metric = 'accuracy' if target_info.task_type == 'classification' else 'mse'
        transfer_effectiveness = improvement.get(primary_metric, 0)

        # 도메인 유사성 계산
        domain_similarity = self._calculate_domain_similarity(source_info, target_info)

        return TransferLearningResult(
            source_dataset=source_info.name,
            target_dataset=target_info.name,
            model_name=model.__class__.__name__,
            baseline_performance=baseline_performance.metrics,
            transfer_performance=transfer_performance.metrics,
            improvement=improvement,
            transfer_effectiveness=transfer_effectiveness,
            domain_similarity=domain_similarity
        )

    def analyze_generalization(self,
                             model_name: str,
                             performance_records: List[ModelPerformanceRecord]) -> GeneralizationAnalysis:
        """
        일반화 성능 분석

        Args:
            model_name: 모델 이름
            performance_records: 성능 기록들

        Returns:
            일반화 분석 결과
        """
        # 해당 모델의 성능만 필터링
        model_records = [r for r in performance_records if r.model_name == model_name]

        if not model_records:
            raise ValueError(f"No performance records found for model: {model_name}")

        # 데이터셋별 성능 정리
        dataset_performances = {}
        for record in model_records:
            dataset_performances[record.dataset_name] = record.metrics

        # 메트릭별 통계 계산
        all_metrics = set()
        for metrics in dataset_performances.values():
            all_metrics.update(metrics.keys())

        mean_performance = {}
        std_performance = {}
        cv_performance = {}
        worst_case_performance = {}
        best_case_performance = {}

        for metric in all_metrics:
            values = [metrics.get(metric, 0) for metrics in dataset_performances.values()]
            values = [v for v in values if v != float('inf') and not np.isnan(v)]

            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)

                mean_performance[metric] = mean_val
                std_performance[metric] = std_val
                cv_performance[metric] = std_val / mean_val if mean_val != 0 else float('inf')

                # 분류/회귀에 따라 worst/best 판단
                if metric in ['accuracy', 'precision', 'recall', 'f1']:
                    worst_case_performance[metric] = min(values)
                    best_case_performance[metric] = max(values)
                else:
                    worst_case_performance[metric] = max(values)
                    best_case_performance[metric] = min(values)

        # 일반화 점수 계산
        generalization_score = self._calculate_generalization_score(dataset_performances)

        # 안정성 점수 계산
        stability_score = self._calculate_stability_score(cv_performance)

        return GeneralizationAnalysis(
            model_name=model_name,
            dataset_performances=dataset_performances,
            mean_performance=mean_performance,
            std_performance=std_performance,
            cv_performance=cv_performance,
            worst_case_performance=worst_case_performance,
            best_case_performance=best_case_performance,
            generalization_score=generalization_score,
            stability_score=stability_score
        )

    def perform_meta_analysis(self,
                            performance_records: List[ModelPerformanceRecord],
                            datasets_info: List[DatasetInfo]) -> MetaAnalysisResult:
        """
        메타 분석 수행

        Args:
            performance_records: 성능 기록들
            datasets_info: 데이터셋 정보들

        Returns:
            메타 분석 결과
        """
        # 모델과 데이터셋 목록
        models_compared = list(set(r.model_name for r in performance_records))
        datasets_analyzed = list(set(r.dataset_name for r in performance_records))

        # 모델별 메트릭별 성능 집계
        aggregate_performance = {}
        for model in models_compared:
            model_records = [r for r in performance_records if r.model_name == model]
            aggregate_performance[model] = {}

            # 메트릭별 평균 계산
            all_metrics = set()
            for record in model_records:
                all_metrics.update(record.metrics.keys())

            for metric in all_metrics:
                values = []
                for record in model_records:
                    if metric in record.metrics:
                        val = record.metrics[metric]
                        if val != float('inf') and not np.isnan(val):
                            values.append(val)

                if values:
                    aggregate_performance[model][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'count': len(values)
                    }

        # 통계적 유의성 검정
        statistical_significance = {}
        effect_sizes = {}
        confidence_intervals = {}

        for metric in all_metrics:
            statistical_significance[metric] = {}
            effect_sizes[metric] = {}
            confidence_intervals[metric] = {}

            # 모든 모델 쌍에 대해 비교
            for i, model1 in enumerate(models_compared):
                for model2 in models_compared[i+1:]:
                    # 데이터 추출
                    model1_values = []
                    model2_values = []

                    for record in performance_records:
                        if record.model_name == model1 and metric in record.metrics:
                            val = record.metrics[metric]
                            if val != float('inf') and not np.isnan(val):
                                model1_values.append(val)

                        if record.model_name == model2 and metric in record.metrics:
                            val = record.metrics[metric]
                            if val != float('inf') and not np.isnan(val):
                                model2_values.append(val)

                    # 통계 검정
                    if len(model1_values) >= 3 and len(model2_values) >= 3:
                        try:
                            t_stat, p_value = stats.ttest_ind(model1_values, model2_values)
                            statistical_significance[metric][f"{model1}_vs_{model2}"] = p_value

                            # 효과 크기 (Cohen's d)
                            cohens_d = self._calculate_cohens_d(model1_values, model2_values)
                            effect_sizes[metric][f"{model1}_vs_{model2}"] = cohens_d

                            # 신뢰구간
                            diff_mean = np.mean(model1_values) - np.mean(model2_values)
                            se_diff = np.sqrt(np.var(model1_values)/len(model1_values) +
                                            np.var(model2_values)/len(model2_values))
                            t_critical = stats.t.ppf(0.975, len(model1_values) + len(model2_values) - 2)
                            ci_lower = diff_mean - t_critical * se_diff
                            ci_upper = diff_mean + t_critical * se_diff
                            confidence_intervals[metric][f"{model1}_vs_{model2}"] = (ci_lower, ci_upper)

                        except:
                            statistical_significance[metric][f"{model1}_vs_{model2}"] = 1.0
                            effect_sizes[metric][f"{model1}_vs_{model2}"] = 0.0
                            confidence_intervals[metric][f"{model1}_vs_{model2}"] = (0.0, 0.0)

        # 모델 순위
        ranking = {}
        for metric in all_metrics:
            model_scores = []
            for model in models_compared:
                if model in aggregate_performance and metric in aggregate_performance[model]:
                    score = aggregate_performance[model][metric]['mean']
                    model_scores.append((model, score))

            # 메트릭 타입에 따라 정렬
            if metric in ['accuracy', 'precision', 'recall', 'f1']:
                model_scores.sort(key=lambda x: x[1], reverse=True)  # 높을수록 좋음
            else:
                model_scores.sort(key=lambda x: x[1])  # 낮을수록 좋음

            ranking[metric] = model_scores

        # 이질성 분석 (I² statistic)
        heterogeneity_analysis = {}
        for metric in all_metrics:
            heterogeneity_analysis[metric] = self._calculate_heterogeneity(
                performance_records, metric
            )

        return MetaAnalysisResult(
            models_compared=models_compared,
            datasets_analyzed=datasets_analyzed,
            aggregate_performance=aggregate_performance,
            statistical_significance=statistical_significance,
            effect_sizes=effect_sizes,
            confidence_intervals=confidence_intervals,
            ranking=ranking,
            heterogeneity_analysis=heterogeneity_analysis
        )

    def analyze_domain_characteristics(self, datasets_info: List[DatasetInfo]) -> Dict[str, Any]:
        """
        도메인 특성 분석

        Args:
            datasets_info: 데이터셋 정보들

        Returns:
            도메인 분석 결과
        """
        domain_analysis = {}

        # 도메인별 그룹화
        domains = {}
        for dataset in datasets_info:
            if dataset.domain not in domains:
                domains[dataset.domain] = []
            domains[dataset.domain].append(dataset)

        # 도메인별 특성 분석
        for domain, datasets in domains.items():
            domain_stats = {
                'n_datasets': len(datasets),
                'avg_samples': np.mean([d.n_samples for d in datasets]),
                'avg_features': np.mean([d.n_features for d in datasets]),
                'task_types': list(set(d.task_type for d in datasets)),
                'avg_data_quality': np.mean([d.data_quality_score for d in datasets]),
                'datasets': [d.name for d in datasets]
            }

            # 분류 문제인 경우 클래스 수 분석
            classification_datasets = [d for d in datasets if d.task_type == 'classification']
            if classification_datasets:
                domain_stats['avg_classes'] = np.mean([d.n_classes for d in classification_datasets])

            domain_analysis[domain] = domain_stats

        # 도메인 간 유사성 매트릭스
        domain_names = list(domains.keys())
        similarity_matrix = np.zeros((len(domain_names), len(domain_names)))

        for i, domain1 in enumerate(domain_names):
            for j, domain2 in enumerate(domain_names):
                if i <= j:
                    # 도메인 간 유사성 계산
                    datasets1 = domains[domain1]
                    datasets2 = domains[domain2]
                    similarity = self._calculate_domain_group_similarity(datasets1, datasets2)
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity

        domain_analysis['similarity_matrix'] = {
            'domains': domain_names,
            'matrix': similarity_matrix.tolist()
        }

        return domain_analysis

    def generate_comprehensive_report(self,
                                    validation_id: str,
                                    datasets_info: List[DatasetInfo],
                                    performance_records: List[ModelPerformanceRecord],
                                    transfer_learning_results: List[TransferLearningResult],
                                    generalization_analyses: List[GeneralizationAnalysis],
                                    meta_analysis: MetaAnalysisResult,
                                    domain_analysis: Dict[str, Any]) -> CrossDatasetValidationReport:
        """
        종합 검증 보고서 생성

        Args:
            validation_id: 검증 ID
            datasets_info: 데이터셋 정보들
            performance_records: 성능 기록들
            transfer_learning_results: 전이학습 결과들
            generalization_analyses: 일반화 분석 결과들
            meta_analysis: 메타 분석 결과
            domain_analysis: 도메인 분석 결과

        Returns:
            종합 검증 보고서
        """
        # 권장사항 생성
        recommendations = self._generate_recommendations(
            datasets_info, performance_records, generalization_analyses, meta_analysis
        )

        # 전체 평가
        overall_assessment = self._generate_overall_assessment(
            generalization_analyses, meta_analysis, domain_analysis
        )

        return CrossDatasetValidationReport(
            validation_id=validation_id,
            validation_timestamp=datetime.now().isoformat(),
            datasets_info=datasets_info,
            model_performances=performance_records,
            transfer_learning_results=transfer_learning_results,
            generalization_analyses=generalization_analyses,
            meta_analysis=meta_analysis,
            domain_analysis=domain_analysis,
            recommendations=recommendations,
            overall_assessment=overall_assessment
        )

    # 보조 메서드들
    def _is_classification(self, y: pd.Series) -> bool:
        """분류 문제인지 판단"""
        return y.dtype == 'object' or len(y.unique()) < 20

    def _preprocess_data(self, X: pd.DataFrame, y: pd.Series,
                        task_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """데이터 전처리"""
        # 수치형 컬럼만 선택 (간단한 전처리)
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_columns].fillna(0)

        # 표준화
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X_numeric)

        # 타겟 변수 인코딩 (분류인 경우)
        if task_type == 'classification' and y.dtype == 'object':
            le = LabelEncoder()
            y_processed = le.fit_transform(y)
        else:
            y_processed = y.values

        return X_processed, y_processed

    def _safe_precision(self, y_true, y_pred):
        """안전한 정밀도 계산"""
        try:
            from sklearn.metrics import precision_score
            return precision_score(y_true, y_pred, average='weighted', zero_division=0)
        except:
            return 0.0

    def _safe_recall(self, y_true, y_pred):
        """안전한 재현율 계산"""
        try:
            from sklearn.metrics import recall_score
            return recall_score(y_true, y_pred, average='weighted', zero_division=0)
        except:
            return 0.0

    def _safe_f1(self, y_true, y_pred):
        """안전한 F1 점수 계산"""
        try:
            from sklearn.metrics import f1_score
            return f1_score(y_true, y_pred, average='weighted', zero_division=0)
        except:
            return 0.0

    def _safe_mape(self, y_true, y_pred):
        """안전한 MAPE 계산"""
        try:
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            non_zero_mask = y_true != 0
            if not np.any(non_zero_mask):
                return float('inf')
            return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        except:
            return float('inf')

    def _calculate_outlier_percentage(self, X: pd.DataFrame) -> float:
        """이상치 비율 계산"""
        try:
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                return 0.0

            total_outliers = 0
            total_values = 0

            for col in numeric_columns:
                data = X[col].dropna()
                if len(data) > 0:
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers = data[(data < lower_bound) | (data > upper_bound)]
                    total_outliers += len(outliers)
                    total_values += len(data)

            return total_outliers / total_values if total_values > 0 else 0.0
        except:
            return 0.0

    def _calculate_data_quality_score(self, missing_values: float, outlier_percentage: float,
                                    n_samples: int, n_features: int) -> float:
        """데이터 품질 점수 계산"""
        score = 1.0

        # 결측값 패널티
        score -= missing_values * 0.5

        # 이상치 패널티
        score -= outlier_percentage * 0.3

        # 표본 크기 보너스/패널티
        if n_samples < 100:
            score -= 0.2
        elif n_samples > 1000:
            score += 0.1

        # 특성 수 적절성
        if n_features / n_samples > 0.1:  # 차원의 저주
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _clone_model(self, model):
        """모델 복제"""
        try:
            from sklearn.base import clone
            return clone(model)
        except:
            # 간단한 폴백
            return model.__class__(**model.get_params() if hasattr(model, 'get_params') else {})

    def _calculate_domain_similarity(self, dataset1: DatasetInfo, dataset2: DatasetInfo) -> float:
        """도메인 유사성 계산"""
        similarity = 0.0

        # 작업 유형 일치
        if dataset1.task_type == dataset2.task_type:
            similarity += 0.3

        # 샘플 크기 유사성
        size_ratio = min(dataset1.n_samples, dataset2.n_samples) / max(dataset1.n_samples, dataset2.n_samples)
        similarity += size_ratio * 0.2

        # 특성 수 유사성
        feature_ratio = min(dataset1.n_features, dataset2.n_features) / max(dataset1.n_features, dataset2.n_features)
        similarity += feature_ratio * 0.2

        # 데이터 품질 유사성
        quality_diff = abs(dataset1.data_quality_score - dataset2.data_quality_score)
        similarity += (1 - quality_diff) * 0.15

        # 도메인 일치
        if dataset1.domain == dataset2.domain:
            similarity += 0.15

        return min(1.0, similarity)

    def _calculate_generalization_score(self, dataset_performances: Dict[str, Dict[str, float]]) -> float:
        """일반화 점수 계산"""
        if len(dataset_performances) < 2:
            return 1.0

        # 주요 메트릭 선택
        primary_metrics = ['accuracy', 'f1', 'mse', 'mae']
        available_metrics = set()
        for metrics in dataset_performances.values():
            available_metrics.update(metrics.keys())

        selected_metrics = [m for m in primary_metrics if m in available_metrics]
        if not selected_metrics:
            selected_metrics = list(available_metrics)[:3]

        generalization_scores = []
        for metric in selected_metrics:
            values = []
            for dataset_metrics in dataset_performances.values():
                if metric in dataset_metrics:
                    val = dataset_metrics[metric]
                    if val != float('inf') and not np.isnan(val):
                        values.append(val)

            if len(values) >= 2:
                cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else float('inf')
                # 낮은 변이계수가 좋은 일반화를 의미
                score = max(0, 1 - cv)
                generalization_scores.append(score)

        return np.mean(generalization_scores) if generalization_scores else 0.5

    def _calculate_stability_score(self, cv_performance: Dict[str, float]) -> float:
        """안정성 점수 계산"""
        if not cv_performance:
            return 0.5

        # 변이계수가 낮을수록 안정적
        stable_scores = []
        for cv in cv_performance.values():
            if cv != float('inf') and not np.isnan(cv):
                # CV < 0.1이면 매우 안정적, CV > 0.5면 불안정
                score = max(0, min(1, 1 - cv / 0.5))
                stable_scores.append(score)

        return np.mean(stable_scores) if stable_scores else 0.5

    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Cohen's d 효과 크기 계산"""
        try:
            n1, n2 = len(group1), len(group2)
            mean1, mean2 = np.mean(group1), np.mean(group2)
            std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
        except:
            return 0.0

    def _calculate_heterogeneity(self, performance_records: List[ModelPerformanceRecord],
                               metric: str) -> float:
        """이질성 지수 계산 (I²)"""
        try:
            # 메트릭별 분산 분석
            values = []
            for record in performance_records:
                if metric in record.metrics:
                    val = record.metrics[metric]
                    if val != float('inf') and not np.isnan(val):
                        values.append(val)

            if len(values) < 3:
                return 0.0

            # 간단한 이질성 지수 (분산 기반)
            total_variance = np.var(values)
            within_group_variance = np.mean([np.var(record.cross_val_scores) for record in performance_records
                                           if len(record.cross_val_scores) > 1])

            if within_group_variance == 0:
                return 1.0

            i_squared = max(0, (total_variance - within_group_variance) / total_variance)
            return i_squared
        except:
            return 0.0

    def _calculate_domain_group_similarity(self, datasets1: List[DatasetInfo],
                                         datasets2: List[DatasetInfo]) -> float:
        """도메인 그룹 간 유사성 계산"""
        if not datasets1 or not datasets2:
            return 0.0

        similarities = []
        for d1 in datasets1:
            for d2 in datasets2:
                similarities.append(self._calculate_domain_similarity(d1, d2))

        return np.mean(similarities)

    def _generate_recommendations(self, datasets_info: List[DatasetInfo],
                                performance_records: List[ModelPerformanceRecord],
                                generalization_analyses: List[GeneralizationAnalysis],
                                meta_analysis: MetaAnalysisResult) -> List[str]:
        """권장사항 생성"""
        recommendations = []

        # 일반화 성능 기반 권장사항
        for analysis in generalization_analyses:
            if analysis.generalization_score < 0.7:
                recommendations.append(
                    f"{analysis.model_name} 모델의 일반화 성능이 낮습니다. "
                    "정규화 기법이나 앙상블 방법을 고려하세요."
                )

            if analysis.stability_score < 0.6:
                recommendations.append(
                    f"{analysis.model_name} 모델의 안정성이 부족합니다. "
                    "하이퍼파라미터 튜닝이나 교차 검증 전략을 재검토하세요."
                )

        # 데이터셋 품질 기반 권장사항
        low_quality_datasets = [d for d in datasets_info if d.data_quality_score < 0.7]
        if low_quality_datasets:
            recommendations.append(
                f"일부 데이터셋({', '.join([d.name for d in low_quality_datasets])})의 "
                "품질이 낮습니다. 전처리 강화를 권장합니다."
            )

        # 메타 분석 기반 권장사항
        for metric, rankings in meta_analysis.ranking.items():
            if rankings:
                best_model = rankings[0][0]
                recommendations.append(
                    f"{metric} 기준으로 {best_model} 모델이 가장 우수한 성능을 보입니다."
                )

        return recommendations

    def _generate_overall_assessment(self, generalization_analyses: List[GeneralizationAnalysis],
                                   meta_analysis: MetaAnalysisResult,
                                   domain_analysis: Dict[str, Any]) -> str:
        """전체 평가 생성"""
        # 평균 일반화 점수
        avg_generalization = np.mean([a.generalization_score for a in generalization_analyses])

        # 평균 안정성 점수
        avg_stability = np.mean([a.stability_score for a in generalization_analyses])

        # 도메인 다양성
        n_domains = len(domain_analysis) - 1  # similarity_matrix 제외

        assessment = f"다중 데이터셋 검증 결과: "

        if avg_generalization >= 0.8:
            assessment += "우수한 일반화 성능을 보입니다. "
        elif avg_generalization >= 0.6:
            assessment += "양호한 일반화 성능을 보입니다. "
        else:
            assessment += "일반화 성능 개선이 필요합니다. "

        if avg_stability >= 0.8:
            assessment += "높은 안정성을 나타냅니다. "
        elif avg_stability >= 0.6:
            assessment += "적절한 안정성을 보입니다. "
        else:
            assessment += "안정성 향상이 필요합니다. "

        assessment += f"{n_domains}개 도메인에서 {len(meta_analysis.models_compared)}개 모델을 비교했습니다."

        return assessment

    def save_validation_report(self, report: CrossDatasetValidationReport) -> str:
        """검증 보고서 저장"""
        filename = f"cross_dataset_validation_{report.validation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / "reports" / filename

        # dataclass를 dict로 변환
        report_dict = self._dataclass_to_dict(report)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

        return str(filepath)

    def _dataclass_to_dict(self, obj) -> Any:
        """dataclass를 딕셔너리로 변환"""
        if hasattr(obj, '__dict__'):
            result = {}
            for key, value in obj.__dict__.items():
                if hasattr(value, '__dict__'):
                    result[key] = self._dataclass_to_dict(value)
                elif isinstance(value, list):
                    result[key] = [self._dataclass_to_dict(item) if hasattr(item, '__dict__') else item
                                 for item in value]
                elif isinstance(value, dict):
                    result[key] = {k: self._dataclass_to_dict(v) if hasattr(v, '__dict__') else v
                                 for k, v in value.items()}
                else:
                    result[key] = value
            return result
        return obj

if __name__ == "__main__":
    # 테스트 예제
    print("Cross-dataset validation system ready!")
    print("This system provides comprehensive multi-dataset validation capabilities.")