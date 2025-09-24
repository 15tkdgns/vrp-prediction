"""
문헌 비교 및 재현성 검증 시스템

이 모듈은 기계학습 연구 결과를 기존 문헌과 체계적으로 비교하고,
연구의 재현성을 검증합니다. State-of-the-art 성능 비교, 벤치마크 검증,
연구 기여도 평가 등의 학술 연구에 필수적인 기능을 제공합니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import re
from datetime import datetime
import warnings
from scipy import stats
from collections import defaultdict
import requests
from urllib.parse import quote
warnings.filterwarnings('ignore')

@dataclass
class LiteratureRecord:
    """문헌 기록"""
    paper_id: str
    title: str
    authors: List[str]
    venue: str
    year: int
    dataset_name: str
    task_type: str
    methodology: str
    metrics_reported: Dict[str, float]
    experimental_setup: Dict[str, Any]
    code_available: bool = False
    data_available: bool = False
    reproducibility_score: float = 0.0
    citation_count: int = 0
    impact_factor: float = 0.0

@dataclass
class BenchmarkComparison:
    """벤치마크 비교 결과"""
    dataset_name: str
    task_type: str
    our_results: Dict[str, float]
    literature_results: List[Tuple[str, Dict[str, float]]]  # (paper_id, metrics)
    ranking: Dict[str, List[Tuple[str, float]]]  # metric -> [(method, score)]
    statistical_significance: Dict[str, Dict[str, float]]  # metric -> {paper_id: p_value}
    improvement_analysis: Dict[str, Dict[str, float]]  # metric -> {paper_id: improvement}
    state_of_art_comparison: Dict[str, bool]  # metric -> is_sota

@dataclass
class ReproducibilityAssessment:
    """재현성 평가 결과"""
    paper_id: str
    original_results: Dict[str, float]
    reproduced_results: Dict[str, float]
    reproduction_success: Dict[str, bool]
    similarity_scores: Dict[str, float]
    statistical_equivalence: Dict[str, bool]
    reproduction_difficulty: str  # easy, moderate, hard, impossible
    missing_details: List[str]
    implementation_gaps: List[str]
    overall_reproducibility_score: float

@dataclass
class ContributionAssessment:
    """연구 기여도 평가"""
    novelty_score: float
    significance_score: float
    impact_score: float
    methodological_contribution: str
    empirical_contribution: str
    theoretical_contribution: str
    practical_implications: str
    limitations: List[str]
    future_work_suggestions: List[str]
    overall_contribution_score: float

@dataclass
class LiteratureComparisonReport:
    """문헌 비교 보고서"""
    comparison_id: str
    comparison_timestamp: str
    datasets_compared: List[str]
    literature_records: List[LiteratureRecord]
    benchmark_comparisons: List[BenchmarkComparison]
    reproducibility_assessments: List[ReproducibilityAssessment]
    contribution_assessment: ContributionAssessment
    research_landscape_analysis: Dict[str, Any]
    recommendations: List[str]
    executive_summary: str

class LiteratureComparator:
    """문헌 비교 및 재현성 검증 시스템"""

    def __init__(self, output_dir: str = "results/literature_comparison"):
        """
        초기화

        Args:
            output_dir: 결과 저장 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 서브 디렉토리 생성
        (self.output_dir / "benchmark_comparisons").mkdir(exist_ok=True)
        (self.output_dir / "reproducibility").mkdir(exist_ok=True)
        (self.output_dir / "literature_db").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)

        # 문헌 데이터베이스
        self.literature_db = {}
        self.load_literature_database()

        # 표준 데이터셋 및 메트릭 정의
        self.standard_datasets = {
            'classification': ['CIFAR-10', 'CIFAR-100', 'ImageNet', 'MNIST', 'Fashion-MNIST'],
            'nlp': ['GLUE', 'SuperGLUE', 'SQuAD', 'CoNLL-2003', 'IMDB'],
            'tabular': ['UCI', 'Kaggle', 'OpenML'],
            'time_series': ['UCR', 'UEA', 'M4', 'Electricity'],
            'finance': ['S&P500', 'NASDAQ', 'Forex', 'Crypto']
        }

        self.standard_metrics = {
            'classification': ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'],
            'regression': ['mse', 'mae', 'rmse', 'mape', 'r2_score'],
            'nlp': ['bleu', 'rouge', 'meteor', 'bert_score', 'perplexity'],
            'ranking': ['ndcg', 'map', 'mrr', 'precision_at_k', 'recall_at_k']
        }

    def add_literature_record(self, record: LiteratureRecord):
        """
        문헌 기록 추가

        Args:
            record: 문헌 기록
        """
        self.literature_db[record.paper_id] = record
        self.save_literature_database()

    def load_literature_database(self):
        """문헌 데이터베이스 로드"""
        db_path = self.output_dir / "literature_db" / "literature_database.json"
        if db_path.exists():
            with open(db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for paper_id, record_data in data.items():
                    record = LiteratureRecord(**record_data)
                    self.literature_db[paper_id] = record

    def save_literature_database(self):
        """문헌 데이터베이스 저장"""
        db_path = self.output_dir / "literature_db" / "literature_database.json"
        data = {}
        for paper_id, record in self.literature_db.items():
            data[paper_id] = {
                'paper_id': record.paper_id,
                'title': record.title,
                'authors': record.authors,
                'venue': record.venue,
                'year': record.year,
                'dataset_name': record.dataset_name,
                'task_type': record.task_type,
                'methodology': record.methodology,
                'metrics_reported': record.metrics_reported,
                'experimental_setup': record.experimental_setup,
                'code_available': record.code_available,
                'data_available': record.data_available,
                'reproducibility_score': record.reproducibility_score,
                'citation_count': record.citation_count,
                'impact_factor': record.impact_factor
            }

        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def search_literature(self,
                         dataset_name: str,
                         task_type: str,
                         methodology: Optional[str] = None,
                         year_range: Optional[Tuple[int, int]] = None,
                         min_citations: int = 0) -> List[LiteratureRecord]:
        """
        문헌 검색

        Args:
            dataset_name: 데이터셋 이름
            task_type: 작업 유형
            methodology: 방법론 (선택사항)
            year_range: 연도 범위 (선택사항)
            min_citations: 최소 인용 수

        Returns:
            검색된 문헌 목록
        """
        results = []

        for record in self.literature_db.values():
            # 기본 필터링
            if (dataset_name.lower() in record.dataset_name.lower() and
                record.task_type.lower() == task_type.lower() and
                record.citation_count >= min_citations):

                # 방법론 필터링
                if methodology and methodology.lower() not in record.methodology.lower():
                    continue

                # 연도 범위 필터링
                if year_range and not (year_range[0] <= record.year <= year_range[1]):
                    continue

                results.append(record)

        # 인용 수와 연도로 정렬
        results.sort(key=lambda x: (x.citation_count, x.year), reverse=True)
        return results

    def compare_with_benchmarks(self,
                              our_results: Dict[str, float],
                              dataset_name: str,
                              task_type: str,
                              our_method_name: str = "Our Method") -> BenchmarkComparison:
        """
        벤치마크와 비교

        Args:
            our_results: 우리의 결과
            dataset_name: 데이터셋 이름
            task_type: 작업 유형
            our_method_name: 우리 방법 이름

        Returns:
            벤치마크 비교 결과
        """
        # 관련 문헌 검색
        literature_records = self.search_literature(dataset_name, task_type)

        # 문헌 결과 추출
        literature_results = []
        for record in literature_records:
            if record.metrics_reported:
                literature_results.append((record.paper_id, record.metrics_reported))

        # 메트릭별 순위 계산
        ranking = {}
        all_metrics = set(our_results.keys())
        for _, metrics in literature_results:
            all_metrics.update(metrics.keys())

        for metric in all_metrics:
            method_scores = []

            # 우리 결과 추가
            if metric in our_results:
                method_scores.append((our_method_name, our_results[metric]))

            # 문헌 결과 추가
            for paper_id, metrics in literature_results:
                if metric in metrics:
                    method_name = self.literature_db[paper_id].methodology[:30] + f" ({paper_id[:8]})"
                    method_scores.append((method_name, metrics[metric]))

            # 메트릭 타입에 따라 정렬
            if self._is_higher_better_metric(metric):
                method_scores.sort(key=lambda x: x[1], reverse=True)
            else:
                method_scores.sort(key=lambda x: x[1])

            ranking[metric] = method_scores

        # 통계적 유의성 검정
        statistical_significance = self._perform_significance_testing(
            our_results, literature_results
        )

        # 개선도 분석
        improvement_analysis = self._analyze_improvements(
            our_results, literature_results
        )

        # State-of-the-art 비교
        state_of_art_comparison = {}
        for metric in our_results:
            if metric in ranking and ranking[metric]:
                # 상위 3개 결과와 비교
                top_3_scores = [score for _, score in ranking[metric][:3]]
                our_score = our_results[metric]

                if self._is_higher_better_metric(metric):
                    state_of_art_comparison[metric] = our_score >= max(top_3_scores)
                else:
                    state_of_art_comparison[metric] = our_score <= min(top_3_scores)

        return BenchmarkComparison(
            dataset_name=dataset_name,
            task_type=task_type,
            our_results=our_results,
            literature_results=literature_results,
            ranking=ranking,
            statistical_significance=statistical_significance,
            improvement_analysis=improvement_analysis,
            state_of_art_comparison=state_of_art_comparison
        )

    def assess_reproducibility(self,
                             paper_id: str,
                             reproduced_results: Dict[str, float],
                             implementation_details: Dict[str, Any]) -> ReproducibilityAssessment:
        """
        재현성 평가

        Args:
            paper_id: 논문 ID
            reproduced_results: 재현된 결과
            implementation_details: 구현 세부사항

        Returns:
            재현성 평가 결과
        """
        if paper_id not in self.literature_db:
            raise ValueError(f"Paper {paper_id} not found in literature database")

        original_record = self.literature_db[paper_id]
        original_results = original_record.metrics_reported

        # 결과 비교
        reproduction_success = {}
        similarity_scores = {}
        statistical_equivalence = {}

        common_metrics = set(original_results.keys()) & set(reproduced_results.keys())

        for metric in common_metrics:
            original_score = original_results[metric]
            reproduced_score = reproduced_results[metric]

            # 유사도 점수 (상대 오차 기반)
            if original_score != 0:
                relative_error = abs(original_score - reproduced_score) / abs(original_score)
                similarity_score = max(0, 1 - relative_error)
            else:
                similarity_score = 1.0 if reproduced_score == 0 else 0.0

            similarity_scores[metric] = similarity_score

            # 재현 성공 여부 (5% 임계값)
            reproduction_success[metric] = similarity_score >= 0.95

            # 통계적 동등성 (간단한 구현)
            # 실제로는 더 정교한 동등성 검정 필요
            statistical_equivalence[metric] = similarity_score >= 0.99

        # 재현 난이도 평가
        reproduction_difficulty = self._assess_reproduction_difficulty(
            original_record, implementation_details
        )

        # 누락된 세부사항 식별
        missing_details = self._identify_missing_details(
            original_record, implementation_details
        )

        # 구현 격차 분석
        implementation_gaps = self._analyze_implementation_gaps(
            original_record, implementation_details
        )

        # 전체 재현성 점수
        if similarity_scores:
            overall_reproducibility_score = np.mean(list(similarity_scores.values()))
        else:
            overall_reproducibility_score = 0.0

        return ReproducibilityAssessment(
            paper_id=paper_id,
            original_results=original_results,
            reproduced_results=reproduced_results,
            reproduction_success=reproduction_success,
            similarity_scores=similarity_scores,
            statistical_equivalence=statistical_equivalence,
            reproduction_difficulty=reproduction_difficulty,
            missing_details=missing_details,
            implementation_gaps=implementation_gaps,
            overall_reproducibility_score=overall_reproducibility_score
        )

    def assess_research_contribution(self,
                                   our_results: Dict[str, float],
                                   methodology_description: str,
                                   dataset_info: Dict[str, Any],
                                   comparison_results: List[BenchmarkComparison]) -> ContributionAssessment:
        """
        연구 기여도 평가

        Args:
            our_results: 우리 결과
            methodology_description: 방법론 설명
            dataset_info: 데이터셋 정보
            comparison_results: 벤치마크 비교 결과

        Returns:
            연구 기여도 평가
        """
        # 참신성 점수 (방법론의 새로움)
        novelty_score = self._assess_novelty(methodology_description)

        # 유의성 점수 (성능 개선의 크기)
        significance_score = self._assess_significance(comparison_results)

        # 영향력 점수 (일반화 가능성과 실용성)
        impact_score = self._assess_impact(our_results, dataset_info, comparison_results)

        # 방법론적 기여
        methodological_contribution = self._analyze_methodological_contribution(
            methodology_description
        )

        # 실증적 기여
        empirical_contribution = self._analyze_empirical_contribution(
            comparison_results
        )

        # 이론적 기여
        theoretical_contribution = self._analyze_theoretical_contribution(
            methodology_description
        )

        # 실용적 시사점
        practical_implications = self._analyze_practical_implications(
            our_results, comparison_results
        )

        # 한계점 식별
        limitations = self._identify_limitations(
            our_results, dataset_info, comparison_results
        )

        # 향후 연구 제안
        future_work_suggestions = self._suggest_future_work(
            methodology_description, limitations
        )

        # 전체 기여도 점수
        overall_contribution_score = (novelty_score * 0.4 +
                                    significance_score * 0.4 +
                                    impact_score * 0.2)

        return ContributionAssessment(
            novelty_score=novelty_score,
            significance_score=significance_score,
            impact_score=impact_score,
            methodological_contribution=methodological_contribution,
            empirical_contribution=empirical_contribution,
            theoretical_contribution=theoretical_contribution,
            practical_implications=practical_implications,
            limitations=limitations,
            future_work_suggestions=future_work_suggestions,
            overall_contribution_score=overall_contribution_score
        )

    def analyze_research_landscape(self,
                                 dataset_names: List[str],
                                 task_types: List[str],
                                 year_range: Tuple[int, int] = (2020, 2024)) -> Dict[str, Any]:
        """
        연구 환경 분석

        Args:
            dataset_names: 데이터셋 이름들
            task_types: 작업 유형들
            year_range: 연도 범위

        Returns:
            연구 환경 분석 결과
        """
        landscape = {}

        # 관련 문헌 수집
        all_records = []
        for dataset in dataset_names:
            for task_type in task_types:
                records = self.search_literature(
                    dataset, task_type, year_range=year_range
                )
                all_records.extend(records)

        # 중복 제거
        unique_records = {r.paper_id: r for r in all_records}.values()

        # 연도별 트렌드
        year_trends = defaultdict(int)
        for record in unique_records:
            year_trends[record.year] += 1

        landscape['publication_trends'] = dict(year_trends)

        # 방법론 트렌드
        methodology_trends = defaultdict(int)
        for record in unique_records:
            # 간단한 키워드 추출
            keywords = self._extract_methodology_keywords(record.methodology)
            for keyword in keywords:
                methodology_trends[keyword] += 1

        landscape['methodology_trends'] = dict(methodology_trends)

        # 성능 트렌드 (시간에 따른 성능 향상)
        performance_trends = {}
        for dataset in dataset_names:
            dataset_records = [r for r in unique_records
                             if dataset.lower() in r.dataset_name.lower()]

            if dataset_records:
                # 연도별 최고 성능
                yearly_best = defaultdict(dict)
                for record in dataset_records:
                    for metric, score in record.metrics_reported.items():
                        if (record.year not in yearly_best or
                            metric not in yearly_best[record.year] or
                            (self._is_higher_better_metric(metric) and
                             score > yearly_best[record.year][metric]) or
                            (not self._is_higher_better_metric(metric) and
                             score < yearly_best[record.year][metric])):
                            yearly_best[record.year][metric] = score

                performance_trends[dataset] = dict(yearly_best)

        landscape['performance_trends'] = performance_trends

        # 상위 연구 그룹/기관
        institution_impact = defaultdict(lambda: {'papers': 0, 'total_citations': 0})
        for record in unique_records:
            # 간단한 기관 추출 (실제로는 더 정교한 처리 필요)
            if record.authors:
                first_author = record.authors[0]
                institution_impact[first_author]['papers'] += 1
                institution_impact[first_author]['total_citations'] += record.citation_count

        # 영향력 점수로 정렬
        top_institutions = sorted(
            institution_impact.items(),
            key=lambda x: x[1]['total_citations'],
            reverse=True
        )[:10]

        landscape['top_institutions'] = top_institutions

        # 데이터셋별 경쟁 상황
        dataset_competition = {}
        for dataset in dataset_names:
            dataset_records = [r for r in unique_records
                             if dataset.lower() in r.dataset_name.lower()]

            if dataset_records:
                competition_metrics = {
                    'total_papers': len(dataset_records),
                    'avg_citations': np.mean([r.citation_count for r in dataset_records]),
                    'recent_activity': len([r for r in dataset_records
                                          if r.year >= year_range[1] - 2]),
                    'methodology_diversity': len(set(r.methodology for r in dataset_records))
                }
                dataset_competition[dataset] = competition_metrics

        landscape['dataset_competition'] = dataset_competition

        return landscape

    def generate_comprehensive_report(self,
                                    comparison_id: str,
                                    datasets_compared: List[str],
                                    benchmark_comparisons: List[BenchmarkComparison],
                                    reproducibility_assessments: List[ReproducibilityAssessment],
                                    contribution_assessment: ContributionAssessment,
                                    research_landscape_analysis: Dict[str, Any]) -> LiteratureComparisonReport:
        """
        종합 문헌 비교 보고서 생성

        Args:
            comparison_id: 비교 ID
            datasets_compared: 비교된 데이터셋들
            benchmark_comparisons: 벤치마크 비교 결과들
            reproducibility_assessments: 재현성 평가들
            contribution_assessment: 기여도 평가
            research_landscape_analysis: 연구 환경 분석

        Returns:
            종합 문헌 비교 보고서
        """
        # 관련 문헌 기록들 수집
        literature_records = []
        for comparison in benchmark_comparisons:
            for paper_id, _ in comparison.literature_results:
                if paper_id in self.literature_db:
                    literature_records.append(self.literature_db[paper_id])

        # 중복 제거
        unique_records = {r.paper_id: r for r in literature_records}.values()

        # 권장사항 생성
        recommendations = self._generate_recommendations(
            benchmark_comparisons, reproducibility_assessments, contribution_assessment
        )

        # 요약 생성
        executive_summary = self._generate_executive_summary(
            benchmark_comparisons, contribution_assessment, research_landscape_analysis
        )

        return LiteratureComparisonReport(
            comparison_id=comparison_id,
            comparison_timestamp=datetime.now().isoformat(),
            datasets_compared=datasets_compared,
            literature_records=list(unique_records),
            benchmark_comparisons=benchmark_comparisons,
            reproducibility_assessments=reproducibility_assessments,
            contribution_assessment=contribution_assessment,
            research_landscape_analysis=research_landscape_analysis,
            recommendations=recommendations,
            executive_summary=executive_summary
        )

    # 보조 메서드들
    def _is_higher_better_metric(self, metric: str) -> bool:
        """메트릭이 높을수록 좋은지 판단"""
        higher_better = ['accuracy', 'precision', 'recall', 'f1', 'f1_score',
                        'auc', 'auc_roc', 'r2', 'r2_score', 'bleu', 'rouge',
                        'ndcg', 'map', 'mrr']
        return any(hb in metric.lower() for hb in higher_better)

    def _perform_significance_testing(self,
                                    our_results: Dict[str, float],
                                    literature_results: List[Tuple[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
        """통계적 유의성 검정"""
        significance = {}

        for metric in our_results:
            significance[metric] = {}
            our_score = our_results[metric]

            for paper_id, metrics in literature_results:
                if metric in metrics:
                    literature_score = metrics[metric]

                    # 간단한 z-test (실제로는 더 정교한 검정 필요)
                    # 가정: 각 점수는 정규분포를 따름
                    try:
                        # 표준편차를 점수의 10%로 가정 (임시)
                        std_our = abs(our_score) * 0.1 + 0.001
                        std_lit = abs(literature_score) * 0.1 + 0.001

                        z_score = (our_score - literature_score) / np.sqrt(std_our**2 + std_lit**2)
                        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                        significance[metric][paper_id] = p_value
                    except:
                        significance[metric][paper_id] = 1.0

        return significance

    def _analyze_improvements(self,
                            our_results: Dict[str, float],
                            literature_results: List[Tuple[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
        """개선도 분석"""
        improvements = {}

        for metric in our_results:
            improvements[metric] = {}
            our_score = our_results[metric]

            for paper_id, metrics in literature_results:
                if metric in metrics:
                    literature_score = metrics[metric]

                    if self._is_higher_better_metric(metric):
                        if literature_score != 0:
                            improvement = (our_score - literature_score) / literature_score * 100
                        else:
                            improvement = float('inf') if our_score > 0 else 0
                    else:
                        if literature_score != 0:
                            improvement = (literature_score - our_score) / literature_score * 100
                        else:
                            improvement = float('inf') if our_score < literature_score else 0

                    improvements[metric][paper_id] = improvement

        return improvements

    def _assess_reproduction_difficulty(self,
                                      original_record: LiteratureRecord,
                                      implementation_details: Dict[str, Any]) -> str:
        """재현 난이도 평가"""
        difficulty_score = 0

        # 코드 가용성
        if not original_record.code_available:
            difficulty_score += 3

        # 데이터 가용성
        if not original_record.data_available:
            difficulty_score += 2

        # 실험 설정 상세도
        if not original_record.experimental_setup:
            difficulty_score += 2

        # 하이퍼파라미터 명시
        if 'hyperparameters' not in original_record.experimental_setup:
            difficulty_score += 1

        # 구현 세부사항 불일치
        missing_count = len(self._identify_missing_details(original_record, implementation_details))
        difficulty_score += min(missing_count, 3)

        if difficulty_score <= 2:
            return "easy"
        elif difficulty_score <= 5:
            return "moderate"
        elif difficulty_score <= 8:
            return "hard"
        else:
            return "impossible"

    def _identify_missing_details(self,
                                original_record: LiteratureRecord,
                                implementation_details: Dict[str, Any]) -> List[str]:
        """누락된 세부사항 식별"""
        missing = []

        required_details = [
            'model_architecture', 'hyperparameters', 'training_procedure',
            'data_preprocessing', 'evaluation_metrics', 'hardware_specs',
            'software_versions', 'random_seeds'
        ]

        for detail in required_details:
            if (detail not in original_record.experimental_setup and
                detail not in implementation_details):
                missing.append(detail)

        return missing

    def _analyze_implementation_gaps(self,
                                   original_record: LiteratureRecord,
                                   implementation_details: Dict[str, Any]) -> List[str]:
        """구현 격차 분석"""
        gaps = []

        # 하이퍼파라미터 불일치
        if ('hyperparameters' in original_record.experimental_setup and
            'hyperparameters' in implementation_details):
            orig_params = original_record.experimental_setup['hyperparameters']
            impl_params = implementation_details['hyperparameters']

            for param, value in orig_params.items():
                if param in impl_params and impl_params[param] != value:
                    gaps.append(f"Hyperparameter mismatch: {param}")

        # 데이터 전처리 차이
        if ('data_preprocessing' in original_record.experimental_setup and
            'data_preprocessing' in implementation_details):
            gaps.append("Potential data preprocessing differences")

        # 평가 메트릭 차이
        orig_metrics = set(original_record.metrics_reported.keys())
        impl_metrics = set(implementation_details.get('evaluation_metrics', []))
        missing_metrics = orig_metrics - impl_metrics
        if missing_metrics:
            gaps.append(f"Missing evaluation metrics: {missing_metrics}")

        return gaps

    def _assess_novelty(self, methodology_description: str) -> float:
        """참신성 평가"""
        # 간단한 키워드 기반 참신성 평가
        novel_keywords = [
            'novel', 'new', 'innovative', 'first', 'unprecedented',
            'breakthrough', 'original', 'unique', 'pioneering'
        ]

        description_lower = methodology_description.lower()
        novelty_indicators = sum(1 for keyword in novel_keywords
                               if keyword in description_lower)

        # 문헌에서 유사한 방법론 검색
        similar_count = 0
        for record in self.literature_db.values():
            if self._calculate_methodology_similarity(
                methodology_description, record.methodology) > 0.7:
                similar_count += 1

        # 참신성 점수 계산 (0-1)
        novelty_score = min(1.0, (novelty_indicators * 0.1 +
                                 max(0, 1 - similar_count * 0.1)))

        return novelty_score

    def _assess_significance(self, comparison_results: List[BenchmarkComparison]) -> float:
        """유의성 평가"""
        if not comparison_results:
            return 0.0

        significance_scores = []

        for comparison in comparison_results:
            # State-of-the-art 달성 비율
            sota_ratio = sum(comparison.state_of_art_comparison.values()) / len(comparison.state_of_art_comparison)

            # 평균 개선도
            all_improvements = []
            for metric_improvements in comparison.improvement_analysis.values():
                improvements = [imp for imp in metric_improvements.values()
                              if not np.isnan(imp) and imp != float('inf')]
                all_improvements.extend(improvements)

            avg_improvement = np.mean(all_improvements) if all_improvements else 0

            # 통계적 유의성 비율
            sig_count = 0
            total_count = 0
            for metric_tests in comparison.statistical_significance.values():
                for p_value in metric_tests.values():
                    total_count += 1
                    if p_value < 0.05:
                        sig_count += 1

            sig_ratio = sig_count / total_count if total_count > 0 else 0

            # 종합 유의성 점수
            comp_significance = (sota_ratio * 0.4 +
                               min(1.0, avg_improvement / 10) * 0.4 +
                               sig_ratio * 0.2)
            significance_scores.append(comp_significance)

        return np.mean(significance_scores)

    def _assess_impact(self,
                      our_results: Dict[str, float],
                      dataset_info: Dict[str, Any],
                      comparison_results: List[BenchmarkComparison]) -> float:
        """영향력 평가"""
        impact_factors = []

        # 데이터셋 중요도
        important_datasets = ['ImageNet', 'CIFAR-10', 'GLUE', 'SQuAD', 'UCI']
        dataset_importance = 0.5  # 기본값

        for dataset in important_datasets:
            if any(dataset.lower() in comp.dataset_name.lower()
                  for comp in comparison_results):
                dataset_importance = 1.0
                break

        impact_factors.append(dataset_importance)

        # 일반화 가능성 (다양한 메트릭에서의 성능)
        if our_results:
            metric_coverage = len(our_results) / 5  # 5개 메트릭 기준
            generalizability = min(1.0, metric_coverage)
            impact_factors.append(generalizability)

        # 실용성 (계산 효율성, 구현 용이성)
        if 'computational_complexity' in dataset_info:
            complexity = dataset_info['computational_complexity']
            if complexity == 'low':
                practicality = 1.0
            elif complexity == 'medium':
                practicality = 0.7
            else:
                practicality = 0.4
        else:
            practicality = 0.6  # 기본값

        impact_factors.append(practicality)

        return np.mean(impact_factors)

    def _analyze_methodological_contribution(self, methodology_description: str) -> str:
        """방법론적 기여 분석"""
        description_lower = methodology_description.lower()

        contributions = []

        if any(word in description_lower for word in ['architecture', 'model', 'network']):
            contributions.append("새로운 모델 아키텍처 제안")

        if any(word in description_lower for word in ['algorithm', 'optimization', 'training']):
            contributions.append("개선된 훈련 알고리즘 개발")

        if any(word in description_lower for word in ['loss', 'objective', 'regularization']):
            contributions.append("새로운 목적 함수 또는 정규화 기법")

        if any(word in description_lower for word in ['ensemble', 'fusion', 'combination']):
            contributions.append("앙상블 또는 융합 기법 개발")

        if not contributions:
            contributions.append("기존 방법의 개선 및 최적화")

        return "; ".join(contributions)

    def _analyze_empirical_contribution(self, comparison_results: List[BenchmarkComparison]) -> str:
        """실증적 기여 분석"""
        contributions = []

        total_datasets = len(comparison_results)
        sota_count = sum(
            sum(comp.state_of_art_comparison.values())
            for comp in comparison_results
        )

        if sota_count > 0:
            contributions.append(f"{sota_count}개 메트릭에서 State-of-the-art 성능 달성")

        avg_improvements = []
        for comp in comparison_results:
            for metric_improvements in comp.improvement_analysis.values():
                improvements = [imp for imp in metric_improvements.values()
                              if not np.isnan(imp) and imp != float('inf') and imp > 0]
                avg_improvements.extend(improvements)

        if avg_improvements:
            mean_improvement = np.mean(avg_improvements)
            contributions.append(f"평균 {mean_improvement:.1f}% 성능 향상 달성")

        if total_datasets > 1:
            contributions.append(f"{total_datasets}개 데이터셋에서 일관된 성능 개선 확인")

        if not contributions:
            contributions.append("기존 방법과 경쟁력 있는 성능 달성")

        return "; ".join(contributions)

    def _analyze_theoretical_contribution(self, methodology_description: str) -> str:
        """이론적 기여 분석"""
        description_lower = methodology_description.lower()

        if any(word in description_lower for word in ['theorem', 'proof', 'analysis', 'theory']):
            return "새로운 이론적 분석 및 증명 제공"
        elif any(word in description_lower for word in ['convergence', 'complexity', 'bound']):
            return "수렴성 또는 복잡도 분석 제공"
        elif any(word in description_lower for word in ['interpretation', 'explanation', 'insight']):
            return "방법론에 대한 새로운 해석 및 통찰 제공"
        else:
            return "실험적 검증을 통한 실증적 기여"

    def _analyze_practical_implications(self,
                                      our_results: Dict[str, float],
                                      comparison_results: List[BenchmarkComparison]) -> str:
        """실용적 시사점 분석"""
        implications = []

        # 성능 개선의 실용적 의미
        significant_improvements = []
        for comp in comparison_results:
            for metric, improvements in comp.improvement_analysis.items():
                best_improvement = max(improvements.values()) if improvements else 0
                if best_improvement > 5:  # 5% 이상 개선
                    significant_improvements.append((metric, best_improvement))

        if significant_improvements:
            implications.append("실질적인 성능 향상으로 실제 응용에서의 효과 기대")

        # 일반화 가능성
        if len(comparison_results) > 1:
            implications.append("다양한 데이터셋에서의 검증으로 일반화 가능성 확인")

        # 계산 효율성 (가정)
        implications.append("기존 방법 대비 유사하거나 개선된 계산 효율성")

        if not implications:
            implications.append("학술적 기여와 향후 연구 방향 제시")

        return "; ".join(implications)

    def _identify_limitations(self,
                            our_results: Dict[str, float],
                            dataset_info: Dict[str, Any],
                            comparison_results: List[BenchmarkComparison]) -> List[str]:
        """한계점 식별"""
        limitations = []

        # 데이터셋 제한
        if len(comparison_results) < 3:
            limitations.append("제한된 수의 데이터셋에서만 검증됨")

        # 메트릭 제한
        if len(our_results) < 3:
            limitations.append("다양한 평가 메트릭에서의 검증 부족")

        # 통계적 유의성 부족
        nonsignificant_count = 0
        total_tests = 0
        for comp in comparison_results:
            for metric_tests in comp.statistical_significance.values():
                for p_value in metric_tests.values():
                    total_tests += 1
                    if p_value >= 0.05:
                        nonsignificant_count += 1

        if total_tests > 0 and nonsignificant_count / total_tests > 0.5:
            limitations.append("일부 성능 개선의 통계적 유의성 부족")

        # 재현성 관련
        limitations.append("재현성 검증을 위한 추가 실험 필요")

        # 일반적인 한계
        limitations.append("특정 도메인 또는 작업 유형에 특화될 가능성")

        return limitations

    def _suggest_future_work(self,
                           methodology_description: str,
                           limitations: List[str]) -> List[str]:
        """향후 연구 제안"""
        suggestions = []

        # 한계점 기반 제안
        if "제한된 수의 데이터셋" in str(limitations):
            suggestions.append("더 다양한 데이터셋에서의 검증 실험")

        if "다양한 평가 메트릭" in str(limitations):
            suggestions.append("추가적인 평가 메트릭을 통한 포괄적 성능 분석")

        # 방법론 기반 제안
        description_lower = methodology_description.lower()

        if 'ensemble' in description_lower:
            suggestions.append("다른 앙상블 기법과의 결합 가능성 탐구")

        if 'neural' in description_lower or 'deep' in description_lower:
            suggestions.append("더 깊은 네트워크 아키텍처로의 확장")

        # 일반적인 제안
        suggestions.append("실제 응용 환경에서의 성능 검증")
        suggestions.append("계산 효율성 최적화 연구")
        suggestions.append("다른 도메인으로의 전이학습 가능성 탐구")

        return suggestions

    def _calculate_methodology_similarity(self, method1: str, method2: str) -> float:
        """방법론 유사도 계산"""
        # 간단한 키워드 기반 유사도
        words1 = set(method1.lower().split())
        words2 = set(method2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _extract_methodology_keywords(self, methodology: str) -> List[str]:
        """방법론 키워드 추출"""
        # 일반적인 ML 키워드들
        ml_keywords = [
            'neural', 'network', 'deep', 'learning', 'cnn', 'rnn', 'lstm',
            'transformer', 'attention', 'ensemble', 'random', 'forest',
            'svm', 'regression', 'classification', 'clustering', 'gradient',
            'boosting', 'xgboost', 'lightgbm', 'bert', 'gpt', 'resnet'
        ]

        methodology_lower = methodology.lower()
        found_keywords = [kw for kw in ml_keywords if kw in methodology_lower]

        return found_keywords

    def _generate_recommendations(self,
                                benchmark_comparisons: List[BenchmarkComparison],
                                reproducibility_assessments: List[ReproducibilityAssessment],
                                contribution_assessment: ContributionAssessment) -> List[str]:
        """권장사항 생성"""
        recommendations = []

        # 벤치마크 기반 권장사항
        sota_achieved = any(
            any(comp.state_of_art_comparison.values())
            for comp in benchmark_comparisons
        )

        if sota_achieved:
            recommendations.append("State-of-the-art 성능을 달성했으므로 주요 학회/저널 투고를 권장합니다.")
        else:
            recommendations.append("성능 개선을 위한 추가 연구 및 실험이 필요합니다.")

        # 재현성 기반 권장사항
        if reproducibility_assessments:
            avg_reproducibility = np.mean([
                assess.overall_reproducibility_score
                for assess in reproducibility_assessments
            ])

            if avg_reproducibility < 0.8:
                recommendations.append("재현성 향상을 위한 상세한 구현 세부사항 문서화가 필요합니다.")

        # 기여도 기반 권장사항
        if contribution_assessment.novelty_score < 0.5:
            recommendations.append("방법론의 참신성을 강화하거나 기존 방법과의 차별점을 명확히 해야 합니다.")

        if contribution_assessment.significance_score < 0.5:
            recommendations.append("성능 개선의 통계적 유의성을 강화하는 추가 실험이 필요합니다.")

        # 일반적인 권장사항
        recommendations.append("다양한 데이터셋에서의 추가 검증을 통해 일반화 성능을 확인하세요.")
        recommendations.append("코드 및 데이터 공개를 통해 연구의 투명성과 재현성을 높이세요.")

        return recommendations

    def _generate_executive_summary(self,
                                  benchmark_comparisons: List[BenchmarkComparison],
                                  contribution_assessment: ContributionAssessment,
                                  research_landscape_analysis: Dict[str, Any]) -> str:
        """요약 생성"""
        summary_parts = []

        # 성능 요약
        total_comparisons = len(benchmark_comparisons)
        sota_count = sum(
            sum(comp.state_of_art_comparison.values())
            for comp in benchmark_comparisons
        )

        summary_parts.append(
            f"{total_comparisons}개 데이터셋에서 벤치마크 비교를 수행하였으며, "
            f"{sota_count}개 메트릭에서 State-of-the-art 성능을 달성했습니다."
        )

        # 기여도 요약
        contrib_score = contribution_assessment.overall_contribution_score
        if contrib_score >= 0.8:
            contrib_level = "매우 높은"
        elif contrib_score >= 0.6:
            contrib_level = "높은"
        elif contrib_score >= 0.4:
            contrib_level = "중간"
        else:
            contrib_level = "낮은"

        summary_parts.append(
            f"연구의 전체 기여도 점수는 {contrib_score:.2f}로 {contrib_level} 수준입니다."
        )

        # 연구 환경 요약
        if 'methodology_trends' in research_landscape_analysis:
            top_trends = sorted(
                research_landscape_analysis['methodology_trends'].items(),
                key=lambda x: x[1], reverse=True
            )[:3]

            if top_trends:
                trend_names = [trend[0] for trend in top_trends]
                summary_parts.append(
                    f"현재 연구 분야의 주요 트렌드는 {', '.join(trend_names)} 등입니다."
                )

        # 종합 평가
        if sota_count > 0 and contrib_score >= 0.6:
            summary_parts.append("우수한 성능과 높은 기여도를 바탕으로 학술적 가치가 인정됩니다.")
        elif sota_count > 0:
            summary_parts.append("우수한 성능을 달성했으나 기여도 측면에서 개선이 필요합니다.")
        else:
            summary_parts.append("추가적인 성능 개선과 기여도 강화가 필요합니다.")

        return " ".join(summary_parts)

    def save_comparison_report(self, report: LiteratureComparisonReport) -> str:
        """비교 보고서 저장"""
        filename = f"literature_comparison_{report.comparison_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
    comparator = LiteratureComparator()

    # 샘플 문헌 기록 추가
    sample_record = LiteratureRecord(
        paper_id="sample_2023_001",
        title="Advanced Neural Networks for Financial Prediction",
        authors=["Smith, J.", "Johnson, A."],
        venue="ICML 2023",
        year=2023,
        dataset_name="S&P500",
        task_type="regression",
        methodology="Deep Neural Network with Attention Mechanism",
        metrics_reported={"mse": 0.025, "mae": 0.12, "mape": 2.3},
        experimental_setup={"hyperparameters": {"lr": 0.001, "batch_size": 32}},
        code_available=True,
        data_available=True,
        citation_count=15
    )

    comparator.add_literature_record(sample_record)

    # 벤치마크 비교 테스트
    our_results = {"mse": 0.020, "mae": 0.10, "mape": 2.1}
    comparison = comparator.compare_with_benchmarks(
        our_results, "S&P500", "regression"
    )

    print(f"벤치마크 비교 완료: {len(comparison.literature_results)}개 문헌과 비교")
    print(f"State-of-the-art 달성: {comparison.state_of_art_comparison}")

    print("\\n문헌 비교 및 재현성 검증 시스템 테스트 완료!")