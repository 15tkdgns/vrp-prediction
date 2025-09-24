"""
검증 보고서 생성 시스템

이 모듈은 방법론 검증 결과를 다양한 형식(Markdown, HTML, LaTeX, PDF)으로
변환하여 학술 논문 및 기술 보고서에 활용할 수 있는 고품질 문서를 생성합니다.
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from .methodology_validator import (
    MethodologyValidationReport, DataSplitValidation, CrossValidationValidation,
    HyperparameterTuningValidation, StatisticalValidation, ModelValidation,
    ReproducibilityValidation, EthicalValidation
)

@dataclass
class ReportConfig:
    """보고서 생성 설정"""
    output_format: str = "markdown"  # markdown, html, latex, pdf
    include_figures: bool = True
    include_recommendations: bool = True
    include_detailed_scores: bool = True
    academic_style: bool = True
    template_style: str = "ieee"  # ieee, nature, acm, minimal

class ValidationReportGenerator:
    """검증 보고서 생성 시스템"""

    def __init__(self, output_dir: str = "results/validation_reports", config: Optional[ReportConfig] = None):
        """
        초기화

        Args:
            output_dir: 보고서 저장 디렉토리
            config: 보고서 생성 설정
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config if config else ReportConfig()

        # 서브 디렉토리 생성
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "markdown").mkdir(exist_ok=True)
        (self.output_dir / "html").mkdir(exist_ok=True)
        (self.output_dir / "latex").mkdir(exist_ok=True)

        # 성적 등급 기준
        self.grade_thresholds = {
            'excellent': 0.9,
            'good': 0.8,
            'satisfactory': 0.7,
            'needs_improvement': 0.6,
            'poor': 0.0
        }

    def generate_markdown_report(self, validation_report: MethodologyValidationReport) -> str:
        """
        Markdown 형식 보고서 생성

        Args:
            validation_report: 검증 보고서 객체

        Returns:
            생성된 마크다운 파일 경로
        """
        content = []

        # 헤더
        content.append(f"# 방법론 검증 보고서")
        content.append(f"**실험 ID:** {validation_report.experiment_id}")
        content.append(f"**검증 일시:** {validation_report.validation_timestamp}")
        content.append(f"**전체 점수:** {validation_report.overall_score:.3f}/1.000")
        content.append("")

        # 전체 요약
        content.append("## 검증 요약")
        content.append(validation_report.validation_summary)
        content.append("")

        # 등급 평가
        grade = self._get_grade(validation_report.overall_score)
        content.append(f"**전체 등급:** {grade}")
        content.append("")

        # 심각한 문제
        if validation_report.critical_issues:
            content.append("## ⚠️ 심각한 문제")
            for issue in validation_report.critical_issues:
                content.append(f"- {issue}")
            content.append("")

        # 세부 검증 결과
        content.append("## 세부 검증 결과")

        # 1. 데이터 분할 검증
        content.extend(self._generate_data_split_section_md(validation_report.data_split_validation))

        # 2. 교차 검증
        content.extend(self._generate_cv_section_md(validation_report.cv_validation))

        # 3. 하이퍼파라미터 튜닝
        content.extend(self._generate_hyperparameter_section_md(validation_report.hyperparameter_validation))

        # 4. 통계적 검증
        content.extend(self._generate_statistical_section_md(validation_report.statistical_validation))

        # 5. 모델 검증
        content.extend(self._generate_model_section_md(validation_report.model_validation))

        # 6. 재현성 검증
        content.extend(self._generate_reproducibility_section_md(validation_report.reproducibility_validation))

        # 7. 윤리적 검증
        content.extend(self._generate_ethical_section_md(validation_report.ethical_validation))

        # 권장사항
        if self.config.include_recommendations and validation_report.recommendations:
            content.append("## 권장사항")
            for i, recommendation in enumerate(validation_report.recommendations, 1):
                content.append(f"{i}. {recommendation}")
            content.append("")

        # 그래프 생성 및 추가
        if self.config.include_figures:
            figures = self._generate_validation_figures(validation_report)
            if figures:
                content.append("## 검증 결과 시각화")
                for figure_path in figures:
                    relative_path = os.path.relpath(figure_path, self.output_dir)
                    content.append(f"![검증 결과]({relative_path})")
                content.append("")

        # 상세 점수 표
        if self.config.include_detailed_scores:
            content.extend(self._generate_detailed_scores_table_md(validation_report))

        # 파일 저장
        filename = f"validation_report_{validation_report.experiment_id}.md"
        filepath = self.output_dir / "markdown" / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\\n".join(content))

        return str(filepath)

    def generate_html_report(self, validation_report: MethodologyValidationReport) -> str:
        """
        HTML 형식 보고서 생성

        Args:
            validation_report: 검증 보고서 객체

        Returns:
            생성된 HTML 파일 경로
        """
        # HTML 템플릿
        html_template = """
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>방법론 검증 보고서</title>
            <style>
                body {{
                    font-family: 'Noto Sans KR', sans-serif;
                    line-height: 1.6;
                    margin: 40px;
                    background-color: #f8f9fa;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 40px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                h1, h2, h3 {{ color: #2c3e50; }}
                h1 {{ border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ border-bottom: 2px solid #95a5a6; padding-bottom: 5px; margin-top: 30px; }}
                .score {{
                    font-size: 24px;
                    font-weight: bold;
                    padding: 10px 20px;
                    border-radius: 5px;
                    display: inline-block;
                    margin: 10px 0;
                }}
                .excellent {{ background-color: #d4edda; color: #155724; }}
                .good {{ background-color: #d1ecf1; color: #0c5460; }}
                .satisfactory {{ background-color: #fff3cd; color: #856404; }}
                .needs_improvement {{ background-color: #f8d7da; color: #721c24; }}
                .poor {{ background-color: #f8d7da; color: #721c24; }}
                .critical-issue {{
                    background-color: #f8d7da;
                    border: 1px solid #f5c6cb;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 10px 0;
                }}
                .recommendation {{
                    background-color: #d1ecf1;
                    border: 1px solid #bee5eb;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 5px 0;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                }}
                .metric-pass {{ color: #28a745; font-weight: bold; }}
                .metric-fail {{ color: #dc3545; font-weight: bold; }}
                .progress-bar {{
                    width: 100%;
                    height: 20px;
                    background-color: #e9ecef;
                    border-radius: 10px;
                    overflow: hidden;
                    margin: 5px 0;
                }}
                .progress-fill {{
                    height: 100%;
                    background: linear-gradient(90deg, #28a745, #20c997, #17a2b8, #6f42c1);
                    transition: width 0.3s ease;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                {content}
            </div>
        </body>
        </html>
        """

        # HTML 컨텐츠 생성
        content = []

        # 헤더
        content.append(f"<h1>방법론 검증 보고서</h1>")
        content.append(f"<p><strong>실험 ID:</strong> {validation_report.experiment_id}</p>")
        content.append(f"<p><strong>검증 일시:</strong> {validation_report.validation_timestamp}</p>")

        # 전체 점수
        grade = self._get_grade(validation_report.overall_score)
        grade_class = grade.lower().replace(' ', '_')
        content.append(f'<div class="score {grade_class}">전체 점수: {validation_report.overall_score:.3f}/1.000 ({grade})</div>')

        # 진행률 바
        progress_percentage = validation_report.overall_score * 100
        content.append(f"""
        <div class="progress-bar">
            <div class="progress-fill" style="width: {progress_percentage}%;"></div>
        </div>
        """)

        # 검증 요약
        content.append(f"<h2>검증 요약</h2>")
        content.append(f"<p>{validation_report.validation_summary}</p>")

        # 심각한 문제
        if validation_report.critical_issues:
            content.append("<h2>⚠️ 심각한 문제</h2>")
            for issue in validation_report.critical_issues:
                content.append(f'<div class="critical-issue">{issue}</div>')

        # 세부 검증 결과 테이블
        content.append("<h2>세부 검증 결과</h2>")
        content.append(self._generate_validation_summary_table_html(validation_report))

        # 권장사항
        if self.config.include_recommendations and validation_report.recommendations:
            content.append("<h2>권장사항</h2>")
            for recommendation in validation_report.recommendations:
                content.append(f'<div class="recommendation">{recommendation}</div>')

        # HTML 파일 저장
        filename = f"validation_report_{validation_report.experiment_id}.html"
        filepath = self.output_dir / "html" / filename

        html_content = html_template.format(content="\\n".join(content))

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(filepath)

    def generate_latex_report(self, validation_report: MethodologyValidationReport) -> str:
        """
        LaTeX 형식 보고서 생성

        Args:
            validation_report: 검증 보고서 객체

        Returns:
            생성된 LaTeX 파일 경로
        """
        content = []

        # LaTeX 문서 헤더
        content.append("\\documentclass[11pt,a4paper]{article}")
        content.append("\\usepackage[utf8]{inputenc}")
        content.append("\\usepackage[T1]{fontenc}")
        content.append("\\usepackage{kotex}")
        content.append("\\usepackage{geometry}")
        content.append("\\usepackage{booktabs}")
        content.append("\\usepackage{xcolor}")
        content.append("\\usepackage{graphicx}")
        content.append("\\usepackage{float}")
        content.append("\\geometry{margin=2.5cm}")
        content.append("")
        content.append("\\title{방법론 검증 보고서}")
        content.append(f"\\author{{실험 ID: {validation_report.experiment_id}}}")
        content.append(f"\\date{{{validation_report.validation_timestamp}}}")
        content.append("")
        content.append("\\begin{document}")
        content.append("\\maketitle")
        content.append("")

        # 전체 요약
        content.append("\\section{검증 요약}")
        grade = self._get_grade(validation_report.overall_score)
        content.append(f"전체 검증 점수: \\textbf{{{validation_report.overall_score:.3f}/1.000}} ({grade})")
        content.append("")
        content.append(validation_report.validation_summary)
        content.append("")

        # 심각한 문제
        if validation_report.critical_issues:
            content.append("\\section{심각한 문제}")
            content.append("\\begin{itemize}")
            for issue in validation_report.critical_issues:
                content.append(f"\\item {self._escape_latex(issue)}")
            content.append("\\end{itemize}")
            content.append("")

        # 세부 검증 결과
        content.append("\\section{세부 검증 결과}")
        content.append(self._generate_validation_table_latex(validation_report))

        # 권장사항
        if self.config.include_recommendations and validation_report.recommendations:
            content.append("\\section{권장사항}")
            content.append("\\begin{enumerate}")
            for recommendation in validation_report.recommendations:
                content.append(f"\\item {self._escape_latex(recommendation)}")
            content.append("\\end{enumerate}")
            content.append("")

        content.append("\\end{document}")

        # 파일 저장
        filename = f"validation_report_{validation_report.experiment_id}.tex"
        filepath = self.output_dir / "latex" / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\\n".join(content))

        return str(filepath)

    def _generate_validation_figures(self, validation_report: MethodologyValidationReport) -> List[str]:
        """
        검증 결과 시각화 생성

        Args:
            validation_report: 검증 보고서

        Returns:
            생성된 그래프 파일 경로들
        """
        figure_paths = []

        # 1. 전체 점수 레이더 차트
        radar_path = self._create_validation_radar_chart(validation_report)
        if radar_path:
            figure_paths.append(radar_path)

        # 2. 점수 막대 그래프
        bar_path = self._create_validation_bar_chart(validation_report)
        if bar_path:
            figure_paths.append(bar_path)

        return figure_paths

    def _create_validation_radar_chart(self, validation_report: MethodologyValidationReport) -> str:
        """레이더 차트 생성"""
        # 각 검증 영역의 점수
        categories = [
            'Data Split', 'Cross Validation', 'Hyperparameter',
            'Statistical', 'Model', 'Reproducibility', 'Ethics'
        ]

        scores = [
            validation_report.data_split_validation.validation_score,
            validation_report.cv_validation.validation_score,
            validation_report.hyperparameter_validation.validation_score,
            validation_report.statistical_validation.validation_score,
            validation_report.model_validation.validation_score,
            validation_report.reproducibility_validation.validation_score,
            validation_report.ethical_validation.validation_score
        ]

        # 각도 계산
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        scores_plot = scores + [scores[0]]  # 닫힌 도형을 위해 첫 번째 값 추가
        angles += angles[:1]

        # 레이더 차트 생성
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

        ax.plot(angles, scores_plot, 'o-', linewidth=2, label='Validation Scores')
        ax.fill(angles, scores_plot, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Methodology Validation Radar Chart', size=16, fontweight='bold', pad=20)
        ax.grid(True)

        # 점수 값 표시
        for angle, score, category in zip(angles[:-1], scores, categories):
            ax.text(angle, score + 0.05, f'{score:.2f}',
                   horizontalalignment='center', verticalalignment='center')

        # 저장
        filename = f"validation_radar_{validation_report.experiment_id}.png"
        filepath = self.output_dir / "figures" / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return str(filepath)

    def _create_validation_bar_chart(self, validation_report: MethodologyValidationReport) -> str:
        """막대 그래프 생성"""
        categories = [
            'Data Split', 'Cross Validation', 'Hyperparameter',
            'Statistical', 'Model', 'Reproducibility', 'Ethics'
        ]

        scores = [
            validation_report.data_split_validation.validation_score,
            validation_report.cv_validation.validation_score,
            validation_report.hyperparameter_validation.validation_score,
            validation_report.statistical_validation.validation_score,
            validation_report.model_validation.validation_score,
            validation_report.reproducibility_validation.validation_score,
            validation_report.ethical_validation.validation_score
        ]

        # 색상 설정 (점수에 따라)
        colors = []
        for score in scores:
            if score >= 0.9:
                colors.append('#28a745')  # 초록
            elif score >= 0.8:
                colors.append('#17a2b8')  # 파랑
            elif score >= 0.7:
                colors.append('#ffc107')  # 노랑
            elif score >= 0.6:
                colors.append('#fd7e14')  # 주황
            else:
                colors.append('#dc3545')  # 빨강

        # 막대 그래프 생성
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(categories, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

        # 점수 값 표시
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('Validation Score', fontsize=12, fontweight='bold')
        ax.set_title('Methodology Validation Scores by Category', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')

        # 임계선 표시
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Minimum Threshold (0.8)')
        ax.legend()

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # 저장
        filename = f"validation_bars_{validation_report.experiment_id}.png"
        filepath = self.output_dir / "figures" / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return str(filepath)

    def _get_grade(self, score: float) -> str:
        """점수에 따른 등급 반환"""
        for grade, threshold in self.grade_thresholds.items():
            if score >= threshold:
                return grade.replace('_', ' ').title()
        return "Poor"

    def _generate_data_split_section_md(self, validation: DataSplitValidation) -> List[str]:
        """데이터 분할 섹션 마크다운 생성"""
        content = []
        content.append("### 1. 데이터 분할 검증")
        content.append(f"**점수:** {validation.validation_score:.3f}/1.000")
        content.append("")
        content.append("| 항목 | 값 | 상태 |")
        content.append("|------|-----|------|")
        content.append(f"| 훈련 세트 크기 | {validation.train_size:,} ({validation.train_ratio:.1%}) | ✓ |")
        content.append(f"| 검증 세트 크기 | {validation.val_size:,} ({validation.val_ratio:.1%}) | ✓ |")
        content.append(f"| 테스트 세트 크기 | {validation.test_size:,} ({validation.test_ratio:.1%}) | ✓ |")
        content.append(f"| 계층화 적용 | {validation.is_stratified} | {'✓' if validation.is_stratified else '✗'} |")
        content.append(f"| 데이터 누출 | {validation.data_leakage_detected} | {'✗' if validation.data_leakage_detected else '✓'} |")
        content.append(f"| 시간 순서 보존 | {validation.temporal_ordering_preserved} | {'✓' if validation.temporal_ordering_preserved else '✗'} |")
        content.append(f"| 독립성 검증 | {validation.independence_verified} | {'✓' if validation.independence_verified else '✗'} |")
        content.append("")
        return content

    def _generate_cv_section_md(self, validation: CrossValidationValidation) -> List[str]:
        """교차 검증 섹션 마크다운 생성"""
        content = []
        content.append("### 2. 교차 검증 검증")
        content.append(f"**점수:** {validation.validation_score:.3f}/1.000")
        content.append("")
        content.append("| 항목 | 값 | 상태 |")
        content.append("|------|-----|------|")
        content.append(f"| 교차 검증 유형 | {validation.cv_type} | ✓ |")
        content.append(f"| 폴드 수 | {validation.n_folds} | {'✓' if validation.n_folds >= 5 else '✗'} |")
        content.append(f"| 계층화 적용 | {validation.is_stratified} | {'✓' if validation.is_stratified else '✗'} |")
        content.append(f"| 중첩 교차 검증 | {validation.nested_cv_used} | {'✓' if validation.nested_cv_used else '✗'} |")
        content.append(f"| 과적합 위험도 | {validation.overfitting_risk_assessment} | {'✓' if validation.overfitting_risk_assessment == 'low' else '⚠️' if validation.overfitting_risk_assessment == 'medium' else '✗'} |")
        content.append("")
        return content

    def _generate_hyperparameter_section_md(self, validation: HyperparameterTuningValidation) -> List[str]:
        """하이퍼파라미터 튜닝 섹션 마크다운 생성"""
        content = []
        content.append("### 3. 하이퍼파라미터 튜닝 검증")
        content.append(f"**점수:** {validation.validation_score:.3f}/1.000")
        content.append("")
        content.append("| 항목 | 값 | 상태 |")
        content.append("|------|-----|------|")
        content.append(f"| 튜닝 방법 | {validation.tuning_method} | ✓ |")
        content.append(f"| 탐색 공간 커버리지 | {validation.search_space_coverage:.1%} | {'✓' if validation.search_space_coverage > 0.5 else '✗'} |")
        content.append(f"| 랜덤 시드 고정 | {validation.random_seed_fixed} | {'✓' if validation.random_seed_fixed else '✗'} |")
        content.append(f"| 별도 검증 세트 | {validation.separate_validation_set} | {'✓' if validation.separate_validation_set else '✗'} |")
        content.append(f"| 중첩 교차 검증 | {validation.nested_cv_for_tuning} | {'✓' if validation.nested_cv_for_tuning else '✗'} |")
        content.append(f"| 수렴 달성 | {validation.convergence_achieved} | {'✓' if validation.convergence_achieved else '✗'} |")
        content.append("")
        return content

    def _generate_statistical_section_md(self, validation: StatisticalValidation) -> List[str]:
        """통계적 검증 섹션 마크다운 생성"""
        content = []
        content.append("### 4. 통계적 검증")
        content.append(f"**점수:** {validation.validation_score:.3f}/1.000")
        content.append("")
        content.append("| 항목 | 값 | 상태 |")
        content.append("|------|-----|------|")
        content.append(f"| 가설 명확성 | {validation.hypothesis_clearly_defined} | {'✓' if validation.hypothesis_clearly_defined else '✗'} |")
        content.append(f"| 적절한 검정 선택 | {validation.appropriate_test_selected} | {'✓' if validation.appropriate_test_selected else '✗'} |")
        content.append(f"| 가정 확인 | {validation.assumptions_checked} | {'✓' if validation.assumptions_checked else '✗'} |")
        content.append(f"| 효과 크기 보고 | {validation.effect_size_reported} | {'✓' if validation.effect_size_reported else '✗'} |")
        content.append(f"| 신뢰구간 제공 | {validation.confidence_intervals_provided} | {'✓' if validation.confidence_intervals_provided else '✗'} |")
        content.append(f"| 다중 비교 보정 | {validation.multiple_comparison_corrected} | {'✓' if validation.multiple_comparison_corrected else '✗'} |")
        content.append(f"| 표본 크기 적절성 | {validation.sample_size_adequate} | {'✓' if validation.sample_size_adequate else '✗'} |")
        content.append("")
        return content

    def _generate_model_section_md(self, validation: ModelValidation) -> List[str]:
        """모델 검증 섹션 마크다운 생성"""
        content = []
        content.append("### 5. 모델 검증")
        content.append(f"**점수:** {validation.validation_score:.3f}/1.000")
        content.append("")
        content.append("| 항목 | 값 | 상태 |")
        content.append("|------|-----|------|")
        content.append(f"| 아키텍처 정당화 | {validation.architecture_justified} | {'✓' if validation.architecture_justified else '✗'} |")
        content.append(f"| 베이스라인 비교 | {validation.baseline_comparison} | {'✓' if validation.baseline_comparison else '✗'} |")
        content.append(f"| Ablation 연구 | {validation.ablation_studies_conducted} | {'✓' if validation.ablation_studies_conducted else '✗'} |")
        content.append(f"| 특성 중요도 분석 | {validation.feature_importance_analyzed} | {'✓' if validation.feature_importance_analyzed else '✗'} |")
        content.append(f"| 해석가능성 | {validation.model_interpretability_addressed} | {'✓' if validation.model_interpretability_addressed else '✗'} |")
        content.append(f"| 계산 복잡도 분석 | {validation.computational_complexity_analyzed} | {'✓' if validation.computational_complexity_analyzed else '✗'} |")
        content.append("")
        return content

    def _generate_reproducibility_section_md(self, validation: ReproducibilityValidation) -> List[str]:
        """재현성 검증 섹션 마크다운 생성"""
        content = []
        content.append("### 6. 재현성 검증")
        content.append(f"**점수:** {validation.validation_score:.3f}/1.000")
        content.append("")
        content.append("| 항목 | 값 | 상태 |")
        content.append("|------|-----|------|")
        content.append(f"| 랜덤 시드 고정 | {validation.random_seeds_fixed} | {'✓' if validation.random_seeds_fixed else '✗'} |")
        content.append(f"| 환경 문서화 | {validation.environment_documented} | {'✓' if validation.environment_documented else '✗'} |")
        content.append(f"| 결정적 전처리 | {validation.data_preprocessing_deterministic} | {'✓' if validation.data_preprocessing_deterministic else '✗'} |")
        content.append(f"| 결정적 훈련 | {validation.model_training_deterministic} | {'✓' if validation.model_training_deterministic else '✗'} |")
        content.append(f"| 결과 재현성 | {validation.results_reproducible} | {'✓' if validation.results_reproducible else '✗'} |")
        content.append(f"| 코드 버전 관리 | {validation.code_version_controlled} | {'✓' if validation.code_version_controlled else '✗'} |")
        content.append("")
        return content

    def _generate_ethical_section_md(self, validation: EthicalValidation) -> List[str]:
        """윤리적 검증 섹션 마크다운 생성"""
        content = []
        content.append("### 7. 윤리적 검증")
        content.append(f"**점수:** {validation.validation_score:.3f}/1.000")
        content.append("")
        content.append("| 항목 | 값 | 상태 |")
        content.append("|------|-----|------|")
        content.append(f"| 편향 평가 | {validation.bias_assessment_conducted} | {'✓' if validation.bias_assessment_conducted else '✗'} |")
        content.append(f"| 공정성 메트릭 | {validation.fairness_metrics_evaluated} | {'✓' if validation.fairness_metrics_evaluated else '✗'} |")
        content.append(f"| 개인정보 고려 | {validation.privacy_considerations_addressed} | {'✓' if validation.privacy_considerations_addressed else '✗'} |")
        content.append(f"| 데이터 동의 | {validation.data_consent_verified} | {'✓' if validation.data_consent_verified else '✗'} |")
        content.append(f"| 위해 식별 | {validation.potential_harms_identified} | {'✓' if validation.potential_harms_identified else '✗'} |")
        content.append("")
        return content

    def _generate_detailed_scores_table_md(self, validation_report: MethodologyValidationReport) -> List[str]:
        """상세 점수 표 마크다운 생성"""
        content = []
        content.append("## 상세 점수표")
        content.append("| 검증 영역 | 점수 | 등급 | 상태 |")
        content.append("|----------|------|------|------|")

        validations = [
            ("데이터 분할", validation_report.data_split_validation.validation_score),
            ("교차 검증", validation_report.cv_validation.validation_score),
            ("하이퍼파라미터 튜닝", validation_report.hyperparameter_validation.validation_score),
            ("통계적 검증", validation_report.statistical_validation.validation_score),
            ("모델 검증", validation_report.model_validation.validation_score),
            ("재현성 검증", validation_report.reproducibility_validation.validation_score),
            ("윤리적 검증", validation_report.ethical_validation.validation_score),
        ]

        for name, score in validations:
            grade = self._get_grade(score)
            status = "✓" if score >= 0.8 else "⚠️" if score >= 0.6 else "✗"
            content.append(f"| {name} | {score:.3f} | {grade} | {status} |")

        content.append(f"| **전체** | **{validation_report.overall_score:.3f}** | **{self._get_grade(validation_report.overall_score)}** | **{'✓' if validation_report.overall_score >= 0.8 else '⚠️' if validation_report.overall_score >= 0.6 else '✗'}** |")
        content.append("")
        return content

    def _generate_validation_summary_table_html(self, validation_report: MethodologyValidationReport) -> str:
        """검증 요약 테이블 HTML 생성"""
        validations = [
            ("데이터 분할", validation_report.data_split_validation.validation_score),
            ("교차 검증", validation_report.cv_validation.validation_score),
            ("하이퍼파라미터 튜닝", validation_report.hyperparameter_validation.validation_score),
            ("통계적 검증", validation_report.statistical_validation.validation_score),
            ("모델 검증", validation_report.model_validation.validation_score),
            ("재현성 검증", validation_report.reproducibility_validation.validation_score),
            ("윤리적 검증", validation_report.ethical_validation.validation_score),
        ]

        html = "<table>"
        html += "<tr><th>검증 영역</th><th>점수</th><th>등급</th><th>상태</th></tr>"

        for name, score in validations:
            grade = self._get_grade(score)
            status_class = "metric-pass" if score >= 0.8 else "metric-fail"
            status_icon = "✓" if score >= 0.8 else "✗"
            html += f'<tr><td>{name}</td><td>{score:.3f}</td><td>{grade}</td><td class="{status_class}">{status_icon}</td></tr>'

        # 전체 점수
        overall_grade = self._get_grade(validation_report.overall_score)
        overall_status_class = "metric-pass" if validation_report.overall_score >= 0.8 else "metric-fail"
        overall_status_icon = "✓" if validation_report.overall_score >= 0.8 else "✗"
        html += f'<tr style="font-weight: bold; background-color: #f8f9fa;"><td>전체</td><td>{validation_report.overall_score:.3f}</td><td>{overall_grade}</td><td class="{overall_status_class}">{overall_status_icon}</td></tr>'

        html += "</table>"
        return html

    def _generate_validation_table_latex(self, validation_report: MethodologyValidationReport) -> str:
        """검증 결과 LaTeX 테이블 생성"""
        latex = "\\begin{table}[H]\\n"
        latex += "\\centering\\n"
        latex += "\\caption{방법론 검증 결과}\\n"
        latex += "\\begin{tabular}{lccc}\\n"
        latex += "\\toprule\\n"
        latex += "검증 영역 & 점수 & 등급 & 상태 \\\\\\n"
        latex += "\\midrule\\n"

        validations = [
            ("데이터 분할", validation_report.data_split_validation.validation_score),
            ("교차 검증", validation_report.cv_validation.validation_score),
            ("하이퍼파라미터 튜닝", validation_report.hyperparameter_validation.validation_score),
            ("통계적 검증", validation_report.statistical_validation.validation_score),
            ("모델 검증", validation_report.model_validation.validation_score),
            ("재현성 검증", validation_report.reproducibility_validation.validation_score),
            ("윤리적 검증", validation_report.ethical_validation.validation_score),
        ]

        for name, score in validations:
            grade = self._get_grade(score)
            status = "통과" if score >= 0.8 else "미흡"
            latex += f"{self._escape_latex(name)} & {score:.3f} & {self._escape_latex(grade)} & {self._escape_latex(status)} \\\\\\n"

        latex += "\\midrule\\n"
        overall_grade = self._get_grade(validation_report.overall_score)
        overall_status = "통과" if validation_report.overall_score >= 0.8 else "미흡"
        latex += f"\\textbf{{전체}} & \\textbf{{{validation_report.overall_score:.3f}}} & \\textbf{{{self._escape_latex(overall_grade)}}} & \\textbf{{{self._escape_latex(overall_status)}}} \\\\\\n"

        latex += "\\bottomrule\\n"
        latex += "\\end{tabular}\\n"
        latex += "\\end{table}\\n"

        return latex

    def _escape_latex(self, text: str) -> str:
        """LaTeX 특수 문자 이스케이프"""
        replacements = {
            '&': '\\&',
            '%': '\\%',
            '$': '\\$',
            '#': '\\#',
            '^': '\\textasciicircum{}',
            '_': '\\_',
            '{': '\\{',
            '}': '\\}',
            '~': '\\textasciitilde{}',
            '\\': '\\textbackslash{}'
        }

        for char, replacement in replacements.items():
            text = text.replace(char, replacement)

        return text

    def generate_all_formats(self, validation_report: MethodologyValidationReport) -> Dict[str, str]:
        """
        모든 형식의 보고서 생성

        Args:
            validation_report: 검증 보고서 객체

        Returns:
            생성된 파일 경로들의 딕셔너리
        """
        paths = {}

        # Markdown
        paths['markdown'] = self.generate_markdown_report(validation_report)

        # HTML
        paths['html'] = self.generate_html_report(validation_report)

        # LaTeX
        paths['latex'] = self.generate_latex_report(validation_report)

        # 그래프
        if self.config.include_figures:
            paths['figures'] = self._generate_validation_figures(validation_report)

        return paths

if __name__ == "__main__":
    # 테스트 예제는 methodology_validator와 함께 실행
    print("Validation report generator ready!")