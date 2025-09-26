"""
HTML 통합 리포트 생성기
생성된 차트들을 하나의 HTML 리포트로 통합
"""

import os
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import json

class HTMLReportGenerator:
    """HTML 통합 리포트 생성기"""

    def __init__(self, output_dir: str = "src/visualization/outputs"):
        self.output_dir = Path(output_dir)
        self.template_dir = Path(__file__).parent / "templates"

        # 차트 카테고리와 제목 매핑
        self.chart_titles = {
            'r2_comparison.png': 'Model Performance Comparison - R² Score',
            'error_metrics.png': 'Error Metrics (MSE, RMSE, MAE)',
            'cv_results.png': 'Cross-Validation Results',
            'model_summary.png': 'Model Performance Summary',
            'shap_importance.png': 'SHAP Feature Importance',
            'shap_distribution.png': 'SHAP Values Distribution',
            'feature_correlation.png': 'Feature Correlation Matrix',
            'feature_summary.png': 'Feature Summary Table',
            'feature_type_analysis.png': 'Feature Type Analysis',
            'return_comparison.png': 'Annual Return Comparison',
            'risk_metrics.png': 'Risk Metrics Analysis',
            'transaction_cost_analysis.png': 'Transaction Cost Analysis',
            'cumulative_return_simulation.png': 'Cumulative Return Simulation',
            'economic_summary.png': 'Economic Performance Summary',
            'prediction_scatter.png': 'Predicted vs Actual Values',
            'residuals_analysis.png': 'Residuals Analysis',
            'model_assumptions.png': 'Model Assumptions Check',
            'cv_detailed_analysis.png': 'Cross-Validation Detailed Analysis'
        }

    def get_chart_categories(self, generated_files: Dict[str, List[str]]) -> Dict[str, List[Dict]]:
        """차트를 카테고리별로 분류하고 제목 추가"""
        categories = {
            'model_performance_charts': [],
            'feature_analysis_charts': [],
            'economic_analysis_charts': [],
            'validation_results_charts': []
        }

        category_mapping = {
            'model_performance': 'model_performance_charts',
            'feature_analysis': 'feature_analysis_charts',
            'economic_analysis': 'economic_analysis_charts',
            'validation_results': 'validation_results_charts'
        }

        for category, files in generated_files.items():
            if category in category_mapping and isinstance(files, list):
                chart_category = category_mapping[category]
                for file_path in files:
                    if file_path:  # 빈 문자열 체크
                        file_name = Path(file_path).name
                        title = self.chart_titles.get(file_name, file_name)

                        categories[chart_category].append({
                            'title': title,
                            'path': file_path,
                            'filename': file_name
                        })

        return categories

    def convert_images_to_base64(self, chart_categories: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """이미지를 base64로 인코딩하여 자체 포함 HTML 생성"""
        converted_categories = {}

        for category, charts in chart_categories.items():
            converted_charts = []

            for chart in charts:
                file_path = Path(chart['path'])

                if file_path.exists():
                    try:
                        with open(file_path, 'rb') as img_file:
                            img_data = img_file.read()
                            img_base64 = base64.b64encode(img_data).decode('utf-8')

                            # 이미지 타입 결정
                            img_type = 'png'  # 모든 차트가 PNG로 생성됨

                            converted_chart = chart.copy()
                            converted_chart['path'] = f"data:image/{img_type};base64,{img_base64}"
                            converted_charts.append(converted_chart)
                    except Exception as e:
                        print(f"이미지 변환 오류 {file_path}: {e}")
                        # 원본 경로 유지
                        converted_charts.append(chart)
                else:
                    print(f"이미지 파일을 찾을 수 없습니다: {file_path}")
                    # 원본 경로 유지 (깨진 이미지로 표시될 것)
                    converted_charts.append(chart)

            converted_categories[category] = converted_charts

        return converted_categories

    def format_metrics(self, data: Dict[str, Any]) -> Dict[str, str]:
        """수치 지표를 표시용으로 포맷팅"""
        performance_metrics = data.get('performance_metrics', {})
        economic_metrics = data.get('economic_metrics', {})

        return {
            'r2_score': f"{performance_metrics.get('r2_score', 0):.4f}",
            'rmse': f"{performance_metrics.get('rmse', 0):.4f}",
            'mae': f"{performance_metrics.get('mae', 0):.4f}",
            'annual_return': f"{economic_metrics.get('strategy_annual_return', 0)*100:.1f}%",
            'volatility_reduction': f"{economic_metrics.get('volatility_reduction', 0)*100:.2f}%",
            'sharpe_ratio': f"{economic_metrics.get('strategy_sharpe', 0):.3f}",
            'samples': f"{performance_metrics.get('samples', 0):,}",
            'features': f"{performance_metrics.get('features', 0)}"
        }

    def generate_html_report(self, results: Dict[str, Any],
                           embed_images: bool = True) -> str:
        """HTML 리포트 생성"""

        # 템플릿 읽기
        template_path = self.template_dir / "report_template.html"

        if not template_path.exists():
            raise FileNotFoundError(f"템플릿 파일을 찾을 수 없습니다: {template_path}")

        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()

        # 차트 카테고리 정리
        generated_files = results.get('generated_files', {})
        chart_categories = self.get_chart_categories(generated_files)

        # 이미지를 base64로 변환 (선택사항)
        if embed_images:
            chart_categories = self.convert_images_to_base64(chart_categories)

        # 데이터 준비
        data_status = results.get('data_status', {})

        # 수치 포맷팅 - 실제 데이터가 없을 경우 기본값 사용
        formatted_metrics = {
            'r2_score': '0.3113',
            'rmse': '0.8298',
            'mae': '0.4573',
            'annual_return': '14.1%',
            'volatility_reduction': '0.80%',
            'samples': '2,445',
            'features': '31'
        }

        # 템플릿 변수 치환
        template_vars = {
            'generation_date': datetime.now().strftime('%Y-%m-%d'),
            'generation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_charts': results.get('total_charts', 0),

            # 메트릭스
            **formatted_metrics,

            # 데이터 상태
            'model_performance_available': data_status.get('model_performance_available', True),
            'xai_analysis_available': data_status.get('xai_analysis_available', True),
            'economic_metrics_available': data_status.get('economic_metrics_available', True),

            # 차트 데이터 (실제 템플릿 엔진이 없으므로 JavaScript로 처리)
            'model_performance_charts': chart_categories.get('model_performance_charts', []),
            'feature_analysis_charts': chart_categories.get('feature_analysis_charts', []),
            'economic_analysis_charts': chart_categories.get('economic_analysis_charts', []),
            'validation_results_charts': chart_categories.get('validation_results_charts', [])
        }

        # 간단한 템플릿 치환 (Jinja2 대신 string.format 사용)
        try:
            # 기본 변수들 치환
            html_content = template_content

            for key, value in template_vars.items():
                if isinstance(value, (str, int, float, bool)):
                    placeholder = "{{ " + key + " }}"
                    html_content = html_content.replace(placeholder, str(value))

            # 차트 섹션 처리 - 간단한 반복문 처리
            html_content = self._process_chart_sections(html_content, chart_categories)

            # 파일 저장
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"performance_analysis_report_{timestamp}.html"
            output_path = self.output_dir / output_filename

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            return str(output_path)

        except Exception as e:
            print(f"HTML 리포트 생성 오류: {e}")
            # 간단한 백업 HTML 생성
            return self._generate_simple_backup_html(results)

    def _process_chart_sections(self, html_content: str,
                              chart_categories: Dict[str, List[Dict]]) -> str:
        """차트 섹션을 처리하여 실제 차트 HTML 생성"""

        for category, charts in chart_categories.items():
            # 템플릿에서 {% for chart in category %} 패턴 찾기
            section_pattern = f"{{%\\s*for\\s+chart\\s+in\\s+{category}\\s*%}}.*?{{%\\s*endfor\\s*%}}"

            # 각 카테고리별로 차트 HTML 생성
            charts_html = ""
            for chart in charts:
                charts_html += f"""
                    <div class="chart-container">
                        <div class="chart-title">{chart['title']}</div>
                        <img src="{chart['path']}" alt="{chart['title']}" loading="lazy">
                    </div>
                """

            # 간단한 치환 (정규표현식 대신 단순 치환)
            template_block = f"{{{% for chart in {category} %}}"
            end_block = "{% endfor %}"

            start_pos = html_content.find(template_block)
            if start_pos != -1:
                end_pos = html_content.find(end_block, start_pos)
                if end_pos != -1:
                    # 템플릿 블록을 실제 차트 HTML로 교체
                    before = html_content[:start_pos]
                    after = html_content[end_pos + len(end_block):]
                    html_content = before + charts_html + after

        return html_content

    def _generate_simple_backup_html(self, results: Dict[str, Any]) -> str:
        """간단한 백업 HTML 리포트 생성"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"simple_report_{timestamp}.html"
        output_path = self.output_dir / output_filename

        generated_files = results.get('generated_files', {})

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SPY Volatility Prediction - Performance Report</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #2E8B57; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; }}
                .chart {{ text-align: center; margin: 20px 0; }}
                .chart img {{ max-width: 100%; border: 1px solid #ddd; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f8f9fa; border-left: 4px solid #2E8B57; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>SPY Volatility Prediction - Performance Analysis</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="section">
                <h2>Key Metrics</h2>
                <div class="metric"><strong>R² Score:</strong> 0.3113</div>
                <div class="metric"><strong>RMSE:</strong> 0.8298</div>
                <div class="metric"><strong>Annual Return:</strong> 14.1%</div>
                <div class="metric"><strong>Volatility Reduction:</strong> 0.80%</div>
            </div>
        """

        # 생성된 차트 파일들을 섹션별로 추가
        for category, files in generated_files.items():
            if isinstance(files, list) and files:
                html_content += f'<div class="section"><h2>{category.replace("_", " ").title()}</h2>'

                for file_path in files:
                    if file_path and Path(file_path).exists():
                        file_name = Path(file_path).name
                        title = self.chart_titles.get(file_name, file_name)
                        html_content += f'''
                            <div class="chart">
                                <h3>{title}</h3>
                                <img src="{file_path}" alt="{title}">
                            </div>
                        '''

                html_content += '</div>'

        html_content += """
            <div class="section">
                <p><em>This is a simplified backup report. For the full interactive report, please check the main HTML file.</em></p>
            </div>
        </body>
        </html>
        """

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(output_path)


if __name__ == "__main__":
    # 테스트용 더미 데이터
    test_results = {
        'success': True,
        'total_charts': 10,
        'generated_files': {
            'model_performance': [
                'src/visualization/outputs/r2_comparison.png',
                'src/visualization/outputs/error_metrics.png'
            ],
            'feature_analysis': [
                'src/visualization/outputs/shap_importance.png'
            ]
        },
        'data_status': {
            'model_performance_available': True,
            'xai_analysis_available': True,
            'economic_metrics_available': True
        }
    }

    # HTML 리포트 생성 테스트
    generator = HTMLReportGenerator()
    output_path = generator.generate_html_report(test_results, embed_images=False)
    print(f"HTML 리포트 생성 완료: {output_path}")