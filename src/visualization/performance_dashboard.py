"""
수치 기반 성능 시각화 시스템 - 메인 대시보드 생성기
모든 차트 생성 모듈을 통합 관리
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json
import logging
from typing import Dict, Any, List

# 현재 디렉토리를 Python path에 추가
current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))

# 차트 생성 모듈들
from src.visualization.data_loaders.data_loader import VisualizationDataLoader
from src.visualization.charts.model_metrics import ModelMetricsCharts
from src.visualization.charts.feature_analysis import FeatureAnalysisCharts
from src.visualization.charts.economic_metrics import EconomicMetricsCharts
from src.visualization.charts.validation_results import ValidationResultsCharts

class PerformanceDashboard:
    """수치 기반 성능 시각화 시스템 메인 컨트롤러"""

    def __init__(self, base_path: str = "/root/workspace", output_dir: str = None):
        self.base_path = Path(base_path)
        self.output_dir = Path(output_dir) if output_dir else self.base_path / "src" / "visualization" / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 로깅 설정
        self.setup_logging()

        # 데이터 로더 초기화
        self.data_loader = VisualizationDataLoader(base_path)

        # 차트 생성기들 초기화
        self.model_charts = ModelMetricsCharts(str(self.output_dir))
        self.feature_charts = FeatureAnalysisCharts(str(self.output_dir))
        self.economic_charts = EconomicMetricsCharts(str(self.output_dir))
        self.validation_charts = ValidationResultsCharts(str(self.output_dir))

        # 생성된 파일들 추적
        self.generated_files = []

        self.logger.info(f"성능 대시보드 초기화 완료 - 출력 디렉토리: {self.output_dir}")

    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.output_dir / "dashboard_generation.log")
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_all_data(self) -> Dict[str, Any]:
        """모든 필요한 데이터 로드"""
        self.logger.info("데이터 로드 중...")

        data = {
            'model_performance': self.data_loader.load_model_performance(),
            'xai_analysis': self.data_loader.load_xai_analysis(),
            'model_comparison': self.data_loader.get_model_comparison_data(),
            'performance_metrics': self.data_loader.get_performance_metrics(),
            'feature_importance': self.data_loader.get_feature_importance_data(),
            'economic_metrics': self.data_loader.get_economic_metrics(),
            'cv_data': self.data_loader.get_cross_validation_data(),
            'data_summary': self.data_loader.get_data_summary()
        }

        # 데이터 유효성 검증
        self.validate_data(data)

        self.logger.info("데이터 로드 완료")
        return data

    def validate_data(self, data: Dict[str, Any]):
        """데이터 유효성 검증"""
        required_keys = ['model_performance', 'performance_metrics']
        missing_keys = [key for key in required_keys if not data.get(key)]

        if missing_keys:
            self.logger.warning(f"누락된 데이터: {missing_keys}")

        # 주요 지표 존재 여부 확인
        metrics = data.get('performance_metrics', {})
        if not metrics.get('r2_score'):
            self.logger.warning("R² 스코어 데이터가 없습니다.")

        feature_importance = data.get('feature_importance', [])
        if not feature_importance:
            self.logger.warning("특성 중요도 데이터가 없습니다.")

    def generate_model_performance_charts(self, data: Dict[str, Any]) -> List[str]:
        """모델 성능 차트 생성"""
        self.logger.info("=== 모델 성능 차트 생성 시작 ===")

        try:
            model_comparison = data.get('model_comparison', {})
            performance_metrics = data.get('performance_metrics', {})
            cv_data = data.get('cv_data', {})

            if not any([model_comparison, performance_metrics]):
                self.logger.error("모델 성능 데이터가 부족합니다.")
                return []

            files = self.model_charts.generate_all_model_charts(
                model_comparison, performance_metrics, cv_data
            )

            self.generated_files.extend(files)
            self.logger.info(f"모델 성능 차트 {len(files)}개 생성 완료")
            return files

        except Exception as e:
            self.logger.error(f"모델 성능 차트 생성 오류: {e}")
            return []

    def generate_feature_analysis_charts(self, data: Dict[str, Any]) -> List[str]:
        """특성 분석 차트 생성"""
        self.logger.info("=== 특성 분석 차트 생성 시작 ===")

        try:
            feature_importance = data.get('feature_importance', [])

            if not feature_importance:
                self.logger.warning("특성 중요도 데이터가 없어 차트를 생성할 수 없습니다.")
                return []

            files = self.feature_charts.generate_all_feature_charts(feature_importance)

            self.generated_files.extend(files)
            self.logger.info(f"특성 분석 차트 {len(files)}개 생성 완료")
            return files

        except Exception as e:
            self.logger.error(f"특성 분석 차트 생성 오류: {e}")
            return []

    def generate_economic_analysis_charts(self, data: Dict[str, Any]) -> List[str]:
        """경제적 분석 차트 생성"""
        self.logger.info("=== 경제적 분석 차트 생성 시작 ===")

        try:
            economic_metrics = data.get('economic_metrics', {})

            if not economic_metrics:
                self.logger.warning("경제적 지표 데이터가 없어 차트를 생성할 수 없습니다.")
                return []

            files = self.economic_charts.generate_all_economic_charts(economic_metrics)

            self.generated_files.extend(files)
            self.logger.info(f"경제적 분석 차트 {len(files)}개 생성 완료")
            return files

        except Exception as e:
            self.logger.error(f"경제적 분석 차트 생성 오류: {e}")
            return []

    def generate_validation_charts(self, data: Dict[str, Any]) -> List[str]:
        """검증 결과 차트 생성"""
        self.logger.info("=== 검증 결과 차트 생성 시작 ===")

        try:
            performance_metrics = data.get('performance_metrics', {})
            cv_data = data.get('cv_data', {})

            if not performance_metrics:
                self.logger.error("성능 지표 데이터가 부족합니다.")
                return []

            files = self.validation_charts.generate_all_validation_charts(
                performance_metrics, cv_data
            )

            self.generated_files.extend(files)
            self.logger.info(f"검증 결과 차트 {len(files)}개 생성 완료")
            return files

        except Exception as e:
            self.logger.error(f"검증 결과 차트 생성 오류: {e}")
            return []

    def create_summary_report(self, data: Dict[str, Any]) -> str:
        """요약 리포트 JSON 생성"""
        timestamp = datetime.now().isoformat()

        summary = {
            'generation_info': {
                'timestamp': timestamp,
                'total_charts': len(self.generated_files),
                'output_directory': str(self.output_dir)
            },
            'data_summary': data.get('data_summary', {}),
            'performance_metrics': data.get('performance_metrics', {}),
            'economic_metrics': data.get('economic_metrics', {}),
            'model_comparison': data.get('model_comparison', {}),
            'generated_files': {
                'total_count': len(self.generated_files),
                'files': [str(Path(f).name) for f in self.generated_files if f]
            }
        }

        # JSON 파일로 저장
        summary_path = self.output_dir / f"dashboard_summary_{timestamp.split('T')[0]}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"요약 리포트 저장: {summary_path}")
        return str(summary_path)

    def generate_all_charts(self) -> Dict[str, Any]:
        """모든 차트 생성 - 메인 실행 함수"""
        start_time = datetime.now()
        self.logger.info("🚀 수치 기반 성능 시각화 시스템 시작")

        try:
            # 1. 데이터 로드
            data = self.load_all_data()

            # 2. 모든 차트 생성
            model_files = self.generate_model_performance_charts(data)
            feature_files = self.generate_feature_analysis_charts(data)
            economic_files = self.generate_economic_analysis_charts(data)
            validation_files = self.generate_validation_charts(data)

            # 3. 요약 리포트 생성
            summary_file = self.create_summary_report(data)

            # 4. 결과 정리
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            results = {
                'success': True,
                'total_charts': len(self.generated_files),
                'duration_seconds': duration,
                'output_directory': str(self.output_dir),
                'generated_files': {
                    'model_performance': model_files,
                    'feature_analysis': feature_files,
                    'economic_analysis': economic_files,
                    'validation_results': validation_files,
                    'summary_report': summary_file
                },
                'data_status': {
                    'model_performance_available': bool(data.get('model_performance')),
                    'xai_analysis_available': bool(data.get('feature_importance')),
                    'economic_metrics_available': bool(data.get('economic_metrics'))
                }
            }

            self.logger.info(f"✅ 시각화 시스템 완료 - {len(self.generated_files)}개 차트, {duration:.1f}초 소요")

            return results

        except Exception as e:
            self.logger.error(f"❌ 시각화 시스템 오류: {e}")
            return {
                'success': False,
                'error': str(e),
                'output_directory': str(self.output_dir)
            }

    def print_results_summary(self, results: Dict[str, Any]):
        """결과 요약 출력"""
        print("\n" + "="*60)
        print("수치 기반 성능 시각화 시스템 - 실행 결과")
        print("="*60)

        if results.get('success'):
            print(f"✅ 성공: {results['total_charts']}개 차트 생성")
            print(f"⏱️  실행 시간: {results['duration_seconds']:.1f}초")
            print(f"📁 출력 디렉토리: {results['output_directory']}")

            print(f"\n📊 생성된 차트:")
            for category, files in results['generated_files'].items():
                if isinstance(files, list) and files:
                    print(f"  • {category}: {len(files)}개")
                    for file in files[:3]:  # 처음 3개만 표시
                        print(f"    - {Path(file).name}")
                    if len(files) > 3:
                        print(f"    - ... 외 {len(files)-3}개")

            print(f"\n📈 데이터 상태:")
            data_status = results.get('data_status', {})
            for key, status in data_status.items():
                status_icon = "✅" if status else "❌"
                print(f"  {status_icon} {key}: {'사용 가능' if status else '없음'}")

        else:
            print(f"❌ 실패: {results.get('error', '알 수 없는 오류')}")

        print("="*60)


def main():
    """메인 실행 함수"""
    try:
        # 대시보드 생성기 초기화
        dashboard = PerformanceDashboard()

        # 모든 차트 생성
        results = dashboard.generate_all_charts()

        # 결과 출력
        dashboard.print_results_summary(results)

        return results

    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
        return {'success': False, 'error': 'User interrupted'}

    except Exception as e:
        print(f"\n예상치 못한 오류 발생: {e}")
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    results = main()
    exit_code = 0 if results.get('success') else 1
    sys.exit(exit_code)