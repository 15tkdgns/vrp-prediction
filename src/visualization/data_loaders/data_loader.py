"""
통합 데이터 로더
성능 데이터, XAI 분석 결과, 백테스트 데이터 등을 로드
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

class VisualizationDataLoader:
    """시각화를 위한 통합 데이터 로더"""

    def __init__(self, base_path: str = "/root/workspace"):
        self.base_path = Path(base_path)
        self.data_path = self.base_path / "data"
        self.logger = logging.getLogger(__name__)

    def load_model_performance(self) -> Dict[str, Any]:
        """모델 성능 데이터 로드"""
        try:
            performance_file = self.data_path / "raw" / "model_performance.json"
            with open(performance_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('ridge_regression_volatility_predictor', {})
        except FileNotFoundError:
            self.logger.error(f"성능 데이터 파일을 찾을 수 없습니다: {performance_file}")
            return {}
        except Exception as e:
            self.logger.error(f"성능 데이터 로드 오류: {e}")
            return {}

    def load_xai_analysis(self) -> Dict[str, Any]:
        """XAI 분석 결과 로드 (가장 최신 파일)"""
        try:
            xai_dir = self.data_path / "xai_analysis"
            xai_files = list(xai_dir.glob("verified_xai_analysis_*.json"))

            if not xai_files:
                # fallback to processed directory
                xai_files = list((self.data_path / "processed").glob("xai_*.json"))

            if not xai_files:
                self.logger.warning("XAI 분석 파일을 찾을 수 없습니다")
                return {}

            # 가장 최신 파일 선택
            latest_file = max(xai_files, key=lambda p: p.stat().st_mtime)

            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"XAI 데이터 로드 오류: {e}")
            return {}

    def get_model_comparison_data(self) -> Dict[str, float]:
        """모델 비교를 위한 R² 스코어 데이터"""
        perf_data = self.load_model_performance()

        if not perf_data:
            return {}

        benchmark_results = perf_data.get('benchmark_results', {})

        return {
            'Ridge': benchmark_results.get('our_model_ridge', 0.3113),
            'HAR': benchmark_results.get('har_benchmark', 0.0088),
            'Random Forest': benchmark_results.get('random_forest', 0.2447),
            'ElasticNet': benchmark_results.get('elasticnet', -0.1773)
        }

    def get_performance_metrics(self) -> Dict[str, float]:
        """주요 성능 지표 추출"""
        perf_data = self.load_model_performance()

        if not perf_data:
            return {}

        return {
            'r2_score': perf_data.get('r2', 0.0),
            'r2_std': perf_data.get('r2_std', 0.0),
            'mse': perf_data.get('mse', 0.0),
            'rmse': perf_data.get('rmse', 0.0),
            'mae': perf_data.get('mae', 0.0),
            'samples': perf_data.get('samples', 0),
            'features': perf_data.get('features', 0)
        }

    def get_feature_importance_data(self) -> List[Dict[str, Any]]:
        """특성 중요도 데이터 추출"""
        xai_data = self.load_xai_analysis()

        if not xai_data:
            return []

        shap_analysis = xai_data.get('shap_analysis', {})
        feature_importance = shap_analysis.get('feature_importance', [])

        # 상위 15개 특성만 반환
        return feature_importance[:15] if len(feature_importance) > 15 else feature_importance

    def get_economic_metrics(self) -> Dict[str, float]:
        """경제적 백테스트 지표 추출"""
        perf_data = self.load_model_performance()

        if not perf_data:
            return {}

        backtest_results = perf_data.get('economic_backtest_results', {})

        return {
            'strategy_annual_return': backtest_results.get('strategy_annual_return', 0.0),
            'benchmark_annual_return': backtest_results.get('benchmark_annual_return', 0.0),
            'strategy_volatility': backtest_results.get('strategy_volatility', 0.0),
            'benchmark_volatility': backtest_results.get('benchmark_volatility', 0.0),
            'strategy_sharpe': backtest_results.get('strategy_sharpe', 0.0),
            'strategy_max_drawdown': backtest_results.get('strategy_max_drawdown', 0.0),
            'total_transaction_costs': backtest_results.get('total_transaction_costs', 0.0),
            'volatility_reduction': backtest_results.get('volatility_reduction', 0.0)
        }

    def get_cross_validation_data(self) -> Dict[str, Any]:
        """교차검증 설정 및 결과 데이터"""
        perf_data = self.load_model_performance()

        if not perf_data:
            return {}

        validation_details = perf_data.get('validation_details', {})
        purged_kfold = validation_details.get('purged_kfold', {})

        return {
            'n_splits': purged_kfold.get('n_splits', 5),
            'purge_length': purged_kfold.get('purge_length', 5),
            'embargo_length': purged_kfold.get('embargo_length', 5),
            'r2_score': perf_data.get('r2', 0.0),
            'r2_std': perf_data.get('r2_std', 0.0)
        }

    def get_data_summary(self) -> Dict[str, Any]:
        """전체 데이터 요약 정보"""
        perf_data = self.load_model_performance()
        xai_data = self.load_xai_analysis()

        return {
            'model_performance_available': bool(perf_data),
            'xai_analysis_available': bool(xai_data),
            'data_period': perf_data.get('data_period', 'Unknown'),
            'samples': perf_data.get('samples', 0),
            'features': perf_data.get('features', 0),
            'cv_method': perf_data.get('cv_method', 'Unknown'),
            'model_type': perf_data.get('method', 'Unknown')
        }

if __name__ == "__main__":
    # 데이터 로더 테스트
    loader = VisualizationDataLoader()

    print("=== 데이터 로더 테스트 ===")
    print(f"모델 비교 데이터: {loader.get_model_comparison_data()}")
    print(f"성능 지표: {loader.get_performance_metrics()}")
    print(f"경제적 지표: {loader.get_economic_metrics()}")
    print(f"데이터 요약: {loader.get_data_summary()}")