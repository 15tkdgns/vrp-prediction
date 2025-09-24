#!/usr/bin/env python3
"""
회귀 기반 가격 예측 파이프라인
MAPE, R², MSE 최적화 통합 시스템
"""

import sys
import os
sys.path.append('/root/workspace/src')

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime

from core.data_processor import DataProcessor
from training.regression_metric_trainer import RegressionMetricTrainer
from evaluation.performance_evaluator import PerformanceEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegressionPipeline:
    """회귀 기반 가격 예측 파이프라인"""

    def __init__(self, config=None):
        self.config = config or self._get_default_config()
        self.data_processor = DataProcessor()
        self.model_trainer = RegressionMetricTrainer(gpu_enabled=self.config['gpu_enabled'])
        self.evaluator = PerformanceEvaluator()

        logger.info("회귀 파이프라인 초기화 완료")

    def _get_default_config(self):
        """기본 설정"""
        return {
            'data_path': '/root/workspace/data/training/sp500_2020_2024_enhanced.csv',
            'target_type': 'return',  # 회귀: 'return', 분류: 'direction'
            'sequence_length': 20,
            'cv_splits': 3,
            'gpu_enabled': True,
            'save_models': True,
            'save_results': True,
            'output_dir': '/root/workspace/data/results/regression'
        }

    def load_and_prepare_data(self):
        """데이터 로딩 및 전처리"""
        logger.info("=== 회귀 데이터 준비 단계 ===")

        # 데이터 로딩
        df = self.data_processor.load_and_validate_data(self.config['data_path'])
        if df is None:
            raise ValueError("데이터 로딩 실패")

        # ML용 데이터 준비 (회귀)
        X, y, feature_cols = self.data_processor.prepare_ml_data(df, self.config['target_type'])

        # 시퀀스 데이터 준비
        X_seq, y_seq, scaler = self.data_processor.prepare_sequence_data(
            df, self.config['sequence_length'], self.config['target_type']
        )

        # 데이터 무결성 검증
        integrity_check = self.data_processor.validate_data_integrity(X.values, y.values)
        logger.info(f"데이터 무결성 검사: {integrity_check}")

        # 학습/검증 분할
        splits = self.data_processor.create_train_val_split(X.values, y.values, self.config['cv_splits'])

        # 타겟 분포 분석
        logger.info(f"회귀 타겟 통계:")
        logger.info(f"  평균: {np.mean(y.values):.6f}")
        logger.info(f"  표준편차: {np.std(y.values):.6f}")
        logger.info(f"  최소값: {np.min(y.values):.6f}")
        logger.info(f"  최대값: {np.max(y.values):.6f}")

        return {
            'flat_data': (X.values, y.values),
            'sequence_data': (X_seq, y_seq),
            'feature_names': feature_cols,
            'splits': splits,
            'scaler': scaler,
            'integrity_check': integrity_check
        }

    def train_regression_models(self, data_dict):
        """회귀 모델 훈련"""
        logger.info("=== 회귀 모델 훈련 단계 ===")

        X, y = data_dict['flat_data']
        X_seq, y_seq = data_dict['sequence_data']
        splits = data_dict['splits']

        all_results = {}

        # 각 폴드별로 훈련 및 평가
        for fold, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"Fold {fold + 1}/{len(splits)} 회귀 훈련 시작")

            # 데이터 분할
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 시퀀스 데이터 분할 (인덱스 조정)
            seq_train_idx = train_idx[train_idx < len(X_seq)]
            seq_val_idx = val_idx[val_idx < len(X_seq)]

            if len(seq_train_idx) > 0 and len(seq_val_idx) > 0:
                X_seq_train, X_seq_val = X_seq[seq_train_idx], X_seq[seq_val_idx]
                sequence_data = (X_seq_train, X_seq_val)
            else:
                sequence_data = None

            # 회귀 모델 훈련
            fold_results = self.model_trainer.train_all_regression_models(
                X_train, X_val, y_train, y_val, sequence_data=sequence_data
            )

            # 결과 누적
            for model_name, result in fold_results.items():
                if model_name not in all_results:
                    all_results[model_name] = {
                        'fold_mse': [],
                        'fold_mae': [],
                        'fold_rmse': [],
                        'fold_mape': [],
                        'fold_r2': [],
                        'fold_direction_accuracy': [],
                        'type': result['type']
                    }

                all_results[model_name]['fold_mse'].append(result['mse'])
                all_results[model_name]['fold_mae'].append(result['mae'])
                all_results[model_name]['fold_rmse'].append(result['rmse'])
                all_results[model_name]['fold_mape'].append(result['mape'])
                all_results[model_name]['fold_r2'].append(result['r2'])
                all_results[model_name]['fold_direction_accuracy'].append(result['direction_accuracy'])

        # 평균 성능 계산
        final_results = {}
        for model_name, data in all_results.items():
            final_results[model_name] = {
                'mean_mse': np.mean(data['fold_mse']),
                'std_mse': np.std(data['fold_mse']),
                'mean_mae': np.mean(data['fold_mae']),
                'std_mae': np.std(data['fold_mae']),
                'mean_rmse': np.mean(data['fold_rmse']),
                'std_rmse': np.std(data['fold_rmse']),
                'mean_mape': np.mean(data['fold_mape']),
                'std_mape': np.std(data['fold_mape']),
                'mean_r2': np.mean(data['fold_r2']),
                'std_r2': np.std(data['fold_r2']),
                'mean_direction_accuracy': np.mean(data['fold_direction_accuracy']),
                'std_direction_accuracy': np.std(data['fold_direction_accuracy']),
                'fold_results': data,
                'type': data['type']
            }

        return final_results

    def analyze_regression_performance(self, results):
        """회귀 성능 분석"""
        logger.info("=== 회귀 성능 분석 ===")

        analysis = {
            'best_by_mse': None,
            'best_by_mae': None,
            'best_by_mape': None,
            'best_by_r2': None,
            'best_by_direction': None,
            'metric_rankings': {},
            'regression_analysis': {}
        }

        # 각 지표별 최고 모델 찾기
        best_mse = float('inf')
        best_mae = float('inf')
        best_mape = float('inf')
        best_r2 = float('-inf')
        best_direction = 0

        for model_name, result in results.items():
            # MSE 기준 (낮을수록 좋음)
            if result['mean_mse'] < best_mse:
                best_mse = result['mean_mse']
                analysis['best_by_mse'] = {
                    'model': model_name,
                    'score': best_mse,
                    'details': result
                }

            # MAE 기준 (낮을수록 좋음)
            if result['mean_mae'] < best_mae:
                best_mae = result['mean_mae']
                analysis['best_by_mae'] = {
                    'model': model_name,
                    'score': best_mae,
                    'details': result
                }

            # MAPE 기준 (낮을수록 좋음)
            if result['mean_mape'] < best_mape:
                best_mape = result['mean_mape']
                analysis['best_by_mape'] = {
                    'model': model_name,
                    'score': best_mape,
                    'details': result
                }

            # R² 기준 (높을수록 좋음)
            if result['mean_r2'] > best_r2:
                best_r2 = result['mean_r2']
                analysis['best_by_r2'] = {
                    'model': model_name,
                    'score': best_r2,
                    'details': result
                }

            # 방향 정확도 기준 (높을수록 좋음)
            if result['mean_direction_accuracy'] > best_direction:
                best_direction = result['mean_direction_accuracy']
                analysis['best_by_direction'] = {
                    'model': model_name,
                    'score': best_direction,
                    'details': result
                }

        # 지표별 순위 매기기
        for metric in ['mean_mse', 'mean_mae', 'mean_mape']:  # 낮을수록 좋음
            ranking = sorted(results.items(),
                           key=lambda x: x[1][metric],
                           reverse=False)
            analysis['metric_rankings'][metric] = ranking

        for metric in ['mean_r2', 'mean_direction_accuracy']:  # 높을수록 좋음
            ranking = sorted(results.items(),
                           key=lambda x: x[1][metric],
                           reverse=True)
            analysis['metric_rankings'][metric] = ranking

        return analysis

    def run_regression_pipeline(self):
        """전체 회귀 파이프라인 실행"""
        logger.info("=" * 80)
        logger.info("🚀 회귀 기반 가격 예측 파이프라인 시작")
        logger.info("=" * 80)

        start_time = datetime.now()

        try:
            # 1. 데이터 준비
            data_dict = self.load_and_prepare_data()

            # 2. 회귀 모델 훈련
            model_results = self.train_regression_models(data_dict)

            # 3. 회귀 성능 분석
            regression_analysis = self.analyze_regression_performance(model_results)

            # 4. 종합 결과
            pipeline_results = {
                'pipeline_config': self.config,
                'execution_time': str(datetime.now() - start_time),
                'data_info': {
                    'feature_count': len(data_dict['feature_names']),
                    'sample_count': len(data_dict['flat_data'][1]),
                    'sequence_length': self.config['sequence_length'],
                    'target_type': self.config['target_type']
                },
                'model_results': model_results,
                'regression_analysis': regression_analysis,
                'data_integrity': data_dict['integrity_check']
            }

            # 5. 모델 저장
            if self.config['save_models']:
                self.model_trainer.save_regression_models()

            # 6. 결과 저장
            if self.config['save_results']:
                self._save_results(pipeline_results)

            # 7. 결과 출력
            self._print_regression_summary(pipeline_results)

            logger.info("✅ 회귀 파이프라인 실행 완료")
            return pipeline_results

        except Exception as e:
            logger.error(f"❌ 회귀 파이프라인 실행 실패: {e}")
            raise

    def _save_results(self, results):
        """결과 저장"""
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON 직렬화를 위한 변환
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj

        def clean_for_json(data):
            if isinstance(data, dict):
                return {k: clean_for_json(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_for_json(item) for item in data]
            else:
                return convert_numpy(data)

        cleaned_results = clean_for_json(results)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"regression_results_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(cleaned_results, f, indent=2)

        logger.info(f"회귀 결과 저장: {results_file}")

    def _print_regression_summary(self, results):
        """회귀 파이프라인 결과 요약 출력"""
        print("\n" + "=" * 80)
        print("📊 회귀 기반 가격 예측 결과")
        print("=" * 80)

        # 기본 정보
        print(f"\n📋 실행 정보:")
        print(f"   실행 시간: {results['execution_time']}")
        print(f"   특징 수: {results['data_info']['feature_count']}")
        print(f"   샘플 수: {results['data_info']['sample_count']}")

        # 지표별 최고 성능
        analysis = results['regression_analysis']

        print(f"\n🏆 지표별 최고 성능:")
        print("-" * 80)

        if analysis['best_by_mse']:
            mse_best = analysis['best_by_mse']
            print(f"MSE 최저:        {mse_best['model']:30s} {mse_best['score']:.8f}")

        if analysis['best_by_mae']:
            mae_best = analysis['best_by_mae']
            print(f"MAE 최저:        {mae_best['model']:30s} {mae_best['score']:.6f}")

        if analysis['best_by_mape']:
            mape_best = analysis['best_by_mape']
            print(f"MAPE 최저:       {mape_best['model']:30s} {mape_best['score']:.2f}%")

        if analysis['best_by_r2']:
            r2_best = analysis['best_by_r2']
            print(f"R² 최고:         {r2_best['model']:30s} {r2_best['score']:.4f}")

        if analysis['best_by_direction']:
            dir_best = analysis['best_by_direction']
            print(f"방향 정확도 최고: {dir_best['model']:30s} {dir_best['score']:.2f}%")

        # 종합 순위 (MAPE 기준)
        print(f"\n📈 종합 성능 순위 (MAPE 기준):")
        print("-" * 80)

        if 'mean_mape' in analysis['metric_rankings']:
            for i, (model_name, result) in enumerate(analysis['metric_rankings']['mean_mape'][:5], 1):
                mape = result['mean_mape']
                r2 = result['mean_r2']
                direction = result['mean_direction_accuracy']
                print(f"{i:2d}. {model_name:30s} | MAPE: {mape:6.2f}% | R²: {r2:7.4f} | 방향: {direction:5.1f}%")

        print("\n" + "=" * 80)

def main():
    """메인 실행 함수"""
    config = {
        'data_path': '/root/workspace/data/training/sp500_2020_2024_enhanced.csv',
        'target_type': 'return',  # 회귀 모드
        'sequence_length': 20,
        'cv_splits': 3,
        'gpu_enabled': True,
        'save_models': True,
        'save_results': True,
        'output_dir': '/root/workspace/data/results/regression'
    }

    # 파이프라인 실행
    pipeline = RegressionPipeline(config)
    results = pipeline.run_regression_pipeline()

    return results

if __name__ == "__main__":
    results = main()