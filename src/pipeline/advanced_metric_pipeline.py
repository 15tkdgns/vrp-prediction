#!/usr/bin/env python3
"""
고급 지표 기반 학습 파이프라인
로그 손실 및 F1-Score 최적화 통합 시스템
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
from training.advanced_metric_trainer import AdvancedMetricTrainer
from evaluation.performance_evaluator import PerformanceEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedMetricPipeline:
    """고급 지표 기반 학습 파이프라인"""

    def __init__(self, config=None):
        self.config = config or self._get_default_config()
        self.data_processor = DataProcessor()
        self.model_trainer = AdvancedMetricTrainer(gpu_enabled=self.config['gpu_enabled'])
        self.evaluator = PerformanceEvaluator()

        logger.info("고급 지표 파이프라인 초기화 완료")

    def _get_default_config(self):
        """기본 설정"""
        return {
            'data_path': '/root/workspace/data/training/sp500_2020_2024_enhanced.csv',
            'target_type': 'direction',
            'sequence_length': 20,
            'cv_splits': 3,
            'gpu_enabled': True,
            'save_models': True,
            'save_results': True,
            'output_dir': '/root/workspace/data/results/advanced_metrics'
        }

    def load_and_prepare_data(self):
        """데이터 로딩 및 전처리"""
        logger.info("=== 고급 지표 데이터 준비 단계 ===")

        # 데이터 로딩
        df = self.data_processor.load_and_validate_data(self.config['data_path'])
        if df is None:
            raise ValueError("데이터 로딩 실패")

        # ML용 데이터 준비
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

        return {
            'flat_data': (X.values, y.values),
            'sequence_data': (X_seq, y_seq),
            'feature_names': feature_cols,
            'splits': splits,
            'scaler': scaler,
            'integrity_check': integrity_check
        }

    def train_advanced_models(self, data_dict):
        """고급 지표 모델 훈련"""
        logger.info("=== 고급 지표 모델 훈련 단계 ===")

        X, y = data_dict['flat_data']
        X_seq, y_seq = data_dict['sequence_data']
        splits = data_dict['splits']

        all_results = {}

        # 각 폴드별로 훈련 및 평가
        for fold, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"Fold {fold + 1}/{len(splits)} 고급 지표 훈련 시작")

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

            # 고급 모델 훈련
            fold_results = self.model_trainer.train_all_advanced_models(
                X_train, X_val, y_train, y_val, sequence_data=sequence_data
            )

            # 결과 누적
            for model_name, result in fold_results.items():
                if model_name not in all_results:
                    all_results[model_name] = {
                        'fold_accuracies': [],
                        'fold_f1_scores': [],
                        'fold_log_losses': [],
                        'fold_aucs': [],
                        'fold_precisions': [],
                        'fold_recalls': [],
                        'type': result['type']
                    }

                all_results[model_name]['fold_accuracies'].append(result['accuracy'])
                all_results[model_name]['fold_f1_scores'].append(result['f1_score'])
                all_results[model_name]['fold_log_losses'].append(result['log_loss'])
                all_results[model_name]['fold_aucs'].append(result['auc'])
                all_results[model_name]['fold_precisions'].append(result['precision'])
                all_results[model_name]['fold_recalls'].append(result['recall'])

        # 평균 성능 계산
        final_results = {}
        for model_name, data in all_results.items():
            final_results[model_name] = {
                'mean_accuracy': np.mean(data['fold_accuracies']),
                'std_accuracy': np.std(data['fold_accuracies']),
                'mean_f1_score': np.mean(data['fold_f1_scores']),
                'std_f1_score': np.std(data['fold_f1_scores']),
                'mean_log_loss': np.mean([x for x in data['fold_log_losses'] if x != float('inf')]),
                'std_log_loss': np.std([x for x in data['fold_log_losses'] if x != float('inf')]),
                'mean_auc': np.mean(data['fold_aucs']),
                'std_auc': np.std(data['fold_aucs']),
                'mean_precision': np.mean(data['fold_precisions']),
                'std_precision': np.std(data['fold_precisions']),
                'mean_recall': np.mean(data['fold_recalls']),
                'std_recall': np.std(data['fold_recalls']),
                'direction_accuracy': np.mean(data['fold_accuracies']) * 100,
                'fold_results': data,
                'type': data['type']
            }

        return final_results

    def analyze_metric_performance(self, results):
        """지표별 성능 분석"""
        logger.info("=== 지표별 성능 분석 ===")

        analysis = {
            'best_by_accuracy': None,
            'best_by_f1': None,
            'best_by_log_loss': None,
            'best_by_auc': None,
            'metric_rankings': {},
            'cross_metric_analysis': {}
        }

        # 각 지표별 최고 모델 찾기
        best_accuracy = 0
        best_f1 = 0
        best_log_loss = float('inf')
        best_auc = 0

        for model_name, result in results.items():
            # 정확도 기준
            if result['mean_accuracy'] > best_accuracy:
                best_accuracy = result['mean_accuracy']
                analysis['best_by_accuracy'] = {
                    'model': model_name,
                    'score': best_accuracy,
                    'details': result
                }

            # F1 점수 기준
            if result['mean_f1_score'] > best_f1:
                best_f1 = result['mean_f1_score']
                analysis['best_by_f1'] = {
                    'model': model_name,
                    'score': best_f1,
                    'details': result
                }

            # 로그 손실 기준 (낮을수록 좋음)
            if result['mean_log_loss'] < best_log_loss and result['mean_log_loss'] != float('inf'):
                best_log_loss = result['mean_log_loss']
                analysis['best_by_log_loss'] = {
                    'model': model_name,
                    'score': best_log_loss,
                    'details': result
                }

            # AUC 기준
            if result['mean_auc'] > best_auc:
                best_auc = result['mean_auc']
                analysis['best_by_auc'] = {
                    'model': model_name,
                    'score': best_auc,
                    'details': result
                }

        # 지표별 순위 매기기
        for metric in ['mean_accuracy', 'mean_f1_score', 'mean_auc']:
            ranking = sorted(results.items(),
                           key=lambda x: x[1][metric],
                           reverse=True)
            analysis['metric_rankings'][metric] = ranking

        # 로그 손실 순위 (낮을수록 좋음)
        log_loss_ranking = sorted(results.items(),
                                key=lambda x: x[1]['mean_log_loss'] if x[1]['mean_log_loss'] != float('inf') else 999,
                                reverse=False)
        analysis['metric_rankings']['mean_log_loss'] = log_loss_ranking

        return analysis

    def run_advanced_pipeline(self):
        """전체 고급 파이프라인 실행"""
        logger.info("=" * 80)
        logger.info("🚀 고급 지표 기반 학습 파이프라인 시작")
        logger.info("=" * 80)

        start_time = datetime.now()

        try:
            # 1. 데이터 준비
            data_dict = self.load_and_prepare_data()

            # 2. 고급 모델 훈련
            model_results = self.train_advanced_models(data_dict)

            # 3. 지표별 성능 분석
            metric_analysis = self.analyze_metric_performance(model_results)

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
                'metric_analysis': metric_analysis,
                'data_integrity': data_dict['integrity_check']
            }

            # 5. 모델 저장
            if self.config['save_models']:
                self.model_trainer.save_advanced_models()

            # 6. 결과 저장
            if self.config['save_results']:
                self._save_results(pipeline_results)

            # 7. 결과 출력
            self._print_advanced_summary(pipeline_results)

            logger.info("✅ 고급 지표 파이프라인 실행 완료")
            return pipeline_results

        except Exception as e:
            logger.error(f"❌ 고급 파이프라인 실행 실패: {e}")
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
        results_file = output_dir / f"advanced_metric_results_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(cleaned_results, f, indent=2)

        logger.info(f"고급 지표 결과 저장: {results_file}")

    def _print_advanced_summary(self, results):
        """고급 파이프라인 결과 요약 출력"""
        print("\n" + "=" * 80)
        print("📊 고급 지표 기반 학습 결과")
        print("=" * 80)

        # 기본 정보
        print(f"\n📋 실행 정보:")
        print(f"   실행 시간: {results['execution_time']}")
        print(f"   특징 수: {results['data_info']['feature_count']}")
        print(f"   샘플 수: {results['data_info']['sample_count']}")

        # 지표별 최고 성능
        analysis = results['metric_analysis']

        print(f"\n🏆 지표별 최고 성능:")
        print("-" * 80)

        if analysis['best_by_accuracy']:
            acc_best = analysis['best_by_accuracy']
            print(f"정확도 최고:     {acc_best['model']:30s} {acc_best['score']*100:.2f}%")

        if analysis['best_by_f1']:
            f1_best = analysis['best_by_f1']
            print(f"F1 점수 최고:    {f1_best['model']:30s} {f1_best['score']:.4f}")

        if analysis['best_by_log_loss']:
            log_best = analysis['best_by_log_loss']
            print(f"로그 손실 최고:  {log_best['model']:30s} {log_best['score']:.4f}")

        if analysis['best_by_auc']:
            auc_best = analysis['best_by_auc']
            print(f"AUC 최고:        {auc_best['model']:30s} {auc_best['score']:.4f}")

        # 종합 순위 (정확도 기준)
        print(f"\n📈 종합 성능 순위 (정확도 기준):")
        print("-" * 80)

        if 'mean_accuracy' in analysis['metric_rankings']:
            for i, (model_name, result) in enumerate(analysis['metric_rankings']['mean_accuracy'][:5], 1):
                acc = result['mean_accuracy'] * 100
                f1 = result['mean_f1_score']
                print(f"{i:2d}. {model_name:30s} | 정확도: {acc:6.2f}% | F1: {f1:.4f}")

        print("\n" + "=" * 80)

def main():
    """메인 실행 함수"""
    config = {
        'data_path': '/root/workspace/data/training/sp500_2020_2024_enhanced.csv',
        'target_type': 'direction',
        'sequence_length': 20,
        'cv_splits': 3,
        'gpu_enabled': True,
        'save_models': True,
        'save_results': True,
        'output_dir': '/root/workspace/data/results/advanced_metrics'
    }

    # 파이프라인 실행
    pipeline = AdvancedMetricPipeline(config)
    results = pipeline.run_advanced_pipeline()

    return results

if __name__ == "__main__":
    results = main()