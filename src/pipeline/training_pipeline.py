#!/usr/bin/env python3
"""
통합 학습 파이프라인
데이터 처리부터 모델 훈련, 평가까지 전체 과정 관리
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
from training.model_trainer import ModelTrainer
from evaluation.performance_evaluator import PerformanceEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingPipeline:
    """통합 학습 파이프라인"""

    def __init__(self, config=None):
        self.config = config or self._get_default_config()
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer(gpu_enabled=self.config['gpu_enabled'])
        self.evaluator = PerformanceEvaluator()

        logger.info("학습 파이프라인 초기화 완료")

    def _get_default_config(self):
        """기본 설정"""
        return {
            'data_path': '/root/workspace/data/training/sp500_2020_2024_enhanced.csv',
            'target_type': 'direction',  # 'direction' or 'return'
            'sequence_length': 20,
            'cv_splits': 3,
            'gpu_enabled': True,
            'save_models': True,
            'save_results': True,
            'output_dir': '/root/workspace/data/results'
        }

    def load_and_prepare_data(self):
        """데이터 로딩 및 전처리"""
        logger.info("=== 데이터 준비 단계 ===")

        # 데이터 로딩
        df = self.data_processor.load_and_validate_data(self.config['data_path'])
        if df is None:
            raise ValueError("데이터 로딩 실패")

        # ML용 데이터 준비
        X, y, feature_cols = self.data_processor.prepare_ml_data(df, self.config['target_type'])

        # 시퀀스 데이터 준비 (딥러닝용)
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

    def train_models(self, data_dict):
        """모델 훈련"""
        logger.info("=== 모델 훈련 단계 ===")

        X, y = data_dict['flat_data']
        X_seq, y_seq = data_dict['sequence_data']
        splits = data_dict['splits']

        all_results = {}

        # 각 폴드별로 훈련 및 평가
        for fold, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"Fold {fold + 1}/{len(splits)} 훈련 시작")

            # 데이터 분할
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 시퀀스 데이터 분할 (인덱스 조정)
            seq_train_idx = train_idx[train_idx < len(X_seq)]
            seq_val_idx = val_idx[val_idx < len(X_seq)]

            if len(seq_train_idx) > 0 and len(seq_val_idx) > 0:
                X_seq_train, X_seq_val = X_seq[seq_train_idx], X_seq[seq_val_idx]
                y_seq_train, y_seq_val = y_seq[seq_train_idx], y_seq[seq_val_idx]
            else:
                X_seq_train = X_seq_val = None
                y_seq_train = y_seq_val = None

            # 모델 훈련
            if X_seq_train is not None:
                fold_results = self.model_trainer.train_all_models(
                    X_train, X_val, y_train, y_val,
                    sequence_data=(X_seq_train, X_seq_val)
                )
            else:
                fold_results = self.model_trainer.train_all_models(
                    X_train, X_val, y_train, y_val,
                    sequence_data=None
                )

            # 개별 모델 평가
            for model_name, result in fold_results.items():
                if model_name not in all_results:
                    all_results[model_name] = {
                        'fold_scores': [],
                        'type': result['type']
                    }
                all_results[model_name]['fold_scores'].append(result['accuracy'])

        # 평균 성능 계산
        final_results = {}
        for model_name, result in all_results.items():
            scores = result['fold_scores']
            final_results[model_name] = {
                'accuracy': np.mean(scores),
                'accuracy_std': np.std(scores),
                'fold_scores': scores,
                'direction_accuracy': np.mean(scores) * 100,  # 방향 정확도
                'type': result['type']
            }

        return final_results

    def evaluate_ensemble_performance(self, data_dict):
        """앙상블 성능 평가"""
        logger.info("=== 앙상블 평가 단계 ===")

        X, y = data_dict['flat_data']
        X_seq, y_seq = data_dict['sequence_data']
        splits = data_dict['splits']

        ensemble_results = []

        for fold, (train_idx, val_idx) in enumerate(splits):
            X_val = X[val_idx]
            y_val = y[val_idx]

            # 앙상블 예측
            ensemble_pred_proba = self.model_trainer.create_ensemble_prediction(X_val)

            if ensemble_pred_proba is not None:
                ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)

                # 성능 평가
                performance = self.evaluator.evaluate_classification_performance(
                    y_val, ensemble_pred, ensemble_pred_proba, f"Ensemble_Fold_{fold+1}"
                )

                ensemble_results.append(performance['accuracy'])

        if ensemble_results:
            return {
                'ensemble_accuracy': np.mean(ensemble_results),
                'ensemble_accuracy_std': np.std(ensemble_results),
                'ensemble_direction_accuracy': np.mean(ensemble_results) * 100,
                'fold_results': ensemble_results
            }
        else:
            return None

    def run_full_pipeline(self):
        """전체 파이프라인 실행"""
        logger.info("=" * 80)
        logger.info("🚀 통합 학습 파이프라인 시작")
        logger.info("=" * 80)

        start_time = datetime.now()

        try:
            # 1. 데이터 준비
            data_dict = self.load_and_prepare_data()

            # 2. 모델 훈련
            model_results = self.train_models(data_dict)

            # 3. 앙상블 평가
            ensemble_results = self.evaluate_ensemble_performance(data_dict)

            # 4. 성능 비교
            model_ranking = self.evaluator.compare_models(model_results)

            # 5. 종합 결과
            pipeline_results = {
                'pipeline_config': self.config,
                'execution_time': str(datetime.now() - start_time),
                'data_info': {
                    'feature_count': len(data_dict['feature_names']),
                    'sample_count': len(data_dict['flat_data'][1]),
                    'sequence_length': self.config['sequence_length'],
                    'target_type': self.config['target_type']
                },
                'individual_models': model_results,
                'ensemble_performance': ensemble_results,
                'model_ranking': model_ranking,
                'data_integrity': data_dict['integrity_check']
            }

            # 6. 모델 저장
            if self.config['save_models']:
                self.model_trainer.save_models()

            # 7. 결과 저장
            if self.config['save_results']:
                self._save_results(pipeline_results)

            # 8. 결과 출력
            self._print_pipeline_summary(pipeline_results)

            logger.info("✅ 파이프라인 실행 완료")
            return pipeline_results

        except Exception as e:
            logger.error(f"❌ 파이프라인 실행 실패: {e}")
            raise

    def _save_results(self, results):
        """결과 저장"""
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # 결과 파일 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"pipeline_results_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"결과 저장: {results_file}")

    def _print_pipeline_summary(self, results):
        """파이프라인 결과 요약 출력"""
        print("\n" + "=" * 80)
        print("📊 학습 파이프라인 실행 결과")
        print("=" * 80)

        # 기본 정보
        print(f"\n📋 실행 정보:")
        print(f"   실행 시간: {results['execution_time']}")
        print(f"   특징 수: {results['data_info']['feature_count']}")
        print(f"   샘플 수: {results['data_info']['sample_count']}")
        print(f"   타겟 타입: {results['data_info']['target_type']}")

        # 데이터 무결성
        integrity = results['data_integrity']
        print(f"\n🛡️ 데이터 안전성:")
        print(f"   특징-타겟 상관관계 안전: {'✅' if integrity['feature_target_correlation_safe'] else '⚠️'}")
        print(f"   최대 상관관계: {integrity['max_correlation']:.3f}")

        # 개별 모델 성능
        print(f"\n🏆 모델 성능 순위:")
        for i, model in enumerate(results['model_ranking'][:5], 1):
            accuracy = model['full_results']['direction_accuracy']
            print(f"   {i}. {model['model_name']}: {accuracy:.2f}%")

        # 앙상블 성능
        if results['ensemble_performance']:
            ensemble_acc = results['ensemble_performance']['ensemble_direction_accuracy']
            print(f"\n🎯 앙상블 성능:")
            print(f"   방향 정확도: {ensemble_acc:.2f}%")

        # 최고 성능
        if results['model_ranking']:
            best_model = results['model_ranking'][0]
            best_accuracy = best_model['full_results']['direction_accuracy']
            print(f"\n🥇 최고 성능:")
            print(f"   모델: {best_model['model_name']}")
            print(f"   방향 정확도: {best_accuracy:.2f}%")

        print("\n" + "=" * 80)

def main():
    """메인 실행 함수"""
    # 설정
    config = {
        'data_path': '/root/workspace/data/training/sp500_2020_2024_enhanced.csv',
        'target_type': 'direction',  # 방향 예측에 집중
        'sequence_length': 20,
        'cv_splits': 3,  # 빠른 실행을 위해 3개 폴드
        'gpu_enabled': True,
        'save_models': True,
        'save_results': True,
        'output_dir': '/root/workspace/data/results'
    }

    # 파이프라인 실행
    pipeline = TrainingPipeline(config)
    results = pipeline.run_full_pipeline()

    return results

if __name__ == "__main__":
    results = main()