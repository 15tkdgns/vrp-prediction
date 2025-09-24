#!/usr/bin/env python3
"""
안전한 모델 학습 시스템
데이터 누출 완전 차단 파이프라인
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
from typing import Dict, List, Tuple, Optional, Any

from core.safe_data_processor import SafeDataProcessor
from validation.auto_leakage_detector import AutoLeakageDetector
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafeLearningPipeline:
    """안전한 모델 학습 파이프라인"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()

        # 핵심 구성요소 초기화
        self.data_processor = SafeDataProcessor()
        self.leakage_detector = AutoLeakageDetector()

        # 모델 설정 (간단하고 안전한 모델만)
        self.safe_models = self._get_safe_models()

        # 결과 저장
        self.training_results = {}
        self.validation_reports = []

        logger.info("안전한 학습 파이프라인 초기화 완료")
        logger.info("CLAUDE.md 준수: 데이터누출 방지, 성능 95% 미만 강제")

    def _get_default_config(self) -> Dict:
        """기본 설정"""
        return {
            'data_path': '/root/workspace/data/training/sp500_2020_2024_enhanced.csv',
            'target_type': 'return',  # 'return' or 'direction'
            'max_features': 20,       # 최대 특징 수 제한
            'cv_splits': 3,           # 교차검증 분할 수
            'enable_auto_stop': True, # 자동 중단 활성화
            'save_results': True,
            'output_dir': '/root/workspace/data/results/safe_learning'
        }

    def _get_safe_models(self) -> Dict:
        """안전한 모델 설정 (복잡성 제한)"""
        return {
            'linear_regression': {
                'model_class': LinearRegression,
                'params': {},
                'description': '기본 선형 회귀'
            },
            'ridge_regression': {
                'model_class': Ridge,
                'params': {
                    'alpha': 1.0,
                    'random_state': 42
                },
                'description': 'L2 정규화 회귀'
            },
            'lasso_regression': {
                'model_class': Lasso,
                'params': {
                    'alpha': 0.01,
                    'random_state': 42,
                    'max_iter': 2000
                },
                'description': 'L1 정규화 회귀'
            },
            'random_forest_simple': {
                'model_class': RandomForestRegressor,
                'params': {
                    'n_estimators': 50,  # 복잡성 제한
                    'max_depth': 5,      # 깊이 제한
                    'min_samples_split': 20,  # 과적합 방지
                    'min_samples_leaf': 10,
                    'random_state': 42,
                    'n_jobs': -1
                },
                'description': '단순 랜덤 포레스트'
            }
        }

    def run_safe_pipeline(self) -> Dict:
        """안전한 파이프라인 전체 실행"""
        logger.info("=" * 80)
        logger.info("🛡️ 안전한 모델 학습 파이프라인 시작")
        logger.info("🚨 데이터 누출 완전 차단 모드")
        logger.info("=" * 80)

        start_time = datetime.now()

        try:
            # 1. 안전한 데이터 준비
            logger.info("1단계: 안전한 데이터 준비")
            data_dict = self._prepare_safe_data()

            # 2. 훈련 전 검증
            logger.info("2단계: 훈련 전 안전성 검증")
            pre_validation_passed = self._validate_before_training(data_dict)

            if not pre_validation_passed:
                raise ValueError("훈련 전 검증 실패 - 데이터 누출 위험 감지")

            # 3. 안전한 모델 훈련
            logger.info("3단계: 안전한 모델 훈련")
            training_results = self._train_safe_models(data_dict)

            # 4. 훈련 후 검증
            logger.info("4단계: 훈련 후 안전성 검증")
            post_validation_passed = self._validate_after_training(training_results)

            if not post_validation_passed:
                logger.warning("훈련 후 검증에서 일부 모델 실패")

            # 5. 결과 정리 및 저장
            logger.info("5단계: 결과 정리")
            pipeline_results = self._compile_results(data_dict, training_results, start_time)

            # 6. 검증 보고서 생성
            validation_report = self.leakage_detector.generate_validation_report()
            pipeline_results['validation_report'] = validation_report

            # 7. 결과 저장
            if self.config['save_results']:
                self._save_results(pipeline_results)

            # 8. 최종 요약 출력
            self._print_final_summary(pipeline_results)

            logger.info("✅ 안전한 파이프라인 완료")
            return pipeline_results

        except Exception as e:
            logger.error(f"❌ 안전한 파이프라인 실패: {e}")
            # 실패해도 검증 보고서는 생성
            try:
                validation_report = self.leakage_detector.generate_validation_report()
                self.validation_reports.append(validation_report)
            except:
                pass
            raise

    def _prepare_safe_data(self) -> Dict:
        """안전한 데이터 준비"""
        data_dict = self.data_processor.prepare_safe_ml_data(
            self.config['data_path'],
            self.config['target_type']
        )

        # 특징 수 제한 (복잡성 감소)
        if data_dict['X'].shape[1] > self.config['max_features']:
            logger.info(f"특징 수 제한: {data_dict['X'].shape[1]} → {self.config['max_features']}")

            # 간단한 분산 기반 특징 선택
            feature_vars = np.var(data_dict['X'], axis=0)
            top_indices = np.argsort(feature_vars)[-self.config['max_features']:]

            data_dict['X'] = data_dict['X'][:, top_indices]
            data_dict['feature_names'] = [data_dict['feature_names'][i] for i in top_indices]

        logger.info(f"최종 데이터: X{data_dict['X'].shape}, 특징 {len(data_dict['feature_names'])}개")

        return data_dict

    def _validate_before_training(self, data_dict: Dict) -> bool:
        """훈련 전 검증"""
        return self.leakage_detector.validate_data_before_training(
            data_dict['X'],
            data_dict['y'],
            data_dict['feature_names']
        )

    def _train_safe_models(self, data_dict: Dict) -> Dict:
        """안전한 모델 훈련"""
        X = data_dict['X']
        y = data_dict['y']
        splits = data_dict['splits']

        all_results = {}

        for model_name, model_config in self.safe_models.items():
            logger.info(f"=== {model_name} 훈련 시작 ===")

            model_results = {
                'fold_r2': [],
                'fold_mse': [],
                'fold_mae': [],
                'fold_direction_acc': [],
                'description': model_config['description'],
                'training_stopped': False
            }

            # 폴드별 훈련
            for fold, (train_idx, val_idx) in enumerate(splits):
                logger.info(f"Fold {fold + 1}/{len(splits)} 훈련")

                # 데이터 분할
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # 안전한 스케일링
                X_train_scaled, X_val_scaled, scaler = self.data_processor.safe_scale_data(X_train, X_val)

                # 모델 훈련
                model = model_config['model_class'](**model_config['params'])
                model.fit(X_train_scaled, y_train)

                # 예측 및 평가
                y_pred = model.predict(X_val_scaled)

                # 성능 지표 계산
                r2 = r2_score(y_val, y_pred)
                mse = mean_squared_error(y_val, y_pred)
                mae = mean_absolute_error(y_val, y_pred)

                # 방향 정확도 계산
                direction_correct = np.sum(np.sign(y_val) == np.sign(y_pred))
                direction_acc = direction_correct / len(y_val) * 100

                # 결과 저장
                model_results['fold_r2'].append(r2)
                model_results['fold_mse'].append(mse)
                model_results['fold_mae'].append(mae)
                model_results['fold_direction_acc'].append(direction_acc)

                # 실시간 검증
                fold_metrics = {
                    'r2': r2,
                    'mse': mse,
                    'mae': mae,
                    'direction_accuracy': direction_acc
                }

                # 자동 중단 검사
                if self.config['enable_auto_stop']:
                    should_continue = self.leakage_detector.validate_during_training(
                        fold, model_name, fold_metrics
                    )

                    if not should_continue:
                        logger.error(f"🛑 {model_name} 훈련 중단 (Fold {fold})")
                        model_results['training_stopped'] = True
                        break

                logger.info(f"Fold {fold} 성능: R²={r2:.4f}, 방향정확도={direction_acc:.1f}%")

            # 평균 성능 계산
            if model_results['fold_r2']:
                model_results['mean_r2'] = np.mean(model_results['fold_r2'])
                model_results['std_r2'] = np.std(model_results['fold_r2'])
                model_results['mean_mse'] = np.mean(model_results['fold_mse'])
                model_results['mean_mae'] = np.mean(model_results['fold_mae'])
                model_results['mean_direction_accuracy'] = np.mean(model_results['fold_direction_acc'])
                model_results['std_direction_accuracy'] = np.std(model_results['fold_direction_acc'])

                logger.info(f"{model_name} 최종 성능:")
                logger.info(f"   R²: {model_results['mean_r2']:.4f} ± {model_results['std_r2']:.4f}")
                logger.info(f"   방향정확도: {model_results['mean_direction_accuracy']:.1f}% ± {model_results['std_direction_accuracy']:.1f}%")

            all_results[model_name] = model_results

        return all_results

    def _validate_after_training(self, training_results: Dict) -> bool:
        """훈련 후 검증"""
        return self.leakage_detector.validate_after_training(training_results)

    def _compile_results(self, data_dict: Dict, training_results: Dict, start_time: datetime) -> Dict:
        """결과 종합"""
        return {
            'pipeline_info': {
                'execution_time': str(datetime.now() - start_time),
                'config': self.config,
                'data_shape': data_dict['X'].shape,
                'target_type': data_dict['target_type'],
                'feature_count': len(data_dict['feature_names']),
                'safety_validated': data_dict['safety_validated']
            },
            'training_results': training_results,
            'data_info': {
                'feature_names': data_dict['feature_names'],
                'splits_count': len(data_dict['splits']),
            },
            'safety_summary': self._generate_safety_summary(training_results)
        }

    def _generate_safety_summary(self, training_results: Dict) -> Dict:
        """안전성 요약 생성"""
        summary = {
            'total_models': len(training_results),
            'completed_models': 0,
            'stopped_models': 0,
            'safe_models': 0,
            'suspicious_models': 0,
            'max_r2': 0,
            'max_direction_acc': 0
        }

        for model_name, results in training_results.items():
            if results.get('training_stopped', False):
                summary['stopped_models'] += 1
            else:
                summary['completed_models'] += 1

            # 최대 성능 추적
            mean_r2 = results.get('mean_r2', 0)
            mean_direction = results.get('mean_direction_accuracy', 0)

            summary['max_r2'] = max(summary['max_r2'], mean_r2)
            summary['max_direction_acc'] = max(summary['max_direction_acc'], mean_direction)

            # 안전성 분류
            if mean_r2 < 0.10 and mean_direction < 60:
                summary['safe_models'] += 1
            else:
                summary['suspicious_models'] += 1

        # CLAUDE.md 준수 확인
        summary['claude_md_compliant'] = (
            summary['max_r2'] < 0.95 and
            summary['max_direction_acc'] < 95.0
        )

        return summary

    def _save_results(self, pipeline_results: Dict) -> None:
        """결과 저장"""
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"safe_learning_results_{timestamp}.json"

        # JSON 직렬화 가능하도록 변환
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return obj

        def clean_for_json(data):
            if isinstance(data, dict):
                return {k: clean_for_json(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_for_json(item) for item in data]
            else:
                return convert_for_json(data)

        cleaned_results = clean_for_json(pipeline_results)

        with open(results_file, 'w') as f:
            json.dump(cleaned_results, f, indent=2)

        logger.info(f"안전한 학습 결과 저장: {results_file}")

    def _print_final_summary(self, pipeline_results: Dict) -> None:
        """최종 요약 출력"""
        print("\n" + "=" * 80)
        print("🛡️ 안전한 모델 학습 파이프라인 결과")
        print("=" * 80)

        # 파이프라인 정보
        info = pipeline_results['pipeline_info']
        print(f"\n📋 실행 정보:")
        print(f"   실행 시간: {info['execution_time']}")
        print(f"   데이터 크기: {info['data_shape']}")
        print(f"   특징 수: {info['feature_count']}")
        print(f"   타겟 타입: {info['target_type']}")

        # 안전성 요약
        safety = pipeline_results['safety_summary']
        print(f"\n🛡️ 안전성 요약:")
        print(f"   총 모델: {safety['total_models']}")
        print(f"   완료된 모델: {safety['completed_models']}")
        print(f"   중단된 모델: {safety['stopped_models']}")
        print(f"   안전한 모델: {safety['safe_models']}")
        print(f"   의심스러운 모델: {safety['suspicious_models']}")

        # 성능 요약
        print(f"\n📊 성능 요약:")
        print(f"   최대 R²: {safety['max_r2']:.4f}")
        print(f"   최대 방향정확도: {safety['max_direction_acc']:.1f}%")

        # CLAUDE.md 준수
        compliance_icon = "✅" if safety['claude_md_compliant'] else "❌"
        print(f"\n📋 CLAUDE.md 준수: {compliance_icon}")

        # 모델별 결과
        print(f"\n🏆 모델별 성능:")
        training_results = pipeline_results['training_results']

        for model_name, results in training_results.items():
            if not results.get('training_stopped', False):
                r2 = results.get('mean_r2', 0)
                direction = results.get('mean_direction_accuracy', 0)
                status = "🛡️" if r2 < 0.10 and direction < 60 else "⚠️"
                print(f"   {status} {model_name:20s}: R²={r2:6.3f}, 방향={direction:5.1f}%")
            else:
                print(f"   🛑 {model_name:20s}: 훈련 중단됨")

        print("\n" + "=" * 80)

        # 권장사항
        if safety['claude_md_compliant']:
            print("✅ 모든 모델이 안전 기준을 만족합니다.")
            print("💡 현실적인 성능 범위 내에서 안전하게 훈련되었습니다.")
        else:
            print("❌ 일부 모델이 안전 기준을 위반했습니다.")
            print("🚨 데이터 누출 가능성이 있으니 재검토가 필요합니다.")

        print("=" * 80)

def main():
    """메인 실행 함수"""
    config = {
        'data_path': '/root/workspace/data/training/sp500_2020_2024_enhanced.csv',
        'target_type': 'return',
        'max_features': 10,  # 특징 수 제한
        'cv_splits': 3,
        'enable_auto_stop': True,
        'save_results': True,
        'output_dir': '/root/workspace/data/results/safe_learning'
    }

    pipeline = SafeLearningPipeline(config)
    results = pipeline.run_safe_pipeline()

    return results

if __name__ == "__main__":
    results = main()