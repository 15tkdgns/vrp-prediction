#!/usr/bin/env python3
"""
종합적인 모델 성능 순위 평가 시스템
다차원 지표 기반 상세 분석
"""

import sys
sys.path.append('/root/workspace')

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from src.optimization.fast_safe_gridsearch import FastSafeGridSearch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveModelRanking:
    """종합적인 모델 순위 평가"""

    def __init__(self):
        self.optimizer = FastSafeGridSearch()

        # 평가 가중치
        self.weights = {
            'r2': 0.25,           # R² 가중치
            'mae': 0.25,          # MAE 가중치 (낮을수록 좋음)
            'direction_acc': 0.30, # 방향 정확도 가중치
            'stability': 0.20     # 안정성 가중치
        }

        logger.info("📊 종합 모델 순위 평가 시스템 초기화")

    def calculate_advanced_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """고급 성능 지표 계산"""

        # 기본 지표
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        # 방향 정확도
        direction_actual = (y_true > 0).astype(int)
        direction_pred = (y_pred > 0).astype(int)
        direction_accuracy = (direction_actual == direction_pred).mean() * 100

        # 상승/하락 별 정확도
        up_mask = y_true > 0
        down_mask = y_true <= 0

        up_accuracy = 0
        down_accuracy = 0

        if np.sum(up_mask) > 0:
            up_accuracy = (direction_actual[up_mask] == direction_pred[up_mask]).mean() * 100

        if np.sum(down_mask) > 0:
            down_accuracy = (direction_actual[down_mask] == direction_pred[down_mask]).mean() * 100

        # 예측 편향성
        prediction_bias = np.mean(y_pred - y_true)

        # 예측 분산
        prediction_variance = np.var(y_pred - y_true)

        # 상관관계
        correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0

        return {
            'r2': r2,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'direction_accuracy': direction_accuracy,
            'up_accuracy': up_accuracy,
            'down_accuracy': down_accuracy,
            'prediction_bias': prediction_bias,
            'prediction_variance': prediction_variance,
            'correlation': correlation
        }

    def evaluate_model_stability(self, model, X: np.ndarray, y: np.ndarray) -> Dict:
        """모델 안정성 평가 (여러 fold 성능 분산)"""

        tscv = TimeSeriesSplit(n_splits=3, test_size=80, gap=2)

        fold_r2_scores = []
        fold_mae_scores = []
        fold_direction_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            # 데이터 분할
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 스케일링
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # 모델 복사 및 훈련
            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(X_train_scaled, y_train)
            y_pred = fold_model.predict(X_val_scaled)

            # 성능 계산
            metrics = self.calculate_advanced_metrics(y_val, y_pred)

            fold_r2_scores.append(metrics['r2'])
            fold_mae_scores.append(metrics['mae'])
            fold_direction_scores.append(metrics['direction_accuracy'])

        # 안정성 지표 (낮은 분산이 좋음)
        r2_stability = 1 / (1 + np.std(fold_r2_scores))
        mae_stability = 1 / (1 + np.std(fold_mae_scores))
        direction_stability = 1 / (1 + np.std(fold_direction_scores))

        overall_stability = (r2_stability + mae_stability + direction_stability) / 3

        return {
            'r2_stability': r2_stability,
            'mae_stability': mae_stability,
            'direction_stability': direction_stability,
            'overall_stability': overall_stability,
            'fold_r2_scores': fold_r2_scores,
            'fold_mae_scores': fold_mae_scores,
            'fold_direction_scores': fold_direction_scores
        }

    def calculate_comprehensive_score(self, metrics: Dict, stability: Dict) -> float:
        """종합 점수 계산"""

        # R² 정규화 (0 기준으로 높을수록 좋음, 최대 0.15)
        r2_normalized = max(0, metrics['r2']) / 0.15

        # MAE 정규화 (낮을수록 좋음, 역수 취함)
        mae_normalized = 1 / (1 + metrics['mae'] * 100)  # 100배로 스케일링

        # 방향 정확도 정규화 (최대 65%)
        direction_normalized = metrics['direction_accuracy'] / 65.0

        # 안정성 정규화 (이미 0~1 범위)
        stability_normalized = stability['overall_stability']

        # 가중 평균
        comprehensive_score = (
            self.weights['r2'] * r2_normalized +
            self.weights['mae'] * mae_normalized +
            self.weights['direction_acc'] * direction_normalized +
            self.weights['stability'] * stability_normalized
        )

        return comprehensive_score

    def run_comprehensive_ranking(self, data_path: str) -> Dict:
        """종합적인 모델 순위 평가 실행"""

        logger.info("🏆 종합적인 모델 순위 평가 시작")

        # 1. 기본 최적화 실행
        optimization_results = self.optimizer.run_fast_optimization(data_path)

        # 2. 데이터 준비
        data_dict = self.optimizer.data_processor.prepare_ultra_safe_data(data_path)
        X, y = data_dict['X'], data_dict['y']

        # 3. 각 모델별 상세 평가
        detailed_rankings = []

        for model_name, result in optimization_results['detailed_results'].items():
            if result['status'] == 'success' and result['cv_results']['status'] == 'safe':
                logger.info(f"\n📊 {model_name} 상세 평가")

                # 모델과 스케일러
                model = result['model']
                scaler = result['scaler']

                # 전체 데이터셋으로 최종 평가
                X_scaled = scaler.transform(X)

                # Hold-out 테스트 (마지막 20% 데이터)
                split_point = int(len(X) * 0.8)
                X_train, X_test = X_scaled[:split_point], X_scaled[split_point:]
                y_train, y_test = y[:split_point], y[split_point:]

                # 최종 모델 훈련
                from sklearn.base import clone
                final_model = clone(model)
                final_model.fit(X_train, y_train)
                y_pred = final_model.predict(X_test)

                # 고급 지표 계산
                advanced_metrics = self.calculate_advanced_metrics(y_test, y_pred)

                # 안정성 평가
                stability_metrics = self.evaluate_model_stability(model, X, y)

                # 종합 점수
                comprehensive_score = self.calculate_comprehensive_score(
                    advanced_metrics, stability_metrics
                )

                detailed_rankings.append({
                    'model_name': model_name,
                    'best_params': result['best_params'],
                    'advanced_metrics': advanced_metrics,
                    'stability_metrics': stability_metrics,
                    'comprehensive_score': comprehensive_score
                })

                logger.info(f"   종합 점수: {comprehensive_score:.3f}")
                logger.info(f"   R²: {advanced_metrics['r2']:.4f}")
                logger.info(f"   MAE: {advanced_metrics['mae']:.4f}")
                logger.info(f"   방향정확도: {advanced_metrics['direction_accuracy']:.1f}%")
                logger.info(f"   안정성: {stability_metrics['overall_stability']:.3f}")

        # 4. 종합 점수 기준 정렬
        detailed_rankings.sort(key=lambda x: x['comprehensive_score'], reverse=True)

        # 5. 최종 순위 출력
        logger.info(f"\n{'='*100}")
        logger.info(f"🏆 최종 종합 순위 (데이터 누출 완전 차단)")
        logger.info(f"{'='*100}")

        for rank, result in enumerate(detailed_rankings, 1):
            model_name = result['model_name']
            metrics = result['advanced_metrics']
            stability = result['stability_metrics']
            score = result['comprehensive_score']

            logger.info(f"\n{rank}위. {model_name} (종합점수: {score:.3f})")
            logger.info(f"   🎯 성능 지표:")
            logger.info(f"      R²: {metrics['r2']:.4f}")
            logger.info(f"      MAE: {metrics['mae']:.4f}")
            logger.info(f"      RMSE: {metrics['rmse']:.4f}")
            logger.info(f"      방향정확도: {metrics['direction_accuracy']:.1f}%")
            logger.info(f"      상승정확도: {metrics['up_accuracy']:.1f}%")
            logger.info(f"      하락정확도: {metrics['down_accuracy']:.1f}%")
            logger.info(f"      상관관계: {metrics['correlation']:.3f}")
            logger.info(f"   🛡️ 안정성 지표:")
            logger.info(f"      전체안정성: {stability['overall_stability']:.3f}")
            logger.info(f"      R²안정성: {stability['r2_stability']:.3f}")
            logger.info(f"      MAE안정성: {stability['mae_stability']:.3f}")
            logger.info(f"      방향안정성: {stability['direction_stability']:.3f}")
            logger.info(f"   ⚙️ 최적 파라미터: {result['best_params']}")

        return {
            'detailed_rankings': detailed_rankings,
            'total_safe_models': len(detailed_rankings),
            'evaluation_weights': self.weights,
            'winner': detailed_rankings[0] if detailed_rankings else None
        }

def main():
    """메인 실행 함수"""
    evaluator = ComprehensiveModelRanking()

    try:
        results = evaluator.run_comprehensive_ranking(
            '/root/workspace/data/training/sp500_2020_2024_enhanced.csv'
        )

        print(f"\n{'='*100}")
        print(f"📊 종합 모델 순위 평가 완료")
        print(f"{'='*100}")

        if results['winner']:
            winner = results['winner']
            print(f"\n🏆 최우수 모델: {winner['model_name']}")
            print(f"   종합 점수: {winner['comprehensive_score']:.3f}")
            print(f"   R²: {winner['advanced_metrics']['r2']:.4f}")
            print(f"   MAE: {winner['advanced_metrics']['mae']:.4f}")
            print(f"   방향정확도: {winner['advanced_metrics']['direction_accuracy']:.1f}%")
            print(f"   안정성: {winner['stability_metrics']['overall_stability']:.3f}")
            print(f"   최적 파라미터: {winner['best_params']}")

        print(f"\n📈 평가 완료 모델: {results['total_safe_models']}개")
        print(f"🛡️ 모든 모델이 데이터 누출 방지 기준 통과")

        return results

    except Exception as e:
        logger.error(f"종합 순위 평가 실패: {e}")
        return None

if __name__ == "__main__":
    result = main()