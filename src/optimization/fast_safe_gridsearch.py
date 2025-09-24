#!/usr/bin/env python3
"""
빠른 안전 GridSearch 최적화 시스템
핵심 모델만 선별하여 효율적 최적화
"""

import sys
sys.path.append('/root/workspace')

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# XGBoost 안전 import
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.core.ultra_safe_data_processor import UltraSafeDataProcessor
from src.validation.auto_leakage_detector import AutoLeakageDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastSafeGridSearch:
    """빠른 안전 GridSearch 시스템"""

    def __init__(self):
        self.MAX_R2 = 0.15
        self.MAX_DIRECTION_ACC = 65.0

        self.data_processor = UltraSafeDataProcessor()
        self.leakage_detector = AutoLeakageDetector()

        logger.info("⚡ 빠른 안전 GridSearch 시스템 초기화")

    def define_fast_model_grids(self) -> Dict[str, Dict]:
        """빠른 실행을 위한 핵심 모델 그리드"""

        grids = {
            'LinearRegression': {
                'model': LinearRegression(),
                'params': {}
            },

            'Ridge': {
                'model': Ridge(random_state=42),
                'params': {
                    'alpha': [0.1, 1.0, 10.0]  # 3개만
                }
            },

            'Lasso': {
                'model': Lasso(random_state=42, max_iter=1000),
                'params': {
                    'alpha': [0.01, 0.1, 1.0]  # 3개만
                }
            },

            'RandomForest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [10, 50],  # 2개만
                    'max_depth': [3, 5],       # 2개만
                    'min_samples_split': [2, 5]  # 2개만
                }
            },

            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100],  # 2개만
                    'learning_rate': [0.1, 0.2],  # 2개만
                    'max_depth': [3, 5]  # 2개만
                }
            }
        }

        # XGBoost 추가 (간소화)
        if XGBOOST_AVAILABLE:
            grids['XGBoost'] = {
                'model': xgb.XGBRegressor(random_state=42, verbosity=0),
                'params': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.1, 0.2],
                    'max_depth': [3, 5]
                }
            }

        logger.info(f"빠른 모델 그리드: {len(grids)}개")
        return grids

    def fast_evaluate_model(self, X: np.ndarray, y: np.ndarray,
                           model: Any, model_name: str) -> Dict:
        """빠른 모델 평가 (2-fold만)"""

        tscv = TimeSeriesSplit(n_splits=2, test_size=100, gap=2)

        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            # 시간 순서 검증
            assert train_idx.max() < val_idx.min()

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 스케일링
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # 모델 훈련
            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(X_train_scaled, y_train)
            y_pred = fold_model.predict(X_val_scaled)

            # 성능 계산
            r2 = r2_score(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)

            # 방향 정확도
            direction_actual = (y_val > 0).astype(int)
            direction_pred = (y_pred > 0).astype(int)
            direction_acc = (direction_actual == direction_pred).mean() * 100

            fold_results.append({
                'fold': fold,
                'r2': r2,
                'mae': mae,
                'direction_accuracy': direction_acc
            })

            # 안전성 검증
            metrics = {'r2': r2, 'direction_accuracy': direction_acc}
            is_safe = self.leakage_detector.validate_during_training(fold, model_name, metrics)

            if not is_safe:
                return {'status': 'unsafe', 'fold_results': fold_results}

        # 평균 성능
        avg_r2 = np.mean([r['r2'] for r in fold_results])
        avg_mae = np.mean([r['mae'] for r in fold_results])
        avg_direction = np.mean([r['direction_accuracy'] for r in fold_results])

        # 최종 안전성
        final_safe = (avg_r2 <= self.MAX_R2 and avg_direction <= self.MAX_DIRECTION_ACC)

        return {
            'status': 'safe' if final_safe else 'unsafe',
            'avg_r2': avg_r2,
            'avg_mae': avg_mae,
            'avg_direction_accuracy': avg_direction,
            'fold_results': fold_results
        }

    def fast_grid_search(self, X: np.ndarray, y: np.ndarray,
                        model_name: str, model_config: Dict) -> Dict:
        """빠른 GridSearch"""
        logger.info(f"⚡ {model_name} 빠른 최적화")

        tscv = TimeSeriesSplit(n_splits=2, test_size=100, gap=2)

        try:
            if model_config['params']:
                # GridSearch (2-fold만)
                grid_search = GridSearchCV(
                    estimator=model_config['model'],
                    param_grid=model_config['params'],
                    cv=tscv,
                    scoring='neg_mean_squared_error',
                    n_jobs=1,
                    verbose=0
                )

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                grid_search.fit(X_scaled, y)

                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_

            else:
                best_model = model_config['model']
                best_params = {}
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                best_model.fit(X_scaled, y)

            # 성능 평가
            cv_results = self.fast_evaluate_model(X, y, best_model, model_name)

            return {
                'model': best_model,
                'scaler': scaler,
                'best_params': best_params,
                'cv_results': cv_results,
                'status': 'success'
            }

        except Exception as e:
            logger.error(f"❌ {model_name} 실패: {e}")
            return {'status': 'failed', 'error': str(e)}

    def run_fast_optimization(self, data_path: str) -> Dict:
        """빠른 종합 최적화"""
        logger.info("⚡ 빠른 안전 최적화 시작")

        # 데이터 준비
        data_dict = self.data_processor.prepare_ultra_safe_data(data_path)
        X, y = data_dict['X'], data_dict['y']

        # 모델 그리드
        model_grids = self.define_fast_model_grids()

        # 최적화 실행
        results = {}
        safe_models = []

        for model_name, model_config in model_grids.items():
            result = self.fast_grid_search(X, y, model_name, model_config)
            results[model_name] = result

            if result['status'] == 'success':
                cv_result = result['cv_results']
                if cv_result['status'] == 'safe':
                    safe_models.append((model_name, cv_result, result['best_params']))
                    logger.info(f"✅ {model_name}: R²={cv_result['avg_r2']:.4f}, 방향정확도={cv_result['avg_direction_accuracy']:.1f}%")
                else:
                    logger.warning(f"⚠️ {model_name}: 안전 기준 초과")

        # 순위 매기기
        if safe_models:
            safe_models_scored = []
            for model_name, cv_result, best_params in safe_models:
                # 종합 점수 계산
                r2_score_norm = max(0, cv_result['avg_r2']) / self.MAX_R2
                direction_score_norm = cv_result['avg_direction_accuracy'] / self.MAX_DIRECTION_ACC
                combined_score = (r2_score_norm * 0.6) + (direction_score_norm * 0.4)

                safe_models_scored.append((model_name, cv_result, best_params, combined_score))

            # 정렬
            safe_models_scored.sort(key=lambda x: x[3], reverse=True)

            logger.info(f"\n🏆 안전한 모델 순위:")
            for rank, (model_name, cv_result, best_params, score) in enumerate(safe_models_scored, 1):
                logger.info(f"{rank}위. {model_name} (점수: {score:.3f})")
                logger.info(f"   R²: {cv_result['avg_r2']:.4f}")
                logger.info(f"   MAE: {cv_result['avg_mae']:.4f}")
                logger.info(f"   방향정확도: {cv_result['avg_direction_accuracy']:.1f}%")
                logger.info(f"   최적 파라미터: {best_params}")

        return {
            'safe_models': len(safe_models),
            'ranking': safe_models_scored if safe_models else [],
            'detailed_results': results
        }

def main():
    """메인 실행"""
    optimizer = FastSafeGridSearch()

    try:
        results = optimizer.run_fast_optimization(
            '/root/workspace/data/training/sp500_2020_2024_enhanced.csv'
        )

        print(f"\n⚡ 빠른 안전 최적화 완료")
        print(f"안전한 모델: {results['safe_models']}개")

        if results['ranking']:
            print(f"\n🏆 1위 모델: {results['ranking'][0][0]}")
            print(f"   종합 점수: {results['ranking'][0][3]:.3f}")
            print(f"   R²: {results['ranking'][0][1]['avg_r2']:.4f}")
            print(f"   방향정확도: {results['ranking'][0][1]['avg_direction_accuracy']:.1f}%")
            print(f"   최적 파라미터: {results['ranking'][0][2]}")

        return results

    except Exception as e:
        logger.error(f"최적화 실패: {e}")
        return None

if __name__ == "__main__":
    result = main()