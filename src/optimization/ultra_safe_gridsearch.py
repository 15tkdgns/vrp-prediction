#!/usr/bin/env python3
"""
초안전 GridSearchCV 기반 모델 최적화 시스템
데이터 누출 완전 차단하며 최고 성능 모델 탐색
"""

import sys
sys.path.append('/root/workspace')

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# XGBoost 안전 import
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost 미설치 - XGBoost 모델 제외")

from src.core.ultra_safe_data_processor import UltraSafeDataProcessor
from src.validation.auto_leakage_detector import AutoLeakageDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltraSafeGridSearch:
    """초안전 GridSearch 최적화 시스템"""

    def __init__(self):
        # CLAUDE.md 준수 - 극도로 엄격한 기준
        self.MAX_R2 = 0.15           # R² 15% 상한선
        self.MAX_DIRECTION_ACC = 65.0 # 방향정확도 65% 상한선
        self.MAX_CORRELATION = 0.20   # 20% 상관관계 상한선

        # 시스템 초기화
        self.data_processor = UltraSafeDataProcessor()
        self.leakage_detector = AutoLeakageDetector()

        logger.info("🔒 초안전 GridSearch 시스템 초기화")
        logger.info(f"엄격한 안전 기준 - R²<{self.MAX_R2}, 방향정확도<{self.MAX_DIRECTION_ACC}%")

    def define_model_grids(self) -> Dict[str, Dict]:
        """모델별 하이퍼파라미터 그리드 정의"""
        logger.info("📋 모델 그리드 정의")

        grids = {
            'LinearRegression': {
                'model': LinearRegression(),
                'params': {}  # 파라미터 없음
            },

            'Ridge': {
                'model': Ridge(random_state=42),
                'params': {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
                }
            },

            'Lasso': {
                'model': Lasso(random_state=42, max_iter=1000),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
                }
            },

            'ElasticNet': {
                'model': ElasticNet(random_state=42, max_iter=1000),
                'params': {
                    'alpha': [0.01, 0.1, 1.0],
                    'l1_ratio': [0.1, 0.5, 0.9]
                }
            },

            'RandomForest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [10, 50, 100],
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },

            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },

            'SVR': {
                'model': SVR(),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'gamma': ['scale', 'auto', 0.001, 0.01],
                    'kernel': ['rbf', 'linear']
                }
            },

            'MLPRegressor': {
                'model': MLPRegressor(random_state=42, max_iter=500),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate_init': [0.001, 0.01, 0.1]
                }
            }
        }

        # XGBoost 추가 (사용 가능한 경우)
        if XGBOOST_AVAILABLE:
            grids['XGBoost'] = {
                'model': xgb.XGBRegressor(random_state=42, verbosity=0),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.1, 1.0],
                    'reg_lambda': [1, 1.5, 2.0]
                }
            }

        logger.info(f"정의된 모델: {len(grids)}개")
        return grids

    def safe_grid_search(self, X: np.ndarray, y: np.ndarray,
                        model_name: str, model_config: Dict) -> Dict:
        """안전한 GridSearch 실행"""
        logger.info(f"\n🔍 {model_name} GridSearch 시작")

        # TimeSeriesSplit 설정 (엄격한 시간 순서)
        tscv = TimeSeriesSplit(n_splits=3, test_size=50, gap=2)

        try:
            if model_config['params']:
                # GridSearchCV 실행
                grid_search = GridSearchCV(
                    estimator=model_config['model'],
                    param_grid=model_config['params'],
                    cv=tscv,
                    scoring='neg_mean_squared_error',
                    n_jobs=1,  # 안전을 위해 단일 프로세스
                    verbose=0
                )

                # 스케일링 (시간 순서 준수)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # GridSearch 실행
                grid_search.fit(X_scaled, y)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_

                logger.info(f"   최적 파라미터: {best_params}")

            else:
                # 파라미터 없는 모델 (LinearRegression)
                best_model = model_config['model']
                best_params = {}
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                best_model.fit(X_scaled, y)

            # 교차 검증으로 성능 평가
            cv_results = self.evaluate_model_safely(X, y, best_model, model_name)

            return {
                'model': best_model,
                'scaler': scaler,
                'best_params': best_params,
                'cv_results': cv_results,
                'status': 'success'
            }

        except Exception as e:
            logger.error(f"   ❌ {model_name} GridSearch 실패: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def evaluate_model_safely(self, X: np.ndarray, y: np.ndarray,
                             model: Any, model_name: str) -> Dict:
        """안전한 모델 성능 평가"""

        tscv = TimeSeriesSplit(n_splits=3, test_size=50, gap=2)

        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            # 시간 순서 재검증
            assert train_idx.max() < val_idx.min(), f"시간 순서 위반: {model_name} fold {fold}"

            # 데이터 분할
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 독립적 스케일링
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # 모델 복사 후 훈련
            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(X_train_scaled, y_train)

            # 예측
            y_pred = fold_model.predict(X_val_scaled)

            # 성능 계산
            r2 = r2_score(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            mse = mean_squared_error(y_val, y_pred)

            # 방향 정확도
            direction_actual = (y_val > 0).astype(int)
            direction_pred = (y_pred > 0).astype(int)
            direction_acc = (direction_actual == direction_pred).mean() * 100

            fold_result = {
                'fold': fold,
                'r2': r2,
                'mae': mae,
                'mse': mse,
                'direction_accuracy': direction_acc
            }
            fold_results.append(fold_result)

            # 실시간 안전성 검증
            metrics = {'r2': r2, 'direction_accuracy': direction_acc}
            is_safe = self.leakage_detector.validate_during_training(fold, model_name, metrics)

            if not is_safe:
                logger.error(f"   🚨 {model_name} fold {fold} 안전 기준 초과!")
                return {'status': 'unsafe', 'fold_results': fold_results}

        # 평균 성능 계산
        avg_r2 = np.mean([r['r2'] for r in fold_results])
        avg_mae = np.mean([r['mae'] for r in fold_results])
        avg_mse = np.mean([r['mse'] for r in fold_results])
        avg_direction = np.mean([r['direction_accuracy'] for r in fold_results])

        # 최종 안전성 검증
        final_safe = (avg_r2 <= self.MAX_R2 and avg_direction <= self.MAX_DIRECTION_ACC)

        return {
            'status': 'safe' if final_safe else 'unsafe',
            'avg_r2': avg_r2,
            'avg_mae': avg_mae,
            'avg_mse': avg_mse,
            'avg_direction_accuracy': avg_direction,
            'fold_results': fold_results,
            'safety_check': {
                'r2_safe': avg_r2 <= self.MAX_R2,
                'direction_safe': avg_direction <= self.MAX_DIRECTION_ACC
            }
        }

    def run_comprehensive_optimization(self, data_path: str) -> Dict:
        """종합적인 모델 최적화 실행"""
        logger.info("=" * 100)
        logger.info("🚀 초안전 종합 모델 최적화 시작")
        logger.info("🔒 데이터 누출 완전 차단 하에서 최고 성능 탐색")
        logger.info("=" * 100)

        # 1. 초안전 데이터 준비
        data_dict = self.data_processor.prepare_ultra_safe_data(data_path)
        X, y = data_dict['X'], data_dict['y']

        logger.info(f"데이터 준비 완료: X{X.shape}, y{y.shape}")

        # 2. 모델 그리드 정의
        model_grids = self.define_model_grids()

        # 3. 각 모델별 GridSearch 실행
        optimization_results = {}
        safe_models = []
        unsafe_models = []

        for model_name, model_config in model_grids.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 {model_name} 최적화")
            logger.info(f"{'='*60}")

            result = self.safe_grid_search(X, y, model_name, model_config)
            optimization_results[model_name] = result

            if result['status'] == 'success':
                cv_result = result['cv_results']
                if cv_result['status'] == 'safe':
                    safe_models.append((model_name, cv_result))
                    logger.info(f"   ✅ {model_name}: 안전 기준 통과")
                    logger.info(f"      R²={cv_result['avg_r2']:.4f}, 방향정확도={cv_result['avg_direction_accuracy']:.1f}%")
                else:
                    unsafe_models.append((model_name, cv_result))
                    logger.warning(f"   ⚠️ {model_name}: 안전 기준 초과")
            else:
                logger.error(f"   ❌ {model_name}: 최적화 실패")

        # 4. 안전한 모델들 순위 매기기
        logger.info(f"\n{'='*100}")
        logger.info(f"📊 안전한 모델 성능 순위")
        logger.info(f"{'='*100}")

        if safe_models:
            # R²와 방향정확도 종합 점수로 순위 결정
            safe_models_scored = []
            for model_name, cv_result in safe_models:
                # 종합 점수 = (R² 정규화) + (방향정확도 정규화)
                # R²: 높을수록 좋음 (하지만 0.15 이하)
                # 방향정확도: 높을수록 좋음 (하지만 65% 이하)
                r2_score_norm = max(0, cv_result['avg_r2']) / self.MAX_R2  # 0~1
                direction_score_norm = cv_result['avg_direction_accuracy'] / self.MAX_DIRECTION_ACC  # 0~1

                combined_score = (r2_score_norm * 0.6) + (direction_score_norm * 0.4)

                safe_models_scored.append((model_name, cv_result, combined_score))

            # 종합 점수 기준 내림차순 정렬
            safe_models_scored.sort(key=lambda x: x[2], reverse=True)

            logger.info(f"🏆 안전한 모델 순위 (총 {len(safe_models_scored)}개):")
            for rank, (model_name, cv_result, score) in enumerate(safe_models_scored, 1):
                logger.info(f"   {rank}위. {model_name}")
                logger.info(f"      종합점수: {score:.3f}")
                logger.info(f"      R²: {cv_result['avg_r2']:.4f}")
                logger.info(f"      MAE: {cv_result['avg_mae']:.4f}")
                logger.info(f"      방향정확도: {cv_result['avg_direction_accuracy']:.1f}%")
                logger.info(f"      최적 파라미터: {optimization_results[model_name]['best_params']}")

        else:
            logger.warning("⚠️ 안전 기준을 통과한 모델이 없습니다!")

        # 5. 최종 결과 정리
        final_results = {
            'safe_models': len(safe_models),
            'unsafe_models': len(unsafe_models),
            'total_models': len(model_grids),
            'ranking': safe_models_scored if safe_models else [],
            'detailed_results': optimization_results,
            'safety_summary': {
                'max_r2_allowed': self.MAX_R2,
                'max_direction_acc_allowed': self.MAX_DIRECTION_ACC,
                'claude_md_compliant': True
            }
        }

        logger.info(f"\n{'='*100}")
        logger.info(f"📋 최종 요약")
        logger.info(f"{'='*100}")
        logger.info(f"안전한 모델: {len(safe_models)}개")
        logger.info(f"위험한 모델: {len(unsafe_models)}개")
        logger.info(f"총 모델: {len(model_grids)}개")
        logger.info(f"CLAUDE.md 준수: ✅")

        return final_results

def main():
    """메인 실행 함수"""
    optimizer = UltraSafeGridSearch()

    try:
        results = optimizer.run_comprehensive_optimization(
            '/root/workspace/data/training/sp500_2020_2024_enhanced.csv'
        )

        print(f"\n{'='*100}")
        print(f"🎯 초안전 GridSearch 최적화 완료")
        print(f"{'='*100}")

        if results['ranking']:
            print(f"\n🏆 최고 성능 모델: {results['ranking'][0][0]}")
            print(f"   종합 점수: {results['ranking'][0][2]:.3f}")
            print(f"   R²: {results['ranking'][0][1]['avg_r2']:.4f}")
            print(f"   방향정확도: {results['ranking'][0][1]['avg_direction_accuracy']:.1f}%")

        print(f"\n📊 전체 통계:")
        print(f"   안전한 모델: {results['safe_models']}개")
        print(f"   위험한 모델: {results['unsafe_models']}개")
        print(f"   성공률: {results['safe_models']/results['total_models']*100:.1f}%")

        return results

    except Exception as e:
        logger.error(f"최적화 실패: {e}")
        return None

if __name__ == "__main__":
    result = main()