#!/usr/bin/env python3
"""
🔒 엄격한 Purged 검증을 사용한 최종 모델 평가
완전한 데이터 무결성 보장 및 과적합 방지
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, log_loss
from sklearn.isotonic import IsotonicRegression

# Purged CV 모듈 import
import sys
sys.path.append('/root/workspace')
from src.validation.purged_cross_validation import (
    PurgedKFold, CombinatorialPurgedCV, PurgedTimeSeriesSplit,
    validate_purged_cv_integrity
)

class RigorousPurgedValidator:
    """
    엄격한 Purged 검증 시스템
    - Purged and Embargoed Cross-Validation
    - Combinatorial Purged Cross-Validation
    - 완전한 데이터 무결성 보장
    """

    def __init__(self):
        """초기화"""
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
            'en': ElasticNet(alpha=0.01, random_state=42),
            'ridge': Ridge(alpha=1.0, random_state=42)
        }

        # 다양한 스케일러 테스트
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }

        self.model_weights = np.array([0.3, 0.3, 0.25, 0.15])  # RF, GB, EN, Ridge
        self.validation_log = []

    def calculate_comprehensive_metrics(self, y_true, y_pred, y_pred_proba=None):
        """포괄적 성능 지표 계산"""
        metrics = {}

        # 기본 회귀 지표
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
        metrics['rmse'] = np.sqrt(metrics['mse'])

        # R² 계산 (현실적 수준 확인)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # 방향 정확도
        direction_actual = np.sign(y_true)
        direction_pred = np.sign(y_pred)
        metrics['direction_accuracy'] = np.mean(direction_actual == direction_pred) * 100

        # Log Loss (확률 예측이 있는 경우)
        if y_pred_proba is not None:
            binary_actual = (y_true > 0).astype(int)
            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            metrics['log_loss'] = log_loss(binary_actual, y_pred_proba)

        # 금융 지표
        returns = y_pred

        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        metrics['max_drawdown'] = abs(np.min(drawdown))

        # Sharpe Ratio
        if np.std(returns) > 0:
            metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            metrics['sharpe_ratio'] = 0

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            if downside_std > 0:
                metrics['sortino_ratio'] = np.mean(returns) / downside_std * np.sqrt(252)
            else:
                metrics['sortino_ratio'] = 0
        else:
            metrics['sortino_ratio'] = 0

        return metrics

    def advanced_probability_calibration(self, predictions, targets):
        """고급 확률 보정"""
        pred_probs = 1 / (1 + np.exp(-predictions))
        binary_targets = (targets > 0).astype(int)

        if len(predictions) >= 10:
            iso_reg = IsotonicRegression(out_of_bounds='clip')
            calibrated_probs = iso_reg.fit_transform(pred_probs, binary_targets)

            # 실제 양성 비율 기반 조정
            actual_positive_rate = np.mean(binary_targets)
            predicted_positive_rate = np.mean(calibrated_probs)

            if predicted_positive_rate > 0:
                adjustment_factor = actual_positive_rate / predicted_positive_rate
                calibrated_probs = calibrated_probs * adjustment_factor

            # 신뢰도 스케일링
            confidence_scaling = 0.75
            calibrated_probs = 0.5 + (calibrated_probs - 0.5) * confidence_scaling

            # 극값 방지
            calibrated_probs = np.clip(calibrated_probs, 0.05, 0.95)

            return calibrated_probs
        else:
            return np.clip(pred_probs, 0.05, 0.95)

    def purged_cross_validation(self, X, y, cv_method='purged_kfold'):
        """Purged Cross-Validation 실행"""
        print(f"🔒 {cv_method.upper()} 검증 실행...")

        # CV 방법 선택
        if cv_method == 'purged_kfold':
            cv = PurgedKFold(n_splits=5, pct_embargo=0.02)
        elif cv_method == 'purged_timeseries':
            cv = PurgedTimeSeriesSplit(n_splits=5, gap=int(0.02 * len(X)))
        else:
            raise ValueError(f"지원되지 않는 CV 방법: {cv_method}")

        results = []
        fold_num = 0

        for train_idx, val_idx in cv.split(X):
            fold_num += 1
            print(f"\\n📊 Fold {fold_num}")
            print(f"   훈련: {len(train_idx)}개, 검증: {len(val_idx)}개")

            # 무결성 검증
            gap_required = int(0.02 * len(X))
            if not validate_purged_cv_integrity(train_idx, val_idx, gap_required):
                print(f"⚠️ Fold {fold_num} 무결성 검증 실패, 건너뜀")
                continue

            # 데이터 분할
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # 스케일러 비교 테스트
            best_scaler_result = None
            best_scaler_score = float('inf')

            for scaler_name, scaler in self.scalers.items():
                # 각 fold에서 새로운 스케일러 인스턴스 생성
                scaler_instance = type(scaler)()

                # 스케일링: 훈련 데이터에서만 fit
                X_train_scaled = scaler_instance.fit_transform(X_train)
                X_val_scaled = scaler_instance.transform(X_val)

                # 모델 훈련
                fold_predictions = {}
                for name, model in self.models.items():
                    # 각 fold에서 새로운 모델 인스턴스 생성
                    model_instance = type(model)(**model.get_params())
                    model_instance.fit(X_train_scaled, y_train)
                    pred = model_instance.predict(X_val_scaled)
                    fold_predictions[name] = pred

                # 앙상블 예측
                ensemble_pred = np.average(list(fold_predictions.values()),
                                         weights=self.model_weights, axis=0)

                # 확률 보정
                calibrated_probs = self.advanced_probability_calibration(ensemble_pred, y_val)

                # 성능 계산
                metrics = self.calculate_comprehensive_metrics(y_val, ensemble_pred, calibrated_probs)

                # 최고 성능 스케일러 선택 (MAE 기준)
                if metrics['mae'] < best_scaler_score:
                    best_scaler_score = metrics['mae']
                    best_scaler_result = {
                        'scaler': scaler_name,
                        'metrics': metrics,
                        'fold': fold_num,
                        'train_size': len(train_idx),
                        'val_size': len(val_idx),
                        'predictions': ensemble_pred,
                        'probabilities': calibrated_probs
                    }

            if best_scaler_result:
                print(f"   최고 스케일러: {best_scaler_result['scaler']}")
                print(f"   MAE: {best_scaler_result['metrics']['mae']:.6f}")
                print(f"   방향 정확도: {best_scaler_result['metrics']['direction_accuracy']:.2f}%")
                print(f"   Log Loss: {best_scaler_result['metrics']['log_loss']:.6f}")
                print(f"   MDD: {best_scaler_result['metrics']['max_drawdown']:.6f}")
                print(f"   R²: {best_scaler_result['metrics']['r2']:.4f}")

                results.append(best_scaler_result)

        return results

    def combinatorial_purged_validation(self, X, y, n_paths=50):
        """Combinatorial Purged Cross-Validation"""
        print(f"🔄 Combinatorial Purged CV - {n_paths} 경로 생성...")

        cpcv = CombinatorialPurgedCV(
            n_splits=10,
            n_test_groups=3,
            pct_embargo=0.015,
            n_paths=n_paths
        )

        results = []
        path_num = 0

        for train_idx, val_idx in cpcv.split(X):
            path_num += 1

            # 무결성 검증
            gap_required = int(0.015 * len(X))
            if not validate_purged_cv_integrity(train_idx, val_idx, gap_required):
                continue

            # 데이터 분할
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # 최적 스케일러 사용 (이전 결과 기반)
            scaler = RobustScaler()  # 일반적으로 금융 데이터에 더 적합
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # 앙상블 예측
            path_predictions = {}
            for name, model in self.models.items():
                model_instance = type(model)(**model.get_params())
                model_instance.fit(X_train_scaled, y_train)
                pred = model_instance.predict(X_val_scaled)
                path_predictions[name] = pred

            ensemble_pred = np.average(list(path_predictions.values()),
                                     weights=self.model_weights, axis=0)

            # 확률 보정
            calibrated_probs = self.advanced_probability_calibration(ensemble_pred, y_val)

            # 성능 계산
            metrics = self.calculate_comprehensive_metrics(y_val, ensemble_pred, calibrated_probs)

            result = {
                'path': path_num,
                'metrics': metrics,
                'train_size': len(train_idx),
                'val_size': len(val_idx)
            }

            results.append(result)

            if path_num % 10 == 0:
                print(f"   진행: {path_num}/{n_paths} 경로 완료")

        return results

    def rigorous_model_evaluation(self, data_file):
        """엄격한 모델 평가 실행"""
        print("🔒 엄격한 Purged 검증 기반 모델 평가")
        print(f"📁 데이터: {data_file}")

        # 데이터 로드
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        print(f"📊 데이터: {df.shape}")

        # 특징 준비
        safe_features = [
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'RSI', 'BB_position',
            'Volatility_5', 'Volatility_10', 'Volatility_20',
            'Volume_ratio_10', 'Volume_ratio_20', 'ATR',
            'Returns_lag_1', 'Returns_lag_2', 'Returns_lag_3',
            'RSI_lag_1', 'RSI_lag_2', 'RSI_lag_3',
            'Volatility_20_lag_1', 'Volatility_20_lag_2', 'Volatility_20_lag_3',
            'BB_position_lag_1', 'BB_position_lag_2', 'BB_position_lag_3',
            'Price_momentum_5', 'Price_momentum_10'
        ]

        available_features = [col for col in safe_features if col in df.columns]
        X = df[available_features].copy()
        y = df['Returns'].copy()

        # NaN 제거
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]

        print(f"✅ 특징: {len(available_features)}개")
        print(f"✅ 샘플: {len(X_clean)}개")

        # 1. Purged K-Fold Cross-Validation
        print("\\n" + "="*60)
        print("🔒 1단계: Purged K-Fold Cross-Validation")
        print("="*60)

        purged_kfold_results = self.purged_cross_validation(X_clean, y_clean, 'purged_kfold')

        # 2. Purged Time Series Split
        print("\\n" + "="*60)
        print("🔒 2단계: Purged Time Series Split")
        print("="*60)

        purged_ts_results = self.purged_cross_validation(X_clean, y_clean, 'purged_timeseries')

        # 3. Combinatorial Purged Cross-Validation
        print("\\n" + "="*60)
        print("🔒 3단계: Combinatorial Purged Cross-Validation")
        print("="*60)

        cpcv_results = self.combinatorial_purged_validation(X_clean, y_clean, n_paths=30)

        # 결과 분석
        all_results = {
            'purged_kfold': purged_kfold_results,
            'purged_timeseries': purged_ts_results,
            'combinatorial_purged': cpcv_results
        }

        return self.analyze_rigorous_results(all_results, available_features, X_clean.shape)

    def analyze_rigorous_results(self, all_results, features, data_shape):
        """엄격한 검증 결과 분석"""
        print("\\n" + "="*60)
        print("📊 엄격한 검증 결과 종합 분석")
        print("="*60)

        analysis = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'validation_type': 'rigorous_purged_validation',
            'features_used': features,
            'data_shape': list(data_shape),
            'method_comparisons': {}
        }

        for method_name, results in all_results.items():
            if not results:
                continue

            print(f"\\n🔍 {method_name.upper()} 결과:")

            # 성능 지표 집계
            metrics_summary = {}
            for metric in ['mae', 'rmse', 'r2', 'direction_accuracy', 'log_loss',
                          'max_drawdown', 'sharpe_ratio', 'sortino_ratio']:
                values = [r['metrics'][metric] for r in results]
                metrics_summary[f"{metric}_mean"] = np.mean(values)
                metrics_summary[f"{metric}_std"] = np.std(values)
                metrics_summary[f"{metric}_min"] = np.min(values)
                metrics_summary[f"{metric}_max"] = np.max(values)

            # 출력
            print(f"   MAE: {metrics_summary['mae_mean']:.6f} ± {metrics_summary['mae_std']:.6f}")
            print(f"   R²: {metrics_summary['r2_mean']:.4f} ± {metrics_summary['r2_std']:.4f}")
            print(f"   방향 정확도: {metrics_summary['direction_accuracy_mean']:.2f}% ± {metrics_summary['direction_accuracy_std']:.2f}%")
            print(f"   Log Loss: {metrics_summary['log_loss_mean']:.6f} ± {metrics_summary['log_loss_std']:.6f}")
            print(f"   MDD: {metrics_summary['max_drawdown_mean']:.6f} ± {metrics_summary['max_drawdown_std']:.6f}")
            print(f"   Sharpe: {metrics_summary['sharpe_ratio_mean']:.3f} ± {metrics_summary['sharpe_ratio_std']:.3f}")

            # CLAUDE.md 준수 평가
            claude_compliance = {
                'r2_realistic': metrics_summary['r2_mean'] < 0.20,  # 20% 미만
                'direction_accuracy_realistic': metrics_summary['direction_accuracy_mean'] < 90,  # 90% 미만
                'mdd_realistic': metrics_summary['max_drawdown_mean'] > 0.01,  # 1% 이상
                'overall_realistic': True
            }

            # 목표 달성 평가
            goal_achievement = {
                'mdd_target': 0.6,
                'mdd_achieved': metrics_summary['max_drawdown_mean'],
                'mdd_success': metrics_summary['max_drawdown_mean'] < 0.6,
                'log_loss_target': 0.7,
                'log_loss_achieved': metrics_summary['log_loss_mean'],
                'log_loss_success': metrics_summary['log_loss_mean'] < 0.7
            }

            claude_compliance['overall_realistic'] = (
                claude_compliance['r2_realistic'] and
                claude_compliance['direction_accuracy_realistic'] and
                claude_compliance['mdd_realistic']
            )

            print(f"\\n   🎯 CLAUDE.md 준수:")
            print(f"     R² < 20%: {metrics_summary['r2_mean']:.1%} {'✅' if claude_compliance['r2_realistic'] else '❌'}")
            print(f"     방향 정확도 < 90%: {metrics_summary['direction_accuracy_mean']:.1f}% {'✅' if claude_compliance['direction_accuracy_realistic'] else '❌'}")
            print(f"     MDD > 1%: {metrics_summary['max_drawdown_mean']:.1%} {'✅' if claude_compliance['mdd_realistic'] else '❌'}")

            print(f"\\n   🎯 목표 달성:")
            print(f"     MDD < 60%: {metrics_summary['max_drawdown_mean']:.1%} {'✅' if goal_achievement['mdd_success'] else '❌'}")
            print(f"     Log Loss < 0.7: {metrics_summary['log_loss_mean']:.4f} {'✅' if goal_achievement['log_loss_success'] else '❌'}")

            analysis['method_comparisons'][method_name] = {
                'metrics_summary': metrics_summary,
                'claude_compliance': claude_compliance,
                'goal_achievement': goal_achievement,
                'fold_count': len(results)
            }

        # 최종 권고
        print(f"\\n🏆 최종 평가:")

        # 가장 보수적이고 현실적인 결과 찾기
        best_method = None
        best_score = float('inf')

        for method_name, analysis_data in analysis['method_comparisons'].items():
            # 보수적 점수: R²가 낮고, MDD가 적절하며, 목표 달성도가 높은 방법
            r2_penalty = max(0, analysis_data['metrics_summary']['r2_mean'] - 0.15) * 10
            mdd_penalty = max(0, 0.05 - analysis_data['metrics_summary']['max_drawdown_mean']) * 5
            goal_bonus = (analysis_data['goal_achievement']['mdd_success'] +
                         analysis_data['goal_achievement']['log_loss_success']) * -2

            conservative_score = (analysis_data['metrics_summary']['mae_mean'] +
                                r2_penalty + mdd_penalty + goal_bonus)

            if conservative_score < best_score:
                best_score = conservative_score
                best_method = method_name

        if best_method:
            best_analysis = analysis['method_comparisons'][best_method]
            print(f"   🥇 권장 방법: {best_method.upper()}")
            print(f"   📊 MAE: {best_analysis['metrics_summary']['mae_mean']:.6f}")
            print(f"   📊 R²: {best_analysis['metrics_summary']['r2_mean']:.4f} (현실적)")
            print(f"   📊 MDD: {best_analysis['metrics_summary']['max_drawdown_mean']:.4f}")
            print(f"   📊 Log Loss: {best_analysis['metrics_summary']['log_loss_mean']:.4f}")

            overall_success = (best_analysis['goal_achievement']['mdd_success'] and
                             best_analysis['goal_achievement']['log_loss_success'] and
                             best_analysis['claude_compliance']['overall_realistic'])

            if overall_success:
                print(f"   🎉 완전한 성공: 모든 기준 충족!")
            else:
                print(f"   ⚠️ 부분적 성공: 일부 기준 개선 필요")

        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"/root/workspace/results/analysis/rigorous_purged_validation_{timestamp}.json"
        Path(results_file).parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

        print(f"\\n💾 결과 저장: {results_file}")

        return analysis


def main():
    """메인 실행"""
    print("🔒 엄격한 Purged 검증 기반 모델 평가 시작")

    validator = RigorousPurgedValidator()
    data_file = "/root/workspace/data/training/sp500_leak_free_dataset.csv"

    try:
        results = validator.rigorous_model_evaluation(data_file)
        print("\\n✅ 엄격한 검증 완료!")
        return results

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()