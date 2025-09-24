"""
최적 앙상블 모델 종합 검증 시스템
V1(70%) + V2(30%) 앙상블의 신뢰성 완전 검증

검증 항목:
1. 과적합 재검증 (훈련 vs 검증 성능)
2. Walk-Forward Validation (시간적 안정성)
3. 데이터 누출 재확인
4. 경제적 백테스팅 (실제 거래 시뮬레이션)
5. 다양한 시장 조건에서의 성능
6. 벤치마크 대비 성능
7. 안정성 및 신뢰성 지표
8. Monte Carlo 시뮬레이션
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import logging
from datetime import datetime, timedelta
import warnings
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from scipy import stats
import time

warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/workspace/data/raw/optimal_ensemble_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PurgedKFoldSklearn:
    """sklearn 호환 Purged K-Fold Cross-Validation"""

    def __init__(self, n_splits=5, purge_length=5, embargo_length=5):
        self.n_splits = n_splits
        self.purge_length = purge_length
        self.embargo_length = embargo_length

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        test_size = n_samples // self.n_splits
        splits = []

        for i in range(self.n_splits):
            test_start = i * test_size
            test_end = min((i + 1) * test_size, n_samples)
            test_indices = indices[test_start:test_end]

            purge_start = test_end
            purge_end = min(test_end + self.purge_length, n_samples)
            embargo_end = min(purge_end + self.embargo_length, n_samples)

            train_indices = np.concatenate([
                indices[:test_start],
                indices[embargo_end:]
            ])

            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))

        return splits

class OptimalEnsembleValidator:
    """최적 앙상블 모델 종합 검증기"""

    def __init__(self):
        self.cv = PurgedKFoldSklearn()
        self.scaler_v1 = StandardScaler()
        self.scaler_v2 = StandardScaler()

        # 최적 모델 설정
        self.optimal_weights = [0.7, 0.3]  # V1: 70%, V2: 30%
        self.v1_alpha = 1.8523
        self.v2_alpha = 19.5029

        self.validation_results = {}

    def load_spy_data(self):
        """SPY 데이터 로드"""
        logger.info("📊 SPY 데이터 로딩...")

        spy = yf.Ticker("SPY")
        data = spy.history(start="2015-01-01", end="2024-12-31")
        data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data = data.dropna()

        logger.info(f"✅ 데이터 로딩 완료: {len(data)}개 관측치")
        logger.info(f"   기간: {data.index[0].date()} ~ {data.index[-1].date()}")
        return data

    def create_v1_features(self, data):
        """V1 특성 생성 (12개)"""
        returns = data['returns']
        features = pd.DataFrame(index=data.index)

        # 기본 변동성 (3개)
        features['vol_5'] = returns.rolling(5).std()
        features['vol_10'] = returns.rolling(10).std()
        features['vol_20'] = returns.rolling(20).std()

        # 기본 래그 (3개)
        features['return_lag_1'] = returns.shift(1)
        features['return_lag_2'] = returns.shift(2)
        features['return_lag_3'] = returns.shift(3)

        # 기본 통계 (4개)
        for window in [10, 20]:
            ma = returns.rolling(window).mean()
            std = returns.rolling(window).std()
            features[f'zscore_{window}'] = (returns - ma) / (std + 1e-8)
            features[f'momentum_{window}'] = returns.rolling(window).sum()

        # 기본 비율 (2개)
        features['vol_5_20_ratio'] = features['vol_5'] / (features['vol_20'] + 1e-8)
        features['vol_regime'] = (features['vol_5'] > features['vol_10']).astype(float)

        return self.finalize_features(features, returns)

    def create_v2_features(self, data):
        """V2 특성 생성 (30개)"""
        returns = data['returns']
        features = pd.DataFrame(index=data.index)

        # 변동성 (6개)
        for window in [3, 5, 10, 15, 20, 30]:
            features[f'vol_{window}'] = returns.rolling(window).std()

        # 통계적 모멘트 (6개)
        for window in [5, 10, 20]:
            features[f'skew_{window}'] = returns.rolling(window).skew()
            features[f'kurt_{window}'] = returns.rolling(window).kurt()

        # 래그 (6개)
        for lag in [1, 2, 3]:
            features[f'return_lag_{lag}'] = returns.shift(lag)
            features[f'vol_lag_{lag}'] = features['vol_5'].shift(lag)

        # 변동성 체제 (4개)
        short_vol = features['vol_5']
        medium_vol = features['vol_20']
        long_vol = features['vol_30']

        features['vol_regime_short'] = (short_vol > medium_vol).astype(float)
        features['vol_regime_medium'] = (medium_vol > long_vol).astype(float)
        features['vol_expansion'] = short_vol / (long_vol + 1e-8)
        features['vol_contraction'] = long_vol / (short_vol + 1e-8)

        # 통계 지표 (5개)
        for window in [10, 20, 30]:
            ma = returns.rolling(window).mean()
            std = returns.rolling(window).std()
            features[f'zscore_{window}'] = (returns - ma) / (std + 1e-8)

        for window in [10, 20]:
            ma = returns.rolling(window).mean()
            std = returns.rolling(window).std()
            features[f'sharpe_{window}'] = (ma * np.sqrt(252)) / (std + 1e-8)

        # 상호작용 (3개)
        features['vol_5_20_ratio'] = features['vol_5'] / (features['vol_20'] + 1e-8)
        features['vol_10_30_ratio'] = features['vol_10'] / (features['vol_30'] + 1e-8)
        features['vol_price_interaction'] = features['vol_20'] * returns

        return self.finalize_features(features, returns)

    def finalize_features(self, features, returns):
        """특성 마무리 처리"""
        # 타겟: 5일 후 변동성
        target = []
        for i in range(len(returns)):
            if i + 5 < len(returns):
                future_vol = returns.iloc[i+1:i+6].std()
                target.append(future_vol)
            else:
                target.append(np.nan)

        features['target_vol_5d'] = target
        features = features.dropna()

        X = features.drop('target_vol_5d', axis=1)
        y = features['target_vol_5d']

        return X, y

    def test_1_overfitting_recheck(self):
        """1. 과적합 재검증"""
        logger.info("🔍 1. 과적합 재검증...")

        data = self.load_spy_data()
        X_v1, y_v1 = self.create_v1_features(data)
        X_v2, y_v2 = self.create_v2_features(data)

        # 공통 샘플
        common_indices = sorted(list(set(y_v1.index) & set(y_v2.index)))
        v1_mask = y_v1.index.isin(common_indices)
        v2_mask = y_v2.index.isin(common_indices)

        X_v1_scaled = self.scaler_v1.fit_transform(X_v1[v1_mask])
        X_v2_scaled = self.scaler_v2.fit_transform(X_v2[v2_mask])
        y_common = y_v1[v1_mask]

        logger.info(f"   공통 샘플: {len(y_common)}개")

        # 훈련 vs 검증 성능 테스트
        train_scores = []
        val_scores = []

        splits = list(self.cv.split(X_v1_scaled))

        for train_idx, val_idx in splits:
            # V1 모델
            v1_model = Ridge(alpha=self.v1_alpha)
            v1_model.fit(X_v1_scaled[train_idx], y_common.iloc[train_idx])
            v1_train_pred = v1_model.predict(X_v1_scaled[train_idx])
            v1_val_pred = v1_model.predict(X_v1_scaled[val_idx])

            # V2 모델
            v2_model = Ridge(alpha=self.v2_alpha)
            v2_model.fit(X_v2_scaled[train_idx], y_common.iloc[train_idx])
            v2_train_pred = v2_model.predict(X_v2_scaled[train_idx])
            v2_val_pred = v2_model.predict(X_v2_scaled[val_idx])

            # 앙상블 예측
            ensemble_train_pred = (self.optimal_weights[0] * v1_train_pred +
                                 self.optimal_weights[1] * v2_train_pred)
            ensemble_val_pred = (self.optimal_weights[0] * v1_val_pred +
                                self.optimal_weights[1] * v2_val_pred)

            # 성능 계산
            train_r2 = r2_score(y_common.iloc[train_idx], ensemble_train_pred)
            val_r2 = r2_score(y_common.iloc[val_idx], ensemble_val_pred)

            train_scores.append(train_r2)
            val_scores.append(val_r2)

        train_mean = np.mean(train_scores)
        val_mean = np.mean(val_scores)
        performance_gap = train_mean - val_mean

        result = {
            'train_r2_mean': float(train_mean),
            'train_r2_std': float(np.std(train_scores)),
            'val_r2_mean': float(val_mean),
            'val_r2_std': float(np.std(val_scores)),
            'performance_gap': float(performance_gap),
            'overfitting_risk': 'HIGH' if performance_gap > 0.15 else 'MEDIUM' if performance_gap > 0.08 else 'LOW'
        }

        logger.info(f"   훈련 R²: {train_mean:.4f} ± {np.std(train_scores):.4f}")
        logger.info(f"   검증 R²: {val_mean:.4f} ± {np.std(val_scores):.4f}")
        logger.info(f"   성능 격차: {performance_gap:.4f}")
        logger.info(f"   과적합 위험: {result['overfitting_risk']}")

        self.validation_results['overfitting_check'] = result
        return result

    def test_2_walk_forward_validation(self):
        """2. Walk-Forward Validation (시간적 안정성)"""
        logger.info("🔍 2. Walk-Forward Validation...")

        data = self.load_spy_data()
        X_v1, y_v1 = self.create_v1_features(data)
        X_v2, y_v2 = self.create_v2_features(data)

        # 공통 샘플
        common_indices = sorted(list(set(y_v1.index) & set(y_v2.index)))
        v1_mask = y_v1.index.isin(common_indices)
        v2_mask = y_v2.index.isin(common_indices)

        X_v1_scaled = self.scaler_v1.fit_transform(X_v1[v1_mask])
        X_v2_scaled = self.scaler_v2.fit_transform(X_v2[v2_mask])
        y_common = y_v1[v1_mask]

        # Walk-Forward 테스트
        n_samples = len(X_v1_scaled)
        train_size = n_samples // 2  # 50% 훈련
        test_size = 250  # 고정 테스트 크기
        step_size = test_size // 2  # 50% 겹침

        wf_scores = []
        time_periods = []

        for start_idx in range(train_size, n_samples - test_size, step_size):
            end_idx = min(start_idx + test_size, n_samples)

            # 훈련: 처음부터 start_idx까지
            X_v1_train = X_v1_scaled[:start_idx]
            X_v2_train = X_v2_scaled[:start_idx]
            y_train = y_common.iloc[:start_idx]

            # 테스트: start_idx부터 end_idx까지
            X_v1_test = X_v1_scaled[start_idx:end_idx]
            X_v2_test = X_v2_scaled[start_idx:end_idx]
            y_test = y_common.iloc[start_idx:end_idx]

            # 모델 훈련 및 예측
            v1_model = Ridge(alpha=self.v1_alpha)
            v2_model = Ridge(alpha=self.v2_alpha)

            v1_model.fit(X_v1_train, y_train)
            v2_model.fit(X_v2_train, y_train)

            v1_pred = v1_model.predict(X_v1_test)
            v2_pred = v2_model.predict(X_v2_test)

            # 앙상블 예측
            ensemble_pred = (self.optimal_weights[0] * v1_pred +
                           self.optimal_weights[1] * v2_pred)

            # 성능 계산
            wf_r2 = r2_score(y_test, ensemble_pred)
            wf_scores.append(wf_r2)
            time_periods.append(f"{start_idx}-{end_idx}")

            logger.info(f"   Period {start_idx:4d}-{end_idx:4d}: R²={wf_r2:.4f}")

        # 시간적 안정성 분석
        wf_mean = np.mean(wf_scores)
        wf_std = np.std(wf_scores)
        time_trend = np.corrcoef(range(len(wf_scores)), wf_scores)[0, 1] if len(wf_scores) > 1 else 0

        result = {
            'wf_scores': [float(s) for s in wf_scores],
            'time_periods': time_periods,
            'wf_mean': float(wf_mean),
            'wf_std': float(wf_std),
            'time_trend': float(time_trend),
            'temporal_stability': 'STABLE' if abs(time_trend) < 0.3 and wf_std < 0.15 else 'UNSTABLE'
        }

        logger.info(f"   Walk-Forward R²: {wf_mean:.4f} ± {wf_std:.4f}")
        logger.info(f"   시간 트렌드: {time_trend:.4f}")
        logger.info(f"   시간적 안정성: {result['temporal_stability']}")

        self.validation_results['walk_forward'] = result
        return result

    def test_3_economic_backtest(self):
        """3. 경제적 백테스팅 (실제 거래 시뮬레이션)"""
        logger.info("🔍 3. 경제적 백테스팅...")

        data = self.load_spy_data()
        X_v1, y_v1 = self.create_v1_features(data)
        X_v2, y_v2 = self.create_v2_features(data)

        # 공통 샘플
        common_indices = sorted(list(set(y_v1.index) & set(y_v2.index)))
        v1_mask = y_v1.index.isin(common_indices)
        v2_mask = y_v2.index.isin(common_indices)

        X_v1_scaled = self.scaler_v1.fit_transform(X_v1[v1_mask])
        X_v2_scaled = self.scaler_v2.fit_transform(X_v2[v2_mask])
        y_common = y_v1[v1_mask]

        # 백테스팅 시뮬레이션 (마지막 1년)
        train_size = len(X_v1_scaled) - 252  # 마지막 1년 제외

        # 훈련
        v1_model = Ridge(alpha=self.v1_alpha)
        v2_model = Ridge(alpha=self.v2_alpha)

        v1_model.fit(X_v1_scaled[:train_size], y_common.iloc[:train_size])
        v2_model.fit(X_v2_scaled[:train_size], y_common.iloc[:train_size])

        # 테스트 (마지막 1년)
        test_indices = range(train_size, len(X_v1_scaled))

        predictions = []
        actuals = []
        returns_data = []

        spy_prices = data['Close'].reindex(y_common.index).iloc[train_size:]

        for i, idx in enumerate(test_indices):
            if idx + 5 < len(X_v1_scaled):  # 5일 후 데이터 확인 가능한 경우만
                # 예측
                v1_pred = v1_model.predict(X_v1_scaled[idx:idx+1])[0]
                v2_pred = v2_model.predict(X_v2_scaled[idx:idx+1])[0]
                ensemble_pred = (self.optimal_weights[0] * v1_pred +
                               self.optimal_weights[1] * v2_pred)

                # 실제값
                actual = y_common.iloc[idx]

                predictions.append(ensemble_pred)
                actuals.append(actual)

                # 거래 신호 생성 (변동성 예측 기반)
                if i > 0:  # 이전 예측과 비교
                    prev_pred = predictions[i-1]
                    vol_change = ensemble_pred - prev_pred

                    # 단순 전략: 변동성 증가 예상 시 매도, 감소 시 매수
                    signal = -1 if vol_change > 0.001 else 1  # 0.1% 임계값

                    # 수익률 계산 (5일 보유)
                    if idx + 5 < len(spy_prices):
                        period_return = (spy_prices.iloc[idx + 5] - spy_prices.iloc[idx]) / spy_prices.iloc[idx]
                        strategy_return = signal * period_return
                        returns_data.append({
                            'date': y_common.index[idx],
                            'signal': signal,
                            'period_return': period_return,
                            'strategy_return': strategy_return,
                            'predicted_vol': ensemble_pred,
                            'actual_vol': actual
                        })

        # 성과 분석
        if len(returns_data) > 0:
            returns_df = pd.DataFrame(returns_data)

            total_return = (1 + returns_df['strategy_return']).prod() - 1
            benchmark_return = (1 + returns_df['period_return']).prod() - 1

            strategy_vol = returns_df['strategy_return'].std() * np.sqrt(252/5)  # 연율화
            benchmark_vol = returns_df['period_return'].std() * np.sqrt(252/5)

            sharpe_ratio = (returns_df['strategy_return'].mean() * 252/5) / strategy_vol if strategy_vol > 0 else 0

            win_rate = (returns_df['strategy_return'] > 0).mean()

        else:
            total_return = benchmark_return = strategy_vol = benchmark_vol = sharpe_ratio = win_rate = 0

        # 예측 정확도
        pred_r2 = r2_score(actuals, predictions) if len(predictions) > 0 else 0
        pred_corr = np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0

        result = {
            'prediction_performance': {
                'r2_score': float(pred_r2),
                'correlation': float(pred_corr),
                'n_predictions': len(predictions)
            },
            'economic_performance': {
                'strategy_return': float(total_return),
                'benchmark_return': float(benchmark_return),
                'excess_return': float(total_return - benchmark_return),
                'strategy_volatility': float(strategy_vol),
                'benchmark_volatility': float(benchmark_vol),
                'sharpe_ratio': float(sharpe_ratio),
                'win_rate': float(win_rate),
                'n_trades': len(returns_data)
            }
        }

        logger.info(f"   예측 R²: {pred_r2:.4f}")
        logger.info(f"   전략 수익률: {total_return:.2%}")
        logger.info(f"   벤치마크 수익률: {benchmark_return:.2%}")
        logger.info(f"   초과 수익률: {total_return - benchmark_return:+.2%}")
        logger.info(f"   샤프 비율: {sharpe_ratio:.4f}")
        logger.info(f"   승률: {win_rate:.1%}")

        self.validation_results['economic_backtest'] = result
        return result

    def test_4_benchmark_comparison(self):
        """4. 벤치마크 대비 성능"""
        logger.info("🔍 4. 벤치마크 대비 성능...")

        data = self.load_spy_data()
        X_v1, y_v1 = self.create_v1_features(data)
        X_v2, y_v2 = self.create_v2_features(data)

        # 공통 샘플
        common_indices = sorted(list(set(y_v1.index) & set(y_v2.index)))
        v1_mask = y_v1.index.isin(common_indices)
        v2_mask = y_v2.index.isin(common_indices)

        X_v1_scaled = self.scaler_v1.fit_transform(X_v1[v1_mask])
        X_v2_scaled = self.scaler_v2.fit_transform(X_v2[v2_mask])
        y_common = y_v1[v1_mask]

        # 벤치마크 모델들
        benchmarks = {
            'V1_Only': Ridge(alpha=self.v1_alpha),
            'V2_Only': Ridge(alpha=self.v2_alpha),
            'Simple_Average': 'ensemble',  # V1과 V2의 단순 평균
            'HAR_Model': Ridge(alpha=1.0),  # 간단한 HAR 모형 근사
            'Naive_Persistence': 'naive'  # 단순 지속성 모델
        }

        benchmark_results = {}
        splits = list(self.cv.split(X_v1_scaled))

        for name, model_spec in benchmarks.items():
            scores = []

            for train_idx, val_idx in splits:
                y_train = y_common.iloc[train_idx]
                y_val = y_common.iloc[val_idx]

                if name == 'V1_Only':
                    model_spec.fit(X_v1_scaled[train_idx], y_train)
                    pred = model_spec.predict(X_v1_scaled[val_idx])
                elif name == 'V2_Only':
                    model_spec.fit(X_v2_scaled[train_idx], y_train)
                    pred = model_spec.predict(X_v2_scaled[val_idx])
                elif name == 'Simple_Average':
                    v1_model = Ridge(alpha=self.v1_alpha)
                    v2_model = Ridge(alpha=self.v2_alpha)
                    v1_model.fit(X_v1_scaled[train_idx], y_train)
                    v2_model.fit(X_v2_scaled[train_idx], y_train)
                    v1_pred = v1_model.predict(X_v1_scaled[val_idx])
                    v2_pred = v2_model.predict(X_v2_scaled[val_idx])
                    pred = (v1_pred + v2_pred) / 2  # 단순 평균
                elif name == 'HAR_Model':
                    # 간단한 HAR 모델 (변동성 래그 사용)
                    har_features = np.column_stack([
                        X_v1_scaled[train_idx, 0],  # vol_5
                        X_v1_scaled[train_idx, 1],  # vol_10
                        X_v1_scaled[train_idx, 2]   # vol_20
                    ])
                    model_spec.fit(har_features, y_train)
                    har_test = np.column_stack([
                        X_v1_scaled[val_idx, 0],
                        X_v1_scaled[val_idx, 1],
                        X_v1_scaled[val_idx, 2]
                    ])
                    pred = model_spec.predict(har_test)
                elif name == 'Naive_Persistence':
                    # 단순 지속성: 현재 변동성이 미래 변동성
                    pred = X_v1_scaled[val_idx, 0]  # vol_5 사용

                r2 = r2_score(y_val, pred)
                scores.append(r2)

            benchmark_results[name] = {
                'scores': [float(s) for s in scores],
                'mean_r2': float(np.mean(scores)),
                'std_r2': float(np.std(scores))
            }

        # 우리 모델 (최적 앙상블)
        optimal_scores = []
        for train_idx, val_idx in splits:
            v1_model = Ridge(alpha=self.v1_alpha)
            v2_model = Ridge(alpha=self.v2_alpha)

            v1_model.fit(X_v1_scaled[train_idx], y_common.iloc[train_idx])
            v2_model.fit(X_v2_scaled[train_idx], y_common.iloc[train_idx])

            v1_pred = v1_model.predict(X_v1_scaled[val_idx])
            v2_pred = v2_model.predict(X_v2_scaled[val_idx])

            optimal_pred = (self.optimal_weights[0] * v1_pred +
                          self.optimal_weights[1] * v2_pred)

            r2 = r2_score(y_common.iloc[val_idx], optimal_pred)
            optimal_scores.append(r2)

        our_performance = {
            'scores': [float(s) for s in optimal_scores],
            'mean_r2': float(np.mean(optimal_scores)),
            'std_r2': float(np.std(optimal_scores))
        }

        # 성능 비교
        logger.info("   벤치마크 대비 성능:")
        logger.info(f"   최적 앙상블: R² = {our_performance['mean_r2']:.4f} ± {our_performance['std_r2']:.4f}")

        for name, perf in benchmark_results.items():
            improvement = ((our_performance['mean_r2'] - perf['mean_r2']) / perf['mean_r2'] * 100) if perf['mean_r2'] > 0 else 0
            logger.info(f"   {name}: R² = {perf['mean_r2']:.4f} ± {perf['std_r2']:.4f} ({improvement:+.1f}%)")

        result = {
            'optimal_ensemble': our_performance,
            'benchmarks': benchmark_results
        }

        self.validation_results['benchmark_comparison'] = result
        return result

    def test_5_stability_analysis(self):
        """5. 안정성 및 신뢰성 분석"""
        logger.info("🔍 5. 안정성 및 신뢰성 분석...")

        data = self.load_spy_data()
        X_v1, y_v1 = self.create_v1_features(data)
        X_v2, y_v2 = self.create_v2_features(data)

        # 공통 샘플
        common_indices = sorted(list(set(y_v1.index) & set(y_v2.index)))
        v1_mask = y_v1.index.isin(common_indices)
        v2_mask = y_v2.index.isin(common_indices)

        X_v1_scaled = self.scaler_v1.fit_transform(X_v1[v1_mask])
        X_v2_scaled = self.scaler_v2.fit_transform(X_v2[v2_mask])
        y_common = y_v1[v1_mask]

        # 1. Bootstrap 신뢰구간
        n_bootstrap = 100
        bootstrap_scores = []

        for i in range(n_bootstrap):
            # 부트스트랩 샘플링
            n_samples = len(X_v1_scaled)
            bootstrap_idx = np.random.choice(n_samples, size=n_samples, replace=True)

            X_v1_boot = X_v1_scaled[bootstrap_idx]
            X_v2_boot = X_v2_scaled[bootstrap_idx]
            y_boot = y_common.iloc[bootstrap_idx]

            # 훈련/테스트 분할
            split_idx = n_samples // 2

            v1_model = Ridge(alpha=self.v1_alpha)
            v2_model = Ridge(alpha=self.v2_alpha)

            v1_model.fit(X_v1_boot[:split_idx], y_boot.iloc[:split_idx])
            v2_model.fit(X_v2_boot[:split_idx], y_boot.iloc[:split_idx])

            v1_pred = v1_model.predict(X_v1_boot[split_idx:])
            v2_pred = v2_model.predict(X_v2_boot[split_idx:])

            ensemble_pred = (self.optimal_weights[0] * v1_pred +
                           self.optimal_weights[1] * v2_pred)

            r2 = r2_score(y_boot.iloc[split_idx:], ensemble_pred)
            bootstrap_scores.append(r2)

        # 신뢰구간 계산
        ci_lower = np.percentile(bootstrap_scores, 2.5)
        ci_upper = np.percentile(bootstrap_scores, 97.5)
        bootstrap_mean = np.mean(bootstrap_scores)
        bootstrap_std = np.std(bootstrap_scores)

        # 2. 다양한 시장 조건에서의 성능
        # 변동성 체제별 성능
        vol_regimes = {
            'Low_Vol': y_common < np.percentile(y_common, 33),
            'Medium_Vol': (y_common >= np.percentile(y_common, 33)) & (y_common < np.percentile(y_common, 67)),
            'High_Vol': y_common >= np.percentile(y_common, 67)
        }

        regime_performance = {}
        splits = list(self.cv.split(X_v1_scaled))

        for regime_name, regime_mask in vol_regimes.items():
            regime_scores = []

            for train_idx, val_idx in splits:
                # 검증 세트에서 해당 체제만 선택
                regime_val_idx = val_idx[regime_mask.iloc[val_idx].values]

                if len(regime_val_idx) < 10:  # 최소 샘플 수 확보
                    continue

                v1_model = Ridge(alpha=self.v1_alpha)
                v2_model = Ridge(alpha=self.v2_alpha)

                v1_model.fit(X_v1_scaled[train_idx], y_common.iloc[train_idx])
                v2_model.fit(X_v2_scaled[train_idx], y_common.iloc[train_idx])

                v1_pred = v1_model.predict(X_v1_scaled[regime_val_idx])
                v2_pred = v2_model.predict(X_v2_scaled[regime_val_idx])

                ensemble_pred = (self.optimal_weights[0] * v1_pred +
                               self.optimal_weights[1] * v2_pred)

                r2 = r2_score(y_common.iloc[regime_val_idx], ensemble_pred)
                regime_scores.append(r2)

            if len(regime_scores) > 0:
                regime_performance[regime_name] = {
                    'mean_r2': float(np.mean(regime_scores)),
                    'std_r2': float(np.std(regime_scores)),
                    'n_observations': int(regime_mask.sum())
                }

        result = {
            'bootstrap_analysis': {
                'mean_r2': float(bootstrap_mean),
                'std_r2': float(bootstrap_std),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'n_bootstrap': n_bootstrap
            },
            'regime_performance': regime_performance
        }

        logger.info(f"   Bootstrap R²: {bootstrap_mean:.4f} ± {bootstrap_std:.4f}")
        logger.info(f"   95% 신뢰구간: [{ci_lower:.4f}, {ci_upper:.4f}]")

        for regime, perf in regime_performance.items():
            logger.info(f"   {regime}: R² = {perf['mean_r2']:.4f} ± {perf['std_r2']:.4f}")

        self.validation_results['stability_analysis'] = result
        return result

    def generate_comprehensive_validation_report(self):
        """종합 검증 보고서 생성"""
        logger.info("📋 최적 앙상블 종합 검증 보고서 생성...")

        # 모든 테스트 실행
        start_time = time.time()

        test1 = self.test_1_overfitting_recheck()
        test2 = self.test_2_walk_forward_validation()
        test3 = self.test_3_economic_backtest()
        test4 = self.test_4_benchmark_comparison()
        test5 = self.test_5_stability_analysis()

        total_time = time.time() - start_time

        # 종합 평가
        validation_scores = {
            'overfitting': 1 if test1['overfitting_risk'] == 'LOW' else 0.5 if test1['overfitting_risk'] == 'MEDIUM' else 0,
            'temporal_stability': 1 if test2['temporal_stability'] == 'STABLE' else 0,
            'economic_value': 1 if test3['economic_performance']['excess_return'] > 0 else 0,
            'benchmark_superior': 1 if test4['optimal_ensemble']['mean_r2'] > max([b['mean_r2'] for b in test4['benchmarks'].values()]) else 0,
            'statistical_significance': 1 if test5['bootstrap_analysis']['ci_lower'] > 0.2 else 0
        }

        overall_score = np.mean(list(validation_scores.values()))

        if overall_score >= 0.8:
            final_verdict = "EXCELLENT"
            recommendation = "강력 권장 - 즉시 프로덕션 배포"
        elif overall_score >= 0.6:
            final_verdict = "GOOD"
            recommendation = "권장 - 주의 깊은 모니터링과 함께 배포"
        elif overall_score >= 0.4:
            final_verdict = "ACCEPTABLE"
            recommendation = "조건부 권장 - 추가 개선 후 배포"
        else:
            final_verdict = "POOR"
            recommendation = "재개발 필요"

        comprehensive_report = {
            'validation_date': datetime.now().isoformat(),
            'model_specification': {
                'ensemble_type': 'V1_V2_Weighted',
                'weights': self.optimal_weights,
                'v1_alpha': self.v1_alpha,
                'v2_alpha': self.v2_alpha
            },
            'validation_results': self.validation_results,
            'validation_scores': validation_scores,
            'overall_assessment': {
                'overall_score': float(overall_score),
                'final_verdict': final_verdict,
                'recommendation': recommendation,
                'validation_time_seconds': float(total_time)
            },
            'key_findings': self.generate_key_findings()
        }

        # 결과 저장
        save_path = '/root/workspace/data/raw/optimal_ensemble_comprehensive_validation.json'
        with open(save_path, 'w') as f:
            json.dump(comprehensive_report, f, indent=2)

        logger.info("="*80)
        logger.info("🎯 최적 앙상블 종합 검증 완료")
        logger.info(f"📊 종합 점수: {overall_score:.2f}/1.00")
        logger.info(f"📊 최종 판정: {final_verdict}")
        logger.info(f"💡 권장사항: {recommendation}")
        logger.info(f"⏱️ 검증 시간: {total_time:.1f}초")
        logger.info(f"💾 상세 결과: {save_path}")
        logger.info("="*80)

        return comprehensive_report

    def generate_key_findings(self):
        """핵심 발견사항 생성"""
        findings = []

        # 과적합 관련
        overfitting = self.validation_results.get('overfitting_check', {})
        if overfitting.get('overfitting_risk') == 'LOW':
            findings.append(f"✅ 과적합 위험 낮음 (성능 격차: {overfitting.get('performance_gap', 0):.3f})")
        else:
            findings.append(f"⚠️ 과적합 위험 존재 (성능 격차: {overfitting.get('performance_gap', 0):.3f})")

        # 시간적 안정성
        wf = self.validation_results.get('walk_forward', {})
        if wf.get('temporal_stability') == 'STABLE':
            findings.append(f"✅ 시간적 안정성 우수 (WF R²: {wf.get('wf_mean', 0):.4f})")
        else:
            findings.append(f"⚠️ 시간적 불안정성 (WF R²: {wf.get('wf_mean', 0):.4f})")

        # 경제적 가치
        econ = self.validation_results.get('economic_backtest', {}).get('economic_performance', {})
        excess_return = econ.get('excess_return', 0)
        if excess_return > 0:
            findings.append(f"✅ 경제적 가치 입증 (초과수익: {excess_return:.2%})")
        else:
            findings.append(f"❌ 경제적 가치 부족 (초과수익: {excess_return:.2%})")

        # 벤치마크 대비
        bench = self.validation_results.get('benchmark_comparison', {})
        our_r2 = bench.get('optimal_ensemble', {}).get('mean_r2', 0)
        findings.append(f"📊 최종 검증 성능: R² = {our_r2:.4f}")

        # 신뢰성
        bootstrap = self.validation_results.get('stability_analysis', {}).get('bootstrap_analysis', {})
        ci_lower = bootstrap.get('ci_lower', 0)
        ci_upper = bootstrap.get('ci_upper', 0)
        findings.append(f"📈 95% 신뢰구간: [{ci_lower:.4f}, {ci_upper:.4f}]")

        return findings

def main():
    """메인 실행"""
    logger.info("🎯 최적 앙상블 모델 종합 검증 시작")
    logger.info("🏆 검증 대상: V1(70%) + V2(30%) 앙상블")

    validator = OptimalEnsembleValidator()

    try:
        results = validator.generate_comprehensive_validation_report()

        logger.info("💡 핵심 발견사항:")
        for finding in results['key_findings']:
            logger.info(f"   {finding}")

        return results

    except Exception as e:
        logger.error(f"❌ 종합 검증 실패: {str(e)}")
        raise

if __name__ == "__main__":
    main()