#!/usr/bin/env python3
"""
State-of-the-Art Volatility Model (Fixed) - R² 0.25+ 목표
데이터 처리 문제 해결 버전
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

# 딥러닝 모델 (선택적 import)
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow 사용 가능 - LSTM 모델 활용")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow 없음 - 전통적인 ML 모델만 사용")

warnings.filterwarnings('ignore')

class SOTAVolatilityPredictorFixed:
    """State-of-the-Art 변동성 예측기 (수정된 버전)"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.performance_history = {}

    def load_comprehensive_data(self):
        """포괄적 데이터 로드 - VIX, 경제지표, 고빈도 프록시"""
        print("📊 포괄적 데이터 로드 중...")

        # SPY 기본 데이터 (더 긴 기간)
        spy = yf.download('SPY', start='2010-01-01', end='2024-12-31', progress=False)
        spy['returns'] = spy['Close'].pct_change()

        # VIX (핵심 지표)
        vix = yf.download('^VIX', start='2010-01-01', end='2024-12-31', progress=False)
        spy['vix'] = vix['Close'].reindex(spy.index, method='ffill')

        # 추가 지표들 (안전하게 로드)
        additional_data = {}
        tickers_info = {
            '^TNX': 'treasury_10y',
            'UUP': 'dollar_index',
            'GLD': 'gold',
            'USO': 'oil'
        }

        for ticker, name in tickers_info.items():
            try:
                data = yf.download(ticker, start='2010-01-01', end='2024-12-31', progress=False)
                additional_data[name] = data['Close'].reindex(spy.index, method='ffill')
                print(f"✅ {name} 로드 성공")
            except Exception as e:
                print(f"⚠️ {name} 로드 실패: {e}")
                additional_data[name] = pd.Series(index=spy.index, dtype=float)

        # 데이터 결합
        for name, series in additional_data.items():
            spy[name] = series

        # 초기 데이터 정리 (기본 컬럼만)
        initial_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'vix']
        spy_clean = spy[initial_columns].dropna()

        # 추가 지표들은 개별적으로 처리
        for name in additional_data.keys():
            if name in spy.columns:
                spy_clean[name] = spy[name].reindex(spy_clean.index, method='ffill')

        print(f"✅ 포괄적 데이터 로드 완료: {len(spy_clean)} 관측치")
        return spy_clean

    def create_stable_features(self, data):
        """안정적인 특성 생성 (NaN 최소화)"""
        print("🔧 안정적인 특성 생성 중...")

        features = pd.DataFrame(index=data.index)
        returns = data['returns']
        prices = data['Close']
        vix = data['vix']

        # 1. 기본 변동성 특성 (확실히 작동)
        for window in [5, 10, 20]:
            features[f'vol_{window}'] = returns.rolling(window, min_periods=min(window//2, 3)).std()
            features[f'vol_{window}_normalized'] = features[f'vol_{window}'] / features[f'vol_{window}'].rolling(252, min_periods=60).mean()

        # 2. VIX 기반 특성 (핵심)
        features['vix_level'] = vix
        features['vix_change'] = vix.pct_change()
        features['vix_ma_5'] = vix.rolling(5, min_periods=3).mean()
        features['vix_ma_20'] = vix.rolling(20, min_periods=10).mean()
        features['vix_regime'] = (vix > vix.rolling(60, min_periods=30).quantile(0.7)).astype(int)

        # 3. 수익률 기반 특성
        for lag in [1, 2, 3, 5]:
            features[f'returns_lag_{lag}'] = returns.shift(lag)
            features[f'abs_returns_lag_{lag}'] = np.abs(returns.shift(lag))

        # 4. 트렌드 특성
        features['price_ma_5'] = prices.rolling(5, min_periods=3).mean()
        features['price_ma_20'] = prices.rolling(20, min_periods=10).mean()
        features['trend_5_20'] = features['price_ma_5'] / features['price_ma_20'] - 1

        # 5. 거래량 특성 (있는 경우)
        if 'Volume' in data.columns:
            volume = data['Volume']
            volume_ma_5 = volume.rolling(5, min_periods=3).mean()
            features['volume_ma_5'] = volume_ma_5
            volume_ratio = volume / volume_ma_5
            if isinstance(volume_ratio, pd.Series):
                features['volume_ratio'] = volume_ratio
            else:
                features['volume_ratio'] = volume_ratio.iloc[:, 0] if hasattr(volume_ratio, 'iloc') else volume_ratio

        # 6. 외부 지표 (안전하게 추가)
        if 'treasury_10y' in data.columns and data['treasury_10y'].notna().sum() > 100:
            treasury = data['treasury_10y']
            features['treasury_level'] = treasury
            features['vix_treasury_spread'] = vix - treasury

        # 7. 고급 변동성 추정량 (간단 버전)
        if 'High' in data.columns and 'Low' in data.columns:
            high = data['High']
            low = data['Low']
            high_valid = int(high.notna().sum())
            low_valid = int(low.notna().sum())
            if high_valid > 100 and low_valid > 100:
                gk_vol = np.log(high / low) ** 2
                features['gk_vol_simple'] = gk_vol.rolling(5, min_periods=3).mean()

        print(f"✅ 안정적인 특성 생성 완료: {len(features.columns)}개")
        return features

    def create_robust_targets(self, data):
        """강건한 타겟 변수 설계"""
        print("🎯 강건한 타겟 변수 생성 중...")

        targets = pd.DataFrame(index=data.index)
        returns = data['returns']

        # 간단하고 안정적인 타겟들만
        for horizon in [1, 3, 5]:
            vol_values = []
            for i in range(len(returns)):
                if i + horizon < len(returns):
                    future_returns = returns.iloc[i+1:i+1+horizon]
                    if len(future_returns) >= horizon:  # 충분한 데이터 확인
                        vol_values.append(future_returns.std() * np.sqrt(252))
                    else:
                        vol_values.append(np.nan)
                else:
                    vol_values.append(np.nan)
            targets[f'target_vol_{horizon}d'] = vol_values

        # 로그 변환 (안전하게)
        for col in targets.columns:
            if 'target_vol' in col:
                log_values = np.log(targets[col] + 1e-8)
                log_values = log_values.replace([np.inf, -np.inf], np.nan)
                targets[f'log_{col}'] = log_values

        print(f"✅ 강건한 타겟 생성 완료: {len(targets.columns)}개")
        return targets

    def smart_feature_selection(self, features, targets, target_name, max_features=15):
        """지능적 특성 선택 (NaN 고려)"""
        print(f"🎯 지능적 특성 선택: {target_name}")

        target_series = targets[target_name]

        # 1. 각 특성별 유효 데이터 비율 계산
        feature_stats = {}
        for col in features.columns:
            feature_series = features[col]

            # 특성과 타겟이 모두 유효한 샘플 수
            valid_mask = feature_series.notna() & target_series.notna()
            valid_count = valid_mask.sum()
            total_count = len(feature_series)
            valid_ratio = valid_count / total_count

            if valid_count > 100:  # 최소 100개 샘플 필요
                # 상관계수 계산
                correlation = feature_series[valid_mask].corr(target_series[valid_mask])
                if not np.isnan(correlation):
                    feature_stats[col] = {
                        'valid_count': valid_count,
                        'valid_ratio': valid_ratio,
                        'correlation': abs(correlation),
                        'score': abs(correlation) * valid_ratio  # 복합 점수
                    }

        if not feature_stats:
            print(f"❌ {target_name}: 유효한 특성이 없음")
            return None, None, 0

        # 2. 특성 선택 (복합 점수 기준)
        sorted_features = sorted(feature_stats.items(),
                               key=lambda x: x[1]['score'], reverse=True)

        print(f"특성 통계 (상위 10개):")
        for i, (feature, stats) in enumerate(sorted_features[:10]):
            print(f"  {i+1:2d}. {feature:25}: 상관={stats['correlation']:.3f}, "
                  f"유효={stats['valid_ratio']:.2f}, 점수={stats['score']:.3f}")

        # 최대 max_features개 선택
        selected_features = [f for f, _ in sorted_features[:max_features]]

        # 3. 최종 데이터셋 구성
        X = features[selected_features].copy()
        y = target_series.copy()

        # 모든 선택된 특성과 타겟이 유효한 샘플만 선택
        valid_mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        print(f"최종 데이터셋: {len(X_clean)} 샘플, {len(selected_features)} 특성")

        return X_clean, y_clean, len(X_clean)

    def train_focused_models(self, X, y, target_name):
        """집중된 모델 훈련"""
        print(f"🤖 집중된 모델 훈련: {target_name}")

        if len(X) < 100:
            print(f"❌ 데이터 부족: {len(X)} 샘플")
            return {}

        models = {}

        # 1. Ridge 회귀 (여러 정규화 강도)
        for alpha in [0.1, 1.0, 10.0]:
            models[f'Ridge_a{alpha}'] = Ridge(alpha=alpha)

        # 2. ElasticNet
        models['ElasticNet'] = ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=3000)

        # 3. 앙상블 모델들
        models['RandomForest'] = RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_split=10,
            random_state=42, n_jobs=-1)

        models['GradientBoosting'] = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42)

        # 4. 신경망 (적당한 크기)
        models['MLP'] = MLPRegressor(
            hidden_layer_sizes=(50, 25), max_iter=1000,
            random_state=42, early_stopping=True, validation_fraction=0.2)

        # 모델 훈련
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        trained_models = {}
        for name, model in models.items():
            try:
                model.fit(X_scaled, y)
                trained_models[name] = {'model': model, 'scaler': scaler}
                print(f"✅ {name} 훈련 완료")
            except Exception as e:
                print(f"❌ {name} 훈련 실패: {e}")

        return trained_models

    def walk_forward_validation(self, X, y, trained_models, target_name):
        """Walk-Forward 검증"""
        print(f"📊 Walk-Forward 검증: {target_name}")

        if len(X) < 200:
            print(f"❌ Walk-Forward 검증에 데이터 부족: {len(X)} 샘플")
            return {}

        # 검증 설정
        initial_window = min(504, len(X) // 3)  # 초기 훈련 윈도우
        refit_frequency = 63  # 재훈련 주기
        prediction_horizon = 21  # 예측 구간

        results = {}

        for model_name, model_info in trained_models.items():
            model = model_info['model']
            scaler = model_info['scaler']

            predictions = []
            actuals = []

            start_idx = initial_window
            while start_idx + prediction_horizon < len(X):
                # 훈련 데이터
                train_end = start_idx
                train_start = max(0, train_end - initial_window)

                X_train = X.iloc[train_start:train_end]
                y_train = y.iloc[train_start:train_end]

                # 테스트 데이터
                test_end = min(start_idx + prediction_horizon, len(X))
                X_test = X.iloc[start_idx:test_end]
                y_test = y.iloc[start_idx:test_end]

                if len(X_train) > 50 and len(X_test) > 0:
                    try:
                        # 재훈련
                        scaler_fold = StandardScaler()
                        X_train_scaled = scaler_fold.fit_transform(X_train)
                        X_test_scaled = scaler_fold.transform(X_test)

                        # 새로운 모델 인스턴스 생성
                        if 'Ridge' in model_name:
                            alpha = float(model_name.split('_a')[1])
                            model_fold = Ridge(alpha=alpha)
                        elif model_name == 'ElasticNet':
                            model_fold = ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=3000)
                        elif model_name == 'RandomForest':
                            model_fold = RandomForestRegressor(
                                n_estimators=100, max_depth=8, min_samples_split=10,
                                random_state=42, n_jobs=-1)
                        elif model_name == 'GradientBoosting':
                            model_fold = GradientBoostingRegressor(
                                n_estimators=100, max_depth=4, learning_rate=0.1,
                                random_state=42)
                        elif model_name == 'MLP':
                            model_fold = MLPRegressor(
                                hidden_layer_sizes=(50, 25), max_iter=1000,
                                random_state=42, early_stopping=True, validation_fraction=0.2)
                        else:
                            continue

                        model_fold.fit(X_train_scaled, y_train)
                        y_pred = model_fold.predict(X_test_scaled)

                        predictions.extend(y_pred)
                        actuals.extend(y_test.values)

                    except Exception as e:
                        print(f"⚠️ {model_name} Fold 실패: {e}")

                start_idx += refit_frequency

            if len(predictions) > 10:
                r2 = r2_score(actuals, predictions)
                mse = mean_squared_error(actuals, predictions)

                results[model_name] = {
                    'r2': r2,
                    'mse': mse,
                    'predictions_count': len(predictions)
                }

                print(f"  {model_name:15}: R² = {r2:6.3f}, MSE = {mse:6.3f}")

        return results

def main():
    """메인 실행 함수"""
    print("🚀 State-of-the-Art Volatility Model (Fixed) - R² 0.25+ 목표")
    print("="*80)
    print("데이터 처리 문제 해결 버전")
    print("="*80)

    predictor = SOTAVolatilityPredictorFixed()

    # 1. 데이터 로드
    data = predictor.load_comprehensive_data()

    # 2. 특성 생성
    features = predictor.create_stable_features(data)

    # 3. 타겟 생성
    targets = predictor.create_robust_targets(data)

    print(f"\n데이터 요약:")
    print(f"  전체 데이터: {len(data)} 관측치")
    print(f"  특성 수: {len(features.columns)}개")
    print(f"  타겟 수: {len(targets.columns)}개")

    # 4. 각 타겟에 대해 모델링
    target_priority = ['target_vol_5d', 'target_vol_3d', 'target_vol_1d', 'log_target_vol_5d']

    best_performance = {'target': None, 'model': None, 'r2': -999}
    all_results = {}

    for target_name in target_priority:
        if target_name not in targets.columns:
            continue

        print(f"\n" + "="*60)
        print(f"🎯 타겟: {target_name}")
        print("="*60)

        # 특성 선택
        X, y, sample_count = predictor.smart_feature_selection(features, targets, target_name)

        if sample_count < 100:
            print(f"❌ {target_name}: 데이터 부족 ({sample_count} 샘플)")
            continue

        # 모델 훈련
        trained_models = predictor.train_focused_models(X, y, target_name)

        if not trained_models:
            continue

        # Walk-Forward 검증
        validation_results = predictor.walk_forward_validation(X, y, trained_models, target_name)

        if validation_results:
            all_results[target_name] = validation_results

            # 최고 성능 추적
            for model_name, stats in validation_results.items():
                if stats['r2'] > best_performance['r2']:
                    best_performance = {
                        'target': target_name,
                        'model': model_name,
                        'r2': stats['r2'],
                        'mse': stats['mse'],
                        'sample_count': sample_count
                    }

    # 5. 최종 결과
    print(f"\n" + "="*80)
    print(f"🏆 최종 결과")
    print("="*80)

    if best_performance['target']:
        print(f"최고 성능:")
        print(f"  타겟: {best_performance['target']}")
        print(f"  모델: {best_performance['model']}")
        print(f"  R²: {best_performance['r2']:.4f}")
        print(f"  MSE: {best_performance['mse']:.4f}")
        print(f"  샘플 수: {best_performance['sample_count']}")

        target_r2 = 0.25
        if best_performance['r2'] >= target_r2:
            print(f"\n🎉 목표 달성! R² ≥ {target_r2}")
        else:
            print(f"\n📈 목표 미달성 (목표: R² ≥ {target_r2})")
            gap = target_r2 - best_performance['r2']
            print(f"   목표까지 갭: {gap:.4f}")
    else:
        print("❌ 모든 모델에서 유효한 성능을 얻지 못함")

    # 6. 결과 저장
    os.makedirs('results', exist_ok=True)

    fixed_results = {
        'version': 'SOTA_Volatility_Model_Fixed',
        'timestamp': datetime.now().isoformat(),
        'goal': 'R² ≥ 0.25',
        'best_performance': best_performance,
        'all_results': all_results,
        'data_summary': {
            'total_observations': len(data),
            'total_features': len(features.columns),
            'total_targets': len(targets.columns)
        }
    }

    with open('results/sota_volatility_model_fixed.json', 'w') as f:
        json.dump(fixed_results, f, indent=2, default=str)

    print(f"\n💾 결과 저장: results/sota_volatility_model_fixed.json")
    print("="*80)

if __name__ == "__main__":
    main()