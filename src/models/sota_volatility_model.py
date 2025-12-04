#!/usr/bin/env python3
"""
State-of-the-Art Volatility Model - R² 0.25+ 목표
최신 학술 연구 기반 고성능 변동성 예측 모델
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

class SOTAVolatilityPredictor:
    """State-of-the-Art 변동성 예측기"""

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

        # 추가 지표들
        try:
            # 10년 국채
            tnx = yf.download('^TNX', start='2010-01-01', end='2024-12-31', progress=False)
            spy['treasury_10y'] = tnx['Close'].reindex(spy.index, method='ffill')

            # 달러 지수 (DXY 대신 UUP 사용)
            dxy = yf.download('UUP', start='2010-01-01', end='2024-12-31', progress=False)
            spy['dollar_index'] = dxy['Close'].reindex(spy.index, method='ffill')

            # 금 (GLD)
            gold = yf.download('GLD', start='2010-01-01', end='2024-12-31', progress=False)
            spy['gold'] = gold['Close'].reindex(spy.index, method='ffill')

            # 원유 (USO)
            oil = yf.download('USO', start='2010-01-01', end='2024-12-31', progress=False)
            spy['oil'] = oil['Close'].reindex(spy.index, method='ffill')

        except Exception as e:
            print(f"⚠️ 일부 지표 로드 실패: {e}")

        spy = spy.dropna()
        print(f"✅ 포괄적 데이터 로드 완료: {len(spy)} 관측치")
        return spy

    def create_realized_volatility_features(self, data):
        """Realized Volatility 기반 고급 특성 (HAR-RV 모델 기반)"""
        print("🔧 Realized Volatility 특성 생성 중...")

        features = pd.DataFrame(index=data.index)
        returns = data['returns']
        prices = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']

        # 1. Realized Volatility (여러 윈도우)
        for window in [1, 5, 22]:  # 일, 주, 월
            rv = returns.rolling(window).std() * np.sqrt(252)  # 연율화
            features[f'rv_{window}'] = rv

        # 2. HAR-RV 모델 특성들
        features['rv_daily'] = features['rv_1']
        features['rv_weekly'] = features['rv_5']
        features['rv_monthly'] = features['rv_22']

        # 로그 변환 (HAR-RV 모델에서 표준)
        for col in ['rv_daily', 'rv_weekly', 'rv_monthly']:
            features[f'log_{col}'] = np.log(features[col] + 1e-8)

        # 3. Garman-Klass 추정량 (고빈도 프록시)
        gk_vol = np.log(high / low) ** 2
        for window in [1, 5, 22]:
            features[f'gk_vol_{window}'] = gk_vol.rolling(window).mean()

        # 4. Parkinson 추정량
        parkinson_vol = np.log(high / low) ** 2 / (4 * np.log(2))
        for window in [1, 5, 22]:
            features[f'parkinson_vol_{window}'] = parkinson_vol.rolling(window).mean()

        # 5. Rogers-Satchell 추정량
        rs_vol = np.log(high / prices) * np.log(high / prices.shift(1)) + \
                 np.log(low / prices) * np.log(low / prices.shift(1))
        for window in [1, 5, 22]:
            features[f'rs_vol_{window}'] = rs_vol.rolling(window).mean()

        print(f"✅ Realized Volatility 특성 생성 완료: {len(features.columns)}개")
        return features

    def create_regime_switching_features(self, data):
        """시장 체제 변화 특성"""
        print("🔧 Regime-Switching 특성 생성 중...")

        features = pd.DataFrame(index=data.index)
        returns = data['returns']
        vix = data['vix']

        # 1. 변동성 체제 분류
        vol_rolling = returns.rolling(22).std()
        vol_percentile = vol_rolling.rolling(252).rank(pct=True)

        features['vol_regime_low'] = (vol_percentile < 0.3).astype(int)
        features['vol_regime_high'] = (vol_percentile > 0.7).astype(int)
        features['vol_regime_normal'] = ((vol_percentile >= 0.3) & (vol_percentile <= 0.7)).astype(int)

        # 2. VIX 체제 분류
        vix_percentile = vix.rolling(252).rank(pct=True)
        features['vix_regime_low'] = (vix_percentile < 0.3).astype(int)
        features['vix_regime_high'] = (vix_percentile > 0.7).astype(int)

        # 3. 트렌드 체제
        price_ma_short = data['Close'].rolling(20).mean()
        price_ma_long = data['Close'].rolling(50).mean()
        features['trend_up'] = (price_ma_short > price_ma_long).astype(int)
        features['trend_down'] = (price_ma_short < price_ma_long).astype(int)

        # 4. 시장 스트레스 지표
        if 'vix' in data.columns and 'treasury_10y' in data.columns:
            stress_indicator = data['vix'] / data['treasury_10y']
            stress_percentile = stress_indicator.rolling(252).rank(pct=True)
            features['stress_regime'] = (stress_percentile > 0.8).astype(int)

        # 5. 체제 지속성 (상태 변화 감지)
        for col in ['vol_regime_high', 'vix_regime_high', 'trend_up']:
            if col in features.columns:
                features[f'{col}_persistence'] = features[col].rolling(5).sum() / 5

        print(f"✅ Regime-Switching 특성 생성 완료: {len(features.columns)}개")
        return features

    def create_sentiment_proxy_features(self, data):
        """감정 지표 프록시 특성 (VIX 기반)"""
        print("🔧 감정 지표 프록시 특성 생성 중...")

        features = pd.DataFrame(index=data.index)
        vix = data['vix']
        returns = data['returns']

        # 1. VIX 구조 특성
        features['vix_level'] = vix / 100
        for window in [5, 10, 22]:
            features[f'vix_ma_{window}'] = vix.rolling(window).mean() / 100

        # 2. VIX 기간 구조 (프록시)
        features['vix_term_structure'] = features['vix_level'] / features['vix_ma_22']

        # 3. VIX의 변동성 (공포의 변동성)
        vix_vol = vix.pct_change().rolling(22).std()
        features['vix_vol'] = vix_vol

        # 4. VIX-실현변동성 스프레드
        realized_vol = returns.rolling(22).std() * np.sqrt(252) / 100
        features['vix_rv_spread'] = features['vix_level'] - realized_vol

        # 5. 시장 스트레스 합성 지표
        if 'treasury_10y' in data.columns:
            features['financial_stress'] = (vix / data['treasury_10y']).rolling(5).mean()

        # 6. 감정 모멘텀
        features['vix_momentum'] = vix.pct_change().rolling(5).mean()

        print(f"✅ 감정 지표 프록시 특성 생성 완료: {len(features.columns)}개")
        return features

    def create_cross_asset_features(self, data):
        """Cross-Asset 특성"""
        print("🔧 Cross-Asset 특성 생성 중...")

        features = pd.DataFrame(index=data.index)

        # 1. 자산 간 상관관계
        assets = ['Close', 'vix', 'treasury_10y', 'dollar_index', 'gold', 'oil']
        available_assets = [asset for asset in assets if asset in data.columns]

        # 2. 롤링 상관관계
        for i, asset1 in enumerate(available_assets):
            for asset2 in available_assets[i+1:]:
                corr = data[asset1].pct_change().rolling(22).corr(data[asset2].pct_change())
                features[f'corr_{asset1}_{asset2}'] = corr

        # 3. 안전자산 스프레드
        if 'gold' in data.columns and 'Close' in data.columns:
            gold_equity_ratio = data['gold'] / data['Close']
            flight_to_quality = gold_equity_ratio.pct_change().rolling(5).mean()
            if isinstance(flight_to_quality, pd.Series):
                features['flight_to_quality'] = flight_to_quality
            else:
                features['flight_to_quality'] = flight_to_quality.iloc[:, 0] if not flight_to_quality.empty else np.nan

        # 4. 달러 강세 효과
        if 'dollar_index' in data.columns:
            dollar_strength = data['dollar_index'].pct_change().rolling(10).mean()
            if isinstance(dollar_strength, pd.Series):
                features['dollar_strength'] = dollar_strength
            else:
                features['dollar_strength'] = dollar_strength.iloc[:, 0] if not dollar_strength.empty else np.nan

        print(f"✅ Cross-Asset 특성 생성 완료: {len(features.columns)}개")
        return features

    def create_enhanced_targets(self, data):
        """향상된 타겟 변수 설계"""
        print("🎯 향상된 타겟 변수 생성 중...")

        targets = pd.DataFrame(index=data.index)
        returns = data['returns']
        high = data['High']
        low = data['Low']
        prices = data['Close']

        # 1. 기본 미래 변동성 (1, 3, 5일)
        for horizon in [1, 3, 5]:
            vol_values = []
            for i in range(len(returns)):
                if i + horizon < len(returns):
                    future_returns = returns.iloc[i+1:i+1+horizon]
                    vol_values.append(future_returns.std() * np.sqrt(252))  # 연율화
                else:
                    vol_values.append(np.nan)
            targets[f'target_rv_{horizon}d'] = vol_values

        # 2. Garman-Klass 기반 미래 변동성
        for horizon in [1, 3, 5]:
            gk_values = []
            for i in range(len(returns)):
                if i + horizon < len(returns):
                    future_high = high.iloc[i+1:i+1+horizon]
                    future_low = low.iloc[i+1:i+1+horizon]
                    future_close = prices.iloc[i+1:i+1+horizon]
                    future_gk = np.mean(np.log(future_high / future_low) ** 2)
                    gk_values.append(future_gk)
                else:
                    gk_values.append(np.nan)
            targets[f'target_gk_{horizon}d'] = gk_values

        # 3. 로그 변환된 타겟 (HAR-RV 스타일)
        for col in targets.columns:
            if 'target_rv' in col:
                targets[f'log_{col}'] = np.log(targets[col] + 1e-8)

        print(f"✅ 향상된 타겟 생성 완료: {len(targets.columns)}개")
        return targets

    def create_lstm_model(self, input_shape):
        """LSTM 모델 생성 (TensorFlow 사용 가능 시)"""
        if not TENSORFLOW_AVAILABLE:
            return None

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(1)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        return model

    def prepare_lstm_data(self, X, y, sequence_length=22):
        """LSTM용 시계열 데이터 준비"""
        X_seq = []
        y_seq = []

        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq)

    def train_sota_models(self, X, y, target_name):
        """SOTA 모델들 훈련"""
        print(f"🤖 SOTA 모델 훈련 중: {target_name}")

        models = {}

        # 1. 강화된 선형 모델들
        models['HAR_Ridge'] = Ridge(alpha=1.0)
        models['HAR_ElasticNet'] = ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=3000)

        # 2. 앙상블 모델들
        models['RF_Enhanced'] = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_split=20,
            min_samples_leaf=10, random_state=42
        )

        models['GBM_Enhanced'] = GradientBoostingRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            min_samples_split=20, min_samples_leaf=10, random_state=42
        )

        # 3. 신경망
        models['MLP_Enhanced'] = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25), activation='relu',
            alpha=0.01, learning_rate='adaptive', max_iter=1000,
            random_state=42
        )

        # 데이터 준비
        combined_data = pd.concat([X, y], axis=1).dropna()
        X_clean = combined_data[X.columns]
        y_clean = combined_data[y.name]

        print(f"훈련 데이터: {len(X_clean)} 샘플, {len(X_clean.columns)} 특성")

        # 스케일링
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_clean)

        # 모델 훈련
        trained_models = {}
        for name, model in models.items():
            try:
                print(f"  훈련 중: {name}")
                model.fit(X_scaled, y_clean)
                trained_models[name] = model

                # 간단한 성능 체크
                y_pred = model.predict(X_scaled)
                train_r2 = r2_score(y_clean, y_pred)
                print(f"    훈련 R²: {train_r2:.4f}")

            except Exception as e:
                print(f"    {name} 훈련 실패: {e}")

        # 4. LSTM 모델 (TensorFlow 사용 가능 시)
        if TENSORFLOW_AVAILABLE and len(X_clean) > 100:
            try:
                print("  훈련 중: LSTM_Enhanced")

                # LSTM 데이터 준비
                X_lstm, y_lstm = self.prepare_lstm_data(X_scaled, y_clean.values)

                if len(X_lstm) > 50:
                    # 훈련/검증 분할
                    split_idx = int(len(X_lstm) * 0.8)
                    X_train_lstm = X_lstm[:split_idx]
                    y_train_lstm = y_lstm[:split_idx]
                    X_val_lstm = X_lstm[split_idx:]
                    y_val_lstm = y_lstm[split_idx:]

                    # LSTM 모델 생성
                    lstm_model = self.create_lstm_model((X_lstm.shape[1], X_lstm.shape[2]))

                    # 조기 중단
                    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

                    # 훈련
                    history = lstm_model.fit(
                        X_train_lstm, y_train_lstm,
                        epochs=50,
                        batch_size=32,
                        validation_data=(X_val_lstm, y_val_lstm),
                        callbacks=[early_stopping],
                        verbose=0
                    )

                    trained_models['LSTM_Enhanced'] = lstm_model
                    print(f"    LSTM 훈련 완료: {len(history.history['loss'])} epochs")

            except Exception as e:
                print(f"    LSTM 훈련 실패: {e}")

        self.models[target_name] = trained_models
        self.scalers[target_name] = scaler

        print(f"✅ {len(trained_models)}개 모델 훈련 완료")
        return trained_models

    def comprehensive_walkforward_validation(self, X, y, target_name):
        """포괄적 Walk-Forward 검증"""
        print(f"🚀 포괄적 Walk-Forward 검증: {target_name}")

        if target_name not in self.models:
            print(f"❌ {target_name}에 대한 훈련된 모델 없음")
            return {}

        models = self.models[target_name]
        scaler = self.scalers[target_name]

        combined_data = pd.concat([X, y], axis=1).dropna()
        X_clean = combined_data[X.columns]
        y_clean = combined_data[y.name]

        # Walk-Forward 설정 (더 엄격하게)
        initial_window = 1000  # 4년
        step_size = 60         # 3개월
        test_window = 20       # 1개월

        results = {}
        for model_name in models.keys():
            results[model_name] = []

        current_start = 0
        fold = 0

        while current_start + initial_window + test_window < len(X_clean):
            fold += 1

            train_end = current_start + initial_window
            test_start = train_end
            test_end = min(test_start + test_window, len(X_clean))

            X_train = X_clean.iloc[current_start:train_end]
            y_train = y_clean.iloc[current_start:train_end]
            X_test = X_clean.iloc[test_start:test_end]
            y_test = y_clean.iloc[test_start:test_end]

            print(f"  Fold {fold}: 훈련 {len(X_train)}, 테스트 {len(X_test)} ({X_test.index[0].strftime('%Y-%m')})")

            # 각 모델 테스트
            for model_name, model in models.items():
                try:
                    if model_name == 'LSTM_Enhanced' and TENSORFLOW_AVAILABLE:
                        # LSTM 별도 처리
                        scaler_fold = RobustScaler()
                        X_train_scaled = scaler_fold.fit_transform(X_train)
                        X_test_scaled = scaler_fold.transform(X_test)

                        X_train_lstm, y_train_lstm = self.prepare_lstm_data(X_train_scaled, y_train.values)
                        X_test_lstm, _ = self.prepare_lstm_data(X_test_scaled, y_test.values)

                        if len(X_train_lstm) > 30 and len(X_test_lstm) > 0:
                            # LSTM 재훈련
                            lstm_fold = self.create_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))
                            lstm_fold.fit(X_train_lstm, y_train_lstm, epochs=30, batch_size=16, verbose=0)

                            y_pred_lstm = lstm_fold.predict(X_test_lstm, verbose=0)
                            y_pred = y_pred_lstm.flatten()
                            y_test_adj = y_test.iloc[22:22+len(y_pred)]  # 시퀀스 길이만큼 조정
                        else:
                            continue

                    else:
                        # 전통적인 모델들
                        scaler_fold = RobustScaler()
                        X_train_scaled = scaler_fold.fit_transform(X_train)
                        X_test_scaled = scaler_fold.transform(X_test)

                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        y_test_adj = y_test

                    # 성능 계산
                    r2 = r2_score(y_test_adj, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test_adj, y_pred))
                    mae = mean_absolute_error(y_test_adj, y_pred)

                    results[model_name].append({
                        'fold': fold,
                        'r2': r2,
                        'rmse': rmse,
                        'mae': mae,
                        'test_period': f"{X_test.index[0].strftime('%Y-%m')} ~ {X_test.index[-1].strftime('%Y-%m')}"
                    })

                    print(f"    {model_name:20}: R²={r2:7.4f}, RMSE={rmse:.5f}")

                except Exception as e:
                    print(f"    {model_name:20}: 오류 - {e}")
                    results[model_name].append({
                        'fold': fold,
                        'r2': -999,
                        'rmse': 999,
                        'mae': 999,
                        'error': str(e)
                    })

            current_start += step_size

        # 결과 요약
        summary = {}
        print(f"\n📊 {target_name} Walk-Forward 결과:")
        print(f"{'Model':<20} {'Mean R²':<10} {'Std R²':<10} {'Max R²':<10} {'Valid'}")
        print("-" * 70)

        for model_name, fold_results in results.items():
            valid_r2s = [r['r2'] for r in fold_results if r['r2'] != -999]

            if valid_r2s:
                mean_r2 = np.mean(valid_r2s)
                std_r2 = np.std(valid_r2s)
                max_r2 = np.max(valid_r2s)
                valid_count = len(valid_r2s)

                summary[model_name] = {
                    'mean_r2': mean_r2,
                    'std_r2': std_r2,
                    'max_r2': max_r2,
                    'valid_folds': valid_count,
                    'total_folds': len(fold_results)
                }

                print(f"{model_name:<20} {mean_r2:<10.4f} {std_r2:<10.4f} {max_r2:<10.4f} {valid_count}/{len(fold_results)}")
            else:
                summary[model_name] = {'mean_r2': -999}
                print(f"{model_name:<20} {'FAILED':<10}")

        self.performance_history[target_name] = summary
        return summary

def main():
    """메인 SOTA 모델 함수"""
    print("🚀 State-of-the-Art Volatility Model - R² 0.25+ 목표")
    print("=" * 80)
    print("최신 학술 연구 기반 고성능 변동성 예측 모델")
    print("=" * 80)

    predictor = SOTAVolatilityPredictor()

    # 1. 포괄적 데이터 로드
    data = predictor.load_comprehensive_data()

    # 2. 고급 특성 생성
    print("\n🔧 고급 특성 엔지니어링...")

    rv_features = predictor.create_realized_volatility_features(data)
    regime_features = predictor.create_regime_switching_features(data)
    sentiment_features = predictor.create_sentiment_proxy_features(data)
    cross_asset_features = predictor.create_cross_asset_features(data)

    # 모든 특성 결합
    all_features = pd.concat([
        rv_features,
        regime_features,
        sentiment_features,
        cross_asset_features
    ], axis=1)

    print(f"전체 특성 수: {len(all_features.columns)}")

    # 3. 향상된 타겟 생성
    targets = predictor.create_enhanced_targets(data)

    # 4. 핵심 타겟들에 대해 모델 훈련 및 검증
    key_targets = ['target_rv_1d', 'target_rv_3d', 'target_rv_5d', 'log_target_rv_5d']

    best_performance = {'target': None, 'model': None, 'r2': -999}

    for target_name in key_targets:
        if target_name not in targets.columns:
            continue

        print(f"\n" + "="*60)
        print(f"🎯 타겟: {target_name}")
        print("="*60)

        target_series = targets[target_name]

        # 특성 선별 (상관관계 기반)
        combined = pd.concat([all_features, target_series], axis=1).dropna()

        if len(combined) < 500:
            print(f"❌ {target_name}: 데이터 부족 ({len(combined)} 샘플)")
            continue

        correlations = combined[all_features.columns].corrwith(combined[target_name]).abs().sort_values(ascending=False)

        # 상위 30개 특성 선택 (더 많은 정보 활용)
        top_features = correlations.head(30).index
        selected_features = all_features[top_features]

        print(f"선택된 특성: {len(selected_features.columns)}개")
        print("상위 10개 특성:")
        for i, (feature, corr) in enumerate(correlations.head(10).items()):
            print(f"  {i+1:2d}. {feature:30}: {corr:.4f}")

        # 5. 모델 훈련
        trained_models = predictor.train_sota_models(selected_features, target_series, target_name)

        if not trained_models:
            continue

        # 6. Walk-Forward 검증
        summary = predictor.comprehensive_walkforward_validation(selected_features, target_series, target_name)

        # 최고 성능 추적
        if summary:
            for model_name, stats in summary.items():
                if stats.get('mean_r2', -999) > best_performance['r2']:
                    best_performance = {
                        'target': target_name,
                        'model': model_name,
                        'r2': stats['mean_r2'],
                        'std_r2': stats.get('std_r2', 0),
                        'max_r2': stats.get('max_r2', 0)
                    }

    # 7. 최종 결과 및 평가
    print(f"\n" + "="*80)
    print(f"🏆 최종 결과")
    print("="*80)

    if best_performance['target']:
        print(f"최고 성능:")
        print(f"  타겟: {best_performance['target']}")
        print(f"  모델: {best_performance['model']}")
        print(f"  평균 R²: {best_performance['r2']:.4f}")
        print(f"  표준편차: {best_performance['std_r2']:.4f}")
        print(f"  최고 R²: {best_performance['max_r2']:.4f}")

        # 목표 달성 여부
        target_r2 = 0.25
        if best_performance['r2'] >= target_r2:
            print(f"\n🎉 목표 달성! R² ≥ {target_r2}")
            improvement = (best_performance['r2'] - 0.0145) / 0.0145 * 100
            print(f"   이전 최고 대비 개선: +{improvement:.1f}%")
        else:
            print(f"\n📈 목표 미달성 (목표: R² ≥ {target_r2})")
            gap = target_r2 - best_performance['r2']
            print(f"   목표까지 갭: {gap:.4f}")

        # 벤치마크 비교
        print(f"\n📊 벤치마크 비교:")
        print(f"   우리 모델: R² = {best_performance['r2']:.4f}")
        print(f"   HAR-RV 모델: R² = 0.119")
        print(f"   하이브리드 HAR-PSO-ESN: R² = 0.635 (1일)")

        if best_performance['r2'] > 0.119:
            improvement_vs_har = (best_performance['r2'] - 0.119) / 0.119 * 100
            print(f"   HAR-RV 대비: +{improvement_vs_har:.1f}%")

    else:
        print("❌ 모든 모델에서 유효한 성능을 얻지 못함")

    # 8. 결과 저장
    os.makedirs('results', exist_ok=True)

    sota_results = {
        'version': 'SOTA_Volatility_Model',
        'timestamp': datetime.now().isoformat(),
        'goal': 'R² ≥ 0.25',
        'best_performance': best_performance,
        'all_performances': predictor.performance_history,
        'total_features_engineered': len(all_features.columns),
        'targets_tested': key_targets,
        'methodology': [
            'Realized Volatility (HAR-RV based)',
            'Regime-Switching Detection',
            'Sentiment Proxy Features',
            'Cross-Asset Features',
            'Enhanced Neural Networks',
            'LSTM with Walk-Forward'
        ]
    }

    with open('results/sota_volatility_model.json', 'w') as f:
        json.dump(sota_results, f, indent=2, default=str)

    print(f"\n💾 결과 저장: results/sota_volatility_model.json")
    print("="*80)

if __name__ == "__main__":
    main()