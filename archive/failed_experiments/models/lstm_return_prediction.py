#!/usr/bin/env python3
"""
LSTM 기반 수익률 예측 모델
목표: R² ≥ 0.3 (실패 가능성 높음 - EMH로 인한 이론적 한계)

경고:
- 수익률 예측은 효율적 시장 가설(EMH)로 인해 본질적으로 어려움
- 학술 연구에서도 out-of-sample R² 음수가 일반적
- 데이터 누출 없이 R² 0.3은 극히 어려움
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow 로그 억제

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow 필요: pip install tensorflow")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
import json
from datetime import datetime

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class PurgedKFold:
    """금융 시계열 특화 교차 검증"""
    def __init__(self, n_splits=5, purge_length=5, embargo_length=5):
        self.n_splits = n_splits
        self.purge_length = purge_length
        self.embargo_length = embargo_length

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        test_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            test_start = i * test_size
            test_end = (i + 1) * test_size if i < self.n_splits - 1 else n_samples
            test_indices = list(range(test_start, test_end))

            train_indices = []
            if test_start > self.purge_length:
                train_indices.extend(range(0, test_start - self.purge_length))
            if test_end + self.embargo_length < n_samples:
                train_indices.extend(range(test_end + self.embargo_length, n_samples))

            yield train_indices, test_indices


def get_spy_data():
    """SPY 데이터 로드"""
    print("📊 SPY 데이터 로드 중...")

    # 기존 데이터셋 사용 (yfinance 이슈 회피)
    dataset_path = 'data/training/multi_modal_sp500_dataset.csv'

    if os.path.exists(dataset_path):
        print(f"   기존 데이터셋 사용: {dataset_path}")
        full_data = pd.read_csv(dataset_path, parse_dates=['Date'])
        full_data = full_data.set_index('Date').sort_index()

        # OHLCV 데이터 추출
        df = pd.DataFrame({
            'open': full_data['open'],
            'high': full_data['high'],
            'low': full_data['low'],
            'close': full_data['close'],
            'volume': full_data['volume']
        })

        print(f"✅ SPY 데이터: {len(df)} 관측치")
        return df

    # fallback: yfinance 시도
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance 필요하지만 데이터셋도 없음")

    print("   yfinance로 데이터 수집 중...")
    data = yf.download("SPY", start="2015-01-01", end="2024-12-31", progress=False)

    if data.empty:
        raise ValueError("SPY 데이터 수집 실패")

    df = pd.DataFrame({
        'open': data['Open'],
        'high': data['High'],
        'low': data['Low'],
        'close': data['Close'],
        'volume': data['Volume']
    })

    print(f"✅ SPY 데이터: {len(df)} 관측치")
    return df


def create_advanced_features(data):
    """고급 시계열 피처 생성 (LSTM 입력용)"""
    print("🔧 고급 시계열 피처 생성...")

    df = data.copy()

    # 기본 수익률
    df['returns'] = np.log(df['close'] / df['close'].shift(1))

    # 1. 가격 기반 피처
    for window in [5, 10, 20, 50]:
        df[f'sma_{window}'] = df['close'].rolling(window).mean()
        df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        df[f'price_to_sma_{window}'] = df['close'] / df[f'sma_{window}']

    # 2. 변동성 피처 (시계열 특화)
    for window in [5, 10, 20]:
        df[f'volatility_{window}'] = df['returns'].rolling(window).std()
        df[f'volatility_ewm_{window}'] = df['returns'].ewm(span=window).std()

    # 3. 모멘텀 피처
    for window in [5, 10, 20]:
        df[f'momentum_{window}'] = df['returns'].rolling(window).sum()
        df[f'roc_{window}'] = (df['close'] - df['close'].shift(window)) / df['close'].shift(window)

    # 4. 통계적 피처
    for window in [5, 10, 20]:
        df[f'mean_return_{window}'] = df['returns'].rolling(window).mean()
        df[f'std_return_{window}'] = df['returns'].rolling(window).std()
        df[f'skew_{window}'] = df['returns'].rolling(window).skew()
        df[f'kurt_{window}'] = df['returns'].rolling(window).kurt()

    # 5. 래그 피처 (과거 정보만)
    for lag in [1, 2, 3, 5, 10]:
        df[f'return_lag_{lag}'] = df['returns'].shift(lag)
        df[f'vol_lag_{lag}'] = df[f'volatility_5'].shift(lag)

    # 6. 거래량 피처
    df['volume_ma_5'] = df['volume'].rolling(5).mean()
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']

    # 7. High-Low 피처
    df['high_low_ratio'] = df['high'] / df['low']
    df['high_low_range'] = (df['high'] - df['low']) / df['close']

    # 8. 시간적 교차 피처
    df['vol_5_to_vol_20'] = df['volatility_5'] / (df['volatility_20'] + 1e-8)
    df['mom_5_to_mom_20'] = df['momentum_5'] / (df['momentum_20'] + 1e-8)

    print(f"✅ 피처 생성 완료: {len(df.columns)}개")
    return df


def create_return_target(data, horizon=5):
    """수익률 타겟 생성 (완전한 시간적 분리)"""
    print(f"🎯 타겟 생성 ({horizon}일 후 평균 수익률)...")

    returns = data['returns']
    target_values = []

    for i in range(len(returns)):
        if i + horizon < len(returns):
            # t+1부터 t+horizon까지의 평균 수익률
            future_window = returns.iloc[i+1:i+1+horizon]
            target_values.append(future_window.mean())
        else:
            target_values.append(np.nan)

    target = pd.Series(target_values, index=data.index, name='target_return_5d')
    print(f"✅ 타겟 생성 완료")
    return target


def create_sequences(X, y, sequence_length=20):
    """시계열 시퀀스 생성 (LSTM 입력용)"""
    X_sequences = []
    y_sequences = []

    for i in range(sequence_length, len(X)):
        X_sequences.append(X[i-sequence_length:i])
        y_sequences.append(y[i])

    return np.array(X_sequences), np.array(y_sequences)


def build_lstm_model(input_shape, learning_rate=0.001):
    """Bidirectional LSTM + Attention 모델"""

    # 입력 레이어
    inputs = layers.Input(shape=input_shape)

    # Bidirectional LSTM layers
    lstm1 = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.2)
    )(inputs)

    lstm2 = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2)
    )(lstm1)

    # Attention mechanism (간단한 버전)
    attention = layers.Dense(1, activation='tanh')(lstm2)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(128)(attention)
    attention = layers.Permute([2, 1])(attention)

    # Apply attention
    lstm2_reshaped = layers.Reshape((input_shape[0], 128))(lstm2)
    merged = layers.Multiply()([lstm2_reshaped, attention])
    merged = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(merged)

    # Dense layers
    dense1 = layers.Dense(64, activation='relu')(merged)
    dense1 = layers.Dropout(0.3)(dense1)

    dense2 = layers.Dense(32, activation='relu')(dense1)
    dense2 = layers.Dropout(0.2)(dense2)

    # Output layer
    outputs = layers.Dense(1)(dense2)

    # 모델 컴파일
    model = models.Model(inputs=inputs, outputs=outputs)

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model


def train_lstm_model():
    """LSTM 모델 훈련 및 평가"""

    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow 필요")

    print("🚀 LSTM 수익률 예측 모델 훈련")
    print("=" * 80)
    print("⚠️ 경고: 수익률 예측은 EMH로 인해 본질적으로 어려움")
    print("⚠️ R² ≥ 0.3 달성 가능성은 매우 낮음")
    print("=" * 80)

    # 1. 데이터 로드 및 피처 생성
    data = get_spy_data()
    data_with_features = create_advanced_features(data)
    target = create_return_target(data_with_features, horizon=5)

    # 2. 데이터 결합
    combined = pd.concat([data_with_features, target], axis=1).dropna()

    # 피처 선택 (타겟과 기본 OHLCV 제외)
    feature_cols = [col for col in combined.columns
                   if col not in ['target_return_5d', 'open', 'high', 'low', 'close', 'volume']]

    X = combined[feature_cols].values
    y = combined['target_return_5d'].values

    print(f"\n💾 데이터:")
    print(f"   샘플 수: {len(X)}")
    print(f"   피처 수: {len(feature_cols)}")

    # 3. Purged K-Fold CV
    print(f"\n🤖 Bidirectional LSTM + Attention 모델 훈련")
    print(f"   시퀀스 길이: 20일")
    print(f"   아키텍처: BiLSTM(128) -> BiLSTM(64) -> Attention -> Dense")
    print("-" * 80)

    purged_cv = PurgedKFold(n_splits=5, purge_length=5, embargo_length=5)

    cv_results = []
    sequence_length = 20

    for fold, (train_idx, val_idx) in enumerate(purged_cv.split(X)):
        print(f"\n📊 Fold {fold+1}/5")

        # 스케일링 (train 기준)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X[train_idx])
        X_val_scaled = scaler.transform(X[val_idx])

        y_train = y[train_idx]
        y_val = y[val_idx]

        # 시퀀스 생성
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, sequence_length)
        X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, sequence_length)

        if len(X_train_seq) < 50 or len(X_val_seq) < 10:
            print(f"   ⚠️ 데이터 부족, fold 스킵")
            continue

        print(f"   훈련 시퀀스: {len(X_train_seq)}, 검증 시퀀스: {len(X_val_seq)}")

        # 모델 구축
        model = build_lstm_model(
            input_shape=(sequence_length, X_train_seq.shape[2]),
            learning_rate=0.001
        )

        # Early stopping
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )

        # 훈련
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )

        # 예측
        y_pred = model.predict(X_val_seq, verbose=0).flatten()

        # 평가
        r2 = r2_score(y_val_seq, y_pred)
        mae = mean_absolute_error(y_val_seq, y_pred)
        mse = mean_squared_error(y_val_seq, y_pred)
        rmse = np.sqrt(mse)

        cv_results.append({
            'fold': fold + 1,
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'train_samples': len(X_train_seq),
            'val_samples': len(X_val_seq),
            'epochs': len(history.history['loss'])
        })

        print(f"   R² = {r2:7.4f}, MAE = {mae:.6f}, RMSE = {rmse:.6f}")
        print(f"   훈련 epochs: {len(history.history['loss'])}")

    # 결과 요약
    print("\n" + "=" * 80)
    print("📊 Cross-Validation 결과")
    print("=" * 80)

    if not cv_results:
        print("❌ 모든 fold 실패")
        return None

    cv_df = pd.DataFrame(cv_results)

    avg_r2 = cv_df['r2'].mean()
    std_r2 = cv_df['r2'].std()
    avg_mae = cv_df['mae'].mean()
    avg_rmse = cv_df['rmse'].mean()

    print(f"\n각 Fold 성능:")
    for _, row in cv_df.iterrows():
        print(f"   Fold {int(row['fold'])}: R² = {row['r2']:7.4f}, "
              f"MAE = {row['mae']:.6f}, RMSE = {row['rmse']:.6f}")

    print(f"\n평균 성능:")
    print(f"   R²:   {avg_r2:7.4f} ± {std_r2:.4f}")
    print(f"   MAE:  {avg_mae:.6f}")
    print(f"   RMSE: {avg_rmse:.6f}")

    # 성능 평가
    print(f"\n🎯 성능 평가:")
    if avg_r2 >= 0.3:
        print(f"   ✅ 목표 달성! R² ≥ 0.3")
        print(f"   ⚠️ 데이터 누출 검증 필수!")
    elif avg_r2 >= 0.15:
        print(f"   📈 양호한 성능 (R² ≥ 0.15)")
        print(f"   ⚠️ 여전히 목표 미달")
    elif avg_r2 >= 0.05:
        print(f"   📊 미약한 예측력 (R² ≥ 0.05)")
    elif avg_r2 > 0:
        print(f"   ⚠️ 매우 약한 예측력 (R² > 0)")
    else:
        print(f"   ❌ 예측력 없음 (R² ≤ 0)")
        print(f"   → 랜덤 추측보다 못함")

    # 전체 데이터로 최종 모델 훈련 (저장용)
    print(f"\n🔨 전체 데이터로 최종 모델 훈련...")

    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)
    X_seq, y_seq = create_sequences(X_scaled, y, sequence_length)

    final_model = build_lstm_model(
        input_shape=(sequence_length, X_seq.shape[2]),
        learning_rate=0.001
    )

    early_stop = callbacks.EarlyStopping(
        monitor='loss',
        patience=15,
        restore_best_weights=True,
        verbose=0
    )

    final_model.fit(
        X_seq, y_seq,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    # 모델 저장
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)

    final_model.save('models/lstm_return_prediction.keras')

    with open('models/lstm_scaler.pkl', 'wb') as f:
        pickle.dump(scaler_final, f)

    with open('models/lstm_feature_names.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)

    # 메타데이터 저장
    metadata = {
        'model_type': 'Bidirectional LSTM + Attention',
        'target': 'target_return_5d',
        'sequence_length': sequence_length,
        'feature_count': len(feature_cols),
        'feature_names': feature_cols,
        'cv_performance': {
            'mean_r2': float(avg_r2),
            'std_r2': float(std_r2),
            'mean_mae': float(avg_mae),
            'mean_rmse': float(avg_rmse),
            'fold_results': cv_results
        },
        'architecture': {
            'lstm1': 'Bidirectional(128)',
            'lstm2': 'Bidirectional(64)',
            'attention': 'Dense attention mechanism',
            'dense': '[64, 32, 1]',
            'dropout': [0.2, 0.2, 0.3, 0.2]
        },
        'training_samples': len(X_seq),
        'trained_date': datetime.now().isoformat(),
        'data_period': '2015-01-01 to 2024-12-31',
        'warning': 'EMH로 인해 수익률 예측은 본질적으로 어려움. 데이터 누출 검증 필수.'
    }

    with open('models/lstm_model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # 성능 데이터 저장
    performance_data = {
        'model_name': 'LSTM Return Predictor',
        'model_type': 'Bidirectional LSTM + Attention',
        'target': 'target_return_5d',
        'test_r2': float(avg_r2),
        'test_mae': float(avg_mae),
        'test_rmse': float(avg_rmse),
        'cv_std': float(std_r2),
        'validation_method': 'Purged K-Fold CV (5-fold)',
        'n_samples': len(X_seq),
        'n_features': len(feature_cols),
        'timestamp': datetime.now().isoformat()
    }

    with open('data/raw/lstm_model_performance.json', 'w') as f:
        json.dump(performance_data, f, indent=2)

    print(f"\n✅ 모델 저장 완료:")
    print(f"   - models/lstm_return_prediction.keras")
    print(f"   - models/lstm_scaler.pkl")
    print(f"   - models/lstm_feature_names.pkl")
    print(f"   - models/lstm_model_metadata.json")
    print(f"   - data/raw/lstm_model_performance.json")

    return metadata, cv_results


if __name__ == "__main__":
    try:
        metadata, results = train_lstm_model()

        print(f"\n" + "=" * 80)
        print(f"✅ LSTM 모델 훈련 완료")
        print(f"=" * 80)

        avg_r2 = metadata['cv_performance']['mean_r2']

        if avg_r2 >= 0.3:
            print(f"🎉 목표 달성: R² = {avg_r2:.4f} ≥ 0.3")
            print(f"⚠️ 데이터 누출 검증을 반드시 수행하세요!")
        else:
            print(f"📊 최종 결과: R² = {avg_r2:.4f}")
            print(f"⚠️ 목표 미달 (R² < 0.3)")
            print(f"→ EMH로 인해 예상된 결과입니다.")

    except Exception as e:
        print(f"\n❌ LSTM 모델 훈련 실패: {e}")
        import traceback
        traceback.print_exc()
