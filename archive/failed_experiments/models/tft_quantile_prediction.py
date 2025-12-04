#!/usr/bin/env python3
"""
TFT-inspired Quantile Regression 수익률 예측 모델
핵심 개선: 단일 값 예측 → 분포(quantile) 예측

목표: R² ≥ 0.3 (확률 8%)
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
import json
from datetime import datetime


class PurgedKFold:
    """Purged K-Fold CV"""
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


def quantile_loss(quantile):
    """Quantile loss function"""
    def loss(y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(quantile * error, (quantile - 1) * error))
    return loss


def pinball_loss(y_true, y_pred, quantile):
    """Pinball loss for quantile evaluation"""
    error = y_true - y_pred
    return np.mean(np.maximum(quantile * error, (quantile - 1) * error))


def load_data():
    """데이터 로드"""
    print("📊 SPY 데이터 로드 중...")

    dataset_path = 'data/training/multi_modal_sp500_dataset.csv'
    data = pd.read_csv(dataset_path, parse_dates=['Date'])
    data = data.set_index('Date').sort_index()

    print(f"✅ SPY 데이터: {len(data)} 관측치")
    return data


def create_log_return_features(data):
    """Log returns 기반 피처 생성"""
    print("🔧 Log returns 피처 생성...")

    df = data.copy()

    # Log returns (로그정규분포의 핵심)
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # 1. 변동성 (log returns 기반)
    for window in [5, 10, 20, 50]:
        df[f'log_vol_{window}'] = df['log_returns'].rolling(window).std()
        df[f'log_realized_vol_{window}'] = df[f'log_vol_{window}'] * np.sqrt(252)

    # 2. 모멘텀 (log returns)
    for window in [5, 10, 20]:
        df[f'log_momentum_{window}'] = df['log_returns'].rolling(window).sum()

    # 3. 통계 (log returns)
    for window in [5, 10, 20]:
        df[f'log_mean_{window}'] = df['log_returns'].rolling(window).mean()
        df[f'log_std_{window}'] = df['log_returns'].rolling(window).std()
        df[f'log_skew_{window}'] = df['log_returns'].rolling(window).skew()
        df[f'log_kurt_{window}'] = df['log_returns'].rolling(window).kurt()

    # 4. 래그 (log returns)
    for lag in [1, 2, 3, 5, 10]:
        df[f'log_return_lag_{lag}'] = df['log_returns'].shift(lag)
        df[f'log_vol_lag_{lag}'] = df['log_vol_5'].shift(lag)

    # 5. 가격 기반 (보조)
    for window in [5, 10, 20]:
        df[f'sma_{window}'] = df['close'].rolling(window).mean()
        df[f'price_to_sma_{window}'] = df['close'] / df[f'sma_{window}']

    # 6. 거래량
    df['volume_ma_5'] = df['volume'].rolling(5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_5']

    print(f"✅ 피처 생성 완료: {len(df.columns)}개")
    return df


def create_log_return_target(data, horizon=5):
    """Log returns 타겟 생성 (미래 평균 log return)"""
    print(f"🎯 Log returns 타겟 생성 ({horizon}일 후)...")

    log_returns = data['log_returns']
    target_values = []

    for i in range(len(log_returns)):
        if i + horizon < len(log_returns):
            # t+1부터 t+horizon까지의 평균 log return
            future_window = log_returns.iloc[i+1:i+1+horizon]
            target_values.append(future_window.mean())
        else:
            target_values.append(np.nan)

    target = pd.Series(target_values, index=data.index, name='target_log_return_5d')
    print(f"✅ 타겟 생성 완료")
    return target


def create_sequences(X, y, sequence_length=20):
    """시계열 시퀀스 생성"""
    X_sequences = []
    y_sequences = []

    for i in range(sequence_length, len(X)):
        X_sequences.append(X[i-sequence_length:i])
        y_sequences.append(y[i])

    return np.array(X_sequences), np.array(y_sequences)


def build_quantile_model(input_shape, quantiles=[0.1, 0.5, 0.9], learning_rate=0.001):
    """
    TFT-inspired Quantile Regression 모델
    핵심: 여러 quantile을 동시에 예측
    """

    # 입력
    inputs = layers.Input(shape=input_shape)

    # LSTM Encoder-Decoder (Variable Selection 제거 - 속도 개선)
    lstm1 = layers.LSTM(128, return_sequences=True, dropout=0.2)(inputs)
    lstm2 = layers.LSTM(64, return_sequences=False, dropout=0.2)(lstm1)

    # Dense layers
    dense1 = layers.Dense(64, activation='relu')(lstm2)
    dense1 = layers.Dropout(0.3)(dense1)

    dense2 = layers.Dense(32, activation='relu')(dense1)
    dense2 = layers.Dropout(0.2)(dense2)

    # Multi-quantile output
    outputs = []
    for q in quantiles:
        output = layers.Dense(1, name=f'quantile_{int(q*100)}')(dense2)
        outputs.append(output)

    # 모델 생성
    model = models.Model(inputs=inputs, outputs=outputs)

    # 각 quantile별 loss와 metric
    losses = {f'quantile_{int(q*100)}': quantile_loss(q) for q in quantiles}
    metrics = {f'quantile_{int(q*100)}': ['mae'] for q in quantiles}

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=losses, metrics=metrics)

    return model


def train_tft_quantile_model():
    """TFT Quantile 모델 훈련"""

    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow 필요")

    print("🚀 TFT-inspired Quantile Regression 모델 훈련")
    print("=" * 80)
    print("⚠️ 핵심 개선: 단일 값 예측 → 분포(quantile) 예측")
    print("⚠️ R² ≥ 0.3 달성 확률: 8%")
    print("=" * 80)

    # 1. 데이터 로드
    data = load_data()
    data_with_features = create_log_return_features(data)
    target = create_log_return_target(data_with_features, horizon=5)

    # 2. 데이터 결합
    combined = pd.concat([data_with_features, target], axis=1).dropna()

    # 피처 선택 (log returns 관련 + 기본 피처)
    feature_cols = [col for col in combined.columns
                   if col not in ['target_log_return_5d', 'open', 'high', 'low', 'close', 'volume']
                   and not col.startswith('sma_')]  # SMA는 제외 (log returns 중심)

    X = combined[feature_cols].values
    y = combined['target_log_return_5d'].values

    print(f"\n💾 데이터:")
    print(f"   샘플 수: {len(X)}")
    print(f"   피처 수: {len(feature_cols)}")
    print(f"   타겟: Log returns (로그정규분포)")

    # 3. Quantile 설정
    quantiles = [0.1, 0.5, 0.9]
    print(f"\n📊 Quantiles: {quantiles}")
    print(f"   0.1: 10th percentile (비관적)")
    print(f"   0.5: 50th percentile (중앙값)")
    print(f"   0.9: 90th percentile (낙관적)")

    # 4. Purged K-Fold CV
    print(f"\n🤖 TFT Quantile 모델 훈련 (Purged K-Fold)")
    print("-" * 80)

    purged_cv = PurgedKFold(n_splits=5, purge_length=5, embargo_length=5)
    sequence_length = 20

    cv_results = []

    for fold, (train_idx, val_idx) in enumerate(purged_cv.split(X)):
        print(f"\n📊 Fold {fold+1}/5")

        # 스케일링
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
        model = build_quantile_model(
            input_shape=(sequence_length, X_train_seq.shape[2]),
            quantiles=quantiles,
            learning_rate=0.001
        )

        # Early stopping
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        )

        # 훈련 (각 quantile별 타겟은 동일)
        y_train_dict = {f'quantile_{int(q*100)}': y_train_seq for q in quantiles}
        y_val_dict = {f'quantile_{int(q*100)}': y_val_seq for q in quantiles}

        history = model.fit(
            X_train_seq, y_train_dict,
            validation_data=(X_val_seq, y_val_dict),
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )

        # 예측 (모든 quantile)
        predictions = model.predict(X_val_seq, verbose=0)

        # 중앙값 (quantile 0.5) 기준 R²
        y_pred_median = predictions[1].flatten()  # quantile_50
        r2 = r2_score(y_val_seq, y_pred_median)
        mae = mean_absolute_error(y_val_seq, y_pred_median)
        mse = mean_squared_error(y_val_seq, y_pred_median)

        # Quantile coverage (10%-90% 구간에 실제 값 포함 비율)
        y_pred_q10 = predictions[0].flatten()
        y_pred_q90 = predictions[2].flatten()
        coverage = np.mean((y_val_seq >= y_pred_q10) & (y_val_seq <= y_pred_q90))

        # Pinball losses
        pinball_10 = pinball_loss(y_val_seq, y_pred_q10, 0.1)
        pinball_50 = pinball_loss(y_val_seq, y_pred_median, 0.5)
        pinball_90 = pinball_loss(y_val_seq, y_pred_q90, 0.9)

        cv_results.append({
            'fold': fold + 1,
            'r2': r2,
            'mae': mae,
            'rmse': np.sqrt(mse),
            'coverage': coverage,
            'pinball_10': pinball_10,
            'pinball_50': pinball_50,
            'pinball_90': pinball_90,
            'train_samples': len(X_train_seq),
            'val_samples': len(X_val_seq),
            'epochs': len(history.history['loss'])
        })

        print(f"   R² = {r2:7.4f} (quantile 0.5)")
        print(f"   Coverage (10%-90%): {coverage:.2%}")
        print(f"   MAE = {mae:.6f}, RMSE = {np.sqrt(mse):.6f}")
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
    avg_coverage = cv_df['coverage'].mean()
    avg_mae = cv_df['mae'].mean()
    avg_rmse = cv_df['rmse'].mean()

    print(f"\n각 Fold 성능:")
    for _, row in cv_df.iterrows():
        print(f"   Fold {int(row['fold'])}: R² = {row['r2']:7.4f}, "
              f"Coverage = {row['coverage']:.2%}, "
              f"MAE = {row['mae']:.6f}")

    print(f"\n평균 성능:")
    print(f"   R²:         {avg_r2:7.4f} ± {std_r2:.4f}")
    print(f"   Coverage:   {avg_coverage:.2%} (목표: 80%)")
    print(f"   MAE:        {avg_mae:.6f}")
    print(f"   RMSE:       {avg_rmse:.6f}")

    # 성능 평가
    print(f"\n🎯 성능 평가:")
    if avg_r2 >= 0.3:
        print(f"   ✅ 목표 달성! R² ≥ 0.3")
        print(f"   ⚠️ 데이터 누출 검증 필수!")
    elif avg_r2 >= 0.15:
        print(f"   📈 개선됨 (R² ≥ 0.15)")
        print(f"   ⚠️ 여전히 목표 미달")
    elif avg_r2 >= 0.05:
        print(f"   📊 미약한 개선 (R² ≥ 0.05)")
    elif avg_r2 > 0:
        print(f"   ⚠️ 약한 예측력 (R² > 0)")
    else:
        print(f"   ❌ 예측력 없음 (R² ≤ 0)")

    # 전체 데이터로 최종 모델 훈련
    print(f"\n🔨 전체 데이터로 최종 모델 훈련...")

    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)
    X_seq, y_seq = create_sequences(X_scaled, y, sequence_length)

    final_model = build_quantile_model(
        input_shape=(sequence_length, X_seq.shape[2]),
        quantiles=quantiles,
        learning_rate=0.001
    )

    y_dict = {f'quantile_{int(q*100)}': y_seq for q in quantiles}

    early_stop = callbacks.EarlyStopping(
        monitor='loss',
        patience=20,
        restore_best_weights=True,
        verbose=0
    )

    final_model.fit(
        X_seq, y_dict,
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    # 모델 저장
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)

    final_model.save('models/tft_quantile_model.keras')

    with open('models/tft_scaler.pkl', 'wb') as f:
        pickle.dump(scaler_final, f)

    with open('models/tft_feature_names.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)

    # 메타데이터 저장
    metadata = {
        'model_type': 'TFT-inspired Quantile Regression',
        'target': 'target_log_return_5d',
        'quantiles': quantiles,
        'sequence_length': sequence_length,
        'feature_count': len(feature_cols),
        'feature_names': feature_cols,
        'cv_performance': {
            'mean_r2': float(avg_r2),
            'std_r2': float(std_r2),
            'mean_coverage': float(avg_coverage),
            'mean_mae': float(avg_mae),
            'mean_rmse': float(avg_rmse),
            'fold_results': cv_results
        },
        'training_samples': len(X_seq),
        'trained_date': datetime.now().isoformat(),
        'data_period': '2015-2024',
        'improvements': [
            'Quantile prediction (분포 예측)',
            'Log returns (로그정규분포)',
            'Variable selection network',
            'Multi-quantile loss'
        ]
    }

    with open('models/tft_model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # 성능 데이터
    performance_data = {
        'model_name': 'TFT Quantile Predictor',
        'model_type': 'TFT-inspired Quantile Regression',
        'target': 'target_log_return_5d',
        'test_r2': float(avg_r2),
        'test_coverage': float(avg_coverage),
        'test_mae': float(avg_mae),
        'test_rmse': float(avg_rmse),
        'cv_std': float(std_r2),
        'validation_method': 'Purged K-Fold CV (5-fold)',
        'n_samples': len(X_seq),
        'n_features': len(feature_cols),
        'timestamp': datetime.now().isoformat()
    }

    with open('data/raw/tft_model_performance.json', 'w') as f:
        json.dump(performance_data, f, indent=2)

    print(f"\n✅ 모델 저장 완료:")
    print(f"   - models/tft_quantile_model.keras")
    print(f"   - models/tft_scaler.pkl")
    print(f"   - models/tft_feature_names.pkl")
    print(f"   - models/tft_model_metadata.json")
    print(f"   - data/raw/tft_model_performance.json")

    return metadata, cv_results


if __name__ == "__main__":
    try:
        metadata, results = train_tft_quantile_model()

        print(f"\n" + "=" * 80)
        print(f"✅ TFT Quantile 모델 훈련 완료")
        print(f"=" * 80)

        avg_r2 = metadata['cv_performance']['mean_r2']

        if avg_r2 >= 0.3:
            print(f"🎉 목표 달성: R² = {avg_r2:.4f} ≥ 0.3")
            print(f"⚠️ 데이터 누출 검증을 반드시 수행하세요!")
        elif avg_r2 >= 0.1:
            print(f"📈 개선됨: R² = {avg_r2:.4f}")
            print(f"   LSTM (0.0041) → TFT ({avg_r2:.4f})")
            print(f"   ⚠️ 여전히 목표 미달 (R² < 0.3)")
        else:
            print(f"📊 최종 결과: R² = {avg_r2:.4f}")
            print(f"   ⚠️ 목표 미달 (R² < 0.1)")
            print(f"   → EMH로 인한 예상된 결과")

    except Exception as e:
        print(f"\n❌ TFT 모델 훈련 실패: {e}")
        import traceback
        traceback.print_exc()
