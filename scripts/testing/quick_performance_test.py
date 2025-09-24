#!/usr/bin/env python3
"""
빠른 통합 모델 성능 테스트
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
import time
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("⚡ 빠른 통합 AI 주식 예측 성능 테스트")
print("=" * 60)
print(f"TensorFlow 버전: {tf.__version__}")
print(f"GPU: {'있음' if tf.config.list_physical_devices('GPU') else '없음'}")
print()

class QuickFeatureEngineering:
    """빠른 특성 공학"""
    
    def create_features(self, data):
        """핵심 특성만 빠르게 생성"""
        features = {}
        
        # 기본 수익률
        for period in [1, 5, 10, 20]:
            features[f'return_{period}'] = data['Close'].pct_change(period)
            features[f'volatility_{period}'] = data['Close'].rolling(period).std()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        features['macd'] = exp1 - exp2
        
        # 모멘텀
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = data['Close'] / data['Close'].shift(period) - 1
        
        # 볼린저 밴드 위치
        sma_20 = data['Close'].rolling(20).mean()
        std_20 = data['Close'].rolling(20).std()
        features['bollinger_position'] = (data['Close'] - sma_20) / std_20
        
        # 가격-거래량 상관관계
        price_changes = data['Close'].pct_change(5)
        volume_changes = data['Volume'].pct_change(5)
        features['price_vol_corr'] = price_changes.rolling(10).corr(volume_changes)
        
        # DataFrame으로 변환
        feature_df = pd.DataFrame(features, index=data.index)
        feature_df = feature_df.fillna(method='ffill').fillna(0)
        
        return feature_df

class SimpleLSTM:
    """간단한 LSTM 모델"""
    
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
    
    def prepare_sequences(self, X, y):
        """시퀀스 데이터 준비"""
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X.iloc[i:i+self.sequence_length].values)
            y_seq.append(y.iloc[i+self.sequence_length])
        return np.array(X_seq), np.array(y_seq)
    
    def fit(self, X, y):
        """모델 학습"""
        # 정규화
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns, index=X.index
        )
        
        # 시퀀스 준비
        X_seq, y_seq = self.prepare_sequences(X_scaled, y)
        
        if len(X_seq) < 50:  # 최소 데이터 요구
            print("   ⚠️ LSTM: 데이터 부족")
            return self
        
        # 모델 생성
        self.model = keras.Sequential([
            layers.LSTM(32, input_shape=(self.sequence_length, X_seq.shape[2])),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # 학습
        self.model.fit(
            X_seq, y_seq,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        return self
    
    def predict(self, X):
        """예측"""
        if self.model is None:
            return np.random.choice([0, 1], len(X))
        
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns, index=X.index
        )
        
        X_seq, _ = self.prepare_sequences(X_scaled, pd.Series(range(len(X))))
        
        if len(X_seq) == 0:
            return np.random.choice([0, 1], len(X))
        
        pred_proba = self.model.predict(X_seq, verbose=0)
        
        # 전체 길이에 맞춰 패딩
        full_pred = np.zeros(len(X))
        full_pred[self.sequence_length:] = pred_proba.flatten()
        full_pred[:self.sequence_length] = pred_proba[0] if len(pred_proba) > 0 else 0.5
        
        return (full_pred > 0.5).astype(int)

def run_quick_test():
    """빠른 성능 테스트"""
    print("📊 SPY 데이터 다운로드 중...")
    
    # 데이터 다운로드 (1년)
    ticker = yf.Ticker('SPY')
    data = ticker.history(period='1y')
    
    print(f"   ✅ 데이터: {len(data)}일 ({data.index[0].date()} ~ {data.index[-1].date()})")
    
    # 특성 생성
    feature_eng = QuickFeatureEngineering()
    features = feature_eng.create_features(data)
    
    # 타겟 생성
    target = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    # 마지막 행 제거
    features = features.iloc[:-1]
    target = target.iloc[:-1]
    
    print(f"   특성 수: {len(features.columns)}")
    print(f"   상승 비율: {target.mean():.1%}")
    print()
    
    # 데이터 분할 (시계열 순서 유지)
    split_idx = int(len(features) * 0.8)
    X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
    
    print(f"   훈련: {len(X_train)}일, 테스트: {len(X_test)}일")
    print()
    
    # 베이스라인
    baseline_random = 0.5000
    baseline_momentum = (y_test.shift(1).fillna(0) == y_test).mean()
    
    print("🎯 모델 성능 테스트")
    print("=" * 50)
    
    results = {}
    
    # 1. Random Forest
    print("1️⃣ Random Forest 학습...")
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)
        results['RandomForest'] = rf_acc
        print(f"   ✅ Random Forest: {rf_acc:.4f}")
    except Exception as e:
        results['RandomForest'] = baseline_random
        print(f"   ❌ Random Forest 실패: {str(e)[:30]}")
    
    # 2. AdaBoost
    print("2️⃣ AdaBoost 학습...")
    try:
        ada = AdaBoostClassifier(n_estimators=50, random_state=42)
        ada.fit(X_train, y_train)
        ada_pred = ada.predict(X_test)
        ada_acc = accuracy_score(y_test, ada_pred)
        results['AdaBoost'] = ada_acc
        print(f"   ✅ AdaBoost: {ada_acc:.4f}")
    except Exception as e:
        results['AdaBoost'] = baseline_random
        print(f"   ❌ AdaBoost 실패: {str(e)[:30]}")
    
    # 3. Logistic Regression
    print("3️⃣ Logistic Regression 학습...")
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train_scaled, y_train)
        lr_pred = lr.predict(X_test_scaled)
        lr_acc = accuracy_score(y_test, lr_pred)
        results['LogisticRegression'] = lr_acc
        print(f"   ✅ Logistic Regression: {lr_acc:.4f}")
    except Exception as e:
        results['LogisticRegression'] = baseline_random
        print(f"   ❌ Logistic Regression 실패: {str(e)[:30]}")
    
    # 4. Simple LSTM
    print("4️⃣ Simple LSTM 학습...")
    try:
        lstm = SimpleLSTM(sequence_length=15)
        lstm.fit(X_train, y_train)
        lstm_pred = lstm.predict(X_test)
        lstm_acc = accuracy_score(y_test, lstm_pred)
        results['SimpleLSTM'] = lstm_acc
        print(f"   ✅ Simple LSTM: {lstm_acc:.4f}")
    except Exception as e:
        results['SimpleLSTM'] = baseline_random
        print(f"   ❌ Simple LSTM 실패: {str(e)[:30]}")
    
    print()
    
    # 결과 분석
    print("🏆 최종 성능 분석")
    print("=" * 50)
    
    # 성능 순위
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    print("📊 모델 성능 순위:")
    for i, (model, acc) in enumerate(sorted_results, 1):
        print(f"   {i}. {model:20s}: {acc:.4f}")
    
    print(f"\n📈 베이스라인 비교:")
    print(f"   Random Walk (50%): {baseline_random:.4f}")
    print(f"   Momentum Strategy: {baseline_momentum:.4f}")
    
    if sorted_results:
        best_model, best_acc = sorted_results[0]
        improvement = best_acc - max(baseline_random, baseline_momentum)
        improvement_pct = (improvement / max(baseline_random, baseline_momentum)) * 100
        
        print(f"   최고 모델 ({best_model}): {best_acc:.4f}")
        print(f"   개선도: {improvement:+.4f} ({improvement_pct:+.1f}%)")
        
        # 목표 달성 여부
        target_min, target_max = 0.60, 0.65
        if best_acc >= target_min:
            if best_acc >= target_max:
                print(f"   🎯 목표 달성! (목표: {target_min:.1%}-{target_max:.1%})")
            else:
                print(f"   🎯 최소 목표 달성! (목표: {target_min:.1%}-{target_max:.1%})")
        else:
            needed = target_min - best_acc
            print(f"   🎯 목표까지 {needed:.4f} ({needed*100:.1f}%p) 부족")
    
    return results

def main():
    """메인 실행"""
    start_time = time.time()
    
    # GPU 메모리 설정
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU 설정 오류: {e}")
    
    # 테스트 실행
    results = run_quick_test()
    
    elapsed = time.time() - start_time
    print(f"\n⏰ 실행 시간: {elapsed:.1f}초")
    print("=" * 60)
    print("✅ 빠른 성능 테스트 완료!")

if __name__ == "__main__":
    main()