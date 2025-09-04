#!/usr/bin/env python3
"""
SPY 예측 모델 고급 성능 개선 실험
다른 연구 참고한 첨단 기법들 적용
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

# Deep Learning
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available, skipping LSTM models")

# Advanced optimization
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False
    print("⚠️ scikit-optimize not available, using GridSearch instead")

class SPYAdvancedExperiment:
    def __init__(self):
        self.spy_data = None
        self.vix_data = None
        self.enhanced_features = None
        self.models = {}
        self.results = {}
        self.scalers = {}
        
    def load_historical_data(self):
        """역사적 데이터 수집"""
        print("📥 고급 실험용 데이터 수집 중...")
        
        try:
            # SPY + VIX 데이터 (2018-2024로 확장)
            spy_raw = yf.download('SPY', start='2018-01-01', end='2025-01-01', auto_adjust=True, progress=False)
            vix_raw = yf.download('^VIX', start='2018-01-01', end='2025-01-01', auto_adjust=True, progress=False)
            
            # MultiIndex 컬럼 정리
            if isinstance(spy_raw.columns, pd.MultiIndex):
                spy_raw.columns = spy_raw.columns.get_level_values(0)
            if isinstance(vix_raw.columns, pd.MultiIndex):
                vix_raw.columns = vix_raw.columns.get_level_values(0)
                
            self.spy_data = spy_raw
            self.vix_data = vix_raw
            
            print(f"✅ SPY 데이터: {len(spy_raw)} 일")
            print(f"✅ VIX 데이터: {len(vix_raw)} 일")
            
            return True
            
        except Exception as e:
            print(f"❌ 데이터 수집 실패: {str(e)}")
            return False
    
    def create_advanced_technical_indicators(self, df):
        """고급 기술적 지표 10개 추가 구현"""
        print("🔧 고급 기술적 지표 생성 중...")
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # 0. RSI (추가)
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
            
        df['rsi'] = calculate_rsi(close)
        
        # 1. Stochastic Oscillator
        def stochastic_k(high, low, close, k_period=14):
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            return k_percent
        
        df['stoch_k'] = stochastic_k(high, low, close)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # 2. Williams %R
        def williams_r(high, low, close, period=14):
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
            return wr
            
        df['williams_r'] = williams_r(high, low, close)
        
        # 3. Commodity Channel Index (CCI)
        def cci(high, low, close, period=20):
            tp = (high + low + close) / 3
            tp_ma = tp.rolling(period).mean()
            mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
            cci = (tp - tp_ma) / (0.015 * mad)
            return cci
            
        df['cci'] = cci(high, low, close)
        
        # 4. Money Flow Index (MFI)
        def mfi(high, low, close, volume, period=14):
            tp = (high + low + close) / 3
            raw_money_flow = tp * volume
            
            positive_flow = raw_money_flow.where(tp.diff() > 0, 0).rolling(period).sum()
            negative_flow = raw_money_flow.where(tp.diff() < 0, 0).rolling(period).sum()
            
            money_ratio = positive_flow / negative_flow
            mfi = 100 - (100 / (1 + money_ratio))
            return mfi
            
        df['mfi'] = mfi(high, low, close, volume)
        
        # 5. Average Directional Index (ADX)
        def adx(high, low, close, period=14):
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            up_move = high.diff()
            down_move = -low.diff()
            
            plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
            minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
            
            tr_smooth = tr.rolling(period).mean()
            plus_dm_smooth = plus_dm.rolling(period).mean()
            minus_dm_smooth = minus_dm.rolling(period).mean()
            
            plus_di = 100 * (plus_dm_smooth / tr_smooth)
            minus_di = 100 * (minus_dm_smooth / tr_smooth)
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(period).mean()
            
            return adx, plus_di, minus_di
            
        df['adx'], df['plus_di'], df['minus_di'] = adx(high, low, close)
        
        # 6. Ultimate Oscillator
        def ultimate_oscillator(high, low, close, period1=7, period2=14, period3=28):
            tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
            bp = close - pd.concat([low, close.shift()], axis=1).min(axis=1)
            
            avg1 = bp.rolling(period1).sum() / tr.rolling(period1).sum()
            avg2 = bp.rolling(period2).sum() / tr.rolling(period2).sum()
            avg3 = bp.rolling(period3).sum() / tr.rolling(period3).sum()
            
            uo = 100 * (4*avg1 + 2*avg2 + avg3) / 7
            return uo
            
        df['ultimate_osc'] = ultimate_oscillator(high, low, close)
        
        # 7. Parabolic SAR
        def parabolic_sar(high, low, af=0.02, max_af=0.2):
            length = len(high)
            dates = high.index
            high_values = high.values
            low_values = low.values
            
            psar = close.copy()
            psaraf = af
            psarep = high_values[0]
            bull = True
            
            for i in range(1, length):
                if bull:
                    psar.iloc[i] = psar.iloc[i-1] + psaraf * (psarep - psar.iloc[i-1])
                    if low_values[i] < psar.iloc[i]:
                        bull = False
                        psar.iloc[i] = psarep
                        psarep = low_values[i]
                        psaraf = af
                    else:
                        if high_values[i] > psarep:
                            psarep = high_values[i]
                            psaraf = min(psaraf + af, max_af)
                else:
                    psar.iloc[i] = psar.iloc[i-1] + psaraf * (psarep - psar.iloc[i-1])
                    if high_values[i] > psar.iloc[i]:
                        bull = True
                        psar.iloc[i] = psarep
                        psarep = high_values[i]
                        psaraf = af
                    else:
                        if low_values[i] < psarep:
                            psarep = low_values[i]
                            psaraf = min(psaraf + af, max_af)
                            
            return psar
            
        df['parabolic_sar'] = parabolic_sar(high, low)
        df['sar_signal'] = (close > df['parabolic_sar']).astype(int)
        
        # 8. VWAP (Volume Weighted Average Price)
        def vwap(high, low, close, volume):
            tp = (high + low + close) / 3
            return (tp * volume).cumsum() / volume.cumsum()
            
        # Reset cumulative calculation daily (simplified version)
        df['vwap'] = vwap(high, low, close, volume)
        df['price_to_vwap'] = close / df['vwap']
        
        # 9. Aroon Indicator
        def aroon(high, low, period=25):
            aroon_up = high.rolling(period).apply(lambda x: (period - x.argmax()) / period * 100)
            aroon_down = low.rolling(period).apply(lambda x: (period - x.argmin()) / period * 100)
            aroon_osc = aroon_up - aroon_down
            return aroon_up, aroon_down, aroon_osc
            
        df['aroon_up'], df['aroon_down'], df['aroon_osc'] = aroon(high, low)
        
        # 10. Keltner Channel
        def keltner_channel(high, low, close, period=20, multiplier=2):
            tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            
            kc_middle = close.ewm(span=period).mean()
            kc_upper = kc_middle + multiplier * atr
            kc_lower = kc_middle - multiplier * atr
            
            return kc_upper, kc_middle, kc_lower
            
        df['kc_upper'], df['kc_middle'], df['kc_lower'] = keltner_channel(high, low, close)
        df['kc_position'] = (close - df['kc_lower']) / (df['kc_upper'] - df['kc_lower'])
        
        print(f"✅ 고급 기술적 지표 {len([col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']])}개 생성 완료")
        
        return df
    
    def detect_market_regime(self, returns, window=60):
        """시장 체제 감지 (Bull/Bear/Sideways)"""
        print("📊 시장 체제 감지 중...")
        
        # 간단한 체제 감지: 이동평균 기반
        returns_ma = returns.rolling(window).mean()
        returns_std = returns.rolling(window).std()
        
        regime = pd.Series(index=returns.index, dtype=int)
        
        # Bull: 평균 수익률 > 0.1% and 변동성 < 평균
        # Bear: 평균 수익률 < -0.1% and 변동성 > 평균  
        # Sideways: 그 외
        
        bull_condition = (returns_ma > 0.001) & (returns_std < returns_std.rolling(250).mean())
        bear_condition = (returns_ma < -0.001) & (returns_std > returns_std.rolling(250).mean())
        
        regime.loc[bull_condition] = 2  # Bull
        regime.loc[bear_condition] = 0   # Bear
        regime.loc[~(bull_condition | bear_condition)] = 1  # Sideways
        
        regime = regime.fillna(1)  # Default to sideways
        
        print(f"✅ Bull 체제: {(regime == 2).sum()}일")
        print(f"✅ Sideways 체제: {(regime == 1).sum()}일") 
        print(f"✅ Bear 체제: {(regime == 0).sum()}일")
        
        return regime
    
    def create_enhanced_features_v2(self):
        """강화된 특성 생성 v2"""
        print("🚀 강화된 특성 생성 v2 중...")
        
        # SPY 데이터에 고급 지표 추가
        spy_features = self.create_advanced_technical_indicators(self.spy_data.copy())
        
        # 기본 수익률
        spy_features['returns'] = spy_features['Close'].pct_change()
        spy_features['log_returns'] = np.log(spy_features['Close'] / spy_features['Close'].shift(1))
        
        # 과거 수익률 시리즈 (1-10일)
        for i in range(1, 11):
            spy_features[f'return_lag_{i}'] = spy_features['returns'].shift(i)
            
        # 롤링 통계량
        for window in [5, 10, 20, 50]:
            spy_features[f'returns_mean_{window}'] = spy_features['returns'].rolling(window).mean()
            spy_features[f'returns_std_{window}'] = spy_features['returns'].rolling(window).std()
            spy_features[f'returns_skew_{window}'] = spy_features['returns'].rolling(window).skew()
            spy_features[f'returns_kurt_{window}'] = spy_features['returns'].rolling(window).kurt()
        
        # VIX 특성
        vix_aligned = self.vix_data.reindex(spy_features.index, method='ffill')
        spy_features['vix'] = vix_aligned['Close']
        spy_features['vix_change'] = spy_features['vix'].pct_change()
        spy_features['vix_ma_5'] = spy_features['vix'].rolling(5).mean()
        spy_features['vix_ma_20'] = spy_features['vix'].rolling(20).mean()
        spy_features['vix_signal'] = (spy_features['vix'] <= 20).astype(int)
        
        # 시장 체제
        regime = self.detect_market_regime(spy_features['returns'])
        spy_features['market_regime'] = regime
        
        # 타겟 변수
        spy_features['future_return'] = spy_features['Close'].shift(-1) / spy_features['Close'] - 1
        spy_features['target'] = (spy_features['future_return'] > 0).astype(int)
        
        # 상호작용 특성 (Feature Interaction)
        spy_features['rsi_vix'] = spy_features['rsi'] * spy_features['vix'] / 100
        spy_features['stoch_adx'] = spy_features['stoch_k'] * spy_features['adx'] / 100
        spy_features['regime_vix'] = spy_features['market_regime'] * spy_features['vix']
        
        self.enhanced_features = spy_features
        print(f"✅ 총 특성 수: {len(spy_features.columns)}개")
        
        # 결측치 처리
        self.enhanced_features = self.enhanced_features.fillna(method='ffill').fillna(method='bfill')
        
    def prepare_advanced_training_data(self):
        """고급 학습 데이터 준비"""
        print("📊 고급 학습 데이터 준비 중...")
        
        # 모든 수치 특성 자동 선택 (target, future_return 제외)
        exclude_cols = ['target', 'future_return', 'Open', 'High', 'Low', 'Close', 'Volume']
        feature_columns = [col for col in self.enhanced_features.columns 
                          if col not in exclude_cols and 
                          self.enhanced_features[col].dtype in ['float64', 'int64']]
        
        # 무한값과 결측값 처리
        clean_data = self.enhanced_features.replace([np.inf, -np.inf], np.nan).dropna()
        
        X = clean_data[feature_columns]
        y = clean_data['target']
        
        # 시계열 분할: 2023년까지 훈련, 2024년 테스트
        train_mask = X.index < '2023-01-01'
        val_mask = (X.index >= '2023-01-01') & (X.index < '2024-01-01')
        test_mask = X.index >= '2024-01-01'
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        print(f"✅ 훈련 데이터: {len(X_train)} 샘플")
        print(f"✅ 검증 데이터: {len(X_val)} 샘플")
        print(f"✅ 테스트 데이터: {len(X_test)} 샘플")
        print(f"✅ 특성 수: {len(feature_columns)}개")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, feature_columns
    
    def train_advanced_ensemble(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """고급 앙상블 모델 (Stacking)"""
        print("🎯 고급 앙상블 모델 (Stacking) 훈련 중...")
        
        # 스케일링
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['advanced_ensemble'] = scaler
        
        # Base models
        base_models = {
            'rf': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
            'et': ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
            'gb': GradientBoostingClassifier(n_estimators=200, max_depth=8, random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(probability=True, random_state=42)
        }
        
        # Base model predictions
        base_predictions_val = np.zeros((len(X_val_scaled), len(base_models)))
        base_predictions_test = np.zeros((len(X_test_scaled), len(base_models)))
        
        for i, (name, model) in enumerate(base_models.items()):
            print(f"  훈련 중: {name}")
            model.fit(X_train_scaled, y_train)
            
            # Validation predictions
            val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            base_predictions_val[:, i] = val_pred_proba
            
            # Test predictions
            test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            base_predictions_test[:, i] = test_pred_proba
        
        # Meta model
        meta_model = LogisticRegression(random_state=42)
        meta_model.fit(base_predictions_val, y_val)
        
        # Final predictions
        final_test_pred_proba = meta_model.predict_proba(base_predictions_test)[:, 1]
        final_test_pred = (final_test_pred_proba > 0.5).astype(int)
        
        test_accuracy = accuracy_score(y_test, final_test_pred)
        test_auc = roc_auc_score(y_test, final_test_pred_proba)
        
        self.models['advanced_ensemble'] = {
            'base_models': base_models,
            'meta_model': meta_model,
            'scaler': scaler
        }
        
        self.results['advanced_ensemble'] = {
            'test_accuracy': test_accuracy,
            'test_auc': test_auc,
            'predictions': final_test_pred,
            'probabilities': final_test_pred_proba
        }
        
        print(f"✅ Stacking 앙상블 - 테스트 정확도: {test_accuracy:.3f}, AUC: {test_auc:.3f}")
        
        return test_accuracy
    
    def train_lstm_model(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """LSTM 시계열 모델"""
        if not TENSORFLOW_AVAILABLE:
            print("⚠️ TensorFlow 미설치로 LSTM 모델 스킵")
            return 0.5
            
        print("🧠 LSTM 시계열 모델 훈련 중...")
        
        # 시계열 데이터를 위한 sequence 생성
        def create_sequences(X, y, seq_length=30):
            sequences = []
            targets = []
            
            for i in range(seq_length, len(X)):
                sequences.append(X[i-seq_length:i])
                targets.append(y[i])
                
            return np.array(sequences), np.array(targets)
        
        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Sequence 생성
        seq_length = 30
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, seq_length)
        X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val.values, seq_length)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, seq_length)
        
        print(f"  LSTM 시퀀스 shape: {X_train_seq.shape}")
        
        # LSTM 모델 구성
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(seq_length, X_train_seq.shape[2])),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            
            Dense(16, activation='relu'),
            Dropout(0.2),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='binary_crossentropy', 
                     metrics=['accuracy'])
        
        # Early stopping
        early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
        
        # 훈련
        history = model.fit(X_train_seq, y_train_seq, 
                           epochs=50, 
                           batch_size=32,
                           validation_data=(X_val_seq, y_val_seq),
                           callbacks=[early_stop],
                           verbose=0)
        
        # 테스트 예측
        test_pred_proba = model.predict(X_test_seq, verbose=0).flatten()
        test_pred = (test_pred_proba > 0.5).astype(int)
        
        test_accuracy = accuracy_score(y_test_seq, test_pred)
        test_auc = roc_auc_score(y_test_seq, test_pred_proba)
        
        self.models['lstm'] = model
        self.scalers['lstm'] = scaler
        
        self.results['lstm'] = {
            'test_accuracy': test_accuracy,
            'test_auc': test_auc,
            'predictions': test_pred,
            'probabilities': test_pred_proba
        }
        
        print(f"✅ LSTM 모델 - 테스트 정확도: {test_accuracy:.3f}, AUC: {test_auc:.3f}")
        
        return test_accuracy
    
    def run_bayesian_optimization(self, X_train, X_val, y_train, y_val):
        """베이지안 최적화를 통한 하이퍼파라미터 튜닝"""
        if not BAYESIAN_OPT_AVAILABLE:
            print("⚠️ scikit-optimize 미설치로 그리드서치 사용")
            return self.run_grid_search(X_train, X_val, y_train, y_val)
            
        print("🔍 베이지안 최적화 하이퍼파라미터 튜닝 중...")
        
        # 검색 공간 정의
        search_spaces = {
            'n_estimators': Integer(100, 500),
            'max_depth': Integer(8, 25),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10),
            'max_features': Real(0.3, 1.0)
        }
        
        # 베이지안 최적화
        rf_opt = BayesSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            search_spaces,
            n_iter=50,
            cv=TimeSeriesSplit(n_splits=3),
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42
        )
        
        # 훈련 + 검증 데이터로 최적화
        X_combined = pd.concat([X_train, X_val])
        y_combined = pd.concat([y_train, y_val])
        
        rf_opt.fit(X_combined, y_combined)
        
        best_params = rf_opt.best_params_
        best_score = rf_opt.best_score_
        
        print(f"✅ 최적 파라미터: {best_params}")
        print(f"✅ 최적 CV AUC: {best_score:.3f}")
        
        self.models['bayesian_optimized'] = rf_opt.best_estimator_
        
        return rf_opt.best_estimator_
    
    def run_grid_search(self, X_train, X_val, y_train, y_val):
        """그리드 서치 대체"""
        print("🔍 그리드서치 하이퍼파라미터 튜닝 중...")
        
        param_grid = {
            'n_estimators': [200, 300],
            'max_depth': [15, 20],
            'min_samples_split': [5, 10],
            'max_features': ['sqrt', 0.7]
        }
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            param_grid,
            cv=TimeSeriesSplit(n_splits=3),
            scoring='roc_auc',
            n_jobs=-1
        )
        
        X_combined = pd.concat([X_train, X_val])
        y_combined = pd.concat([y_train, y_val])
        
        rf_grid.fit(X_combined, y_combined)
        
        print(f"✅ 최적 파라미터: {rf_grid.best_params_}")
        print(f"✅ 최적 CV AUC: {rf_grid.best_score_:.3f}")
        
        self.models['grid_optimized'] = rf_grid.best_estimator_
        
        return rf_grid.best_estimator_
    
    def evaluate_all_models(self, X_test, y_test):
        """모든 모델 종합 평가"""
        print("📊 모든 모델 종합 평가 중...")
        
        evaluation_results = {}
        
        # Optimized RF 평가
        if 'grid_optimized' in self.models or 'bayesian_optimized' in self.models:
            opt_model = self.models.get('bayesian_optimized', self.models.get('grid_optimized'))
            
            opt_pred = opt_model.predict(X_test)
            opt_proba = opt_model.predict_proba(X_test)[:, 1]
            opt_accuracy = accuracy_score(y_test, opt_pred)
            opt_auc = roc_auc_score(y_test, opt_proba)
            
            evaluation_results['optimized_rf'] = {
                'accuracy': opt_accuracy,
                'auc': opt_auc
            }
            
            print(f"✅ 최적화된 RF - 정확도: {opt_accuracy:.3f}, AUC: {opt_auc:.3f}")
        
        return evaluation_results
    
    def create_final_report_v2(self):
        """최종 고급 실험 보고서"""
        print("📝 고급 실험 최종 보고서 생성 중...")
        
        report = {
            'experiment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'experiment_type': 'Advanced SPY Prediction with State-of-Art Techniques',
            'data_period': '2018-2024',
            'techniques_applied': [
                'Advanced Technical Indicators (10+ new indicators)',
                'Market Regime Detection',
                'Stacking Ensemble',
                'LSTM Deep Learning' if TENSORFLOW_AVAILABLE else 'LSTM (not available)',
                'Bayesian Optimization' if BAYESIAN_OPT_AVAILABLE else 'Grid Search',
                'Feature Engineering v2',
                'Robust Scaling'
            ],
            'results': {},
            'performance_comparison': {}
        }
        
        # 결과 수집
        for model_name, result in self.results.items():
            report['results'][model_name] = {
                'test_accuracy': float(result['test_accuracy']),
                'test_auc': float(result.get('test_auc', 0))
            }
        
        # 최고 성능 찾기
        best_model = max(self.results.keys(), key=lambda k: self.results[k]['test_accuracy'])
        best_accuracy = self.results[best_model]['test_accuracy']
        
        report['best_model'] = best_model
        report['best_accuracy'] = float(best_accuracy)
        report['improvement_vs_baseline'] = float(best_accuracy - 0.472)  # 원래 47.2%
        report['improvement_vs_previous'] = float(best_accuracy - 0.496)  # 이전 VIX 49.6%
        
        # 저장
        with open('data/raw/spy_advanced_experiment_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        return report
    
    def run_advanced_experiment(self):
        """고급 실험 전체 실행"""
        print("🚀 SPY 고급 성능 개선 실험 시작!")
        print("=" * 60)
        
        # 데이터 수집
        if not self.load_historical_data():
            return
            
        # 고급 특성 생성
        self.create_enhanced_features_v2()
        
        # 데이터 준비
        X_train, X_val, X_test, y_train, y_val, y_test, feature_columns = self.prepare_advanced_training_data()
        
        # 모델 훈련
        ensemble_acc = self.train_advanced_ensemble(X_train, X_val, X_test, y_train, y_val, y_test)
        
        if TENSORFLOW_AVAILABLE:
            lstm_acc = self.train_lstm_model(X_train, X_val, X_test, y_train, y_val, y_test)
        
        # 하이퍼파라미터 최적화 후 평가
        optimized_model = self.run_bayesian_optimization(X_train, X_val, y_train, y_val)
        evaluation_results = self.evaluate_all_models(X_test, y_test)
        
        # 최종 보고서
        report = self.create_final_report_v2()
        
        print("\n" + "=" * 60)
        print("🏆 고급 실험 결과 요약:")
        print(f"📊 고급 앙상블: {self.results['advanced_ensemble']['test_accuracy']:.1%}")
        
        if 'lstm' in self.results:
            print(f"🧠 LSTM 모델: {self.results['lstm']['test_accuracy']:.1%}")
            
        if evaluation_results:
            for model, result in evaluation_results.items():
                print(f"🔍 {model}: {result['accuracy']:.1%}")
        
        print(f"\n🚀 최고 성능: {report['best_accuracy']:.1%} ({report['best_model']})")
        print(f"📈 기준선 대비 개선: {report['improvement_vs_baseline']*100:+.1f}%")
        print(f"📈 이전 실험 대비: {report['improvement_vs_previous']*100:+.1f}%")
        
        print(f"\n✅ 고급 실험 완료! 상세 보고서: data/raw/spy_advanced_experiment_report.json")

def main():
    experiment = SPYAdvancedExperiment()
    experiment.run_advanced_experiment()

if __name__ == "__main__":
    main()