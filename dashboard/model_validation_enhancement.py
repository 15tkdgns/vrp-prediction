#!/usr/bin/env python3
"""
SPY 모델 정확도 향상 및 검증 강화
- 오버피팅 방지 강화
- 데이터 누수 완전 방지
- 모델 오류 분석
- 추가 정확도 향상 기법
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
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, validation_curve, learning_curve
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import seaborn as sns

# Advanced validation
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.inspection import permutation_importance

# Deep Learning (if available)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, GRU
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.regularizers import l1_l2
    tf.get_logger().setLevel('ERROR')
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class ModelValidationEnhancement:
    def __init__(self):
        self.spy_data = None
        self.vix_data = None
        self.enhanced_features = None
        self.models = {}
        self.results = {}
        self.validation_results = {}
        
    def load_and_validate_data(self):
        """데이터 로드 및 품질 검증"""
        print("📥 데이터 로드 및 품질 검증 중...")
        
        try:
            # 더 긴 기간으로 확장 (2017-2024)
            spy_raw = yf.download('SPY', start='2017-01-01', end='2025-01-01', auto_adjust=True, progress=False)
            vix_raw = yf.download('^VIX', start='2017-01-01', end='2025-01-01', auto_adjust=True, progress=False)
            
            # MultiIndex 컬럼 정리
            if isinstance(spy_raw.columns, pd.MultiIndex):
                spy_raw.columns = spy_raw.columns.get_level_values(0)
            if isinstance(vix_raw.columns, pd.MultiIndex):
                vix_raw.columns = vix_raw.columns.get_level_values(0)
                
            # 데이터 품질 검증
            print(f"📊 SPY 데이터: {len(spy_raw)} 일")
            print(f"📊 VIX 데이터: {len(vix_raw)} 일")
            
            # 결측치 검사
            spy_missing = spy_raw.isnull().sum().sum()
            vix_missing = vix_raw.isnull().sum().sum()
            
            print(f"❓ SPY 결측치: {spy_missing}")
            print(f"❓ VIX 결측치: {vix_missing}")
            
            # 이상치 검사 (극단값)
            spy_outliers = ((spy_raw['Close'] - spy_raw['Close'].mean()).abs() > 3 * spy_raw['Close'].std()).sum()
            print(f"⚠️ SPY 이상치 (3σ 초과): {spy_outliers}")
            
            self.spy_data = spy_raw
            self.vix_data = vix_raw
            
            return True
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {str(e)}")
            return False
    
    def create_leak_proof_features(self):
        """완전한 데이터 누수 방지 특성 생성"""
        print("🔒 누수 방지 특성 생성 중...")
        
        spy_features = self.spy_data.copy()
        
        # 기본 수익률 (t+1 예측을 위해 t시점 데이터만 사용)
        spy_features['returns'] = spy_features['Close'].pct_change()
        spy_features['log_returns'] = np.log(spy_features['Close'] / spy_features['Close'].shift(1))
        
        # 과거 수익률 시리즈 (1-20일 전)
        for i in range(1, 21):
            spy_features[f'return_lag_{i}'] = spy_features['returns'].shift(i)
        
        # 기술적 지표 (모두 과거 데이터만 사용)
        def safe_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.shift(1)  # 1일 지연으로 누수 방지
            
        spy_features['rsi'] = safe_rsi(spy_features['Close'])
        
        # 이동평균 (과거만)
        for period in [5, 10, 20, 50, 200]:
            spy_features[f'ma_{period}'] = spy_features['Close'].rolling(period).mean().shift(1)
            spy_features[f'price_to_ma_{period}'] = spy_features['Close'].shift(1) / spy_features[f'ma_{period}']
        
        # 볼린저 밴드 (과거만)
        bb_period = 20
        spy_features['bb_middle'] = spy_features['Close'].rolling(bb_period).mean().shift(1)
        bb_std = spy_features['Close'].rolling(bb_period).std().shift(1)
        spy_features['bb_upper'] = spy_features['bb_middle'] + (bb_std * 2)
        spy_features['bb_lower'] = spy_features['bb_middle'] - (bb_std * 2)
        spy_features['bb_position'] = (spy_features['Close'].shift(1) - spy_features['bb_lower']) / (spy_features['bb_upper'] - spy_features['bb_lower'])
        
        # VIX 특성 (과거만)
        vix_aligned = self.vix_data.reindex(spy_features.index, method='ffill')
        spy_features['vix'] = vix_aligned['Close'].shift(1)  # 1일 지연
        spy_features['vix_change'] = spy_features['vix'].pct_change()
        spy_features['vix_ma_5'] = spy_features['vix'].rolling(5).mean()
        spy_features['vix_signal'] = (spy_features['vix'] <= 20).astype(int)
        
        # 거래량 지표 (과거만)
        spy_features['volume_ma'] = spy_features['Volume'].rolling(20).mean().shift(1)
        spy_features['volume_ratio'] = spy_features['Volume'].shift(1) / spy_features['volume_ma']
        
        # 변동성 지표 (과거만)
        for period in [5, 10, 20]:
            spy_features[f'volatility_{period}'] = spy_features['returns'].rolling(period).std().shift(1)
            spy_features[f'returns_mean_{period}'] = spy_features['returns'].rolling(period).mean().shift(1)
        
        # 타겟 변수: t+1 시점의 수익률 방향
        spy_features['future_return'] = spy_features['Close'].shift(-1) / spy_features['Close'] - 1
        spy_features['target'] = (spy_features['future_return'] > 0).astype(int)
        
        # 날짜 특성 (순환적 인코딩)
        spy_features['month'] = pd.to_datetime(spy_features.index).month
        spy_features['day_of_week'] = pd.to_datetime(spy_features.index).dayofweek
        spy_features['month_sin'] = np.sin(2 * np.pi * spy_features['month'] / 12)
        spy_features['month_cos'] = np.cos(2 * np.pi * spy_features['month'] / 12)
        spy_features['dow_sin'] = np.sin(2 * np.pi * spy_features['day_of_week'] / 7)
        spy_features['dow_cos'] = np.cos(2 * np.pi * spy_features['day_of_week'] / 7)
        
        self.enhanced_features = spy_features
        print(f"✅ 누수 방지 특성 {len(spy_features.columns)}개 생성 완료")
        
        # 결측치 처리 (forward fill만 사용)
        self.enhanced_features = self.enhanced_features.fillna(method='ffill')
        
        return True
    
    def validate_data_leakage(self):
        """데이터 누수 검증"""
        print("🔍 데이터 누수 검증 중...")
        
        validation_results = {
            'feature_future_correlation': {},
            'temporal_consistency': True,
            'target_leakage_check': True
        }
        
        # 특성과 미래 수익률 간 상관관계 검사 (높으면 누수 의심)
        if 'future_return' in self.enhanced_features.columns:
            future_returns = self.enhanced_features['future_return'].dropna()
            
            feature_cols = [col for col in self.enhanced_features.columns 
                          if col not in ['target', 'future_return', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            for feature in feature_cols:
                if self.enhanced_features[feature].dtype in ['float64', 'int64']:
                    # 같은 시점 데이터로 상관관계 계산 (누수 검사)
                    aligned_data = pd.concat([
                        self.enhanced_features[feature],
                        future_returns
                    ], axis=1).dropna()
                    
                    if len(aligned_data) > 100:
                        correlation = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
                        validation_results['feature_future_correlation'][feature] = abs(correlation)
        
        # 의심스러운 높은 상관관계 (>0.8) 체크
        high_corr_features = {k: v for k, v in validation_results['feature_future_correlation'].items() 
                            if v > 0.8}
        
        if high_corr_features:
            print("⚠️ 데이터 누수 의심 특성들:")
            for feature, corr in high_corr_features.items():
                print(f"   {feature}: {corr:.3f}")
            validation_results['target_leakage_check'] = False
        else:
            print("✅ 데이터 누수 검사 통과")
            
        return validation_results
    
    def prepare_robust_training_data(self):
        """강건한 학습 데이터 준비"""
        print("📊 강건한 학습 데이터 준비 중...")
        
        # 특성 선택 (누수 없는 특성만)
        exclude_cols = ['target', 'future_return', 'Open', 'High', 'Low', 'Close', 'Volume', 'month', 'day_of_week']
        feature_columns = [col for col in self.enhanced_features.columns 
                          if col not in exclude_cols and 
                          self.enhanced_features[col].dtype in ['float64', 'int64']]
        
        # 무한값과 결측값 처리
        clean_data = self.enhanced_features.replace([np.inf, -np.inf], np.nan).dropna()
        
        X = clean_data[feature_columns]
        y = clean_data['target']
        
        # 엄격한 시계열 분할
        # 2017-2020: 훈련용
        # 2021-2022: 검증용  
        # 2023-2024: 테스트용
        train_mask = X.index < '2021-01-01'
        val_mask = (X.index >= '2021-01-01') & (X.index < '2023-01-01')
        test_mask = X.index >= '2023-01-01'
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        # 클래스 불균형 확인
        train_class_dist = y_train.value_counts()
        val_class_dist = y_val.value_counts()
        test_class_dist = y_test.value_counts()
        
        print(f"📊 훈련 데이터: {len(X_train)} 샘플")
        print(f"   클래스 분포: {dict(train_class_dist)}")
        print(f"📊 검증 데이터: {len(X_val)} 샘플")  
        print(f"   클래스 분포: {dict(val_class_dist)}")
        print(f"📊 테스트 데이터: {len(X_test)} 샘플")
        print(f"   클래스 분포: {dict(test_class_dist)}")
        print(f"📊 특성 수: {len(feature_columns)}개")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, feature_columns
    
    def detect_overfitting_early(self, model, X_train, X_val, y_train, y_val, model_name):
        """오버피팅 조기 감지"""
        print(f"🔍 {model_name} 오버피팅 검사 중...")
        
        # 학습 곡선 분석
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_train, y_train, 
            train_sizes=train_sizes,
            cv=TimeSeriesSplit(n_splits=3),
            scoring='accuracy',
            n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # 오버피팅 지표 계산
        final_gap = train_mean[-1] - val_mean[-1]
        max_gap = np.max(train_mean - val_mean)
        
        overfitting_detected = final_gap > 0.1 or max_gap > 0.15
        
        overfitting_analysis = {
            'final_train_score': train_mean[-1],
            'final_val_score': val_mean[-1],
            'final_gap': final_gap,
            'max_gap': max_gap,
            'overfitting_detected': overfitting_detected,
            'train_sizes': train_sizes_abs,
            'train_scores': train_scores,
            'val_scores': val_scores
        }
        
        if overfitting_detected:
            print(f"⚠️ {model_name} 오버피팅 감지!")
            print(f"   최종 격차: {final_gap:.3f}")
            print(f"   최대 격차: {max_gap:.3f}")
        else:
            print(f"✅ {model_name} 오버피팅 없음")
            
        return overfitting_analysis
    
    def train_regularized_models(self, X_train, X_val, X_test, y_train, y_val, y_test, feature_columns):
        """정규화가 강화된 모델들 훈련"""
        print("🎯 정규화 강화 모델 훈련 중...")
        
        # 클래스 가중치 계산
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        # 스케일링
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        models_config = {
            'regularized_rf': {
                'model': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,  # 더 제한적
                    min_samples_split=20,  # 더 높게
                    min_samples_leaf=10,   # 더 높게
                    max_features=0.3,      # 더 제한적
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                ),
                'use_scaling': False
            },
            'regularized_gb': {
                'model': GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=6,           # 더 제한적
                    learning_rate=0.05,    # 더 낮게
                    subsample=0.8,         # 샘플링으로 정규화
                    max_features=0.5,      # 특성 샘플링
                    random_state=42
                ),
                'use_scaling': False
            },
            'ridge_lr': {
                'model': RidgeClassifier(
                    alpha=1.0,             # L2 정규화
                    class_weight='balanced',
                    random_state=42
                ),
                'use_scaling': True
            },
            'regularized_svm': {
                'model': SVC(
                    C=0.1,                 # 더 강한 정규화
                    kernel='rbf',
                    class_weight='balanced',
                    probability=True,
                    random_state=42
                ),
                'use_scaling': True
            }
        }
        
        results = {}
        
        for name, config in models_config.items():
            print(f"\n🔧 {name} 훈련 중...")
            
            model = config['model']
            
            # 적절한 데이터 사용
            if config['use_scaling']:
                X_tr, X_v, X_te = X_train_scaled, X_val_scaled, X_test_scaled
            else:
                X_tr, X_v, X_te = X_train, X_val, X_test
            
            # 모델 훈련
            model.fit(X_tr, y_train)
            
            # 예측
            train_pred = model.predict(X_tr)
            val_pred = model.predict(X_v)
            test_pred = model.predict(X_te)
            
            # 성능 계산
            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            test_acc = accuracy_score(y_test, test_pred)
            
            # AUC 계산 (확률 예측 가능한 경우)
            try:
                if hasattr(model, 'predict_proba'):
                    test_proba = model.predict_proba(X_te)[:, 1]
                elif hasattr(model, 'decision_function'):
                    test_proba = model.decision_function(X_te)
                else:
                    test_proba = test_pred
                    
                test_auc = roc_auc_score(y_test, test_proba)
            except:
                test_auc = 0.5
            
            # 오버피팅 검사
            overfitting_analysis = self.detect_overfitting_early(
                model, X_tr, X_v, y_train, y_val, name
            )
            
            results[name] = {
                'model': model,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'test_accuracy': test_acc,
                'test_auc': test_auc,
                'overfitting_analysis': overfitting_analysis,
                'scaler': scaler if config['use_scaling'] else None
            }
            
            print(f"   훈련 정확도: {train_acc:.3f}")
            print(f"   검증 정확도: {val_acc:.3f}")
            print(f"   테스트 정확도: {test_acc:.3f}")
            print(f"   테스트 AUC: {test_auc:.3f}")
            
        self.models.update(results)
        return results
    
    def train_enhanced_lstm(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """강화된 LSTM 모델"""
        if not TENSORFLOW_AVAILABLE:
            print("⚠️ TensorFlow 미설치로 LSTM 스킵")
            return {}
            
        print("🧠 강화된 LSTM 모델 훈련 중...")
        
        # 시퀀스 생성
        def create_sequences(X, y, seq_length=60):  # 더 긴 시퀀스
            sequences = []
            targets = []
            
            for i in range(seq_length, len(X)):
                sequences.append(X[i-seq_length:i])
                targets.append(y[i])
                
            return np.array(sequences), np.array(targets)
        
        # 스케일링
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # 시퀀스 생성
        seq_length = 60
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, seq_length)
        X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val.values, seq_length)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, seq_length)
        
        print(f"   LSTM 시퀀스 shape: {X_train_seq.shape}")
        
        # 클래스 가중치
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_seq), y=y_train_seq)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        # 강화된 LSTM 모델
        model = Sequential([
            # 첫 번째 LSTM 레이어 (더 많은 정규화)
            LSTM(128, return_sequences=True, input_shape=(seq_length, X_train_seq.shape[2]),
                 dropout=0.3, recurrent_dropout=0.3,
                 kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
            BatchNormalization(),
            
            # 두 번째 LSTM 레이어
            LSTM(64, return_sequences=True, 
                 dropout=0.3, recurrent_dropout=0.3,
                 kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
            BatchNormalization(),
            
            # 세 번째 LSTM 레이어
            LSTM(32, return_sequences=False,
                 dropout=0.3, recurrent_dropout=0.3,
                 kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
            BatchNormalization(),
            
            # Dense 레이어들
            Dense(32, activation='relu', 
                  kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
            Dropout(0.4),
            BatchNormalization(),
            
            Dense(16, activation='relu',
                  kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
            Dropout(0.3),
            
            # 출력 레이어
            Dense(1, activation='sigmoid')
        ])
        
        # 컴파일 (더 낮은 학습률)
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # 콜백 설정 (더 엄격한 조기 종료)
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1),
            ModelCheckpoint('best_lstm_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
        ]
        
        # 훈련
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=100,  # 더 많은 에포크
            batch_size=32,
            validation_data=(X_val_seq, y_val_seq),
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=0
        )
        
        # 예측
        train_pred_proba = model.predict(X_train_seq, verbose=0).flatten()
        val_pred_proba = model.predict(X_val_seq, verbose=0).flatten()
        test_pred_proba = model.predict(X_test_seq, verbose=0).flatten()
        
        train_pred = (train_pred_proba > 0.5).astype(int)
        val_pred = (val_pred_proba > 0.5).astype(int)
        test_pred = (test_pred_proba > 0.5).astype(int)
        
        # 성능 계산
        train_acc = accuracy_score(y_train_seq, train_pred)
        val_acc = accuracy_score(y_val_seq, val_pred)
        test_acc = accuracy_score(y_test_seq, test_pred)
        test_auc = roc_auc_score(y_test_seq, test_pred_proba)
        
        # 오버피팅 분석 (히스토리 기반)
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        train_accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        
        # 마지막 10 에포크 평균으로 오버피팅 판단
        final_epochs = 10
        final_train_acc = np.mean(train_accuracy[-final_epochs:])
        final_val_acc = np.mean(val_accuracy[-final_epochs:])
        overfitting_gap = final_train_acc - final_val_acc
        
        lstm_results = {
            'enhanced_lstm': {
                'model': model,
                'scaler': scaler,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,  
                'test_accuracy': test_acc,
                'test_auc': test_auc,
                'overfitting_gap': overfitting_gap,
                'overfitting_detected': overfitting_gap > 0.1,
                'history': history.history,
                'seq_length': seq_length
            }
        }
        
        print(f"✅ Enhanced LSTM - 테스트 정확도: {test_acc:.3f}, AUC: {test_auc:.3f}")
        print(f"   오버피팅 격차: {overfitting_gap:.3f}")
        
        self.models.update(lstm_results)
        return lstm_results
    
    def analyze_model_errors(self, X_test, y_test):
        """모델 오류 상세 분석"""
        print("🔍 모델 오류 상세 분석 중...")
        
        error_analysis = {}
        
        for model_name, model_data in self.models.items():
            if 'model' not in model_data:
                continue
                
            model = model_data['model']
            
            try:
                # 예측 (LSTM은 별도 처리 필요)
                if 'lstm' in model_name.lower():
                    # LSTM은 시퀀스 데이터 필요
                    continue
                    
                # 스케일링 적용 여부
                if model_data.get('scaler') is not None:
                    X_test_processed = model_data['scaler'].transform(X_test)
                else:
                    X_test_processed = X_test
                    
                test_pred = model.predict(X_test_processed)
                
                # 오류 분석
                errors = (y_test != test_pred)
                error_rate = errors.sum() / len(y_test)
                
                # 특성별 오류 패턴 분석
                error_data = X_test[errors]
                correct_data = X_test[~errors]
                
                feature_error_analysis = {}
                for feature in X_test.columns:
                    if X_test[feature].dtype in ['float64', 'int64']:
                        error_mean = error_data[feature].mean()
                        correct_mean = correct_data[feature].mean()
                        difference = abs(error_mean - correct_mean)
                        
                        feature_error_analysis[feature] = {
                            'error_mean': error_mean,
                            'correct_mean': correct_mean,
                            'difference': difference
                        }
                
                # 가장 문제되는 특성들
                problematic_features = sorted(feature_error_analysis.items(), 
                                           key=lambda x: x[1]['difference'], reverse=True)[:10]
                
                error_analysis[model_name] = {
                    'error_rate': error_rate,
                    'total_errors': errors.sum(),
                    'problematic_features': problematic_features,
                    'confusion_matrix': confusion_matrix(y_test, test_pred).tolist()
                }
                
                print(f"\n📊 {model_name} 오류 분석:")
                print(f"   오류율: {error_rate:.3f}")
                print(f"   문제 특성 Top 3:")
                for i, (feature, analysis) in enumerate(problematic_features[:3]):
                    print(f"     {i+1}. {feature}: 차이 {analysis['difference']:.4f}")
                    
            except Exception as e:
                print(f"❌ {model_name} 오류 분석 실패: {str(e)}")
                continue
        
        return error_analysis
    
    def create_validation_report(self, leak_validation, error_analysis):
        """종합 검증 보고서 생성"""
        print("📝 종합 검증 보고서 생성 중...")
        
        report = {
            'validation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_leakage_validation': leak_validation,
            'model_performance': {},
            'overfitting_analysis': {},
            'error_analysis': error_analysis,
            'recommendations': []
        }
        
        # 모델 성능 정리
        for model_name, model_data in self.models.items():
            if 'test_accuracy' in model_data:
                report['model_performance'][model_name] = {
                    'test_accuracy': float(model_data['test_accuracy']),
                    'test_auc': float(model_data.get('test_auc', 0)),
                    'overfitting_detected': model_data.get('overfitting_analysis', {}).get('overfitting_detected', False)
                }
        
        # 최고 성능 모델
        best_model = max(report['model_performance'].keys(), 
                        key=lambda k: report['model_performance'][k]['test_accuracy'])
        best_accuracy = report['model_performance'][best_model]['test_accuracy']
        
        # 권장사항 생성
        recommendations = []
        
        if leak_validation['target_leakage_check']:
            recommendations.append("✅ 데이터 누수 방지 성공")
        else:
            recommendations.append("⚠️ 데이터 누수 의심 특성들 제거 필요")
            
        # 오버피팅 권장사항
        overfitting_models = [name for name, data in report['model_performance'].items() 
                            if data['overfitting_detected']]
        if overfitting_models:
            recommendations.append(f"⚠️ 오버피팅 모델들: {overfitting_models}")
            recommendations.append("🔧 정규화 강화 또는 모델 복잡도 감소 필요")
        else:
            recommendations.append("✅ 모든 모델에서 오버피팅 통제됨")
            
        # 성능 개선 권장사항
        if best_accuracy < 0.60:
            recommendations.append("🎯 60% 돌파를 위한 추가 기법 필요")
            recommendations.append("🔬 특성 엔지니어링 강화 또는 앙상블 적용")
        
        report['recommendations'] = recommendations
        report['best_model'] = best_model
        report['best_accuracy'] = best_accuracy
        
        # 보고서 저장
        with open('data/raw/model_validation_enhancement_report.json', 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        print(f"✅ 검증 보고서 저장: data/raw/model_validation_enhancement_report.json")
        
        return report
    
    def run_enhanced_validation(self):
        """전체 강화된 검증 프로세스 실행"""
        print("🔬 SPY 모델 강화된 검증 프로세스 시작!")
        print("=" * 60)
        
        # 1. 데이터 로드 및 검증
        if not self.load_and_validate_data():
            return
            
        # 2. 누수 방지 특성 생성
        if not self.create_leak_proof_features():
            return
            
        # 3. 데이터 누수 검증
        leak_validation = self.validate_data_leakage()
        
        # 4. 강건한 학습 데이터 준비
        X_train, X_val, X_test, y_train, y_val, y_test, feature_columns = self.prepare_robust_training_data()
        
        # 5. 정규화 강화 모델 훈련
        regularized_results = self.train_regularized_models(X_train, X_val, X_test, y_train, y_val, y_test, feature_columns)
        
        # 6. 강화된 LSTM 훈련
        lstm_results = self.train_enhanced_lstm(X_train, X_val, X_test, y_train, y_val, y_test)
        
        # 7. 모델 오류 분석
        error_analysis = self.analyze_model_errors(X_test, y_test)
        
        # 8. 종합 보고서 생성
        validation_report = self.create_validation_report(leak_validation, error_analysis)
        
        print("\n" + "=" * 60)
        print("🏆 강화된 검증 결과:")
        print(f"📊 최고 모델: {validation_report['best_model']}")
        print(f"🎯 최고 정확도: {validation_report['best_accuracy']:.1%}")
        
        if leak_validation['target_leakage_check']:
            print("✅ 데이터 누수 방지 성공")
        else:
            print("⚠️ 데이터 누수 의심사항 있음")
            
        overfitting_count = sum(1 for data in validation_report['model_performance'].values() 
                               if data['overfitting_detected'])
        print(f"🔍 오버피팅 모델 수: {overfitting_count}")
        
        print("\n📋 주요 권장사항:")
        for rec in validation_report['recommendations']:
            print(f"   {rec}")
            
        print(f"\n✅ 강화된 검증 완료!")

def main():
    validator = ModelValidationEnhancement()
    validator.run_enhanced_validation()

if __name__ == "__main__":
    main()