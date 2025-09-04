#!/usr/bin/env python3
"""
SPY 머신러닝 모델링 파이프라인
데이터 누출 방지 및 오버피팅 방지를 위한 엄격한 시계열 모델링

주요 특징:
- 시간 순서 엄수 (No data leakage)
- 교차 검증 (TimeSeriesSplit)
- 다양한 정규화 기법
- 상세한 실험 로깅
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# Technical analysis
import talib
from sklearn.preprocessing import MinMaxScaler

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Utilities
import json
import pickle
from datetime import datetime
import logging
import os

class SPYMLPipeline:
    """SPY 데이터 머신러닝 파이프라인"""
    
    def __init__(self, data_path="data/raw/spy_data_2020_2025.csv"):
        """파이프라인 초기화"""
        self.data_path = data_path
        self.raw_data = None
        self.features_data = None
        self.X_train = None
        self.X_val = None 
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # 모델 저장용
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        
        # 실험 로깅
        self.experiment_log = {
            'start_time': datetime.now().isoformat(),
            'data_info': {},
            'feature_engineering': {},
            'model_experiments': {},
            'final_results': {}
        }
        
        # 로깅 설정
        self.setup_logging()
        
    def setup_logging(self):
        """로깅 설정"""
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/spy_ml_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_and_explore_data(self):
        """데이터 로드 및 탐색"""
        self.logger.info("=== 데이터 로드 및 탐색 시작 ===")
        
        # 데이터 로드
        self.raw_data = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        self.logger.info(f"원본 데이터 크기: {self.raw_data.shape}")
        self.logger.info(f"데이터 기간: {self.raw_data.index[0]} ~ {self.raw_data.index[-1]}")
        
        # 기본 정보 기록
        self.experiment_log['data_info'] = {
            'total_samples': len(self.raw_data),
            'date_range': f"{self.raw_data.index[0]} to {self.raw_data.index[-1]}",
            'features': list(self.raw_data.columns),
            'missing_values': self.raw_data.isnull().sum().to_dict()
        }
        
        # 기본 통계
        self.logger.info("기본 통계:")
        self.logger.info(f"평균 종가: ${self.raw_data['Close'].mean():.2f}")
        self.logger.info(f"종가 표준편차: ${self.raw_data['Close'].std():.2f}")
        self.logger.info(f"최고가: ${self.raw_data['High'].max():.2f}")
        self.logger.info(f"최저가: ${self.raw_data['Low'].min():.2f}")
        
        return self.raw_data
    
    def create_target_variable(self, prediction_horizon=1, threshold=0.001):
        """
        타겟 변수 생성 - 데이터 누출 방지
        
        Args:
            prediction_horizon: 예측 일수 (기본 1일)
            threshold: 상승/하락 분류 임계값 (기본 0.1%)
        """
        self.logger.info(f"=== 타겟 변수 생성 (예측 기간: {prediction_horizon}일, 임계값: {threshold*100:.1f}%) ===")
        
        # 미래 수익률 계산 (데이터 누출 방지를 위해 shift 사용)
        future_returns = self.raw_data['Close'].pct_change(periods=prediction_horizon).shift(-prediction_horizon)
        
        # 분류 타겟: 상승(1), 하락(0)
        binary_target = (future_returns > threshold).astype(int)
        
        # 3클래스 타겟: 상승(2), 횡보(1), 하락(0) 
        multi_target = np.where(future_returns > threshold, 2,
                               np.where(future_returns < -threshold, 0, 1))
        
        self.raw_data['Future_Return'] = future_returns
        self.raw_data['Binary_Target'] = binary_target
        self.raw_data['Multi_Target'] = multi_target
        
        # 통계 기록
        target_stats = {
            'binary_distribution': binary_target.value_counts().to_dict(),
            'multi_distribution': pd.Series(multi_target).value_counts().to_dict(),
            'mean_future_return': float(future_returns.mean()),
            'future_return_std': float(future_returns.std())
        }
        
        self.experiment_log['target_creation'] = target_stats
        
        self.logger.info(f"이진 타겟 분포: {target_stats['binary_distribution']}")
        self.logger.info(f"다중 타겟 분포: {target_stats['multi_distribution']}")
        
        return binary_target, multi_target
        
    def engineer_features(self, lookback_periods=[5, 10, 20, 50]):
        """
        특성 공학 - 데이터 누출 방지 엄격 적용
        
        Args:
            lookback_periods: 기술적 지표 계산용 기간들
        """
        self.logger.info("=== 특성 공학 시작 ===")
        
        df = self.raw_data.copy()
        
        # 1. 기본 가격 특성 (현재 시점 기준)
        df['Open_Close_Ratio'] = df['Open'] / df['Close']
        df['High_Low_Ratio'] = df['High'] / df['Low'] 
        df['Volume_MA_Ratio'] = df['Volume'] / df['Volume'].rolling(20, min_periods=1).mean()
        
        # 2. 수익률 특성 (과거 데이터만 사용)
        for period in [1, 2, 3, 5, 10]:
            df[f'Return_{period}d'] = df['Close'].pct_change(periods=period)
            df[f'Volume_Change_{period}d'] = df['Volume'].pct_change(periods=period)
        
        # 3. 이동평균 특성
        for period in lookback_periods:
            df[f'SMA_{period}'] = df['Close'].rolling(period, min_periods=1).mean()
            df[f'Price_SMA_Ratio_{period}'] = df['Close'] / df[f'SMA_{period}']
            df[f'Volume_SMA_{period}'] = df['Volume'].rolling(period, min_periods=1).mean()
        
        # 4. 기술적 지표 (TA-Lib 사용)
        try:
            # RSI
            df['RSI_14'] = talib.RSI(df['Close'].values, timeperiod=14)
            df['RSI_30'] = talib.RSI(df['Close'].values, timeperiod=30)
            
            # MACD
            macd, macdsignal, macdhist = talib.MACD(df['Close'].values)
            df['MACD'] = macd
            df['MACD_Signal'] = macdsignal
            df['MACD_Hist'] = macdhist
            
            # 볼린저 밴드
            upper, middle, lower = talib.BBANDS(df['Close'].values, timeperiod=20)
            df['BB_Upper'] = upper
            df['BB_Middle'] = middle
            df['BB_Lower'] = lower
            df['BB_Width'] = (upper - lower) / middle
            df['BB_Position'] = (df['Close'] - lower) / (upper - lower)
            
            # ATR (Average True Range)
            df['ATR'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
            
            # Stochastic
            slowk, slowd = talib.STOCH(df['High'].values, df['Low'].values, df['Close'].values)
            df['Stoch_K'] = slowk
            df['Stoch_D'] = slowd
            
            # Williams %R
            df['Williams_R'] = talib.WILLR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
            
            self.logger.info("기술적 지표 생성 완료")
            
        except Exception as e:
            self.logger.warning(f"기술적 지표 생성 중 오류: {e}")
            
        # 5. 통계적 특성
        for period in [5, 10, 20]:
            df[f'Volatility_{period}d'] = df['Close'].pct_change().rolling(period).std()
            df[f'Price_Std_{period}d'] = df['Close'].rolling(period).std()
            df[f'High_Low_Spread_{period}d'] = (df['High'] - df['Low']).rolling(period).mean()
        
        # 6. 라그 특성 (과거 값들)
        for lag in [1, 2, 3, 5]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
            df[f'Return_Lag_{lag}'] = df['Close'].pct_change().shift(lag)
        
        # 7. 시간 기반 특성
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['IsMonthStart'] = df.index.is_month_start.astype(int)
        df['IsMonthEnd'] = df.index.is_month_end.astype(int)
        
        self.features_data = df
        
        # 특성 개수 기록
        feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 
                                                                'Future_Return', 'Binary_Target', 'Multi_Target']]
        
        self.experiment_log['feature_engineering'] = {
            'total_features': len(feature_cols),
            'feature_categories': {
                'price_ratios': 3,
                'returns': len([c for c in feature_cols if 'Return_' in c]),
                'moving_averages': len([c for c in feature_cols if 'SMA_' in c]),
                'technical_indicators': len([c for c in feature_cols if any(ti in c for ti in ['RSI', 'MACD', 'BB_', 'ATR', 'Stoch', 'Williams'])]),
                'statistical': len([c for c in feature_cols if any(stat in c for stat in ['Volatility', 'Std', 'Spread'])]),
                'lags': len([c for c in feature_cols if 'Lag_' in c]),
                'time_based': len([c for c in feature_cols if c in ['DayOfWeek', 'Month', 'Quarter', 'IsMonthStart', 'IsMonthEnd']])
            }
        }
        
        self.logger.info(f"총 {len(feature_cols)}개 특성 생성 완료")
        self.logger.info(f"특성 카테고리별 개수: {self.experiment_log['feature_engineering']['feature_categories']}")
        
        return self.features_data
    
    def prepare_ml_data(self, test_size=0.2, val_size=0.1, target_type='binary'):
        """
        머신러닝용 데이터 준비 - 시간 순서 엄수
        
        Args:
            test_size: 테스트 세트 비율
            val_size: 검증 세트 비율  
            target_type: 'binary' 또는 'multi'
        """
        self.logger.info("=== 머신러닝 데이터 준비 ===")
        
        # 결측값 제거 (시간 순서 유지)
        df_clean = self.features_data.dropna()
        self.logger.info(f"결측값 제거 후 데이터 크기: {df_clean.shape}")
        
        # 특성과 타겟 분리
        feature_cols = [col for col in df_clean.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 
                                                                       'Future_Return', 'Binary_Target', 'Multi_Target']]
        
        X = df_clean[feature_cols]
        y = df_clean['Binary_Target'] if target_type == 'binary' else df_clean['Multi_Target']
        
        # 시간 순서 기반 분할 (데이터 누출 방지)
        n_samples = len(X)
        train_end = int(n_samples * (1 - test_size - val_size))
        val_end = int(n_samples * (1 - test_size))
        
        self.X_train = X.iloc[:train_end]
        self.X_val = X.iloc[train_end:val_end]  
        self.X_test = X.iloc[val_end:]
        
        self.y_train = y.iloc[:train_end]
        self.y_val = y.iloc[train_end:val_end]
        self.y_test = y.iloc[val_end:]
        
        # 데이터 분할 정보 기록
        split_info = {
            'total_samples': n_samples,
            'train_samples': len(self.X_train),
            'val_samples': len(self.X_val),
            'test_samples': len(self.X_test),
            'train_period': f"{self.X_train.index[0]} to {self.X_train.index[-1]}",
            'val_period': f"{self.X_val.index[0]} to {self.X_val.index[-1]}",
            'test_period': f"{self.X_test.index[0]} to {self.X_test.index[-1]}",
            'feature_count': len(feature_cols),
            'target_type': target_type
        }
        
        self.experiment_log['data_split'] = split_info
        
        self.logger.info(f"훈련 세트: {len(self.X_train)}개 ({split_info['train_period']})")
        self.logger.info(f"검증 세트: {len(self.X_val)}개 ({split_info['val_period']})")
        self.logger.info(f"테스트 세트: {len(self.X_test)}개 ({split_info['test_period']})")
        self.logger.info(f"특성 개수: {len(feature_cols)}")
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def scale_features(self, scaler_type='robust'):
        """특성 스케일링 - 데이터 누출 방지"""
        self.logger.info(f"=== 특성 스케일링 ({scaler_type}) ===")
        
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        # 훈련 데이터로만 스케일러 학습 (데이터 누출 방지)
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_val_scaled = scaler.transform(self.X_val)
        X_test_scaled = scaler.transform(self.X_test)
        
        # DataFrame으로 변환
        self.X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.X_train.columns, index=self.X_train.index)
        self.X_val_scaled = pd.DataFrame(X_val_scaled, columns=self.X_val.columns, index=self.X_val.index)
        self.X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.X_test.columns, index=self.X_test.index)
        
        self.scalers['main'] = scaler
        
        self.logger.info(f"{scaler_type} 스케일링 완료")
        
        return self.X_train_scaled, self.X_val_scaled, self.X_test_scaled
    
    def select_features(self, method='rfe', k=50):
        """
        특성 선택 - 오버피팅 방지
        
        Args:
            method: 'univariate', 'rfe', 'importance'
            k: 선택할 특성 개수
        """
        self.logger.info(f"=== 특성 선택 ({method}, k={k}) ===")
        
        if method == 'univariate':
            selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'rfe':
            # RFE with LogisticRegression (빠른 특성 선택용)
            estimator = LogisticRegression(random_state=42, max_iter=1000)
            selector = RFE(estimator=estimator, n_features_to_select=k)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # 훈련 데이터로만 특성 선택기 학습
        X_train_selected = selector.fit_transform(self.X_train_scaled, self.y_train)
        X_val_selected = selector.transform(self.X_val_scaled)
        X_test_selected = selector.transform(self.X_test_scaled)
        
        # 선택된 특성 이름
        if hasattr(selector, 'get_support'):
            selected_features = self.X_train_scaled.columns[selector.get_support()].tolist()
        else:
            selected_features = [f"feature_{i}" for i in range(k)]
        
        # DataFrame으로 변환
        self.X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features, index=self.X_train.index)
        self.X_val_selected = pd.DataFrame(X_val_selected, columns=selected_features, index=self.X_val.index)
        self.X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features, index=self.X_test.index)
        
        self.feature_selectors['main'] = selector
        
        # 특성 선택 정보 기록
        feature_selection_info = {
            'method': method,
            'original_features': len(self.X_train_scaled.columns),
            'selected_features': k,
            'selected_feature_names': selected_features[:10]  # 상위 10개만 기록
        }
        
        self.experiment_log['feature_selection'] = feature_selection_info
        
        self.logger.info(f"특성 선택 완료: {len(self.X_train_scaled.columns)} → {k}개")
        self.logger.info(f"선택된 상위 특성들: {selected_features[:10]}")
        
        return self.X_train_selected, self.X_val_selected, self.X_test_selected
    
    def save_experiment_log(self):
        """실험 로그 저장"""
        os.makedirs('models', exist_ok=True)
        
        self.experiment_log['end_time'] = datetime.now().isoformat()
        
        log_file = f"models/experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(log_file, 'w') as f:
            json.dump(self.experiment_log, f, indent=2, default=str)
        
        self.logger.info(f"실험 로그 저장: {log_file}")
        
        return log_file

# 사용 예시
if __name__ == "__main__":
    # 파이프라인 초기화
    pipeline = SPYMLPipeline()
    
    # 1. 데이터 로드 및 탐색
    data = pipeline.load_and_explore_data()
    
    # 2. 타겟 변수 생성
    binary_target, multi_target = pipeline.create_target_variable(prediction_horizon=1, threshold=0.001)
    
    # 3. 특성 공학
    features = pipeline.engineer_features(lookback_periods=[5, 10, 20, 50])
    
    # 4. ML 데이터 준비
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.prepare_ml_data(
        test_size=0.2, val_size=0.1, target_type='binary'
    )
    
    # 5. 특성 스케일링
    X_train_scaled, X_val_scaled, X_test_scaled = pipeline.scale_features(scaler_type='robust')
    
    # 6. 특성 선택  
    X_train_selected, X_val_selected, X_test_selected = pipeline.select_features(method='rfe', k=30)
    
    # 7. 실험 로그 저장
    log_file = pipeline.save_experiment_log()
    
    print("=== SPY ML 파이프라인 완료 ===")
    print(f"실험 로그: {log_file}")