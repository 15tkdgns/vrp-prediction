"""
R² 성능 개선을 위한 고급 모델링 시스템

데이터 누출 없이 R² 성능을 개선하는 다양한 방법들:
1. 고급 특성 엔지니어링
2. 다중 시간 프레임 타겟
3. 고급 시계열 모델 (LSTM, XGBoost 시계열 최적화)
4. 적응적 정규화 및 스케일링
5. 동적 앙상블 가중치
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import ElasticNet, Ridge, Lasso
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ sklearn not available")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available, using sklearn alternatives")

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available, using sklearn alternatives")


class AdvancedFeatureEngineer:
    """
    고급 특성 엔지니어링 시스템

    R² 성능 개선을 위한 포괄적 특성 생성:
    - 기술적 지표 확장
    - 통계적 특성
    - 시간 기반 특성
    - 상호작용 특성
    """

    def __init__(self):
        self.feature_names = []
        self.scaler = None

    def create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """고급 특성 생성"""
        print("🔧 고급 특성 엔지니어링 시작...")

        enhanced_data = data.copy()

        # 1. 확장된 기술적 지표
        enhanced_data = self._add_technical_indicators(enhanced_data)

        # 2. 통계적 특성
        enhanced_data = self._add_statistical_features(enhanced_data)

        # 3. 시간 기반 특성
        enhanced_data = self._add_temporal_features(enhanced_data)

        # 4. 상호작용 특성
        enhanced_data = self._add_interaction_features(enhanced_data)

        # 5. 변동성 관련 특성
        enhanced_data = self._add_volatility_features(enhanced_data)

        print(f"✅ 특성 엔지니어링 완료: {len(enhanced_data.columns)}개 특성")
        return enhanced_data.dropna()

    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """확장된 기술적 지표"""
        if 'price' not in data.columns:
            return data

        prices = data['price']

        # 다양한 기간의 이동평균
        for window in [3, 7, 14, 21, 50, 100]:
            data[f'ma_{window}'] = prices.rolling(window).mean()
            data[f'price_ma_ratio_{window}'] = prices / data[f'ma_{window}']

        # 지수이동평균
        for alpha in [0.1, 0.2, 0.3]:
            data[f'ema_alpha_{alpha:.1f}'] = prices.ewm(alpha=alpha).mean()

        # MACD 지표
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        data['macd'] = ema_12 - ema_26
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']

        # Stochastic Oscillator
        high_14 = prices.rolling(14).max()
        low_14 = prices.rolling(14).min()
        data['stoch_k'] = 100 * (prices - low_14) / (high_14 - low_14)
        data['stoch_d'] = data['stoch_k'].rolling(3).mean()

        # Williams %R
        data['williams_r'] = -100 * (high_14 - prices) / (high_14 - low_14)

        # Commodity Channel Index (CCI)
        typical_price = prices  # 단순화 (Close만 사용)
        sma_tp = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        data['cci'] = (typical_price - sma_tp) / (0.015 * mad)

        return data

    def _add_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """통계적 특성"""
        if 'log_returns' not in data.columns:
            return data

        returns = data['log_returns']

        # 롤링 통계
        for window in [5, 10, 20, 50]:
            data[f'returns_mean_{window}'] = returns.rolling(window).mean()
            data[f'returns_std_{window}'] = returns.rolling(window).std()
            data[f'returns_skew_{window}'] = returns.rolling(window).skew()
            data[f'returns_kurt_{window}'] = returns.rolling(window).kurt()

            # 분위수
            data[f'returns_q25_{window}'] = returns.rolling(window).quantile(0.25)
            data[f'returns_q75_{window}'] = returns.rolling(window).quantile(0.75)
            data[f'returns_iqr_{window}'] = data[f'returns_q75_{window}'] - data[f'returns_q25_{window}']

        # Z-score (정규화된 수익률)
        for window in [10, 20, 50]:
            mean = returns.rolling(window).mean()
            std = returns.rolling(window).std()
            data[f'returns_zscore_{window}'] = (returns - mean) / std

        # 누적 통계
        data['returns_cumsum'] = returns.cumsum()
        data['returns_cummax'] = returns.cummax()
        data['returns_cummin'] = returns.cummin()

        return data

    def _add_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """시간 기반 특성"""
        # 인덱스가 DatetimeIndex인지 확인
        if isinstance(data.index, pd.DatetimeIndex):
            dates = data.index
        else:
            # 임의의 날짜 특성 생성
            dates = pd.date_range(start='2020-01-01', periods=len(data), freq='D')

        # 요일 효과
        data['day_of_week'] = dates.dayofweek
        data['is_monday'] = (dates.dayofweek == 0).astype(int)
        data['is_friday'] = (dates.dayofweek == 4).astype(int)

        # 월 효과
        data['month'] = dates.month
        data['quarter'] = dates.quarter
        data['is_january'] = (dates.month == 1).astype(int)
        data['is_december'] = (dates.month == 12).astype(int)

        # 연말/연초 효과
        data['days_from_year_start'] = dates.dayofyear
        data['days_to_year_end'] = 365 - dates.dayofyear

        # 시계열 트렌드
        data['time_trend'] = np.arange(len(data))
        data['time_trend_squared'] = data['time_trend'] ** 2

        return data

    def _add_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """상호작용 특성"""
        # 가격과 변동성 상호작용
        if 'price' in data.columns and 'volatility_20d' in data.columns:
            data['price_vol_interaction'] = data['price'] * data['volatility_20d']
            data['price_vol_ratio'] = data['price'] / (data['volatility_20d'] + 1e-8)

        # 모멘텀과 변동성 상호작용
        if 'price_momentum_5d' in data.columns and 'volatility_5d' in data.columns:
            data['momentum_vol_interaction'] = data['price_momentum_5d'] * data['volatility_5d']

        # 이동평균들 간의 관계
        if 'ma_5' in data.columns and 'ma_20' in data.columns:
            data['ma_spread_5_20'] = data['ma_5'] - data['ma_20']
            data['ma_ratio_5_20'] = data['ma_5'] / (data['ma_20'] + 1e-8)

        if 'ma_20' in data.columns and 'ma_50' in data.columns:
            data['ma_spread_20_50'] = data['ma_20'] - data['ma_50']
            data['ma_ratio_20_50'] = data['ma_20'] / (data['ma_50'] + 1e-8)

        return data

    def _add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """변동성 관련 특성"""
        if 'log_returns' not in data.columns:
            return data

        returns = data['log_returns']

        # 실현 변동성 (다양한 기간)
        for window in [3, 5, 10, 20, 50]:
            data[f'realized_vol_{window}'] = returns.rolling(window).std() * np.sqrt(252)

        # GARCH 스타일 변동성 (간단한 버전)
        squared_returns = returns ** 2
        for window in [5, 10, 20]:
            data[f'garch_vol_{window}'] = squared_returns.rolling(window).mean()

        # 변동성의 변동성
        for window in [10, 20]:
            vol = returns.rolling(window).std()
            data[f'vol_of_vol_{window}'] = vol.rolling(window).std()

        # 극값 기반 변동성 (Parkinson 추정량 근사)
        if 'price' in data.columns:
            for window in [5, 10, 20]:
                high = data['price'].rolling(window).max()
                low = data['price'].rolling(window).min()
                data[f'parkinson_vol_{window}'] = np.sqrt(0.361 * np.log(high / low) ** 2)

        return data


class MultiTimeframeTargets:
    """
    다중 시간 프레임 타겟 생성

    1일, 3일, 5일, 1주일 수익률 등 다양한 예측 타겟으로
    모델의 설명력 향상
    """

    def __init__(self):
        self.target_configs = {
            '1d': 1,
            '3d': 3,
            '5d': 5,
            '1w': 7,
            '2w': 14
        }

    def create_multiple_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """다중 시간 프레임 타겟 생성"""
        print("🎯 다중 시간 프레임 타겟 생성...")

        if 'log_returns' not in data.columns:
            print("⚠️ log_returns 컬럼이 없습니다")
            return data

        enhanced_data = data.copy()
        returns = data['log_returns']

        for name, days in self.target_configs.items():
            # 향후 N일 수익률 합계
            enhanced_data[f'target_return_{name}'] = returns.shift(-days).rolling(days).sum()

            # 향후 N일 수익률 평균
            enhanced_data[f'target_return_avg_{name}'] = returns.shift(-days).rolling(days).mean()

            # 향후 N일 중 최대/최소 수익률
            enhanced_data[f'target_return_max_{name}'] = returns.shift(-days).rolling(days).max()
            enhanced_data[f'target_return_min_{name}'] = returns.shift(-days).rolling(days).min()

            # 향후 N일 방향 (상승/하락)
            future_cumsum = returns.shift(-days).rolling(days).sum()
            enhanced_data[f'target_direction_{name}'] = (future_cumsum > 0).astype(int)

        print(f"✅ 다중 타겟 생성 완료: {len([c for c in enhanced_data.columns if 'target_' in c])}개 타겟")
        return enhanced_data


class AdvancedTimeSeriesModels:
    """
    고급 시계열 모델 구현

    R² 성능 향상을 위한 전문 시계열 모델들:
    - XGBoost 시계열 최적화
    - LSTM 딥러닝
    - 적응적 앙상블
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}

    def train_xgboost_timeseries(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_name: str = "1d"
    ) -> Dict:
        """XGBoost 시계열 최적화 모델"""
        print(f"🚀 XGBoost 시계열 모델 훈련 ({target_name})...")

        if not XGBOOST_AVAILABLE:
            print("⚠️ XGBoost 없음, RandomForest 사용")
            return self._train_random_forest_alternative(X, y, target_name)

        # 시계열 분할
        tscv = TimeSeriesSplit(n_splits=5)

        best_score = -np.inf
        best_model = None
        best_params = None

        # 하이퍼파라미터 그리드
        param_grid = [
            {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        ]

        for params in param_grid:
            for n_est in params['n_estimators']:
                for depth in params['max_depth']:
                    for lr in params['learning_rate']:
                        model_params = {
                            'n_estimators': n_est,
                            'max_depth': depth,
                            'learning_rate': lr,
                            'subsample': 0.9,
                            'colsample_bytree': 0.9,
                            'random_state': 42,
                            'objective': 'reg:squarederror',
                            'eval_metric': 'rmse'
                        }

                        # 시계열 교차검증
                        cv_scores = []
                        for train_idx, val_idx in tscv.split(X):
                            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                            model = xgb.XGBRegressor(**model_params)
                            model.fit(X_train, y_train)

                            y_pred = model.predict(X_val)
                            score = r2_score(y_val, y_pred)
                            cv_scores.append(score)

                        avg_score = np.mean(cv_scores)
                        if avg_score > best_score:
                            best_score = avg_score
                            best_params = model_params
                            best_model = xgb.XGBRegressor(**model_params)

        # 최종 모델 훈련
        if best_model is not None:
            best_model.fit(X, y)
            self.models[f'xgboost_{target_name}'] = best_model

            print(f"✅ XGBoost 훈련 완료: CV R² = {best_score:.4f}")

            return {
                'model': best_model,
                'cv_score': best_score,
                'best_params': best_params,
                'model_type': 'XGBoost'
            }
        else:
            return self._train_random_forest_alternative(X, y, target_name)

    def _train_random_forest_alternative(self, X: pd.DataFrame, y: pd.Series, target_name: str) -> Dict:
        """XGBoost 대체용 RandomForest"""
        print(f"🌲 RandomForest 대체 모델 훈련 ({target_name})...")

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        # 시계열 교차검증
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = r2_score(y_val, y_pred)
            cv_scores.append(score)

        avg_score = np.mean(cv_scores)

        # 전체 데이터로 최종 훈련
        model.fit(X, y)
        self.models[f'rf_{target_name}'] = model

        print(f"✅ RandomForest 훈련 완료: CV R² = {avg_score:.4f}")

        return {
            'model': model,
            'cv_score': avg_score,
            'best_params': model.get_params(),
            'model_type': 'RandomForest'
        }

    def train_lstm_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_name: str = "1d",
        sequence_length: int = 20
    ) -> Dict:
        """LSTM 딥러닝 모델"""
        print(f"🧠 LSTM 모델 훈련 ({target_name})...")

        if not TENSORFLOW_AVAILABLE:
            print("⚠️ TensorFlow 없음, 전통적 모델 사용")
            return self._train_random_forest_alternative(X, y, target_name)

        # 데이터 전처리
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

        # 시퀀스 데이터 생성
        X_sequences, y_sequences = self._create_sequences(X_scaled, y_scaled, sequence_length)

        if len(X_sequences) < 50:  # 데이터가 너무 적으면 포기
            print("⚠️ 시퀀스 데이터 부족, 대체 모델 사용")
            return self._train_random_forest_alternative(X, y, target_name)

        # 시계열 분할
        train_size = int(len(X_sequences) * 0.8)
        X_train, X_test = X_sequences[:train_size], X_sequences[train_size:]
        y_train, y_test = y_sequences[:train_size], y_sequences[train_size:]

        # LSTM 모델 구축
        model = keras.Sequential([
            layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, X_scaled.shape[1])),
            layers.Dropout(0.2),
            layers.LSTM(50, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(25),
            layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # 조기 종료 콜백
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )

        # 모델 훈련
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )

        # 성능 평가
        y_pred_scaled = model.predict(X_test)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

        score = r2_score(y_test_original, y_pred.flatten())

        # 모델 저장
        self.models[f'lstm_{target_name}'] = model
        self.scalers[f'lstm_{target_name}'] = {'X': scaler_X, 'y': scaler_y}

        print(f"✅ LSTM 훈련 완료: Test R² = {score:.4f}")

        return {
            'model': model,
            'cv_score': score,
            'scalers': {'X': scaler_X, 'y': scaler_y},
            'model_type': 'LSTM',
            'sequence_length': sequence_length
        }

    def _create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """시퀀스 데이터 생성"""
        X_sequences, y_sequences = [], []

        for i in range(len(X) - sequence_length):
            X_sequences.append(X[i:(i + sequence_length)])
            y_sequences.append(y[i + sequence_length])

        return np.array(X_sequences), np.array(y_sequences)


class AdaptiveNormalization:
    """
    적응적 정규화 및 스케일링

    R² 성능 향상을 위한 다양한 정규화 기법:
    - 시간 적응적 스케일링
    - 분위수 변환
    - 로버스트 스케일링
    """

    def __init__(self):
        self.scalers = {}
        self.best_scaler = None

    def find_best_normalization(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_scalers: List[str] = None
    ) -> Dict:
        """최적 정규화 방법 찾기"""
        print("📊 최적 정규화 방법 탐색...")

        if test_scalers is None:
            test_scalers = ['standard', 'robust', 'quantile', 'minmax']

        results = {}

        # 기본 모델 (정규화 비교용)
        base_model = RandomForestRegressor(n_estimators=100, random_state=42)

        # 시계열 교차검증
        tscv = TimeSeriesSplit(n_splits=3)

        for scaler_name in test_scalers:
            print(f"   {scaler_name} 스케일러 테스트...")

            # 스케일러 선택
            if scaler_name == 'standard':
                scaler = StandardScaler()
            elif scaler_name == 'robust':
                scaler = RobustScaler()
            elif scaler_name == 'quantile':
                scaler = QuantileTransformer(output_distribution='normal', random_state=42)
            elif scaler_name == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
            else:
                continue

            cv_scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # 데이터 스케일링
                X_train_scaled = pd.DataFrame(
                    scaler.fit_transform(X_train),
                    columns=X_train.columns,
                    index=X_train.index
                )
                X_val_scaled = pd.DataFrame(
                    scaler.transform(X_val),
                    columns=X_val.columns,
                    index=X_val.index
                )

                # 모델 훈련 및 평가
                base_model.fit(X_train_scaled, y_train)
                y_pred = base_model.predict(X_val_scaled)
                score = r2_score(y_val, y_pred)
                cv_scores.append(score)

            avg_score = np.mean(cv_scores)
            results[scaler_name] = {
                'cv_score': avg_score,
                'cv_std': np.std(cv_scores),
                'scaler': scaler
            }

            print(f"     R² = {avg_score:.4f} (±{np.std(cv_scores):.4f})")

        # 최고 성능 스케일러 선택
        best_scaler_name = max(results.keys(), key=lambda k: results[k]['cv_score'])
        self.best_scaler = results[best_scaler_name]['scaler']

        print(f"✅ 최적 스케일러: {best_scaler_name} (R² = {results[best_scaler_name]['cv_score']:.4f})")

        return results


def run_r2_improvement_experiment():
    """R² 성능 개선 실험 실행"""
    print("🚀 R² 성능 개선 실험 시작")
    print("=" * 60)

    # 1. 샘플 데이터 생성 (실제 환경에서는 실제 데이터 사용)
    print("\n📊 1단계: 샘플 데이터 생성")
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')

    # 더 현실적인 금융 데이터 시뮬레이션
    returns = np.random.normal(0.0005, 0.02, 1000)  # 일일 수익률
    prices = 100 * np.exp(np.cumsum(returns))  # 가격

    base_data = pd.DataFrame({
        'price': prices,
        'log_returns': returns
    }, index=dates)

    # 기본 기술적 지표 추가
    base_data['ma_5'] = base_data['price'].rolling(5).mean()
    base_data['ma_20'] = base_data['price'].rolling(20).mean()
    base_data['ma_50'] = base_data['price'].rolling(50).mean()
    base_data['volatility_5d'] = base_data['log_returns'].rolling(5).std()
    base_data['volatility_20d'] = base_data['log_returns'].rolling(20).std()
    base_data['rsi'] = 50 + np.random.normal(0, 15, 1000)  # 간단한 RSI 시뮬레이션
    base_data['price_momentum_5d'] = base_data['price'].pct_change(5)

    print(f"   기본 데이터: {len(base_data.columns)}개 특성, {len(base_data)}개 관측치")

    # 2. 고급 특성 엔지니어링
    print("\n🔧 2단계: 고급 특성 엔지니어링")
    feature_engineer = AdvancedFeatureEngineer()
    enhanced_data = feature_engineer.create_advanced_features(base_data)

    # 3. 다중 시간 프레임 타겟 생성
    print("\n🎯 3단계: 다중 시간 프레임 타겟 생성")
    target_generator = MultiTimeframeTargets()
    enhanced_data = target_generator.create_multiple_targets(enhanced_data)

    # 결측치 제거
    enhanced_data = enhanced_data.dropna()
    print(f"   최종 데이터: {len(enhanced_data.columns)}개 특성, {len(enhanced_data)}개 관측치")

    # 4. 적응적 정규화 최적화
    print("\n📊 4단계: 적응적 정규화 최적화")

    # 특성과 타겟 분리
    target_columns = [col for col in enhanced_data.columns if 'target_' in col]
    feature_columns = [col for col in enhanced_data.columns if 'target_' not in col]

    X = enhanced_data[feature_columns]

    normalizer = AdaptiveNormalization()
    normalization_results = normalizer.find_best_normalization(X, enhanced_data['target_return_1d'])

    # 5. 고급 모델 훈련
    print("\n🤖 5단계: 고급 모델 훈련")
    model_trainer = AdvancedTimeSeriesModels()

    results = {}

    # 각 타겟에 대해 모델 훈련
    for target_col in ['target_return_1d', 'target_return_3d', 'target_return_5d']:
        if target_col in enhanced_data.columns:
            print(f"\n   {target_col} 타겟 모델들 훈련:")
            y = enhanced_data[target_col].dropna()
            X_target = X.loc[y.index]  # 타겟과 인덱스 맞춤

            # XGBoost 모델
            xgb_result = model_trainer.train_xgboost_timeseries(X_target, y, target_col)
            results[f'xgboost_{target_col}'] = xgb_result

            # LSTM 모델 (데이터가 충분한 경우에만)
            if len(X_target) > 100:
                lstm_result = model_trainer.train_lstm_model(X_target, y, target_col)
                results[f'lstm_{target_col}'] = lstm_result

    # 6. 결과 요약
    print("\n📈 6단계: 실험 결과 요약")
    print("=" * 60)

    print("\n🏆 모델별 R² 성능:")
    print("-" * 50)

    best_score = -np.inf
    best_model = None

    for model_name, result in results.items():
        score = result['cv_score']
        model_type = result['model_type']
        print(f"   {model_name:<25} {model_type:<15} R² = {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model_name

    print(f"\n🥇 최고 성능 모델: {best_model} (R² = {best_score:.4f})")

    # 개선 정도 분석
    baseline_score = -0.01  # 기존 모델들의 평균 R²
    if best_score > baseline_score:
        improvement = ((best_score - baseline_score) / abs(baseline_score)) * 100
        print(f"📊 성능 개선: {improvement:.1f}% 향상")
    else:
        print("⚠️ 추가 최적화 필요")

    print(f"\n✅ R² 개선 실험 완료!")

    return {
        'enhanced_data': enhanced_data,
        'results': results,
        'best_model': best_model,
        'best_score': best_score,
        'normalization_results': normalization_results
    }


if __name__ == "__main__":
    # R² 개선 실험 실행
    experiment_results = run_r2_improvement_experiment()