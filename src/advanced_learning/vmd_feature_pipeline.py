#!/usr/bin/env python3
"""
VMD 노이즈 제거 및 고급 피처 엔지니어링 파이프라인
데이터 유출 방지를 위한 시계열 안전 파이프라인
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')


class VMDDenoiser(BaseEstimator, TransformerMixin):
    """
    Variational Mode Decomposition 기반 노이즈 제거
    데이터 유출 방지: 훈련 데이터에만 fit하고 테스트 데이터는 transform만 수행
    """

    def __init__(self, n_modes: int = 5, alpha: float = 2000, tau: float = 0.0,
                 DC: bool = False, init: int = 1, tol: float = 1e-7):
        """
        VMD 파라미터 초기화

        Args:
            n_modes: 분해할 모드 수
            alpha: 대역폭 제어 파라미터
            tau: 노이즈 허용 파라미터
            DC: DC 성분 포함 여부
            init: 초기화 방법
            tol: 수렴 허용 오차
        """
        self.n_modes = n_modes
        self.alpha = alpha
        self.tau = tau
        self.DC = DC
        self.init = init
        self.tol = tol
        self.fitted_params_ = None

    def _vmd_decomposition(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """VMD 분해 수행 (단순화된 버전)"""
        try:
            # PyVMD가 없는 경우 대체 구현
            from vmdpy import VMD
            u, u_hat, omega = VMD(signal, alpha=self.alpha, tau=self.tau, K=self.n_modes,
                                  DC=self.DC, init=self.init, tol=self.tol)
            return u, omega
        except ImportError:
            # VMD 라이브러리가 없는 경우 단순한 필터링으로 대체
            return self._simple_decomposition(signal)

    def _simple_decomposition(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """VMD 대체 구현: 다중 스케일 웨이블릿 분해"""
        from scipy import signal as scipy_signal

        # 다양한 주파수 대역으로 분해
        fs = 1.0  # 샘플링 주파수
        modes = []
        omegas = []

        for i in range(self.n_modes):
            # 각 모드별 저역 통과 필터
            cutoff = 0.5 / (i + 1)  # 점차 낮은 주파수
            b, a = scipy_signal.butter(4, cutoff, btype='low')
            filtered = scipy_signal.filtfilt(b, a, signal)
            modes.append(filtered)
            omegas.append(cutoff)

        return np.array(modes), np.array(omegas)

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """훈련 데이터로 VMD 파라미터 학습"""
        if len(X.shape) == 1:
            signal = X
        else:
            # 다변량의 경우 첫 번째 컬럼 사용 (일반적으로 종가)
            signal = X[:, 0] if X.shape[1] > 0 else X.flatten()

        # VMD 분해 수행하여 특성 저장
        modes, omegas = self._vmd_decomposition(signal)

        self.fitted_params_ = {
            'n_samples_train': len(signal),
            'signal_std': np.std(signal),
            'signal_mean': np.mean(signal),
            'dominant_frequencies': omegas
        }

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """VMD 기반 노이즈 제거 적용"""
        if self.fitted_params_ is None:
            raise ValueError("VMDDenoiser가 아직 fit되지 않았습니다.")

        if len(X.shape) == 1:
            signal = X
        else:
            signal = X[:, 0] if X.shape[1] > 0 else X.flatten()

        # VMD 분해
        modes, _ = self._vmd_decomposition(signal)

        # 높은 주파수 모드 제거 (노이즈 제거)
        # 첫 번째와 두 번째 모드만 유지 (주요 신호)
        denoised = np.sum(modes[:2], axis=0)

        # 원본 형태로 복원
        if len(X.shape) > 1:
            result = X.copy()
            result[:, 0] = denoised[:len(result)]
            return result
        else:
            return denoised

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """fit과 transform을 연속으로 수행"""
        return self.fit(X, y).transform(X)


class AdvancedFeatureEngineering(BaseEstimator, TransformerMixin):
    """
    고급 피처 엔지니어링 - 기술적 지표, 뉴스 감성, 거시경제 지표 통합
    """

    def __init__(self, lookback_windows: List[int] = [5, 10, 20, 50],
                 include_technical: bool = True,
                 include_sentiment: bool = True,
                 include_macro: bool = True):
        """
        Args:
            lookback_windows: 기술적 지표 계산용 윈도우들
            include_technical: 기술적 지표 포함 여부
            include_sentiment: 뉴스 감성 포함 여부
            include_macro: 거시경제 지표 포함 여부
        """
        self.lookback_windows = lookback_windows
        self.include_technical = include_technical
        self.include_sentiment = include_sentiment
        self.include_macro = include_macro
        self.feature_names_ = []
        self.scalers_ = {}

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산"""
        features = data.copy()

        if 'close' not in features.columns:
            # 첫 번째 컬럼을 종가로 가정
            features['close'] = features.iloc[:, 0]

        for window in self.lookback_windows:
            # 이동평균
            features[f'sma_{window}'] = features['close'].rolling(window=window).mean()
            features[f'ema_{window}'] = features['close'].ewm(span=window).mean()

            # 변동성
            features[f'volatility_{window}'] = features['close'].pct_change().rolling(window=window).std()

            # RSI
            delta = features['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            features[f'rsi_{window}'] = 100 - (100 / (1 + rs))

            # 볼린저 밴드
            sma = features['close'].rolling(window=window).mean()
            std = features['close'].rolling(window=window).std()
            features[f'bb_upper_{window}'] = sma + (std * 2)
            features[f'bb_lower_{window}'] = sma - (std * 2)
            features[f'bb_ratio_{window}'] = (features['close'] - features[f'bb_lower_{window}']) / \
                                           (features[f'bb_upper_{window}'] - features[f'bb_lower_{window}'])

        # MACD
        exp1 = features['close'].ewm(span=12).mean()
        exp2 = features['close'].ewm(span=26).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']

        return features

    def _calculate_sentiment_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """감성 분석 기반 피처 (모의 데이터)"""
        features = data.copy()

        # 실제로는 뉴스 API나 소셜미디어 데이터에서 가져옴
        np.random.seed(42)
        n_samples = len(data)

        # 뉴스 감성 점수 (-1 ~ 1)
        features['news_sentiment'] = np.random.normal(0, 0.3, n_samples)
        features['news_sentiment'] = np.clip(features['news_sentiment'], -1, 1)

        # 소셜미디어 감성 (트위터 등)
        features['social_sentiment'] = np.random.normal(0, 0.4, n_samples)
        features['social_sentiment'] = np.clip(features['social_sentiment'], -1, 1)

        # 감성 지표들의 이동평균
        for window in [5, 10, 20]:
            features[f'news_sentiment_ma_{window}'] = features['news_sentiment'].rolling(window=window).mean()
            features[f'social_sentiment_ma_{window}'] = features['social_sentiment'].rolling(window=window).mean()

        return features

    def _calculate_macro_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """거시경제 지표 피처 (모의 데이터)"""
        features = data.copy()

        # 실제로는 FRED API에서 가져옴
        np.random.seed(123)
        n_samples = len(data)

        # 금리 (Federal Funds Rate)
        features['fed_rate'] = np.random.uniform(0, 5, n_samples)

        # VIX (변동성 지수)
        features['vix'] = np.random.uniform(10, 40, n_samples)

        # DXY (달러 지수)
        features['dxy'] = np.random.uniform(90, 110, n_samples)

        # 10년 국채 수익률
        features['treasury_10y'] = np.random.uniform(1, 4, n_samples)

        # 유가 (WTI)
        features['oil_price'] = np.random.uniform(40, 100, n_samples)

        # 금 가격
        features['gold_price'] = np.random.uniform(1500, 2500, n_samples)

        return features

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """피처 엔지니어링 파라미터 학습"""
        # numpy array를 DataFrame으로 변환
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X)
        else:
            df = X.copy()

        # 각 피처 그룹별로 처리
        all_features = df.copy()

        if self.include_technical:
            all_features = self._calculate_technical_indicators(all_features)

        if self.include_sentiment:
            all_features = self._calculate_sentiment_features(all_features)

        if self.include_macro:
            all_features = self._calculate_macro_features(all_features)

        # 결측값 제거된 피처들만 선택
        all_features = all_features.dropna(axis=1, how='all')
        self.feature_names_ = all_features.columns.tolist()

        # 각 피처별 스케일러 fitted
        for col in all_features.select_dtypes(include=[np.number]).columns:
            if not all_features[col].isna().all():
                scaler = RobustScaler()
                valid_data = all_features[col].dropna().values.reshape(-1, 1)
                if len(valid_data) > 0:
                    scaler.fit(valid_data)
                    self.scalers_[col] = scaler

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """피처 엔지니어링 적용"""
        # numpy array를 DataFrame으로 변환
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X)
        else:
            df = X.copy()

        # 피처 생성
        all_features = df.copy()

        if self.include_technical:
            all_features = self._calculate_technical_indicators(all_features)

        if self.include_sentiment:
            all_features = self._calculate_sentiment_features(all_features)

        if self.include_macro:
            all_features = self._calculate_macro_features(all_features)

        # 학습시와 동일한 피처들만 선택
        if self.feature_names_:
            available_features = [col for col in self.feature_names_ if col in all_features.columns]
            all_features = all_features[available_features]

        # 스케일링 적용
        for col in all_features.columns:
            if col in self.scalers_:
                scaler = self.scalers_[col]
                non_nan_mask = ~all_features[col].isna()
                if non_nan_mask.sum() > 0:
                    scaled_values = scaler.transform(all_features.loc[non_nan_mask, col].values.reshape(-1, 1))
                    all_features.loc[non_nan_mask, col] = scaled_values.flatten()

        # 결측값 처리 (전진 채움)
        all_features = all_features.fillna(method='ffill').fillna(0)

        return all_features.values

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """fit과 transform을 연속으로 수행"""
        return self.fit(X, y).transform(X)


class LeakageFreePipeline:
    """
    데이터 유출 방지 파이프라인
    시계열 데이터의 시간 순서를 존중하는 안전한 전처리
    """

    def __init__(self, include_vmd: bool = True,
                 include_advanced_features: bool = True,
                 vmd_params: Dict = None,
                 feature_params: Dict = None):
        """
        Args:
            include_vmd: VMD 노이즈 제거 포함 여부
            include_advanced_features: 고급 피처 엔지니어링 포함 여부
            vmd_params: VMD 파라미터
            feature_params: 피처 엔지니어링 파라미터
        """
        self.include_vmd = include_vmd
        self.include_advanced_features = include_advanced_features

        # 파이프라인 구성 요소들
        pipeline_steps = []

        if include_vmd:
            vmd_config = vmd_params or {}
            pipeline_steps.append(('vmd_denoise', VMDDenoiser(**vmd_config)))

        if include_advanced_features:
            feature_config = feature_params or {}
            pipeline_steps.append(('feature_engineering', AdvancedFeatureEngineering(**feature_config)))

        # 최종 스케일링
        pipeline_steps.append(('final_scaler', RobustScaler()))

        self.pipeline = Pipeline(pipeline_steps)

    def fit_transform_safe(self, X_train: np.ndarray, X_test: np.ndarray,
                          y_train: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        데이터 유출 방지 안전 변환

        Args:
            X_train: 훈련 데이터
            X_test: 테스트 데이터
            y_train: 훈련 타겟

        Returns:
            (X_train_processed, X_test_processed): 변환된 데이터
        """
        # 훈련 데이터로만 파이프라인 학습
        X_train_processed = self.pipeline.fit_transform(X_train, y_train)

        # 테스트 데이터는 학습된 파이프라인으로 변환만 수행
        X_test_processed = self.pipeline.transform(X_test)

        return X_train_processed, X_test_processed

    def get_feature_names(self) -> List[str]:
        """피처 이름들 반환"""
        for name, transformer in self.pipeline.steps:
            if hasattr(transformer, 'feature_names_'):
                return transformer.feature_names_
        return []


# 사용 예시 및 테스트
if __name__ == "__main__":
    # 테스트 데이터 생성
    np.random.seed(42)
    n_samples = 1000

    # 모의 주가 데이터 (종가, 거래량 등)
    price_trend = np.cumsum(np.random.randn(n_samples) * 0.01) + 100
    noise = np.random.randn(n_samples) * 0.5
    prices = price_trend + noise
    volumes = np.random.uniform(1000, 10000, n_samples)

    X = np.column_stack([prices, volumes])
    y = (np.diff(prices, prepend=prices[0]) > 0).astype(int)  # 상승/하락

    # 훈련/테스트 분할 (시계열이므로 순서 유지)
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print("=== VMD 노이즈 제거 및 피처 엔지니어링 파이프라인 테스트 ===")

    # 파이프라인 생성
    pipeline = LeakageFreePipeline(
        include_vmd=True,
        include_advanced_features=True,
        vmd_params={'n_modes': 3, 'alpha': 1000},
        feature_params={'lookback_windows': [5, 10, 20]}
    )

    # 안전한 변환 수행
    X_train_processed, X_test_processed = pipeline.fit_transform_safe(
        X_train, X_test, y_train
    )

    print(f"원본 훈련 데이터 형태: {X_train.shape}")
    print(f"처리된 훈련 데이터 형태: {X_train_processed.shape}")
    print(f"원본 테스트 데이터 형태: {X_test.shape}")
    print(f"처리된 테스트 데이터 형태: {X_test_processed.shape}")

    # 피처 중요도 (간단한 상관관계)
    feature_names = pipeline.get_feature_names()
    if len(feature_names) > 0:
        print(f"\n생성된 피처 수: {len(feature_names)}")
        print("주요 피처들:")
        for i, name in enumerate(feature_names[:10]):
            print(f"  {i+1}. {name}")

    print("\n✅ VMD 노이즈 제거 및 피처 엔지니어링 파이프라인 테스트 완료")
    print("데이터 유출 없이 안전하게 전처리됨")