#!/usr/bin/env python3
"""
고급 특성 공학 v3.0
- 전통적 기술적 분석 지표 고도화
- 대안 데이터 특성 통합
- 고급 통계적 특성 (entropy, fractal dimension, etc.)
- 시계열 분해 및 주파수 도메인 특성
- 교차 상관 및 상호작용 특성
- 동적 특성 (regime-aware features)
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import entropy
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.feature_selection import mutual_info_regression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available - some advanced features will be skipped")

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not available - using custom technical indicators")


class AdvancedFeatureEngineerV3:
    """고급 특성 공학 v3.0 클래스"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.feature_cache = {}
        
        if SKLEARN_AVAILABLE:
            self.scaler = RobustScaler()
            self.pca = PCA(n_components=0.95)  # 95% variance
        
    def _get_default_config(self) -> Dict:
        """기본 설정 반환"""
        return {
            'windows': [5, 10, 20, 50, 200],  # 다양한 시간 윈도우
            'technical_indicators': {
                'momentum': ['rsi', 'macd', 'stoch', 'williams_r', 'roc'],
                'trend': ['sma', 'ema', 'bollinger', 'adx', 'aroon'],
                'volume': ['obv', 'ad_line', 'chaikin_osc', 'volume_sma'],
                'volatility': ['atr', 'keltner', 'donchian', 'volatility_ratio']
            },
            'statistical_features': {
                'basic': ['mean', 'std', 'skewness', 'kurtosis'],
                'advanced': ['entropy', 'hurst_exponent', 'fractal_dimension'],
                'rolling': [10, 20, 50]
            },
            'frequency_features': {
                'fft_components': 10,
                'wavelet_levels': 3,
                'spectral_features': ['peak_freq', 'spectral_entropy', 'bandwidth']
            },
            'regime_detection': {
                'methods': ['volatility_regime', 'trend_regime', 'correlation_regime'],
                'lookback_periods': [20, 50, 100]
            },
            'interaction_features': {
                'cross_correlations': True,
                'feature_interactions': True,
                'polynomial_features': 2
            }
        }
    
    def engineer_all_features(self, 
                            price_data: pd.Series, 
                            volume_data: Optional[pd.Series] = None,
                            alternative_data: Optional[Dict] = None) -> pd.DataFrame:
        """모든 고급 특성 생성"""
        
        print("🔧 고급 특성 공학 v3.0 시작")
        print("=" * 50)
        
        features = {}
        
        # 1. 전통적 기술적 지표 (고도화)
        print("📊 기술적 지표 생성 중...")
        tech_features = self.create_technical_indicators(price_data, volume_data)
        features.update(tech_features)
        
        # 2. 고급 통계적 특성
        print("📈 통계적 특성 생성 중...")
        stat_features = self.create_statistical_features(price_data)
        features.update(stat_features)
        
        # 3. 주파수 도메인 특성
        print("🌊 주파수 특성 생성 중...")
        freq_features = self.create_frequency_features(price_data)
        features.update(freq_features)
        
        # 4. 시계열 분해 특성
        print("🔍 시계열 분해 중...")
        decomp_features = self.create_decomposition_features(price_data)
        features.update(decomp_features)
        
        # 5. 체제 감지 특성
        print("🎯 체제 감지 특성 생성 중...")
        regime_features = self.create_regime_features(price_data, volume_data)
        features.update(regime_features)
        
        # 6. 대안 데이터 통합 (있는 경우)
        if alternative_data:
            print("🌐 대안 데이터 통합 중...")
            alt_features = self.integrate_alternative_data(alternative_data)
            features.update(alt_features)
        
        # 7. 교차 상관 및 상호작용 특성
        print("🔄 상호작용 특성 생성 중...")
        interaction_features = self.create_interaction_features(features)
        features.update(interaction_features)
        
        # 8. 동적 특성 (시간에 따라 변화)
        print("⚡ 동적 특성 생성 중...")
        dynamic_features = self.create_dynamic_features(price_data, features)
        features.update(dynamic_features)
        
        # DataFrame으로 변환
        feature_df = pd.DataFrame(features, index=price_data.index[-len(list(features.values())[0]):])
        
        # 결측치 처리
        feature_df = self.handle_missing_values(feature_df)
        
        print(f"✅ 특성 공학 완료: {feature_df.shape[1]}개 특성 생성")
        return feature_df
    
    def create_technical_indicators(self, 
                                   price_data: pd.Series, 
                                   volume_data: Optional[pd.Series] = None) -> Dict:
        """고도화된 기술적 지표"""
        
        features = {}
        
        # 기본 가격 변환
        returns = price_data.pct_change().fillna(0)
        log_returns = np.log(price_data / price_data.shift(1)).fillna(0)
        
        # 1. 모멘텀 지표들
        for window in self.config['windows']:
            # RSI variations
            rsi = self._calculate_rsi(price_data, window)
            features[f'rsi_{window}'] = rsi
            features[f'rsi_divergence_{window}'] = rsi - rsi.rolling(window//2).mean()
            
            # MACD variations
            if window >= 20:
                macd, macd_signal = self._calculate_macd(price_data, window//2, window, window//3)
                features[f'macd_{window}'] = macd
                features[f'macd_signal_{window}'] = macd_signal
                features[f'macd_histogram_{window}'] = macd - macd_signal
            
            # Rate of Change with multiple timeframes
            features[f'roc_{window}'] = (price_data / price_data.shift(window) - 1).fillna(0)
            
            # Stochastic Oscillator
            if len(price_data) >= window:
                high = price_data.rolling(window).max()
                low = price_data.rolling(window).min()
                features[f'stoch_k_{window}'] = ((price_data - low) / (high - low)).fillna(0)
        
        # 2. 트렌드 지표들
        for window in self.config['windows']:
            # Moving averages
            sma = price_data.rolling(window).mean()
            ema = price_data.ewm(span=window).mean()
            
            features[f'sma_{window}'] = sma
            features[f'price_to_sma_{window}'] = (price_data / sma - 1).fillna(0)
            features[f'ema_{window}'] = ema
            features[f'price_to_ema_{window}'] = (price_data / ema - 1).fillna(0)
            
            # Moving average crossovers
            if window > 5:
                short_ma = price_data.rolling(window//4).mean()
                features[f'ma_cross_{window}'] = ((short_ma > sma).astype(int) * 2 - 1)
            
            # Bollinger Bands
            std = price_data.rolling(window).std()
            bb_upper = sma + 2 * std
            bb_lower = sma - 2 * std
            features[f'bb_position_{window}'] = ((price_data - bb_lower) / (bb_upper - bb_lower)).fillna(0.5)
            features[f'bb_width_{window}'] = ((bb_upper - bb_lower) / sma).fillna(0)
            
            # ADX (trend strength)
            if len(price_data) >= window + 14:
                adx = self._calculate_adx(price_data, price_data, price_data, window)
                features[f'adx_{window}'] = adx
        
        # 3. 변동성 지표들
        for window in self.config['windows']:
            # Average True Range
            atr = self._calculate_atr(price_data, price_data, price_data, window)
            features[f'atr_{window}'] = atr
            features[f'atr_ratio_{window}'] = atr / price_data
            
            # Realized volatility
            features[f'realized_vol_{window}'] = returns.rolling(window).std() * np.sqrt(252)
            
            # Volatility of volatility
            vol = returns.rolling(window//2).std()
            features[f'vol_of_vol_{window}'] = vol.rolling(window//2).std()
        
        # 4. 볼륨 지표들 (볼륨 데이터가 있는 경우)
        if volume_data is not None:
            for window in self.config['windows']:
                # On-Balance Volume
                obv = self._calculate_obv(price_data, volume_data)
                features[f'obv_sma_{window}'] = obv.rolling(window).mean()
                
                # Volume-Price Trend
                vpt = self._calculate_vpt(price_data, volume_data)
                features[f'vpt_{window}'] = vpt.rolling(window).mean()
                
                # Chaikin Money Flow
                if len(price_data) >= window:
                    cmf = self._calculate_cmf(price_data, price_data, price_data, volume_data, window)
                    features[f'cmf_{window}'] = cmf
        
        return features
    
    def create_statistical_features(self, price_data: pd.Series) -> Dict:
        """고급 통계적 특성"""
        
        features = {}
        returns = price_data.pct_change().fillna(0)
        log_returns = np.log(price_data / price_data.shift(1)).fillna(0)
        
        for window in self.config['statistical_features']['rolling']:
            # 기본 통계량
            features[f'returns_mean_{window}'] = returns.rolling(window).mean()
            features[f'returns_std_{window}'] = returns.rolling(window).std()
            features[f'returns_skew_{window}'] = returns.rolling(window).skew()
            features[f'returns_kurt_{window}'] = returns.rolling(window).kurt()
            
            # 고차 모멘트
            features[f'returns_sem_{window}'] = returns.rolling(window).sem()  # 표준 오차
            features[f'returns_var_{window}'] = returns.rolling(window).var()  # 분산
            
            # 분위수 기반 특성
            features[f'returns_median_{window}'] = returns.rolling(window).median()
            features[f'returns_q25_{window}'] = returns.rolling(window).quantile(0.25)
            features[f'returns_q75_{window}'] = returns.rolling(window).quantile(0.75)
            features[f'returns_iqr_{window}'] = (
                returns.rolling(window).quantile(0.75) - 
                returns.rolling(window).quantile(0.25)
            )
            
            # Entropy (정보 이론적 측도)
            entropy_values = []
            for i in range(window, len(returns)):
                window_returns = returns.iloc[i-window:i]
                if len(window_returns) > 0 and window_returns.std() > 0:
                    # 히스토그램 기반 엔트로피
                    hist, _ = np.histogram(window_returns, bins=10, density=True)
                    hist = hist[hist > 0]  # 0이 아닌 값들만
                    if len(hist) > 0:
                        ent = entropy(hist)
                    else:
                        ent = 0
                else:
                    ent = 0
                entropy_values.append(ent)
            
            # 패딩을 통해 길이 맞추기
            entropy_series = [0] * window + entropy_values
            features[f'entropy_{window}'] = entropy_series[-len(price_data):]
            
            # Hurst Exponent (추세 지속성)
            hurst_values = []
            for i in range(window, len(price_data)):
                window_prices = price_data.iloc[i-window:i]
                if len(window_prices) >= 10:
                    hurst = self._calculate_hurst_exponent(window_prices)
                else:
                    hurst = 0.5
                hurst_values.append(hurst)
            
            hurst_series = [0.5] * window + hurst_values
            features[f'hurst_{window}'] = hurst_series[-len(price_data):]
            
            # Autocorrelation
            for lag in [1, 2, 5, 10]:
                if lag < window:
                    autocorr_values = []
                    for i in range(window, len(returns)):
                        window_returns = returns.iloc[i-window:i]
                        if len(window_returns) > lag:
                            autocorr = window_returns.autocorr(lag=lag)
                            autocorr = autocorr if not np.isnan(autocorr) else 0
                        else:
                            autocorr = 0
                        autocorr_values.append(autocorr)
                    
                    autocorr_series = [0] * window + autocorr_values
                    features[f'autocorr_lag{lag}_{window}'] = autocorr_series[-len(price_data):]
        
        return features
    
    def create_frequency_features(self, price_data: pd.Series) -> Dict:
        """주파수 도메인 특성"""
        
        features = {}
        returns = price_data.pct_change().fillna(0)
        
        # FFT 기반 특성
        for window in [50, 100, 200]:
            if len(returns) >= window:
                fft_features = []
                spectral_entropy_values = []
                peak_freq_values = []
                
                for i in range(window, len(returns)):
                    window_returns = returns.iloc[i-window:i].values
                    
                    # FFT 계산
                    fft_values = fft(window_returns)
                    fft_magnitude = np.abs(fft_values)
                    freqs = fftfreq(len(window_returns))
                    
                    # 주요 주파수 성분들 (저주파 5개)
                    dominant_freqs = fft_magnitude[1:6]  # DC 성분 제외
                    fft_features.append(dominant_freqs)
                    
                    # Spectral Entropy
                    power_spectrum = fft_magnitude ** 2
                    power_spectrum = power_spectrum[power_spectrum > 0]
                    if len(power_spectrum) > 0:
                        power_spectrum = power_spectrum / np.sum(power_spectrum)
                        spec_entropy = entropy(power_spectrum)
                    else:
                        spec_entropy = 0
                    spectral_entropy_values.append(spec_entropy)
                    
                    # Peak Frequency
                    peak_idx = np.argmax(fft_magnitude[1:len(fft_magnitude)//2]) + 1
                    peak_freq = freqs[peak_idx] if peak_idx < len(freqs) else 0
                    peak_freq_values.append(abs(peak_freq))
                
                # FFT 성분들을 개별 특성으로 추가
                fft_array = np.array(fft_features)
                for j in range(min(5, fft_array.shape[1])):
                    fft_series = [0] * window + fft_array[:, j].tolist()
                    features[f'fft_comp_{j+1}_{window}'] = fft_series[-len(price_data):]
                
                # 스펙트럼 엔트로피
                spec_entropy_series = [0] * window + spectral_entropy_values
                features[f'spectral_entropy_{window}'] = spec_entropy_series[-len(price_data):]
                
                # 피크 주파수
                peak_freq_series = [0] * window + peak_freq_values
                features[f'peak_frequency_{window}'] = peak_freq_series[-len(price_data):]
        
        return features
    
    def create_decomposition_features(self, price_data: pd.Series) -> Dict:
        """시계열 분해 특성"""
        
        features = {}
        
        # 간단한 트렌드-사이클 분해
        for window in [20, 50, 100]:
            if len(price_data) >= window * 2:
                # 트렌드 성분 (긴 기간 이동평균)
                trend = price_data.rolling(window, center=True).mean()
                features[f'trend_{window}'] = trend.fillna(method='ffill').fillna(method='bfill')
                
                # 순환 성분 (가격 - 트렌드)
                cycle = price_data - trend
                features[f'cycle_{window}'] = cycle.fillna(0)
                
                # 잔차 성분 (단기 변동)
                short_ma = price_data.rolling(window//4).mean()
                residual = price_data - short_ma
                features[f'residual_{window}'] = residual.fillna(0)
                
                # 트렌드 강도
                trend_strength = abs(trend.pct_change()).rolling(window//2).mean()
                features[f'trend_strength_{window}'] = trend_strength.fillna(0)
        
        # 고주파/저주파 분리
        returns = price_data.pct_change().fillna(0)
        
        # 단순 고주파/저주파 필터
        for cutoff in [5, 10, 20]:
            # 저주파 (트렌드)
            low_freq = returns.rolling(cutoff).mean()
            features[f'low_freq_{cutoff}'] = low_freq.fillna(0)
            
            # 고주파 (노이즈)
            high_freq = returns - low_freq
            features[f'high_freq_{cutoff}'] = high_freq.fillna(0)
            
            # 고주파/저주파 비율
            features[f'hf_lf_ratio_{cutoff}'] = (
                high_freq.rolling(cutoff).std() / (low_freq.rolling(cutoff).std() + 1e-8)
            ).fillna(1)
        
        return features
    
    def create_regime_features(self, 
                              price_data: pd.Series, 
                              volume_data: Optional[pd.Series] = None) -> Dict:
        """체제 감지 특성"""
        
        features = {}
        returns = price_data.pct_change().fillna(0)
        
        for window in self.config['regime_detection']['lookback_periods']:
            # 1. 변동성 체제
            rolling_vol = returns.rolling(window).std()
            vol_median = rolling_vol.rolling(window*2).median()
            
            # 고변동성 체제 (1), 저변동성 체제 (0)
            high_vol_regime = (rolling_vol > vol_median * 1.5).astype(int)
            features[f'high_vol_regime_{window}'] = high_vol_regime.fillna(0)
            
            # 변동성 체제 변화
            vol_regime_change = high_vol_regime.diff().fillna(0)
            features[f'vol_regime_change_{window}'] = vol_regime_change
            
            # 2. 트렌드 체제
            price_ma = price_data.rolling(window).mean()
            trend_direction = (price_data > price_ma).astype(int)
            features[f'uptrend_regime_{window}'] = trend_direction.fillna(0)
            
            # 트렌드 강도
            trend_slope = (price_ma - price_ma.shift(window//4)) / (window//4)
            trend_strength = abs(trend_slope) / price_data
            features[f'trend_strength_{window}'] = trend_strength.fillna(0)
            
            # 3. 상관관계 체제 (시장과의 상관관계 변화)
            if len(returns) >= window * 2:
                rolling_correlation = []
                market_proxy = returns  # 자기 자신과의 지연 상관관계
                
                for i in range(window, len(returns)):
                    if i >= window * 2:
                        current_window = returns.iloc[i-window:i]
                        past_window = returns.iloc[i-window*2:i-window]
                        
                        if len(current_window) > 5 and len(past_window) > 5:
                            corr = np.corrcoef(current_window, past_window)[0, 1]
                            corr = corr if not np.isnan(corr) else 0
                        else:
                            corr = 0
                    else:
                        corr = 0
                    
                    rolling_correlation.append(corr)
                
                correlation_series = [0] * window + rolling_correlation
                features[f'correlation_regime_{window}'] = correlation_series[-len(price_data):]
        
        # 체제 전환 감지
        for window in [20, 50]:
            # 가격 체제 (상승/하락/횡보)
            returns_window = returns.rolling(window).mean()
            vol_window = returns.rolling(window).std()
            
            # 체제 분류
            regime = pd.Series(index=returns.index, dtype=int)
            regime[returns_window > vol_window] = 1  # 상승 체제
            regime[returns_window < -vol_window] = -1  # 하락 체제
            regime[(returns_window >= -vol_window) & (returns_window <= vol_window)] = 0  # 횡보
            
            features[f'price_regime_{window}'] = regime.fillna(0)
            
            # 체제 지속성
            regime_persistence = regime.rolling(window//4).apply(
                lambda x: len(x[x == x.iloc[-1]]) / len(x) if len(x) > 0 else 0
            )
            features[f'regime_persistence_{window}'] = regime_persistence.fillna(0)
        
        return features
    
    def integrate_alternative_data(self, alternative_data: Dict) -> Dict:
        """대안 데이터 통합"""
        
        features = {}
        
        try:
            # 뉴스 감성 특성
            if 'news_sentiment' in alternative_data:
                news = alternative_data['news_sentiment']
                features['news_sentiment_score'] = news.get('sentiment_score', 0.5)
                features['news_count'] = min(news.get('news_count', 0) / 20, 1.0)
                features['news_sentiment_raw'] = news.get('overall_sentiment', 0.0)
            
            # 경제 지표 특성
            if 'economic_indicators' in alternative_data:
                econ = alternative_data['economic_indicators']
                features['economic_score'] = econ.get('economic_score', 0.5)
                
                # VIX 특성
                vix_data = econ.get('vix', {})
                if vix_data:
                    features['vix_level'] = min((vix_data.get('value', 20) - 10) / 30, 1.0)
                    features['vix_change'] = vix_data.get('change', 0)
                    features['vix_high'] = 1 if vix_data.get('value', 20) > 25 else 0
            
            # 공포/탐욕 지수 특성
            if 'market_fear_greed' in alternative_data:
                fg = alternative_data['market_fear_greed']
                features['fear_greed_index'] = fg.get('composite_fear_greed', 50) / 100
                features['market_regime_fear'] = 1 if 'fear' in fg.get('market_regime', '') else 0
                features['market_regime_greed'] = 1 if 'greed' in fg.get('market_regime', '') else 0
                features['vix_score'] = fg.get('vix_score', 50) / 100
                features['market_momentum'] = fg.get('market_momentum', 50) / 100
            
            # 소셜 감성 특성
            if 'social_sentiment' in alternative_data:
                social = alternative_data['social_sentiment']
                features['social_sentiment'] = (social.get('overall_social_sentiment', 0) + 1) / 2
                features['social_volume'] = min(social.get('social_volume', 0) / 100, 1.0)
            
            # 교차 자산 신호
            if 'cross_asset_signals' in alternative_data:
                cross = alternative_data['cross_asset_signals']
                features['crypto_correlation'] = (cross.get('crypto_correlation', 0) + 1) / 2
                features['bond_equity_ratio'] = (cross.get('bond_equity_ratio', 0) + 1) / 2
                features['cross_asset_score'] = (cross.get('cross_asset_score', 0) + 1) / 2
            
            # 변동성 지표
            if 'volatility_indicators' in alternative_data:
                vol = alternative_data['volatility_indicators']
                features['realized_volatility'] = min(vol.get('realized_volatility', 20) / 50, 1.0)
                features['volatility_regime_high'] = 1 if vol.get('volatility_regime') == 'high' else 0
                features['volatility_risk_premium'] = vol.get('volatility_risk_premium', 0)
            
            # 종합 신호
            if 'composite_scores' in alternative_data:
                comp = alternative_data['composite_scores']
                features['alt_bullish_signal'] = comp.get('bullish_score', 0)
                features['alt_bearish_signal'] = comp.get('bearish_score', 0)
                features['alt_uncertainty'] = comp.get('uncertainty_score', 0)
                features['alt_confidence'] = comp.get('confidence_level', 0.5)
        
        except Exception as e:
            print(f"⚠️ 대안 데이터 통합 오류: {e}")
        
        return features
    
    def create_interaction_features(self, base_features: Dict) -> Dict:
        """상호작용 특성 생성"""
        
        features = {}
        
        if not self.config['interaction_features']['feature_interactions']:
            return features
        
        # 주요 특성들 선별
        feature_keys = list(base_features.keys())
        selected_keys = []
        
        # 패턴 기반으로 중요한 특성들 선별
        important_patterns = ['rsi', 'macd', 'bb_position', 'atr', 'vol_regime', 'trend']
        for key in feature_keys:
            if any(pattern in key for pattern in important_patterns):
                selected_keys.append(key)
        
        # 너무 많으면 제한
        selected_keys = selected_keys[:20]
        
        try:
            # 상호작용 특성 생성 (선택된 특성들만)
            for i, key1 in enumerate(selected_keys):
                for j, key2 in enumerate(selected_keys[i+1:], i+1):
                    if i < 10 and j < 15:  # 계산량 제한
                        val1 = np.array(base_features[key1])
                        val2 = np.array(base_features[key2])
                        
                        # 곱셈 상호작용
                        features[f'{key1}_x_{key2}'] = val1 * val2
                        
                        # 비율 상호작용 (분모가 0이 아닌 경우)
                        val2_safe = np.where(np.abs(val2) > 1e-8, val2, 1e-8)
                        features[f'{key1}_div_{key2}'] = val1 / val2_safe
        
        except Exception as e:
            print(f"⚠️ 상호작용 특성 생성 오류: {e}")
        
        return features
    
    def create_dynamic_features(self, price_data: pd.Series, base_features: Dict) -> Dict:
        """동적 특성 (시간에 따라 적응)"""
        
        features = {}
        returns = price_data.pct_change().fillna(0)
        
        # 적응형 특성들
        for window in [20, 50]:
            # 1. 적응형 베타 (시장 대비 민감도)
            market_returns = returns  # 자기 자신을 시장 프록시로 사용
            rolling_beta = []
            
            for i in range(window, len(returns)):
                y = returns.iloc[i-window:i]
                x = market_returns.iloc[i-window:i]
                
                if len(y) > 5 and x.std() > 0:
                    covariance = np.cov(y, x)[0, 1]
                    variance_x = np.var(x)
                    beta = covariance / variance_x if variance_x > 0 else 1.0
                else:
                    beta = 1.0
                
                rolling_beta.append(beta)
            
            beta_series = [1.0] * window + rolling_beta
            features[f'adaptive_beta_{window}'] = beta_series[-len(price_data):]
            
            # 2. 적응형 변동성 (GARCH 스타일)
            adaptive_vol = []
            ewm_factor = 2 / (window + 1)
            
            for i in range(1, len(returns)):
                if i == 1:
                    vol = abs(returns.iloc[i])
                else:
                    prev_vol = adaptive_vol[-1] if adaptive_vol else abs(returns.iloc[i-1])
                    vol = ewm_factor * (returns.iloc[i] ** 2) + (1 - ewm_factor) * prev_vol
                adaptive_vol.append(np.sqrt(vol))
            
            adaptive_vol_series = [0] + adaptive_vol
            features[f'adaptive_vol_{window}'] = adaptive_vol_series[-len(price_data):]
            
            # 3. 모멘텀 지속성 (적응형)
            momentum_persistence = []
            
            for i in range(window, len(returns)):
                recent_momentum = returns.iloc[i-window//4:i].mean()
                past_momentum = returns.iloc[i-window:i-window//4].mean()
                
                if abs(past_momentum) > 1e-8:
                    persistence = recent_momentum / past_momentum
                else:
                    persistence = 1.0
                
                momentum_persistence.append(persistence)
            
            persistence_series = [1.0] * window + momentum_persistence
            features[f'momentum_persistence_{window}'] = persistence_series[-len(price_data):]
        
        return features
    
    def handle_missing_values(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """결측치 처리"""
        
        # 1. Forward fill 후 backward fill
        feature_df = feature_df.fillna(method='ffill').fillna(method='bfill')
        
        # 2. 여전히 남은 결측치는 중간값으로 채움
        for col in feature_df.columns:
            if feature_df[col].isnull().any():
                median_val = feature_df[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                feature_df[col] = feature_df[col].fillna(median_val)
        
        # 3. 무한대 값 처리
        feature_df = feature_df.replace([np.inf, -np.inf], 0)
        
        return feature_df
    
    # =====================================================
    # Helper 메서드들
    # =====================================================
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """MACD 계산"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range 계산"""
        # 단순화: high, low, close가 같다고 가정
        tr = close.diff().abs()
        atr = tr.rolling(window=window).mean()
        return atr.fillna(0)
    
    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """ADX 계산 (단순화 버전)"""
        # 단순화된 ADX 계산
        price_change = close.diff().abs()
        adx = price_change.rolling(window=window).mean() / close * 100
        return adx.fillna(0)
    
    def _calculate_obv(self, prices: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume 계산"""
        direction = np.where(prices.diff() > 0, 1, np.where(prices.diff() < 0, -1, 0))
        obv = (direction * volume).cumsum()
        return obv
    
    def _calculate_vpt(self, prices: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume-Price Trend 계산"""
        vpt = (volume * prices.pct_change()).cumsum()
        return vpt.fillna(0)
    
    def _calculate_cmf(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
        """Chaikin Money Flow 계산"""
        # 단순화: high, low, close가 같다고 가정
        money_flow_volume = volume
        cmf = money_flow_volume.rolling(window=window).sum() / volume.rolling(window=window).sum()
        return cmf.fillna(0)
    
    def _calculate_hurst_exponent(self, prices: pd.Series) -> float:
        """Hurst Exponent 계산"""
        try:
            lags = range(2, min(20, len(prices)//2))
            tau = []
            
            for lag in lags:
                # 평균 절대 편차 계산
                pp = np.subtract(prices.iloc[lag:], prices.iloc[:-lag])
                mad = np.mean(np.abs(pp))
                tau.append(mad)
            
            if len(tau) > 1:
                # 로그-로그 회귀
                lags_log = np.log(list(lags))
                tau_log = np.log(tau)
                
                # 선형 회귀로 기울기 구하기
                poly = np.polyfit(lags_log, tau_log, 1)
                hurst = poly[0]
                
                # 0과 1 사이로 제한
                return max(0, min(1, hurst))
            else:
                return 0.5
        except:
            return 0.5


def main():
    """메인 테스트 함수"""
    
    print("🔧 고급 특성 공학 v3.0 테스트")
    print("=" * 60)
    
    try:
        # 테스트 데이터 생성
        import yfinance as yf
        
        print("📊 SPY 데이터 다운로드 중...")
        spy = yf.Ticker('SPY')
        data = spy.history(period='1y')  # 1년 데이터
        
        if data.empty:
            print("❌ 데이터 다운로드 실패")
            return
        
        price_data = data['Close']
        volume_data = data['Volume']
        
        print(f"✅ 데이터 로드 완료: {len(price_data)}일")
        
        # 특성 공학기 초기화
        engineer = AdvancedFeatureEngineerV3()
        
        # 모든 특성 생성
        feature_df = engineer.engineer_all_features(
            price_data=price_data,
            volume_data=volume_data,
            alternative_data=None  # 대안 데이터는 선택사항
        )
        
        print(f"\n📊 특성 공학 결과:")
        print(f"   생성된 특성: {feature_df.shape[1]}개")
        print(f"   데이터 기간: {feature_df.shape[0]}일")
        print(f"   결측치: {feature_df.isnull().sum().sum()}개")
        
        # 특성 카테고리별 요약
        categories = {
            'technical': ['rsi', 'macd', 'sma', 'ema', 'bb_', 'atr'],
            'statistical': ['mean', 'std', 'skew', 'kurt', 'entropy', 'hurst'],
            'frequency': ['fft', 'spectral', 'peak_freq'],
            'regime': ['regime', 'vol_regime', 'trend_regime'],
            'interaction': ['_x_', '_div_'],
            'dynamic': ['adaptive', 'momentum_persistence']
        }
        
        print(f"\n📋 특성 카테고리별 개수:")
        for category, patterns in categories.items():
            count = sum(1 for col in feature_df.columns if any(pattern in col for pattern in patterns))
            print(f"   {category:12}: {count:3}개")
        
        # 샘플 특성값 출력
        print(f"\n🔍 최근 특성값 샘플:")
        sample_features = feature_df.columns[:10]
        for feature in sample_features:
            recent_value = feature_df[feature].iloc[-1]
            print(f"   {feature:25}: {recent_value:8.4f}")
        
        print(f"\n✅ 고급 특성 공학 v3.0 테스트 완료!")
        
        return feature_df
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()