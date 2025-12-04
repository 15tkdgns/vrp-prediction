#!/usr/bin/env python3
"""
변동성 패턴 발견 및 분석
목표: 숨겨진 패턴을 찾아 R² 0.33 → 0.40+ 달성
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class VolatilityPatternDiscovery:
    """변동성 패턴 심층 분석"""

    def __init__(self, ticker="SPY", start_date="2015-01-01", end_date="2024-12-31"):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.patterns = {}

    def load_data(self):
        """데이터 로드 및 기본 변동성 계산"""
        print("📂 데이터 로드 중...")

        spy = yf.Ticker(self.ticker)
        df = spy.history(start=self.start_date, end=self.end_date)
        df.index = pd.to_datetime(df.index).tz_localize(None)

        # 기본 계산
        df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std()
        df['volatility_5d'] = df['returns'].rolling(5).std()
        df['volatility_60d'] = df['returns'].rolling(60).std()

        # 타겟
        df['target_vol'] = df['volatility_5d'].shift(-5)

        df = df.dropna()
        self.data = df

        print(f"✅ 데이터: {len(df)} 샘플")
        return True

    def analyze_autocorrelation(self):
        """자기상관 패턴 분석"""
        print("\n🔍 패턴 1: 변동성 자기상관 분석...")

        vol = self.data['volatility'].values

        # 다양한 lag에 대한 자기상관
        lags = range(1, 61)
        autocorrs = [pd.Series(vol).autocorr(lag=lag) for lag in lags]

        # 강한 자기상관 lag 찾기
        strong_lags = [(lag, corr) for lag, corr in zip(lags, autocorrs) if abs(corr) > 0.3]

        print(f"   강한 자기상관 lag: {len(strong_lags)}개")
        for lag, corr in strong_lags[:10]:
            print(f"     Lag {lag}: {corr:.4f}")

        self.patterns['strong_autocorr_lags'] = strong_lags

        # 발견: 특정 lag에서 강한 상관관계 → 해당 lag 특성 추가 가능
        return autocorrs

    def analyze_volatility_cycles(self):
        """변동성 사이클 패턴 분석"""
        print("\n🔍 패턴 2: 변동성 사이클 분석...")

        vol = self.data['volatility'].values

        # Peak detection (고변동 사이클)
        peaks, peak_props = find_peaks(vol, distance=20, prominence=0.001)

        # Trough detection (저변동 사이클)
        troughs, trough_props = find_peaks(-vol, distance=20, prominence=0.001)

        print(f"   고변동 피크: {len(peaks)}개")
        print(f"   저변동 저점: {len(troughs)}개")

        # 사이클 길이 계산
        if len(peaks) > 1:
            cycle_lengths = np.diff(peaks)
            avg_cycle = np.mean(cycle_lengths)
            print(f"   평균 사이클 길이: {avg_cycle:.1f}일")

            self.patterns['avg_cycle_length'] = avg_cycle

        # 발견: 고변동 → 저변동 전환 패턴 존재
        return peaks, troughs

    def analyze_volatility_persistence(self):
        """변동성 지속성 패턴"""
        print("\n🔍 패턴 3: 변동성 지속성 분석...")

        df = self.data.copy()

        # 현재 변동성 vs 미래 변동성
        df['vol_change'] = df['target_vol'] - df['volatility_5d']

        # 변동성이 높을 때 vs 낮을 때 지속성 차이
        high_vol_mask = df['volatility_5d'] > df['volatility_5d'].median()

        persistence_high = df[high_vol_mask]['vol_change'].mean()
        persistence_low = df[~high_vol_mask]['vol_change'].mean()

        print(f"   고변동 시 변화량: {persistence_high:.6f}")
        print(f"   저변동 시 변화량: {persistence_low:.6f}")
        print(f"   비대칭성: {abs(persistence_high/persistence_low):.2f}배")

        self.patterns['persistence_asymmetry'] = abs(persistence_high/persistence_low)

        # 발견: 고변동은 빠르게 하락, 저변동은 천천히 상승
        return persistence_high, persistence_low

    def analyze_return_volatility_relation(self):
        """수익률-변동성 관계 패턴"""
        print("\n🔍 패턴 4: 수익률-변동성 비선형 관계...")

        df = self.data.copy()

        # 누적 수익률 구간별 미래 변동성
        df['cumulative_returns_5d'] = df['returns'].rolling(5).sum()

        # 수익률을 10개 구간으로 분할
        df['return_decile'] = pd.qcut(df['cumulative_returns_5d'], q=10, labels=False, duplicates='drop')

        # 각 구간별 평균 미래 변동성
        decile_vol = df.groupby('return_decile')['target_vol'].mean()

        print(f"   수익률 구간별 미래 변동성:")
        for decile, vol in decile_vol.items():
            print(f"     Decile {decile}: {vol:.6f}")

        # U-shape 확인 (극단 수익률 → 높은 변동성)
        edge_vol = (decile_vol.iloc[0] + decile_vol.iloc[-1]) / 2
        center_vol = decile_vol.iloc[4:6].mean()
        u_shape_ratio = edge_vol / center_vol

        print(f"   U-shape 비율: {u_shape_ratio:.2f}")

        self.patterns['u_shape_ratio'] = u_shape_ratio

        # 발견: 극단 수익률(상승/하락) → 높은 미래 변동성
        return decile_vol

    def analyze_volume_volatility_relation(self):
        """거래량-변동성 관계"""
        print("\n🔍 패턴 5: 거래량-변동성 관계...")

        df = self.data.copy()

        # 거래량 이상치 (평균 대비)
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['volume_spike'] = (df['volume_ratio'] > 1.5).astype(int)

        # 거래량 급증 후 변동성 변화
        spike_data = df[df['volume_spike'] == 1]
        normal_data = df[df['volume_spike'] == 0]

        spike_future_vol = spike_data['target_vol'].mean()
        normal_future_vol = normal_data['target_vol'].mean()

        print(f"   거래량 급증 후 변동성: {spike_future_vol:.6f}")
        print(f"   정상 거래량 후 변동성: {normal_future_vol:.6f}")
        print(f"   비율: {spike_future_vol/normal_future_vol:.2f}배")

        self.patterns['volume_spike_effect'] = spike_future_vol / normal_future_vol

        # 발견: 거래량 급증 → 미래 변동성 증가
        return spike_future_vol, normal_future_vol

    def analyze_intraday_patterns(self):
        """일중 패턴 (High-Low spread)"""
        print("\n🔍 패턴 6: 일중 가격 범위 패턴...")

        df = self.data.copy()

        # True Range (Wilder)
        df['high_low'] = df['High'] - df['Low']
        df['high_close'] = abs(df['High'] - df['Close'].shift(1))
        df['low_close'] = abs(df['Low'] - df['Close'].shift(1))
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)

        # ATR (Average True Range)
        df['atr_14'] = df['true_range'].rolling(14).mean()
        df['atr_ratio'] = df['atr_14'] / df['Close']

        # ATR vs 미래 변동성 상관관계
        corr = df[['atr_ratio', 'target_vol']].corr().iloc[0, 1]

        print(f"   ATR vs 미래 변동성 상관: {corr:.4f}")

        self.patterns['atr_correlation'] = corr

        # 발견: ATR은 미래 변동성의 강력한 선행지표
        return corr

    def analyze_gap_patterns(self):
        """갭 패턴 (overnight 변동)"""
        print("\n🔍 패턴 7: Overnight Gap 패턴...")

        df = self.data.copy()

        # Gap = Open - Previous Close
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['gap_size'] = df['gap'].abs()

        # 큰 갭 발생 후 변동성
        large_gap_mask = df['gap_size'] > df['gap_size'].quantile(0.9)

        large_gap_vol = df[large_gap_mask]['target_vol'].mean()
        normal_gap_vol = df[~large_gap_mask]['target_vol'].mean()

        print(f"   큰 갭 후 변동성: {large_gap_vol:.6f}")
        print(f"   정상 갭 후 변동성: {normal_gap_vol:.6f}")
        print(f"   비율: {large_gap_vol/normal_gap_vol:.2f}배")

        self.patterns['gap_effect'] = large_gap_vol / normal_gap_vol

        return large_gap_vol, normal_gap_vol

    def analyze_skewness_kurtosis(self):
        """수익률 분포 왜도/첨도 패턴"""
        print("\n🔍 패턴 8: 수익률 분포 형태 패턴...")

        df = self.data.copy()

        # 롤링 왜도 (skewness)
        df['rolling_skew'] = df['returns'].rolling(20).skew()

        # 롤링 첨도 (kurtosis)
        df['rolling_kurt'] = df['returns'].rolling(20).kurt()

        # 고첨도 (fat tail) → 미래 변동성
        high_kurt_mask = df['rolling_kurt'] > df['rolling_kurt'].quantile(0.75)

        high_kurt_vol = df[high_kurt_mask]['target_vol'].mean()
        normal_kurt_vol = df[~high_kurt_mask]['target_vol'].mean()

        print(f"   고첨도 후 변동성: {high_kurt_vol:.6f}")
        print(f"   정상 첨도 후 변동성: {normal_kurt_vol:.6f}")
        print(f"   비율: {high_kurt_vol/normal_kurt_vol:.2f}배")

        self.patterns['kurtosis_effect'] = high_kurt_vol / normal_kurt_vol

        # 발견: Fat tail (극단 이벤트) → 높은 미래 변동성
        return high_kurt_vol, normal_kurt_vol

    def analyze_volatility_of_volatility(self):
        """변동성의 변동성 (vol-of-vol)"""
        print("\n🔍 패턴 9: 변동성의 변동성...")

        df = self.data.copy()

        # 변동성의 변동성
        df['vol_of_vol'] = df['volatility_5d'].rolling(20).std()

        # Vol-of-vol vs 미래 변동성
        corr = df[['vol_of_vol', 'target_vol']].corr().iloc[0, 1]

        print(f"   Vol-of-vol vs 미래 변동성 상관: {corr:.4f}")

        # 높은 vol-of-vol → 불안정 시장
        high_vov_mask = df['vol_of_vol'] > df['vol_of_vol'].median()

        high_vov_vol = df[high_vov_mask]['target_vol'].mean()
        low_vov_vol = df[~high_vov_mask]['target_vol'].mean()

        print(f"   높은 vol-of-vol 후: {high_vov_vol:.6f}")
        print(f"   낮은 vol-of-vol 후: {low_vov_vol:.6f}")
        print(f"   비율: {high_vov_vol/low_vov_vol:.2f}배")

        self.patterns['vol_of_vol_effect'] = high_vov_vol / low_vov_vol

        return corr

    def analyze_momentum_volatility(self):
        """모멘텀-변동성 관계"""
        print("\n🔍 패턴 10: 가격 모멘텀과 변동성...")

        df = self.data.copy()

        # 가격 모멘텀 (20일)
        df['momentum_20'] = (df['Close'] / df['Close'].shift(20) - 1)

        # 강한 모멘텀 vs 약한 모멘텀
        strong_momentum_mask = df['momentum_20'].abs() > df['momentum_20'].abs().quantile(0.75)

        strong_mom_vol = df[strong_momentum_mask]['target_vol'].mean()
        weak_mom_vol = df[~strong_momentum_mask]['target_vol'].mean()

        print(f"   강한 모멘텀 후 변동성: {strong_mom_vol:.6f}")
        print(f"   약한 모멘텀 후 변동성: {weak_mom_vol:.6f}")
        print(f"   비율: {strong_mom_vol/weak_mom_vol:.2f}배")

        self.patterns['momentum_effect'] = strong_mom_vol / weak_mom_vol

        return strong_mom_vol, weak_mom_vol

    def summarize_patterns(self):
        """발견된 패턴 요약"""
        print("\n" + "="*60)
        print("📊 발견된 핵심 패턴 요약")
        print("="*60)

        # 효과 크기 순 정렬
        effects = {
            'Volume Spike Effect': self.patterns.get('volume_spike_effect', 1.0),
            'Gap Effect': self.patterns.get('gap_effect', 1.0),
            'Kurtosis Effect': self.patterns.get('kurtosis_effect', 1.0),
            'Vol-of-Vol Effect': self.patterns.get('vol_of_vol_effect', 1.0),
            'Momentum Effect': self.patterns.get('momentum_effect', 1.0),
            'U-Shape Ratio': self.patterns.get('u_shape_ratio', 1.0),
            'Persistence Asymmetry': self.patterns.get('persistence_asymmetry', 1.0),
        }

        sorted_effects = sorted(effects.items(), key=lambda x: abs(x[1]-1.0), reverse=True)

        print("\n효과 크기 순위:")
        for i, (name, effect) in enumerate(sorted_effects, 1):
            deviation = abs(effect - 1.0) * 100
            print(f"{i}. {name:25s}: {effect:.2f}배 (편차 {deviation:.1f}%)")

        # 새로운 특성 제안
        print("\n" + "="*60)
        print("💡 새로운 특성 제안")
        print("="*60)

        new_features = [
            "1. ATR (Average True Range) 기반 특성",
            "2. Volume spike 지표",
            "3. Overnight gap size",
            "4. Rolling skewness/kurtosis",
            "5. Volatility-of-volatility",
            "6. 수익률 극단값 카운트",
            "7. 가격 모멘텀 강도",
            "8. 특정 lag 자기상관 특성",
            "9. 변동성 사이클 위치",
            "10. U-shape 수익률 변환"
        ]

        for feature in new_features:
            print(f"   {feature}")

        return sorted_effects

    def run_analysis(self):
        """전체 분석 실행"""
        print("="*60)
        print("🔬 변동성 패턴 심층 분석")
        print("="*60)

        self.load_data()

        # 10가지 패턴 분석
        self.analyze_autocorrelation()
        self.analyze_volatility_cycles()
        self.analyze_volatility_persistence()
        self.analyze_return_volatility_relation()
        self.analyze_volume_volatility_relation()
        self.analyze_intraday_patterns()
        self.analyze_gap_patterns()
        self.analyze_skewness_kurtosis()
        self.analyze_volatility_of_volatility()
        self.analyze_momentum_volatility()

        # 요약
        sorted_effects = self.summarize_patterns()

        print("\n" + "="*60)
        print("✅ 패턴 분석 완료")
        print("="*60)

        return sorted_effects

if __name__ == "__main__":
    analyzer = VolatilityPatternDiscovery()
    analyzer.run_analysis()
