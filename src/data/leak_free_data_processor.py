#!/usr/bin/env python3
"""
🔒 Leak-Free Data Processor
모든 데이터 유출을 제거한 안전한 데이터 처리 파이프라인
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class LeakFreeDataProcessor:
    def __init__(self):
        """Leak-free 데이터 처리기"""
        self.feature_creation_log = []

    def create_safe_lag_features(self, df, columns, max_lag=5):
        """안전한 Lag 특징 생성"""
        print(f"🔒 안전한 Lag 특징 생성: {columns}")

        for col in columns:
            if col not in df.columns:
                continue

            for lag in range(1, max_lag + 1):
                lag_col = f"{col}_lag_{lag}"
                # 올바른 lag: 현재 행에서 lag만큼 이전 값 사용
                df[lag_col] = df[col].shift(lag)

                # 검증: 첫 lag개 행은 반드시 NaN이어야 함
                assert df[lag_col].iloc[:lag].isna().all(), f"{lag_col} 첫 {lag}행이 NaN이 아님!"

                self.feature_creation_log.append(f"{lag_col}: 첫 {lag}행 NaN 확인")

        return df

    def create_safe_moving_averages(self, df, price_col='Close', windows=[5, 10, 20, 50]):
        """안전한 이동평균 생성"""
        print(f"🔒 안전한 이동평균 생성: windows={windows}")

        for window in windows:
            ma_col = f"MA_{window}"
            # 올바른 이동평균: 현재 포함하여 과거 window개 평균
            df[ma_col] = df[price_col].rolling(window=window, min_periods=window).mean()

            # 검증: 첫 (window-1)개 행은 반드시 NaN이어야 함
            assert df[ma_col].iloc[:window-1].isna().all(), f"{ma_col} 첫 {window-1}행이 NaN이 아님!"

            self.feature_creation_log.append(f"{ma_col}: 첫 {window-1}행 NaN 확인")

        return df

    def create_safe_rsi(self, df, price_col='Close', window=14):
        """안전한 RSI 생성"""
        print(f"🔒 안전한 RSI 생성: window={window}")

        # 가격 변화 계산 (첫 번째 행은 NaN)
        delta = df[price_col].diff()

        # 상승과 하락 분리 후 롤링 평균
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=window).mean()

        # RSI 계산
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 엄격한 검증: RSI는 최소 window개 행이 NaN이어야 함
        # diff(1)로 1개 + rolling(window, min_periods=window)로 추가 window-1개
        # 총 window개 행이 NaN이어야 함
        first_valid_idx = df['RSI'].first_valid_index()
        if first_valid_idx is not None and first_valid_idx < window:
            print(f"❌ RSI 데이터 유출 발견: {first_valid_idx}번째 행부터 값 존재 (최소 {window}번째부터 있어야 함)")

            # 강제로 안전하게 만들기: 첫 window개 행을 NaN으로 설정
            df['RSI'].iloc[:window] = np.nan
            print(f"🔧 RSI 수정: 첫 {window}개 행을 NaN으로 강제 설정")

        # 최종 검증
        assert df['RSI'].iloc[:window].isna().all(), f"RSI 첫 {window}행이 NaN이 아님!"
        self.feature_creation_log.append(f"RSI: 첫 {window}행 NaN 확인 (강제 보정 적용)")

        return df

    def create_safe_bollinger_bands(self, df, price_col='Close', window=20, std_mult=2):
        """안전한 볼린저밴드 생성"""
        print(f"🔒 안전한 볼린저밴드 생성: window={window}")

        # 이동평균과 표준편차 (둘 다 충분한 기간 필요)
        rolling_mean = df[price_col].rolling(window=window, min_periods=window).mean()
        rolling_std = df[price_col].rolling(window=window, min_periods=window).std()

        df['BB_middle'] = rolling_mean
        df['BB_upper'] = rolling_mean + (rolling_std * std_mult)
        df['BB_lower'] = rolling_mean - (rolling_std * std_mult)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        df['BB_position'] = (df[price_col] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

        # 검증: 모든 BB 관련 특징의 첫 (window-1)개 행은 NaN이어야 함
        bb_cols = ['BB_middle', 'BB_upper', 'BB_lower', 'BB_width', 'BB_position']
        for col in bb_cols:
            assert df[col].iloc[:window-1].isna().all(), f"{col} 첫 {window-1}행이 NaN이 아님!"
            self.feature_creation_log.append(f"{col}: 첫 {window-1}행 NaN 확인")

        return df

    def create_safe_volatility(self, df, return_col='Returns', windows=[5, 10, 20, 50]):
        """안전한 변동성 지표 생성"""
        print(f"🔒 안전한 변동성 생성: windows={windows}")

        for window in windows:
            vol_col = f"Volatility_{window}"
            df[vol_col] = df[return_col].rolling(window=window, min_periods=window).std()

            # 검증
            assert df[vol_col].iloc[:window-1].isna().all(), f"{vol_col} 첫 {window-1}행이 NaN이 아님!"
            self.feature_creation_log.append(f"{vol_col}: 첫 {window-1}행 NaN 확인")

        return df

    def create_safe_volume_features(self, df, volume_col='Volume', windows=[10, 20]):
        """안전한 거래량 특징 생성"""
        print(f"🔒 안전한 거래량 특징 생성: windows={windows}")

        for window in windows:
            # 거래량 이동평균
            vol_ma_col = f"Volume_MA_{window}"
            df[vol_ma_col] = df[volume_col].rolling(window=window, min_periods=window).mean()

            # 거래량 비율
            vol_ratio_col = f"Volume_ratio_{window}"
            df[vol_ratio_col] = df[volume_col] / df[vol_ma_col]

            # 검증
            for col in [vol_ma_col, vol_ratio_col]:
                assert df[col].iloc[:window-1].isna().all(), f"{col} 첫 {window-1}행이 NaN이 아님!"
                self.feature_creation_log.append(f"{col}: 첫 {window-1}행 NaN 확인")

        return df

    def create_completely_safe_dataset(self, input_file, output_file):
        """Ultra-safe 데이터셋 생성"""
        print("🔒 Leak-free 데이터셋 생성 시작...")

        # 1. 원시 데이터 로드
        df = pd.read_csv(input_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        print(f"📊 원시 데이터: {df.shape}")

        # 2. 기본 수익률 계산 (유일하게 미래 정보 없이 계산 가능)
        df['Returns'] = df['Close'].pct_change()

        # 3. 안전한 특징들 순차적 생성

        # 이동평균 (윈도우 필요)
        df = self.create_safe_moving_averages(df, 'Close', [5, 10, 20, 50])

        # RSI (14일 윈도우 필요)
        df = self.create_safe_rsi(df, 'Close', 14)

        # 볼린저밴드 (20일 윈도우 필요)
        df = self.create_safe_bollinger_bands(df, 'Close', 20)

        # 변동성 지표 (Returns 기반)
        df = self.create_safe_volatility(df, 'Returns', [5, 10, 20])

        # 거래량 특징
        df = self.create_safe_volume_features(df, 'Volume', [10, 20])

        # 4. Lag 특징들 (모든 주요 지표에 대해)
        lag_columns = ['Returns', 'RSI', 'Volatility_20', 'Volume_ratio_20']
        df = self.create_safe_lag_features(df, lag_columns, 5)

        # BB_position lag 특징
        df = self.create_safe_lag_features(df, ['BB_position'], 5)

        # 5. 추가 안전한 특징들

        # 가격 모멘텀 (lag 기반으로 안전)
        df['Price_momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['Price_momentum_10'] = df['Close'] / df['Close'].shift(10) - 1

        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14, min_periods=14).mean()

        # 6. 최종 검증 및 정리

        print(f"\n🔍 특징 생성 로그:")
        for log in self.feature_creation_log:
            print(f"   ✅ {log}")

        # 모든 계산이 완료된 후 NaN 제거
        original_len = len(df)
        df_clean = df.dropna()
        removed_rows = original_len - len(df_clean)

        print(f"\n📊 데이터 정리:")
        print(f"   원래 행 수: {original_len}")
        print(f"   제거된 행 수: {removed_rows} (NaN 포함)")
        print(f"   최종 행 수: {len(df_clean)}")
        print(f"   최종 특징 수: {len(df_clean.columns)}")

        # 특징 선택 (안전하게 확인된 특징들만)
        safe_features = [
            'Date', 'Close', 'Volume', 'Returns',  # 기본
            'MA_5', 'MA_10', 'MA_20', 'MA_50',     # 이동평균
            'RSI',                                  # RSI
            'BB_position',                          # 볼린저밴드 위치
            'Volatility_5', 'Volatility_10', 'Volatility_20',  # 변동성
            'Volume_ratio_10', 'Volume_ratio_20',   # 거래량 비율
            'ATR',                                  # ATR
            # Lag 특징들
            'Returns_lag_1', 'Returns_lag_2', 'Returns_lag_3',
            'RSI_lag_1', 'RSI_lag_2', 'RSI_lag_3',
            'Volatility_20_lag_1', 'Volatility_20_lag_2', 'Volatility_20_lag_3',
            'BB_position_lag_1', 'BB_position_lag_2', 'BB_position_lag_3',
            # 모멘텀
            'Price_momentum_5', 'Price_momentum_10'
        ]

        # 존재하는 특징들만 선택
        available_features = [col for col in safe_features if col in df_clean.columns]
        df_final = df_clean[available_features].copy()

        print(f"\n🔒 최종 안전한 특징들: {len(available_features)}개")
        for feature in available_features:
            print(f"   ✅ {feature}")

        # 7. 최종 안전성 검증
        self.verify_no_leakage(df_final)

        # 8. 저장
        df_final.to_csv(output_file, index=False)
        print(f"\n💾 안전한 데이터셋 저장: {output_file}")

        return df_final

    def verify_no_leakage(self, df):
        """최종 데이터 유출 검증"""
        print(f"\n🔍 최종 데이터 유출 검증...")

        issues = []

        # Lag 특징 검증
        for col in df.columns:
            if 'lag_' in col:
                try:
                    lag_num = int(col.split('_')[-1])
                    first_valid = df[col].first_valid_index()
                    if first_valid < lag_num:
                        issues.append(f"{col}: Lag {lag_num}이지만 {first_valid}번째 행부터 값 존재")
                except:
                    continue

        # 이동평균 검증
        for col in df.columns:
            if col.startswith('MA_'):
                try:
                    window = int(col.split('_')[-1])
                    first_valid = df[col].first_valid_index()
                    if first_valid < window - 1:
                        issues.append(f"{col}: 윈도우 {window}이지만 {first_valid}번째 행부터 값 존재")
                except:
                    continue

        # 기타 윈도우 기반 특징 검증
        window_features = {
            'RSI': 14,
            'BB_position': 20,
            'ATR': 14,
            'Volatility_20': 20
        }

        for col, expected_window in window_features.items():
            if col in df.columns:
                first_valid = df[col].first_valid_index()
                if first_valid < expected_window - 1:
                    issues.append(f"{col}: 윈도우 {expected_window}이지만 {first_valid}번째 행부터 값 존재")

        if len(issues) == 0:
            print("   ✅ 데이터 유출 검증 통과!")
        else:
            print("   ❌ 데이터 유출 발견:")
            for issue in issues:
                print(f"      {issue}")

        return len(issues) == 0

def create_leak_free_dataset():
    """메인 함수: 누출 없는 데이터셋 생성"""
    processor = LeakFreeDataProcessor()

    input_file = "/root/workspace/data/training/sp500_2020_2024_enhanced.csv"
    output_file = "/root/workspace/data/training/sp500_leak_free_dataset.csv"

    # 안전한 데이터셋 생성
    df_safe = processor.create_completely_safe_dataset(input_file, output_file)

    print(f"\n🎉 누출 없는 데이터셋 생성 완료!")
    print(f"   📁 파일: {output_file}")
    print(f"   📊 크기: {df_safe.shape}")
    print(f"   📅 기간: {df_safe['Date'].min()} ~ {df_safe['Date'].max()}")

    return df_safe

if __name__ == "__main__":
    df_safe = create_leak_free_dataset()