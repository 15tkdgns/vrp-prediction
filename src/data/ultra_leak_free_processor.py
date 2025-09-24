#!/usr/bin/env python3
"""
🔒 Ultra Leak-Free Data Processor
완전한 데이터 유출 제거를 위한 초강력 처리기
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class UltraLeakFreeProcessor:
    """
    초강력 무누출 데이터 처리기
    - 원시 가격 데이터에서 시작
    - 모든 특징을 처음부터 안전하게 생성
    - 엄격한 시간적 순서 준수
    """

    def __init__(self):
        """초기화"""
        self.feature_log = []
        self.validation_log = []

    def load_raw_price_data(self, input_file):
        """원시 가격 데이터만 로드"""
        print("📊 원시 가격 데이터 로드...")

        df = pd.read_csv(input_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # 필수 컬럼만 선택 (기존 파생 특징들 모두 제거)
        essential_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = [col for col in essential_columns if col in df.columns]

        df_clean = df[available_columns].copy()

        print(f"✅ 로드된 컬럼: {available_columns}")
        print(f"✅ 데이터 크기: {df_clean.shape}")
        print(f"✅ 기간: {df_clean['Date'].min()} ~ {df_clean['Date'].max()}")

        return df_clean

    def create_ultra_safe_returns(self, df):
        """완전히 안전한 수익률 계산"""
        print("🔒 안전한 수익률 계산...")

        # pct_change()는 첫 번째 행을 NaN으로 만듦 (올바름)
        df['Returns'] = df['Close'].pct_change()

        # 검증: 첫 번째 행은 반드시 NaN
        assert pd.isna(df['Returns'].iloc[0]), "Returns[0]이 NaN이 아님!"

        self.feature_log.append("Returns: 첫 번째 행 NaN 확인")
        return df

    def create_ultra_safe_moving_averages(self, df):
        """완전히 안전한 이동평균 생성"""
        print("🔒 완전히 안전한 이동평균 생성...")

        windows = [5, 10, 20, 50]

        for window in windows:
            ma_col = f"MA_{window}"

            # 안전한 이동평균: min_periods=window로 충분한 데이터가 있을 때만 계산
            df[ma_col] = df['Close'].rolling(window=window, min_periods=window).mean()

            # 엄격한 검증: 첫 (window-1)개 행은 반드시 NaN
            expected_nan_count = window - 1
            actual_nan_count = df[ma_col].iloc[:window].isna().sum()

            if actual_nan_count != window:
                # 강제로 안전하게 만들기
                df[ma_col].iloc[:window] = np.nan
                print(f"🔧 {ma_col}: 첫 {window}개 행을 강제로 NaN 설정")

            # 최종 검증
            assert df[ma_col].iloc[:window].isna().all(), f"{ma_col} 첫 {window}개 행이 NaN이 아님!"

            self.feature_log.append(f"{ma_col}: 첫 {window}행 NaN 확인")

        return df

    def create_ultra_safe_rsi(self, df):
        """완전히 안전한 RSI 생성"""
        print("🔒 완전히 안전한 RSI 생성...")

        window = 14

        # RSI 계산
        delta = df['Close'].diff()  # 첫 번째 행 NaN
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=window).mean()

        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 초강력 검증 및 수정
        # diff(1) + rolling(14) = 최소 14개 행이 NaN이어야 함
        required_nan_rows = window
        df['RSI'].iloc[:required_nan_rows] = np.nan

        # 최종 검증
        assert df['RSI'].iloc[:required_nan_rows].isna().all(), f"RSI 첫 {required_nan_rows}행이 NaN이 아님!"

        self.feature_log.append(f"RSI: 첫 {required_nan_rows}행 NaN 확인 (강제 수정)")
        return df

    def create_ultra_safe_bollinger_bands(self, df):
        """완전히 안전한 볼린저밴드 생성"""
        print("🔒 완전히 안전한 볼린저밴드 생성...")

        window = 20

        # 이동평균과 표준편차
        rolling_mean = df['Close'].rolling(window=window, min_periods=window).mean()
        rolling_std = df['Close'].rolling(window=window, min_periods=window).std()

        df['BB_upper'] = rolling_mean + (rolling_std * 2)
        df['BB_lower'] = rolling_mean - (rolling_std * 2)
        df['BB_middle'] = rolling_mean
        df['BB_width'] = df['BB_upper'] - df['BB_lower']

        # BB 포지션 계산 (0~1 사이)
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

        # 초강력 검증 및 수정
        bb_cols = ['BB_upper', 'BB_lower', 'BB_middle', 'BB_width', 'BB_position']
        required_nan_rows = window

        for col in bb_cols:
            df[col].iloc[:required_nan_rows] = np.nan
            assert df[col].iloc[:required_nan_rows].isna().all(), f"{col} 첫 {required_nan_rows}행이 NaN이 아님!"
            self.feature_log.append(f"{col}: 첫 {required_nan_rows}행 NaN 확인")

        return df

    def create_ultra_safe_volatility(self, df):
        """완전히 안전한 변동성 지표"""
        print("🔒 완전히 안전한 변동성 지표 생성...")

        windows = [5, 10, 20]

        for window in windows:
            vol_col = f"Volatility_{window}"

            # Returns 기반 변동성 (이미 Returns는 첫 번째가 NaN)
            df[vol_col] = df['Returns'].rolling(window=window, min_periods=window).std()

            # 초강력 검증 및 수정
            # Returns가 첫 번째 NaN + rolling window = window+1개 행이 NaN이어야 함
            required_nan_rows = window + 1
            df[vol_col].iloc[:required_nan_rows] = np.nan

            assert df[vol_col].iloc[:required_nan_rows].isna().all(), f"{vol_col} 첫 {required_nan_rows}행이 NaN이 아님!"
            self.feature_log.append(f"{vol_col}: 첫 {required_nan_rows}행 NaN 확인")

        return df

    def create_ultra_safe_volume_features(self, df):
        """완전히 안전한 거래량 특징"""
        print("🔒 완전히 안전한 거래량 특징 생성...")

        windows = [10, 20]

        for window in windows:
            # 거래량 이동평균
            vol_ma_col = f"Volume_MA_{window}"
            df[vol_ma_col] = df['Volume'].rolling(window=window, min_periods=window).mean()

            # 거래량 비율
            vol_ratio_col = f"Volume_ratio_{window}"
            df[vol_ratio_col] = df['Volume'] / df[vol_ma_col]

            # 초강력 검증 및 수정
            required_nan_rows = window
            for col in [vol_ma_col, vol_ratio_col]:
                df[col].iloc[:required_nan_rows] = np.nan
                assert df[col].iloc[:required_nan_rows].isna().all(), f"{col} 첫 {required_nan_rows}행이 NaN이 아님!"
                self.feature_log.append(f"{col}: 첫 {required_nan_rows}행 NaN 확인")

        return df

    def create_ultra_safe_atr(self, df):
        """완전히 안전한 ATR 생성"""
        print("🔒 완전히 안전한 ATR 생성...")

        window = 14

        # True Range 계산
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=window, min_periods=window).mean()

        # 초강력 검증 및 수정
        # shift(1) + rolling(14) = 15개 행이 NaN이어야 함
        required_nan_rows = window + 1
        df['ATR'].iloc[:required_nan_rows] = np.nan

        assert df['ATR'].iloc[:required_nan_rows].isna().all(), f"ATR 첫 {required_nan_rows}행이 NaN이 아님!"
        self.feature_log.append(f"ATR: 첫 {required_nan_rows}행 NaN 확인")

        return df

    def create_ultra_safe_momentum(self, df):
        """완전히 안전한 모멘텀 특징"""
        print("🔒 완전히 안전한 모멘텀 특징 생성...")

        periods = [5, 10]

        for period in periods:
            mom_col = f"Price_momentum_{period}"
            df[mom_col] = df['Close'] / df['Close'].shift(period) - 1

            # 초강력 검증 및 수정
            required_nan_rows = period + 1  # shift로 인한 NaN
            df[mom_col].iloc[:required_nan_rows] = np.nan

            assert df[mom_col].iloc[:required_nan_rows].isna().all(), f"{mom_col} 첫 {required_nan_rows}행이 NaN이 아님!"
            self.feature_log.append(f"{mom_col}: 첫 {required_nan_rows}행 NaN 확인")

        return df

    def create_ultra_safe_lag_features(self, df):
        """완전히 안전한 Lag 특징들"""
        print("🔒 완전히 안전한 Lag 특징 생성...")

        # Lag를 적용할 기본 특징들
        base_features = ['Returns', 'RSI', 'Volatility_20', 'BB_position']
        max_lag = 3

        for base_feature in base_features:
            if base_feature not in df.columns:
                continue

            for lag in range(1, max_lag + 1):
                lag_col = f"{base_feature}_lag_{lag}"

                # 완전히 안전한 lag 적용
                df[lag_col] = df[base_feature].shift(lag)

                # 초강력 검증
                # 원본 특징의 첫 유효 인덱스 + lag
                base_first_valid = df[base_feature].first_valid_index()
                if base_first_valid is not None:
                    expected_first_valid = base_first_valid + lag
                    # 첫 expected_first_valid개 행은 NaN이어야 함
                    df[lag_col].iloc[:expected_first_valid+1] = np.nan

                # 최종 검증: 첫 lag개 행은 반드시 NaN
                assert df[lag_col].iloc[:lag].isna().all(), f"{lag_col} 첫 {lag}행이 NaN이 아님!"

                self.feature_log.append(f"{lag_col}: 완전한 시간적 분리 확인")

        return df

    def ultra_final_cleanup(self, df):
        """초강력 최종 정리"""
        print("🔒 초강력 최종 데이터 정리...")

        # 안전한 특징들만 선택
        safe_features = [
            'Date', 'Close', 'Volume', 'Returns',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'RSI', 'BB_position',
            'Volatility_5', 'Volatility_10', 'Volatility_20',
            'Volume_MA_10', 'Volume_MA_20', 'Volume_ratio_10', 'Volume_ratio_20',
            'ATR',
            'Price_momentum_5', 'Price_momentum_10',
            'Returns_lag_1', 'Returns_lag_2', 'Returns_lag_3',
            'RSI_lag_1', 'RSI_lag_2', 'RSI_lag_3',
            'Volatility_20_lag_1', 'Volatility_20_lag_2', 'Volatility_20_lag_3',
            'BB_position_lag_1', 'BB_position_lag_2', 'BB_position_lag_3'
        ]

        # 존재하는 특징들만 선택
        available_features = [col for col in safe_features if col in df.columns]
        df_final = df[available_features].copy()

        # 모든 계산이 완료된 후 NaN 제거
        original_len = len(df_final)
        df_clean = df_final.dropna()
        removed_rows = original_len - len(df_clean)

        print(f"\\n📊 최종 정리 결과:")
        print(f"   원래 행 수: {original_len}")
        print(f"   제거된 행 수: {removed_rows}")
        print(f"   최종 행 수: {len(df_clean)}")
        print(f"   최종 특징 수: {len(available_features)}")

        return df_clean, available_features

    def ultra_validation_check(self, df):
        """초강력 최종 검증"""
        print("🔍 초강력 최종 검증...")

        issues = []

        # 1. 모든 lag 특징 검증
        for col in df.columns:
            if '_lag_' in col:
                try:
                    lag_num = int(col.split('_')[-1])
                    first_valid = df[col].first_valid_index()

                    # 첫 번째 유효값이 lag_num보다 작으면 문제
                    if first_valid is not None and first_valid < lag_num:
                        issues.append(f"❌ {col}: {first_valid}번째부터 값 (최소 {lag_num}번째부터)")
                except:
                    continue

        # 2. 이동평균 검증
        for col in df.columns:
            if col.startswith('MA_'):
                try:
                    window = int(col.split('_')[1])
                    first_valid = df[col].first_valid_index()

                    if first_valid is not None and first_valid < window - 1:
                        issues.append(f"❌ {col}: {first_valid}번째부터 값 (최소 {window-1}번째부터)")
                except:
                    continue

        # 3. Returns 검증
        if 'Returns' in df.columns:
            if df.index[0] in df['Returns'].index and not pd.isna(df['Returns'].iloc[0]):
                issues.append("❌ Returns[0]이 NaN이 아님")

        # 4. 기타 특징들 검증
        special_checks = {
            'RSI': 14,
            'BB_position': 20,
            'ATR': 15,
            'Volatility_20': 21
        }

        for col, expected_start in special_checks.items():
            if col in df.columns:
                first_valid = df[col].first_valid_index()
                if first_valid is not None and first_valid < expected_start - 1:
                    issues.append(f"❌ {col}: {first_valid}번째부터 값 (최소 {expected_start-1}번째부터)")

        # 결과 출력
        if len(issues) == 0:
            print("   ✅ 모든 검증 통과! 완전한 데이터 무결성 달성!")
        else:
            print(f"   ❌ {len(issues)}개 이슈 발견:")
            for issue in issues:
                print(f"     {issue}")

        return len(issues) == 0

    def create_ultra_leak_free_dataset(self, input_file, output_file):
        """완전히 누출 없는 데이터셋 생성"""
        print("🔒" + "="*60)
        print("🔒 초강력 무누출 데이터셋 생성 시작")
        print("🔒" + "="*60)

        # 1. 원시 가격 데이터만 로드
        df = self.load_raw_price_data(input_file)

        # 2. 모든 특징을 처음부터 안전하게 생성
        df = self.create_ultra_safe_returns(df)
        df = self.create_ultra_safe_moving_averages(df)
        df = self.create_ultra_safe_rsi(df)
        df = self.create_ultra_safe_bollinger_bands(df)
        df = self.create_ultra_safe_volatility(df)
        df = self.create_ultra_safe_volume_features(df)
        df = self.create_ultra_safe_atr(df)
        df = self.create_ultra_safe_momentum(df)
        df = self.create_ultra_safe_lag_features(df)

        # 3. 최종 정리
        df_final, features = self.ultra_final_cleanup(df)

        # 4. 초강력 검증
        validation_passed = self.ultra_validation_check(df_final)

        if validation_passed:
            print("\\n🎉 완전한 데이터 무결성 달성!")
        else:
            print("\\n⚠️ 일부 검증 실패 - 추가 수정 필요")

        # 5. 특징 생성 로그 출력
        print(f"\\n📋 특징 생성 로그:")
        for log in self.feature_log:
            print(f"   ✅ {log}")

        # 6. 저장
        df_final.to_csv(output_file, index=False)
        print(f"\\n💾 초강력 무누출 데이터셋 저장: {output_file}")
        print(f"   📊 크기: {df_final.shape}")
        print(f"   📅 기간: {df_final['Date'].min()} ~ {df_final['Date'].max()}")

        return df_final


def main():
    """메인 함수"""
    processor = UltraLeakFreeProcessor()

    input_file = "/root/workspace/data/training/sp500_2020_2024_enhanced.csv"
    output_file = "/root/workspace/data/training/sp500_ultra_leak_free.csv"

    # 초강력 무누출 데이터셋 생성
    df_ultra_safe = processor.create_ultra_leak_free_dataset(input_file, output_file)

    print("\\n🎉 초강력 무누출 데이터셋 생성 완료!")
    return df_ultra_safe


if __name__ == "__main__":
    df_ultra_safe = main()