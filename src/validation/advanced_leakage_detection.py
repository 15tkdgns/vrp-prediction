#!/usr/bin/env python3
"""
🔍 고급 데이터 유출 탐지 시스템
미묘한 look-ahead bias와 정보 유출 탐지
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class AdvancedLeakageDetector:
    """
    고급 데이터 유출 탐지기
    - 미묘한 look-ahead bias 탐지
    - 특징 간 시간적 일관성 검증
    - 정보 유출 패턴 분석
    """

    def __init__(self):
        """초기화"""
        self.leakage_issues = []
        self.suspicious_patterns = []

    def analyze_feature_timing(self, df):
        """특징별 시간적 정보 사용 분석"""
        print("🔍 특징별 시간적 정보 사용 분석...")

        issues = []

        # 1. 이동평균 검증 - 더 엄격한 기준
        ma_features = [col for col in df.columns if col.startswith('MA_')]
        for ma_col in ma_features:
            window = int(ma_col.split('_')[1])

            # 첫 번째 유효값 위치 확인
            first_valid = df[ma_col].first_valid_index()
            expected_first_valid = window - 1  # 0-indexed

            if first_valid < expected_first_valid:
                issues.append(f"❌ {ma_col}: {first_valid}번째부터 값 (최소 {expected_first_valid}번째부터 있어야 함)")

                # 실제 계산 검증
                manual_ma = df['Close'].rolling(window=window, min_periods=window).mean()
                if not df[ma_col].equals(manual_ma):
                    issues.append(f"❌ {ma_col}: 계산 방식이 안전하지 않음")

        # 2. RSI 계산 검증
        if 'RSI' in df.columns:
            rsi_first_valid = df['RSI'].first_valid_index()
            # RSI는 diff(1) + rolling(14) = 14개 행이 NaN이어야 함
            expected_rsi_start = 14

            if rsi_first_valid < expected_rsi_start:
                issues.append(f"❌ RSI: {rsi_first_valid}번째부터 값 (최소 {expected_rsi_start}번째부터 있어야 함)")

        # 3. Lag 특징 검증 - 매우 엄격
        lag_features = [col for col in df.columns if '_lag_' in col]
        for lag_col in lag_features:
            try:
                lag_num = int(lag_col.split('_')[-1])
                first_valid = df[lag_col].first_valid_index()

                if first_valid < lag_num:
                    issues.append(f"❌ {lag_col}: {first_valid}번째부터 값 (최소 {lag_num}번째부터 있어야 함)")

                # 원본 특징과의 일관성 검증
                base_feature = '_'.join(lag_col.split('_')[:-2])
                if base_feature in df.columns:
                    # 수동으로 lag 계산
                    manual_lag = df[base_feature].shift(lag_num)
                    if not df[lag_col].fillna(-999).equals(manual_lag.fillna(-999)):
                        issues.append(f"❌ {lag_col}: lag 계산이 올바르지 않음")

            except ValueError:
                continue

        # 4. 볼린저밴드 검증
        bb_features = [col for col in df.columns if col.startswith('BB_')]
        for bb_col in bb_features:
            if bb_col.endswith('position'):
                first_valid = df[bb_col].first_valid_index()
                expected_start = 19  # 20일 윈도우 - 1

                if first_valid < expected_start:
                    issues.append(f"❌ {bb_col}: {first_valid}번째부터 값 (최소 {expected_start}번째부터 있어야 함)")

        # 5. ATR 검증
        if 'ATR' in df.columns:
            atr_first_valid = df['ATR'].first_valid_index()
            expected_atr_start = 14  # 14일 윈도우

            if atr_first_valid < expected_atr_start:
                issues.append(f"❌ ATR: {atr_first_valid}번째부터 값 (최소 {expected_atr_start}번째부터 있어야 함)")

        return issues

    def detect_future_information_usage(self, df):
        """미래 정보 사용 탐지"""
        print("🔍 미래 정보 사용 패턴 탐지...")

        issues = []

        # 1. Returns와 특징 간의 부자연스러운 상관관계 검증
        if 'Returns' in df.columns:
            for col in df.columns:
                if col != 'Returns' and col != 'Date' and col != 'Close':
                    # 현재 수익률과 현재 특징의 상관관계가 너무 높으면 의심
                    corr = df['Returns'].corr(df[col])
                    if abs(corr) > 0.7:  # 70% 이상 상관관계는 의심스러움
                        issues.append(f"⚠️ {col}: Returns와 과도한 상관관계 {corr:.3f}")

        # 2. 특징값의 급격한 변화 패턴 검증
        for col in df.columns:
            if col not in ['Date', 'Returns', 'Close', 'Volume']:
                if df[col].dtype in ['float64', 'int64']:
                    # 첫 번째 유효값이 0이거나 비현실적으로 큰 값인지 확인
                    first_valid_idx = df[col].first_valid_index()
                    if first_valid_idx is not None:
                        first_val = df[col].iloc[first_valid_idx]
                        if abs(first_val) > 1000 or first_val == 0:
                            issues.append(f"⚠️ {col}: 첫 번째 값이 의심스러움 {first_val}")

        # 3. 시간적 일관성 검증
        # 각 특징이 시간에 따라 점진적으로 변화하는지 확인
        for col in df.columns:
            if col.startswith('MA_') or col == 'RSI' or col.startswith('Volatility_'):
                if df[col].dtype in ['float64', 'int64']:
                    # 급격한 점프가 있는지 확인
                    diff = df[col].diff().abs()
                    if len(diff.dropna()) > 0:
                        extreme_changes = diff > diff.quantile(0.99) * 5
                        if extreme_changes.sum() > len(df) * 0.01:  # 1% 이상이 극값이면 의심
                            issues.append(f"⚠️ {col}: 비정상적인 급격한 변화 패턴")

        return issues

    def analyze_target_leakage(self, df):
        """타겟 변수 관련 데이터 유출 검증"""
        print("🔍 타겟 변수 관련 데이터 유출 분석...")

        issues = []

        if 'Returns' in df.columns and 'Close' in df.columns:
            # 1. Returns 계산 방식 검증
            manual_returns = df['Close'].pct_change()

            # 첫 번째 Returns 값은 NaN이어야 함
            if not pd.isna(df['Returns'].iloc[0]):
                issues.append("❌ Returns[0]이 NaN이 아님 - 이전 가격 정보 사용 의심")

            # 계산 일관성 검증
            diff_count = (df['Returns'].fillna(-999) != manual_returns.fillna(-999)).sum()
            if diff_count > 0:
                issues.append(f"❌ Returns 계산이 올바르지 않음 - {diff_count}개 불일치")

        # 2. 미래 가격 정보 유출 검증
        # Price momentum 특징들 검증
        momentum_cols = [col for col in df.columns if 'momentum' in col.lower()]
        for mom_col in momentum_cols:
            if '_5' in mom_col:
                expected_start = 5
            elif '_10' in mom_col:
                expected_start = 10
            else:
                continue

            first_valid = df[mom_col].first_valid_index()
            if first_valid < expected_start:
                issues.append(f"❌ {mom_col}: {first_valid}번째부터 값 (최소 {expected_start}번째부터 있어야 함)")

        return issues

    def comprehensive_leakage_audit(self, data_file):
        """포괄적 데이터 유출 감사"""
        print("🔒 포괄적 데이터 유출 감사 시작")
        print(f"📁 파일: {data_file}")

        # 데이터 로드
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        print(f"📊 데이터: {df.shape}")

        all_issues = []

        # 1. 특징별 시간적 정보 사용 분석
        timing_issues = self.analyze_feature_timing(df)
        all_issues.extend(timing_issues)

        # 2. 미래 정보 사용 탐지
        future_info_issues = self.detect_future_information_usage(df)
        all_issues.extend(future_info_issues)

        # 3. 타겟 변수 관련 유출 검증
        target_issues = self.analyze_target_leakage(df)
        all_issues.extend(target_issues)

        # 결과 정리
        print(f"\\n📋 고급 데이터 유출 감사 결과:")
        print(f"   발견된 이슈: {len(all_issues)}개")

        if len(all_issues) == 0:
            print("   ✅ 추가 데이터 유출 발견되지 않음")
        else:
            print("   ❌ 추가 데이터 유출 발견:")
            for i, issue in enumerate(all_issues, 1):
                print(f"     {i}. {issue}")

        # 심각도별 분류
        critical_issues = [issue for issue in all_issues if issue.startswith('❌')]
        warning_issues = [issue for issue in all_issues if issue.startswith('⚠️')]

        print(f"\\n📊 심각도별 분류:")
        print(f"   🚨 Critical (수정 필요): {len(critical_issues)}개")
        print(f"   ⚠️ Warning (검토 필요): {len(warning_issues)}개")

        # 권고사항
        if len(critical_issues) > 0:
            print(f"\\n🔧 권고사항:")
            print(f"   1. Critical 이슈 우선 수정 필요")
            print(f"   2. 특징 생성 로직 재검토")
            print(f"   3. 더 보수적인 윈도우/Lag 적용")
        elif len(warning_issues) > 0:
            print(f"\\n💡 권고사항:")
            print(f"   1. Warning 이슈 검토 권장")
            print(f"   2. 현재 성능이 과적합일 가능성 검토")
        else:
            print(f"\\n✅ 데이터 무결성 우수")

        return {
            'total_issues': len(all_issues),
            'critical_issues': len(critical_issues),
            'warning_issues': len(warning_issues),
            'all_issues': all_issues,
            'timing_issues': timing_issues,
            'future_info_issues': future_info_issues,
            'target_issues': target_issues
        }


def main():
    """메인 실행"""
    detector = AdvancedLeakageDetector()
    data_file = "/root/workspace/data/training/sp500_ultra_leak_free.csv"

    results = detector.comprehensive_leakage_audit(data_file)
    return results


if __name__ == "__main__":
    results = main()