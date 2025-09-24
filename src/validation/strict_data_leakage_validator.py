#!/usr/bin/env python3
"""
엄격한 데이터 누출 검증 시스템
R² 93%는 데이터 누출 의심 - 철저한 재검증 필요
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrictDataLeakageValidator:
    """엄격한 데이터 누출 검증"""

    def __init__(self):
        self.suspicious_r2_threshold = 0.7  # 70% 이상 의심
        self.data_issues = []

    def load_and_inspect_data(self, data_path):
        """데이터 로딩 및 세부 검사"""
        logger.info("=== 엄격한 데이터 누출 검증 시작 ===")

        df = pd.read_csv(data_path)
        logger.info(f"원본 데이터: {df.shape}")

        # 컬럼 상세 분석
        logger.info(f"전체 컬럼 수: {len(df.columns)}")

        # 의심스러운 컬럼명 검사
        suspicious_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['future', 'next', 'tomorrow', 'ahead']):
                suspicious_cols.append(col)
                self.data_issues.append(f"의심스러운 컬럼명: {col}")

        if suspicious_cols:
            logger.warning(f"의심스러운 컬럼명 발견: {suspicious_cols}")

        return df

    def check_target_construction(self, df):
        """타겟 변수 구성 방식 검증"""
        logger.info("1. 타겟 변수 구성 검증")

        if 'Close' in df.columns:
            # 수익률 계산 검증
            manual_returns = df['Close'].pct_change().dropna()

            # 기존 수익률 컬럼과 비교
            if 'Returns' in df.columns:
                existing_returns = df['Returns'].dropna()

                # 길이 맞추기
                min_len = min(len(manual_returns), len(existing_returns))
                corr = np.corrcoef(
                    manual_returns.iloc[:min_len],
                    existing_returns.iloc[:min_len]
                )[0, 1]

                logger.info(f"   수익률 계산 일치도: {corr:.6f}")

                if abs(corr - 1.0) > 0.001:  # 99.9% 이상 일치해야 함
                    self.data_issues.append(f"수익률 계산 불일치: 상관관계 {corr:.6f}")
                    logger.warning(f"   ⚠️ 수익률 계산 불일치: {corr:.6f}")

            # 극단값 검사
            extreme_threshold = 0.1  # 10% 이상 변동
            extreme_count = (abs(manual_returns) > extreme_threshold).sum()
            extreme_pct = extreme_count / len(manual_returns) * 100

            logger.info(f"   극단적 수익률 (>10%): {extreme_count}개 ({extreme_pct:.1f}%)")

            # 연속 극단값 검사 (조작 의심)
            extreme_mask = abs(manual_returns) > extreme_threshold
            consecutive_extremes = 0
            max_consecutive = 0

            for is_extreme in extreme_mask:
                if is_extreme:
                    consecutive_extremes += 1
                    max_consecutive = max(max_consecutive, consecutive_extremes)
                else:
                    consecutive_extremes = 0

            if max_consecutive > 3:
                self.data_issues.append(f"연속 극단값 {max_consecutive}개 감지")
                logger.warning(f"   ⚠️ 연속 극단값: {max_consecutive}개")

    def check_feature_timing(self, df):
        """특징 변수 시간 정보 누출 검사"""
        logger.info("2. 특징 변수 시간 누출 검사")

        # 날짜 인덱스 확인
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            date_diffs = df['Date'].diff().dropna()

            # 날짜 간격 일관성 확인
            common_diff = date_diffs.mode()[0] if len(date_diffs.mode()) > 0 else None
            irregular_dates = (date_diffs != common_diff).sum()

            logger.info(f"   일반적 날짜 간격: {common_diff}")
            logger.info(f"   불규칙 날짜: {irregular_dates}개")

            if irregular_dates > len(date_diffs) * 0.1:  # 10% 이상 불규칙
                self.data_issues.append(f"불규칙 날짜 간격: {irregular_dates}개")

        # 미래 정보가 포함된 특징 검사
        feature_cols = [col for col in df.columns
                       if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        logger.info(f"   분석할 특징 변수: {len(feature_cols)}개")

        # 각 특징이 현재/과거 정보만 사용하는지 확인
        if 'Close' in df.columns:
            current_close = df['Close']

            suspicious_features = []
            for col in feature_cols:
                if col in df.columns and df[col].dtype in ['float64', 'int64']:
                    # 미래 가격과의 상관관계 확인
                    future_close = df['Close'].shift(-1)  # 다음날 종가

                    # NaN 제거 후 상관관계 계산
                    valid_idx = ~(df[col].isna() | future_close.isna())
                    if valid_idx.sum() > 10:
                        corr_future = np.corrcoef(
                            df[col][valid_idx],
                            future_close[valid_idx]
                        )[0, 1]

                        if abs(corr_future) > 0.5:  # 미래 정보와 높은 상관관계
                            suspicious_features.append((col, corr_future))
                            self.data_issues.append(f"미래 정보 누출 의심: {col} (상관관계: {corr_future:.4f})")

            if suspicious_features:
                logger.warning(f"   ⚠️ 미래 정보 누출 의심: {len(suspicious_features)}개")
                for col, corr in suspicious_features[:5]:
                    logger.warning(f"      {col}: {corr:.4f}")

    def validate_model_performance(self, X, y):
        """모델 성능 현실성 검증"""
        logger.info("3. 모델 성능 현실성 검증")

        # TimeSeriesSplit으로 엄격한 시간 분할
        tscv = TimeSeriesSplit(n_splits=5, test_size=100)  # 작은 테스트 셋

        r2_scores = []
        mse_scores = []

        scaler = StandardScaler()

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            # 엄격한 시간 순서 확인
            if train_idx.max() >= test_idx.min():
                self.data_issues.append(f"Fold {fold}: 시간 순서 위반")
                logger.error(f"   ❌ Fold {fold}: 시간 순서 위반")
                continue

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # 데이터 표준화 (훈련 데이터만 사용)
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 단순 선형 회귀로 기준 성능 측정
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            r2_scores.append(r2)
            mse_scores.append(mse)

            logger.info(f"   Fold {fold}: R² = {r2:.4f}, MSE = {mse:.8f}")

            # 의심스러운 고성능 검사
            if r2 > self.suspicious_r2_threshold:
                self.data_issues.append(f"Fold {fold}: 의심스러운 고성능 R² {r2:.4f}")
                logger.error(f"   ❌ Fold {fold}: 의심스러운 고성능 R² {r2:.4f}")

        avg_r2 = np.mean(r2_scores) if r2_scores else 0
        avg_mse = np.mean(mse_scores) if mse_scores else 0
        std_r2 = np.std(r2_scores) if r2_scores else 0

        logger.info(f"   평균 R²: {avg_r2:.4f} ± {std_r2:.4f}")
        logger.info(f"   평균 MSE: {avg_mse:.8f}")

        # 기준선과 비교
        baseline_r2 = 0.0  # 예상 기준선

        if avg_r2 > self.suspicious_r2_threshold:
            self.data_issues.append(f"전체 평균 R² {avg_r2:.4f} > {self.suspicious_r2_threshold} (데이터 누출 의심)")
            logger.error(f"   ❌ 전체 평균 R² {avg_r2:.4f} - 데이터 누출 의심")

        return avg_r2, std_r2, avg_mse

    def check_information_coefficient(self, X, y):
        """정보 계수 (IC) 검증"""
        logger.info("4. 정보 계수 검증")

        # 각 특징의 정보 계수 계산
        ic_values = []

        for i in range(X.shape[1]):
            feature = X[:, i]

            # 순위 상관관계 (Spearman) 계산
            from scipy.stats import spearmanr

            valid_idx = ~(np.isnan(feature) | np.isnan(y))
            if valid_idx.sum() > 10:
                ic, p_value = spearmanr(feature[valid_idx], y[valid_idx])
                ic_values.append(abs(ic))

                # 과도한 정보 계수 검사
                if abs(ic) > 0.1:  # 10% 이상의 정보 계수는 의심스러움
                    logger.warning(f"   특징 {i}: IC = {ic:.4f} (높은 예측력)")

        max_ic = max(ic_values) if ic_values else 0
        avg_ic = np.mean(ic_values) if ic_values else 0

        logger.info(f"   최대 정보 계수: {max_ic:.4f}")
        logger.info(f"   평균 정보 계수: {avg_ic:.4f}")

        if max_ic > 0.15:  # 15% 이상은 매우 의심스러움
            self.data_issues.append(f"과도한 정보 계수: {max_ic:.4f}")
            logger.error(f"   ❌ 과도한 정보 계수: {max_ic:.4f}")

        return max_ic, avg_ic

    def generate_synthetic_comparison(self, X_shape, y_shape):
        """합성 데이터와의 비교"""
        logger.info("5. 합성 데이터 비교")

        # 완전 랜덤 데이터 생성
        np.random.seed(42)
        X_random = np.random.randn(*X_shape)
        y_random = np.random.randn(*y_shape)

        # 랜덤 데이터로 성능 측정
        random_r2, _, _ = self.validate_model_performance(X_random, y_random)

        logger.info(f"   랜덤 데이터 R²: {random_r2:.4f}")

        # 실제 데이터가 랜덤보다 너무 좋으면 의심
        return random_r2

    def run_comprehensive_validation(self, data_path):
        """종합 검증 실행"""
        logger.info("=" * 80)
        logger.info("🔍 엄격한 데이터 누출 검증")
        logger.info("=" * 80)

        # 1. 데이터 로딩 및 검사
        df = self.load_and_inspect_data(data_path)

        # 2. 타겟 변수 검증
        self.check_target_construction(df)

        # 3. 특징 변수 시간 누출 검사
        self.check_feature_timing(df)

        # 4. 간단한 특징 생성 (검증용)
        df = df.dropna()
        if len(df) < 100:
            logger.error("데이터가 너무 적음")
            return None

        # 안전한 특징만 사용
        df['returns'] = df['Close'].pct_change()
        df['returns_lag1'] = df['returns'].shift(1)
        df['returns_lag2'] = df['returns'].shift(2)
        df['vol_5'] = df['returns'].rolling(5).std()

        df = df.dropna()

        feature_cols = ['returns_lag1', 'returns_lag2', 'vol_5']
        X = df[feature_cols].values
        y = df['returns'].values

        logger.info(f"검증 데이터: X{X.shape}, y{y.shape}")

        # 5. 모델 성능 검증
        actual_r2, r2_std, actual_mse = self.validate_model_performance(X, y)

        # 6. 정보 계수 검증
        max_ic, avg_ic = self.check_information_coefficient(X, y)

        # 7. 합성 데이터 비교
        random_r2 = self.generate_synthetic_comparison(X.shape, y.shape)

        # 8. 최종 평가
        self._print_validation_results(actual_r2, r2_std, max_ic, random_r2)

        return self._get_final_assessment(actual_r2)

    def _print_validation_results(self, actual_r2, r2_std, max_ic, random_r2):
        """검증 결과 출력"""
        print("\n" + "=" * 80)
        print("📋 엄격한 데이터 누출 검증 결과")
        print("=" * 80)

        print(f"\n📊 핵심 지표:")
        print(f"   실제 데이터 R²: {actual_r2:.4f} ± {r2_std:.4f}")
        print(f"   랜덤 데이터 R²: {random_r2:.4f}")
        print(f"   최대 정보 계수: {max_ic:.4f}")
        print(f"   의심 임계값: R² > {self.suspicious_r2_threshold}")

        print(f"\n⚠️ 발견된 문제점:")
        if self.data_issues:
            for i, issue in enumerate(self.data_issues, 1):
                print(f"   {i}. {issue}")
        else:
            print("   문제점 없음")

        # 최종 판정
        is_suspicious = (actual_r2 > self.suspicious_r2_threshold or
                        len(self.data_issues) > 3 or
                        max_ic > 0.15)

        if is_suspicious:
            print(f"\n❌ 결론: 데이터 누출 의심됨")
            print(f"   - R² {actual_r2:.4f}는 금융 데이터로는 비현실적")
            print(f"   - 추가 검증 및 모델 재구성 필요")
        else:
            print(f"\n✅ 결론: 데이터 누출 없음")
            print(f"   - R² {actual_r2:.4f}는 합리적 범위")

        print("=" * 80)

    def _get_final_assessment(self, actual_r2):
        """최종 평가"""
        is_suspicious = (actual_r2 > self.suspicious_r2_threshold or
                        len(self.data_issues) > 3)

        return {
            'is_suspicious': is_suspicious,
            'actual_r2': actual_r2,
            'issues_count': len(self.data_issues),
            'issues': self.data_issues,
            'recommendation': 'REJECT' if is_suspicious else 'ACCEPT'
        }

def main():
    """메인 실행"""
    validator = StrictDataLeakageValidator()
    data_path = '/root/workspace/data/training/sp500_2020_2024_enhanced.csv'

    result = validator.run_comprehensive_validation(data_path)

    if result:
        print(f"\n🎯 최종 권고사항: {result['recommendation']}")
        if result['is_suspicious']:
            print("🚨 경고: 93% R² 성능은 데이터 누출 의심 - 모델 재검토 필요")

    return result

if __name__ == "__main__":
    result = main()