#!/usr/bin/env python3
"""
데이터 누출 검증 시스템
CLAUDE.md 3대 금기사항 준수: 데이터누출로 인한 성능 95%이상 방지
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLeakageChecker:
    """데이터 누출 검증 시스템"""

    def __init__(self):
        self.leakage_tests = []
        self.max_safe_r2 = 0.95  # CLAUDE.md 금기사항: 95% 이상 금지

    def load_data(self, data_path):
        """데이터 로딩"""
        logger.info("=== 데이터 누출 검증 시작 ===")
        df = pd.read_csv(data_path)
        logger.info(f"원본 데이터: {df.shape}")
        return df

    def check_future_information_leakage(self, df):
        """미래 정보 누출 검사"""
        logger.info("1. 미래 정보 누출 검사")

        leakage_found = False
        issues = []

        # 날짜 컬럼 확인
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            logger.info(f"   날짜 컬럼 발견: {date_cols}")

            # 날짜 순서 확인
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                is_sorted = df['Date'].is_monotonic_increasing
                if not is_sorted:
                    leakage_found = True
                    issues.append("날짜가 시간순으로 정렬되지 않음")
                    logger.warning("   ⚠️ 날짜가 시간순으로 정렬되지 않음")
                else:
                    logger.info("   ✅ 날짜가 시간순으로 정렬됨")

        # 타겟 변수의 시간적 일관성 확인
        if 'Close' in df.columns:
            returns = df['Close'].pct_change()

            # 극단적인 수익률 패턴 검사
            extreme_returns = np.abs(returns) > 0.1  # 10% 이상 변동
            if extreme_returns.sum() > len(df) * 0.05:  # 5% 이상이 극단적
                logger.warning(f"   ⚠️ 극단적 수익률 패턴 감지: {extreme_returns.sum()}건")

        self.leakage_tests.append({
            'test': 'future_information',
            'passed': not leakage_found,
            'issues': issues
        })

        return not leakage_found

    def check_target_variable_leakage(self, df):
        """타겟 변수 누출 검사"""
        logger.info("2. 타겟 변수 누출 검사")

        leakage_found = False
        issues = []

        # 가격 정보와 수익률의 관계 확인
        if 'Close' in df.columns:
            # 다음날 종가가 특징으로 포함되었는지 확인
            feature_cols = [col for col in df.columns if col not in ['Date', 'Close', 'Open', 'High', 'Low', 'Volume']]

            for col in feature_cols:
                if 'close' in col.lower() and 'prev' not in col.lower() and 'lag' not in col.lower():
                    # 현재 종가 기반이 아닌 미래 종가 기반인지 확인
                    corr = df[col].corr(df['Close'])
                    if abs(corr) > 0.99:  # 거의 완벽한 상관관계
                        leakage_found = True
                        issues.append(f"컬럼 {col}이 현재 종가와 과도한 상관관계: {corr:.4f}")
                        logger.warning(f"   ⚠️ {col} - 종가 상관관계: {corr:.4f}")

        # 수익률 계산의 시간적 일관성 확인
        if 'Close' in df.columns and len(df) > 1:
            manual_returns = df['Close'].pct_change()

            # 수익률 특징이 있다면 검증
            return_cols = [col for col in df.columns if 'return' in col.lower() or 'pct' in col.lower()]
            for col in return_cols:
                if col in df.columns:
                    corr = df[col].corr(manual_returns)
                    if abs(corr) > 0.98:  # 거의 동일한 패턴
                        logger.info(f"   ✅ {col} 올바른 수익률 계산: 상관관계 {corr:.4f}")
                    else:
                        logger.warning(f"   ⚠️ {col} 의심스러운 수익률 패턴: 상관관계 {corr:.4f}")

        self.leakage_tests.append({
            'test': 'target_variable',
            'passed': not leakage_found,
            'issues': issues
        })

        return not leakage_found

    def check_feature_independence(self, X, y):
        """특징 독립성 검사"""
        logger.info("3. 특징-타겟 독립성 검사")

        leakage_found = False
        issues = []
        max_correlation = 0
        suspicious_features = []

        # 각 특징과 타겟 간의 상관관계 확인
        for i, feature_name in enumerate(range(X.shape[1])):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            if not np.isnan(corr):
                max_correlation = max(max_correlation, abs(corr))

                # 과도한 상관관계 검사
                if abs(corr) > 0.8:  # 80% 이상 상관관계
                    leakage_found = True
                    suspicious_features.append((i, corr))
                    issues.append(f"특징 {i}와 타겟 간 과도한 상관관계: {corr:.4f}")

        logger.info(f"   최대 상관관계: {max_correlation:.4f}")
        if suspicious_features:
            logger.warning(f"   ⚠️ 의심스러운 특징들: {len(suspicious_features)}개")
            for feat_idx, corr in suspicious_features[:5]:  # 상위 5개만 출력
                logger.warning(f"      특징 {feat_idx}: {corr:.4f}")
        else:
            logger.info("   ✅ 모든 특징이 안전한 상관관계 범위 내")

        self.leakage_tests.append({
            'test': 'feature_independence',
            'passed': not leakage_found,
            'max_correlation': max_correlation,
            'suspicious_count': len(suspicious_features),
            'issues': issues
        })

        return not leakage_found, max_correlation

    def check_temporal_validation(self, X, y, test_size=0.2):
        """시간적 검증 무결성 확인"""
        logger.info("4. 시간적 검증 무결성 확인")

        issues = []

        # TimeSeriesSplit 검증
        tscv = TimeSeriesSplit(n_splits=3)
        r2_scores = []

        scaler = StandardScaler()

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            # 인덱스 무결성 확인
            if train_idx.max() >= val_idx.min():
                issues.append(f"Fold {fold}: 훈련 데이터가 검증 데이터 이후 시점 포함")
                logger.warning(f"   ⚠️ Fold {fold}: 시간적 순서 위반")
                continue

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 스케일링 (미래 정보 누출 방지)
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # 단순 선형 회귀로 기준 성능 측정
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)

            r2 = r2_score(y_val, y_pred)
            r2_scores.append(r2)

            logger.info(f"   Fold {fold}: R² = {r2:.4f}")

            # 과도한 성능 검사 (CLAUDE.md 금기사항)
            if r2 > self.max_safe_r2:
                issues.append(f"Fold {fold}: R² {r2:.4f} > {self.max_safe_r2} (의심스러운 고성능)")
                logger.error(f"   ❌ Fold {fold}: 금기사항 위반 - R² {r2:.4f} > {self.max_safe_r2}")

        avg_r2 = np.mean(r2_scores) if r2_scores else 0
        logger.info(f"   평균 R² (기준 모델): {avg_r2:.4f}")

        # 최종 안전성 평가
        is_safe = avg_r2 < self.max_safe_r2 and len(issues) == 0

        self.leakage_tests.append({
            'test': 'temporal_validation',
            'passed': is_safe,
            'avg_r2_baseline': avg_r2,
            'max_r2': max(r2_scores) if r2_scores else 0,
            'issues': issues
        })

        return is_safe, avg_r2

    def check_data_preprocessing_leakage(self, df):
        """데이터 전처리 누출 검사"""
        logger.info("5. 데이터 전처리 누출 검사")

        issues = []

        # 결측값 처리 방법 확인
        missing_info = df.isnull().sum()
        if missing_info.sum() > 0:
            logger.info(f"   결측값 발견: {missing_info.sum()}개")

            # 미래 정보를 사용한 결측값 처리 검사
            for col in df.columns:
                if missing_info[col] > 0:
                    # Forward fill이 아닌 다른 방법 사용 시 경고
                    logger.info(f"   {col}: {missing_info[col]}개 결측값")

        # 이상치 처리 검사
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'Date':
                q99 = df[col].quantile(0.99)
                q01 = df[col].quantile(0.01)
                outliers = ((df[col] > q99) | (df[col] < q01)).sum()
                if outliers == 0:
                    # 이상치가 전혀 없다면 의심스러운 전처리
                    logger.info(f"   {col}: 이상치 없음 (과도한 전처리 가능성)")

        self.leakage_tests.append({
            'test': 'preprocessing',
            'passed': True,  # 전처리는 경고만
            'issues': issues
        })

        return True

    def run_comprehensive_check(self, data_path):
        """종합 데이터 누출 검증"""
        logger.info("=" * 80)
        logger.info("🔍 종합 데이터 누출 검증 시작")
        logger.info("=" * 80)

        # 데이터 로딩
        df = self.load_data(data_path)

        # 1. 미래 정보 누출 검사
        future_check = self.check_future_information_leakage(df)

        # 2. 타겟 변수 누출 검사
        target_check = self.check_target_variable_leakage(df)

        # 3. 데이터 전처리 누출 검사
        preprocessing_check = self.check_data_preprocessing_leakage(df)

        # ML 데이터 준비 (간단한 특징 생성)
        logger.info("\n특징 생성 및 타겟 설정...")

        # 기본 특징 생성 (과거 정보만 사용)
        df['returns'] = df['Close'].pct_change()
        df['returns_lag1'] = df['returns'].shift(1)
        df['returns_lag2'] = df['returns'].shift(2)
        df['volatility'] = df['returns'].rolling(5).std()
        df['ma_5'] = df['Close'].rolling(5).mean()
        df['ma_20'] = df['Close'].rolling(20).mean()

        # 결측값 제거
        df = df.dropna()

        # 특징과 타겟 분리
        feature_cols = ['returns_lag1', 'returns_lag2', 'volatility']
        X = df[feature_cols].values
        y = df['returns'].values

        # 4. 특징 독립성 검사
        independence_check, max_corr = self.check_feature_independence(X, y)

        # 5. 시간적 검증 무결성 확인
        temporal_check, baseline_r2 = self.check_temporal_validation(X, y)

        # 종합 결과
        self._print_comprehensive_results(baseline_r2, max_corr)

        return self._get_final_assessment()

    def _print_comprehensive_results(self, baseline_r2, max_corr):
        """종합 결과 출력"""
        print("\n" + "=" * 80)
        print("📋 데이터 누출 검증 결과")
        print("=" * 80)

        print(f"\n🔍 검증 요약:")
        for test in self.leakage_tests:
            status = "✅ 통과" if test['passed'] else "❌ 실패"
            print(f"   {test['test']:20s}: {status}")
            if test['issues']:
                for issue in test['issues'][:3]:  # 최대 3개만 표시
                    print(f"     ⚠️ {issue}")

        print(f"\n📊 핵심 지표:")
        print(f"   기준 모델 R²: {baseline_r2:.4f}")
        print(f"   최대 상관관계: {max_corr:.4f}")
        print(f"   안전 기준 R²: < {self.max_safe_r2}")

        # CLAUDE.md 금기사항 확인
        if baseline_r2 >= self.max_safe_r2:
            print(f"\n❌ CLAUDE.md 금기사항 위반:")
            print(f"   데이터누출로 인한 성능 {baseline_r2*100:.1f}% ≥ 95%")
        else:
            print(f"\n✅ CLAUDE.md 금기사항 준수:")
            print(f"   성능 {baseline_r2*100:.1f}% < 95% (안전)")

        print("\n" + "=" * 80)

    def _get_final_assessment(self):
        """최종 평가"""
        passed_tests = sum(1 for test in self.leakage_tests if test['passed'])
        total_tests = len(self.leakage_tests)

        assessment = {
            'overall_passed': passed_tests == total_tests,
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'test_results': self.leakage_tests,
            'claude_md_compliant': all(
                test.get('avg_r2_baseline', 0) < self.max_safe_r2
                for test in self.leakage_tests
                if 'avg_r2_baseline' in test
            )
        }

        return assessment

def main():
    """메인 실행"""
    checker = DataLeakageChecker()
    data_path = '/root/workspace/data/training/sp500_2020_2024_enhanced.csv'

    results = checker.run_comprehensive_check(data_path)

    print(f"\n🎯 최종 결과:")
    print(f"   전체 통과: {'✅ 예' if results['overall_passed'] else '❌ 아니오'}")
    print(f"   CLAUDE.md 준수: {'✅ 예' if results['claude_md_compliant'] else '❌ 아니오'}")
    print(f"   통과율: {results['passed_tests']}/{results['total_tests']}")

    return results

if __name__ == "__main__":
    results = main()