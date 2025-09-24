#!/usr/bin/env python3
"""
안전한 데이터 처리 시스템
데이터 누출 사전 방지 기준 준수
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafeDataProcessor:
    """안전한 데이터 처리 시스템"""

    def __init__(self):
        # 데이터 누출 방지 기준
        self.MAX_R2 = 0.15
        self.MAX_DIRECTION_ACC = 65.0
        self.MAX_CORRELATION = 0.30
        self.MAX_IC = 0.08
        self.MIN_LAG_DAYS = 1

        logger.info("안전한 데이터 처리 시스템 초기화")
        logger.info(f"허용 기준 - R²: <{self.MAX_R2}, 방향정확도: <{self.MAX_DIRECTION_ACC}%")

    def load_data(self, data_path: str) -> pd.DataFrame:
        """데이터 로딩 및 기본 검증"""
        logger.info("=== 안전한 데이터 로딩 ===")

        df = pd.read_csv(data_path)
        logger.info(f"원본 데이터: {df.shape}")

        # 필수 컬럼 확인
        required_cols = ['Date', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"필수 컬럼 누락: {missing_cols}")

        # 날짜 정렬 확인
        df['Date'] = pd.to_datetime(df['Date'])
        if not df['Date'].is_monotonic_increasing:
            logger.warning("날짜 순서 정렬")
            df = df.sort_values('Date').reset_index(drop=True)

        # 기본 정보 출력
        logger.info(f"데이터 기간: {df['Date'].min()} ~ {df['Date'].max()}")
        logger.info(f"총 거래일: {len(df)}일")

        return df

    def create_safe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """안전한 특징 생성 (미래 정보 누출 없음)"""
        logger.info("=== 안전한 특징 생성 ===")

        df = df.copy()

        # 1. 기본 가격 특징 (최소 1일 지연)
        logger.info("1. 과거 가격 특징 생성")
        df['close_lag1'] = df['Close'].shift(1)
        df['close_lag2'] = df['Close'].shift(2)
        df['close_lag3'] = df['Close'].shift(3)
        df['close_lag5'] = df['Close'].shift(5)

        # 2. 안전한 수익률 특징 (지연 적용)
        logger.info("2. 과거 수익률 특징 생성")
        df['returns'] = df['Close'].pct_change()
        df['returns_lag1'] = df['returns'].shift(1)
        df['returns_lag2'] = df['returns'].shift(2)
        df['returns_lag3'] = df['returns'].shift(3)
        df['returns_lag5'] = df['returns'].shift(5)

        # 3. 안전한 이동평균 (과거 데이터만 사용)
        logger.info("3. 과거 기반 이동평균 생성")
        df['ma5_lag1'] = df['close_lag1'].rolling(5).mean()
        df['ma10_lag1'] = df['close_lag1'].rolling(10).mean()
        df['ma20_lag1'] = df['close_lag1'].rolling(20).mean()

        # 4. 안전한 변동성 (과거 데이터만 사용)
        logger.info("4. 과거 기반 변동성 생성")
        df['vol5_lag1'] = df['returns_lag1'].rolling(5).std()
        df['vol10_lag1'] = df['returns_lag1'].rolling(10).std()
        df['vol20_lag1'] = df['returns_lag1'].rolling(20).std()

        # 5. 안전한 모멘텀 특징 (추가 지연 생성 후)
        logger.info("5. 추가 과거 가격 생성")
        df['close_lag10'] = df['Close'].shift(10)

        logger.info("6. 과거 기반 모멘텀 생성")
        df['momentum_5_lag1'] = (df['close_lag1'] / df['close_lag5']) - 1
        df['momentum_10_lag1'] = (df['close_lag1'] / df['close_lag10']) - 1

        # 6. 안전한 상대 특징
        logger.info("7. 과거 기반 상대 특징 생성")
        df['price_ma_ratio_lag1'] = df['close_lag1'] / df['ma20_lag1']
        df['ma_crossover_lag1'] = (df['ma5_lag1'] > df['ma10_lag1']).astype(float)

        # 8. 안전한 통계 특징
        logger.info("8. 과거 기반 통계 특징 생성")
        df['returns_mean_5_lag1'] = df['returns_lag1'].rolling(5).mean()
        df['returns_std_5_lag1'] = df['returns_lag1'].rolling(5).std()
        df['returns_skew_10_lag1'] = df['returns_lag1'].rolling(10).skew()

        # 결측값 정리
        feature_cols = [col for col in df.columns if col.endswith('_lag1') or 'momentum_' in col or 'crossover' in col]
        logger.info(f"생성된 안전한 특징: {len(feature_cols)}개")

        return df

    def validate_feature_safety(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> bool:
        """특징 안전성 검증"""
        logger.info("=== 특징 안전성 검증 ===")

        issues = []

        # 1. 미래 정보 누출 검사
        logger.info("1. 미래 정보 누출 검사")
        future_leak_features = []

        for col in feature_cols:
            if col in df.columns and df[col].notna().sum() > 10:
                # 현재 타겟과 상관관계 (동시점 누출 검사)
                current_corr = abs(df[col].corr(df[target_col]))

                if current_corr > self.MAX_CORRELATION:
                    future_leak_features.append((col, current_corr))
                    issues.append(f"과도한 동시점 상관관계: {col} ({current_corr:.3f})")

        if future_leak_features:
            logger.error(f"미래 정보 누출 의심: {len(future_leak_features)}개")
            for col, corr in future_leak_features[:5]:
                logger.error(f"   {col}: {corr:.3f}")

        # 2. 특징명 검증 (지연 확인)
        logger.info("2. 특징명 안전성 검증")
        unsafe_features = []

        for col in feature_cols:
            # 지연이 포함되지 않은 특징 검사
            if not any(lag in col for lag in ['_lag', 'momentum_', 'crossover']):
                unsafe_features.append(col)
                issues.append(f"지연 없는 특징: {col}")

        if unsafe_features:
            logger.error(f"지연 없는 특징: {len(unsafe_features)}개")

        # 3. 컬럼명 기반 검증
        logger.info("3. 컬럼명 기반 안전성 검증")
        forbidden_keywords = ['future', 'next', 'ahead', 'forward', 'current']

        for col in feature_cols:
            for keyword in forbidden_keywords:
                if keyword in col.lower():
                    issues.append(f"금지된 키워드 포함: {col} ({keyword})")

        # 결과 출력
        is_safe = len(issues) == 0

        if is_safe:
            logger.info("✅ 모든 특징이 안전성 검증 통과")
        else:
            logger.error(f"❌ 안전성 검증 실패: {len(issues)}개 문제")
            for issue in issues[:10]:  # 최대 10개만 출력
                logger.error(f"   {issue}")

        return is_safe

    def create_safe_target(self, df: pd.DataFrame, target_type: str = 'return') -> pd.Series:
        """안전한 타겟 변수 생성"""
        logger.info(f"=== 안전한 타겟 생성 (타입: {target_type}) ===")

        if target_type == 'return':
            # 다음날 수익률 (미래 정보이지만 예측 타겟으로 허용)
            target = df['Close'].pct_change().shift(-1)
            logger.info("타겟: 다음날 수익률")

        elif target_type == 'direction':
            # 다음날 방향 (상승=1, 하락=0)
            returns = df['Close'].pct_change().shift(-1)
            target = (returns > 0).astype(int)
            logger.info("타겟: 다음날 방향 (상승/하락)")

        else:
            raise ValueError(f"지원하지 않는 타겟 타입: {target_type}")

        # 타겟 통계
        valid_target = target.dropna()
        logger.info(f"타겟 통계:")
        logger.info(f"   유효 샘플: {len(valid_target)}개")

        if target_type == 'return':
            logger.info(f"   평균: {valid_target.mean():.6f}")
            logger.info(f"   표준편차: {valid_target.std():.6f}")
            logger.info(f"   범위: [{valid_target.min():.6f}, {valid_target.max():.6f}]")
        else:
            logger.info(f"   상승: {valid_target.sum()}개 ({valid_target.mean()*100:.1f}%)")
            logger.info(f"   하락: {len(valid_target)-valid_target.sum()}개 ({(1-valid_target.mean())*100:.1f}%)")

        return target

    def create_safe_splits(self, X: np.ndarray, y: np.ndarray, n_splits: int = 3) -> List[Tuple[np.ndarray, np.ndarray]]:
        """안전한 시간 분할"""
        logger.info(f"=== 안전한 시간 분할 (splits: {n_splits}) ===")

        # TimeSeriesSplit 사용 (필수)
        tscv = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=max(100, len(X) // 10),  # 최소 100개 또는 10%
            gap=1  # 최소 1일 간격
        )

        splits = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            # 시간 순서 검증
            if train_idx.max() >= val_idx.min():
                raise ValueError(f"Fold {fold}: 시간 순서 위반")

            logger.info(f"Fold {fold}: 훈련 {len(train_idx)}, 검증 {len(val_idx)}")
            logger.info(f"   훈련 범위: {train_idx.min()}-{train_idx.max()}")
            logger.info(f"   검증 범위: {val_idx.min()}-{val_idx.max()}")
            logger.info(f"   간격: {val_idx.min() - train_idx.max()}일")

            splits.append((train_idx, val_idx))

        return splits

    def safe_scale_data(self, X_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """안전한 데이터 스케일링 (미래 정보 누출 없음)"""

        # 훈련 데이터만으로 스케일러 학습
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # 검증 데이터는 훈련 기준으로 변환
        X_val_scaled = scaler.transform(X_val)

        return X_train_scaled, X_val_scaled, scaler

    def prepare_safe_ml_data(self, data_path: str, target_type: str = 'return') -> Dict:
        """안전한 ML 데이터 준비 (전체 파이프라인)"""
        logger.info("=" * 80)
        logger.info("🛡️ 안전한 ML 데이터 준비 시작")
        logger.info("=" * 80)

        # 1. 데이터 로딩
        df = self.load_data(data_path)

        # 2. 안전한 특징 생성
        df = self.create_safe_features(df)

        # 3. 안전한 타겟 생성
        target = self.create_safe_target(df, target_type)

        # 4. 특징 선택 (안전한 특징만)
        safe_feature_cols = [col for col in df.columns
                           if (col.endswith('_lag1') or
                               'momentum_' in col or
                               'crossover' in col or
                               col.endswith('_lag2') or
                               col.endswith('_lag3') or
                               col.endswith('_lag5') or
                               col.endswith('_lag10') or
                               'ratio_lag1' in col)]

        logger.info(f"선택된 안전한 특징: {len(safe_feature_cols)}개")

        # 5. 특징 안전성 검증
        is_safe = self.validate_feature_safety(df, safe_feature_cols, 'returns')

        if not is_safe:
            raise ValueError("특징 안전성 검증 실패 - 데이터 누출 위험")

        # 6. 결측값 처리 및 데이터 정리
        df['target'] = target
        df = df.dropna()

        if len(df) < 200:
            raise ValueError(f"데이터가 너무 적음: {len(df)}개 (최소 200개 필요)")

        X = df[safe_feature_cols].values
        y = df['target'].values

        logger.info(f"최종 데이터: X{X.shape}, y{y.shape}")

        # 7. 안전한 시간 분할
        splits = self.create_safe_splits(X, y, n_splits=3)

        # 8. 최종 검증
        self._final_safety_check(X, y, safe_feature_cols)

        logger.info("✅ 안전한 ML 데이터 준비 완료")

        return {
            'X': X,
            'y': y,
            'feature_names': safe_feature_cols,
            'splits': splits,
            'target_type': target_type,
            'data_shape': X.shape,
            'safety_validated': True
        }

    def _final_safety_check(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> None:
        """최종 안전성 검사"""
        logger.info("=== 최종 안전성 검사 ===")

        # 1. 상관관계 재검증
        max_corr = 0
        for i in range(X.shape[1]):
            corr = abs(np.corrcoef(X[:, i], y)[0, 1])
            if not np.isnan(corr):
                max_corr = max(max_corr, corr)

        logger.info(f"최대 특징-타겟 상관관계: {max_corr:.3f}")

        if max_corr > self.MAX_CORRELATION:
            raise ValueError(f"과도한 상관관계: {max_corr:.3f} > {self.MAX_CORRELATION}")

        # 2. 기본 통계 검증
        logger.info("기본 통계:")
        logger.info(f"   타겟 평균: {np.mean(y):.6f}")
        logger.info(f"   타겟 표준편차: {np.std(y):.6f}")
        logger.info(f"   특징 평균 범위: [{np.mean(X, axis=0).min():.3f}, {np.mean(X, axis=0).max():.3f}]")

        # 3. 무한값/NaN 검사
        if np.any(np.isinf(X)) or np.any(np.isnan(X)):
            raise ValueError("특징에 무한값 또는 NaN 포함")

        if np.any(np.isinf(y)) or np.any(np.isnan(y)):
            raise ValueError("타겟에 무한값 또는 NaN 포함")

        logger.info("✅ 최종 안전성 검사 통과")

def main():
    """테스트 실행"""
    processor = SafeDataProcessor()

    data_path = '/root/workspace/data/training/sp500_2020_2024_enhanced.csv'

    try:
        # 안전한 회귀 데이터 준비
        result = processor.prepare_safe_ml_data(data_path, target_type='return')

        print(f"\n✅ 안전한 데이터 준비 성공:")
        print(f"   데이터 크기: {result['data_shape']}")
        print(f"   특징 수: {len(result['feature_names'])}")
        print(f"   분할 수: {len(result['splits'])}")
        print(f"   안전성 검증: {result['safety_validated']}")

        return result

    except Exception as e:
        print(f"\n❌ 안전한 데이터 준비 실패: {e}")
        return None

if __name__ == "__main__":
    result = main()