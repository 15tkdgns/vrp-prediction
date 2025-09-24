#!/usr/bin/env python3
"""
🔒 Purged and Embargoed Cross-Validation for Financial Time Series
금융 시계열을 위한 정제 및 금지 교차 검증

Reference: "Advances in Financial Machine Learning" by Marcos López de Prado
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Generator, Optional
import warnings
warnings.filterwarnings('ignore')

class PurgedKFold:
    """
    정제된 K-Fold 교차 검증
    - 훈련/검증 사이에 시간적 gap 적용
    - 겹치는 샘플 제거 (purging)
    - 금지 기간 적용 (embargo)
    """

    def __init__(self, n_splits: int = 5, pct_embargo: float = 0.01):
        """
        Args:
            n_splits: 분할 개수
            pct_embargo: 금지 기간 비율 (전체 데이터의 %)
        """
        self.n_splits = n_splits
        self.pct_embargo = pct_embargo

    def split(self, X: pd.DataFrame, y: pd.Series = None,
              groups: Optional[pd.Series] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        시간적 순서를 고려한 분할 생성

        Args:
            X: 특징 데이터
            y: 타겟 데이터 (사용되지 않음, sklearn 호환성을 위해)
            groups: 그룹 정보 (사용되지 않음)

        Yields:
            (train_indices, test_indices)
        """
        if isinstance(X, pd.DataFrame):
            indices = X.index.values
        else:
            indices = np.arange(len(X))

        n_samples = len(indices)
        embargo_size = int(self.pct_embargo * n_samples)

        # 각 fold의 크기 계산
        test_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            # 테스트 세트 인덱스 계산
            test_start = i * test_size
            test_end = test_start + test_size

            # 마지막 fold는 남은 모든 데이터 사용
            if i == self.n_splits - 1:
                test_end = n_samples

            test_indices = indices[test_start:test_end]

            # 훈련 세트: 테스트 세트 이전의 모든 데이터
            # 단, embargo 기간 제외
            train_end = max(0, test_start - embargo_size)
            train_indices = indices[:train_end]

            # 유효한 분할인지 확인
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """분할 개수 반환"""
        return self.n_splits


class CombinatorialPurgedCV:
    """
    조합 정제 교차 검증 (Combinatorial Purged Cross-Validation)
    - 수백 개의 가상 백테스트 경로 생성
    - 성능의 분포 확인
    - 과적합 위험 최소화
    """

    def __init__(self, n_splits: int = 10, n_test_groups: int = 4,
                 pct_embargo: float = 0.01, n_paths: int = 100):
        """
        Args:
            n_splits: 전체 분할 개수
            n_test_groups: 각 경로에서 사용할 테스트 그룹 수
            pct_embargo: 금지 기간 비율
            n_paths: 생성할 백테스트 경로 수
        """
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.pct_embargo = pct_embargo
        self.n_paths = n_paths

    def _generate_test_combinations(self) -> List[List[int]]:
        """테스트 그룹 조합 생성"""
        from itertools import combinations

        # 가능한 모든 조합 생성
        all_combinations = list(combinations(range(self.n_splits), self.n_test_groups))

        # n_paths개만큼 랜덤 선택
        if len(all_combinations) > self.n_paths:
            np.random.seed(42)  # 재현성을 위한 시드
            selected_indices = np.random.choice(len(all_combinations),
                                              self.n_paths, replace=False)
            selected_combinations = [all_combinations[i] for i in selected_indices]
        else:
            selected_combinations = all_combinations

        return selected_combinations

    def split(self, X: pd.DataFrame, y: pd.Series = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        조합 교차 검증 분할 생성

        Args:
            X: 특징 데이터
            y: 타겟 데이터

        Yields:
            (train_indices, test_indices) for each path
        """
        if isinstance(X, pd.DataFrame):
            indices = X.index.values
        else:
            indices = np.arange(len(X))

        n_samples = len(indices)
        embargo_size = int(self.pct_embargo * n_samples)

        # 전체 데이터를 n_splits개의 그룹으로 분할
        group_size = n_samples // self.n_splits
        groups = []

        for i in range(self.n_splits):
            start_idx = i * group_size
            end_idx = start_idx + group_size

            # 마지막 그룹은 남은 모든 데이터 포함
            if i == self.n_splits - 1:
                end_idx = n_samples

            groups.append(indices[start_idx:end_idx])

        # 테스트 그룹 조합 생성
        test_combinations = self._generate_test_combinations()

        for test_group_indices in test_combinations:
            # 테스트 세트 구성
            test_indices = np.concatenate([groups[i] for i in test_group_indices])

            # 훈련 세트 구성: 테스트 그룹 이전의 모든 데이터
            # embargo 기간 고려
            min_test_start = min([groups[i][0] for i in test_group_indices])
            train_end = max(0, min_test_start - embargo_size)

            train_indices = indices[:train_end]

            # 유효한 분할인지 확인
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices


class PurgedTimeSeriesSplit:
    """
    정제된 시계열 분할
    - 각 fold에서 적절한 purging과 embargo 적용
    - 시간적 순서 엄격히 준수
    """

    def __init__(self, n_splits: int = 5, max_train_size: Optional[int] = None,
                 test_size: Optional[int] = None, gap: int = 0):
        """
        Args:
            n_splits: 분할 개수
            max_train_size: 최대 훈련 크기
            test_size: 테스트 크기
            gap: 훈련과 테스트 사이의 간격 (purging gap)
        """
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
        """시계열 분할 생성"""
        n_samples = len(X)

        # 기본 테스트 크기 계산
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size

        for i in range(self.n_splits):
            # 테스트 세트 범위
            test_start = (i + 1) * test_size
            test_end = test_start + test_size

            if test_end > n_samples:
                break

            # 훈련 세트 범위 (gap 고려)
            train_end = test_start - self.gap

            if self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)
            else:
                train_start = 0

            # 유효한 분할인지 확인
            if train_start < train_end and test_start < test_end:
                train_indices = np.arange(train_start, train_end)
                test_indices = np.arange(test_start, test_end)

                yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """분할 개수 반환"""
        return self.n_splits


def validate_purged_cv_integrity(train_indices: np.ndarray, test_indices: np.ndarray,
                                gap: int = 0) -> bool:
    """
    정제된 교차 검증의 무결성 검증

    Args:
        train_indices: 훈련 인덱스
        test_indices: 테스트 인덱스
        gap: 요구되는 최소 간격

    Returns:
        bool: 무결성 통과 여부
    """
    # 1. 중복 확인
    if len(np.intersect1d(train_indices, test_indices)) > 0:
        print("❌ 훈련/테스트 세트 중복 발견")
        return False

    # 2. 시간적 순서 확인
    max_train_idx = np.max(train_indices) if len(train_indices) > 0 else -1
    min_test_idx = np.min(test_indices) if len(test_indices) > 0 else float('inf')

    actual_gap = min_test_idx - max_train_idx - 1

    if actual_gap < gap:
        print(f"❌ 간격 부족: 요구={gap}, 실제={actual_gap}")
        return False

    # 3. 연속성 확인
    if len(train_indices) > 1:
        train_sorted = np.sort(train_indices)
        if not np.array_equal(train_sorted, np.arange(train_sorted[0], train_sorted[-1] + 1)):
            print("⚠️ 훈련 인덱스가 연속적이지 않음")

    print(f"✅ 무결성 검증 통과: gap={actual_gap}, 훈련={len(train_indices)}, 테스트={len(test_indices)}")
    return True


def demonstrate_purged_cv():
    """Purged CV 시연"""
    print("🔒 Purged Cross-Validation 시연")

    # 샘플 데이터 생성
    np.random.seed(42)
    n_samples = 1000
    X = pd.DataFrame(np.random.randn(n_samples, 5),
                     columns=[f'feature_{i}' for i in range(5)])
    y = pd.Series(np.random.randn(n_samples))

    print(f"📊 샘플 데이터: {X.shape}")

    # 1. 기본 PurgedKFold
    print("\\n🔄 PurgedKFold 테스트:")
    pkf = PurgedKFold(n_splits=5, pct_embargo=0.02)

    for fold, (train_idx, test_idx) in enumerate(pkf.split(X), 1):
        print(f"Fold {fold}: 훈련={len(train_idx)}, 테스트={len(test_idx)}")
        validate_purged_cv_integrity(train_idx, test_idx, gap=int(0.02 * n_samples))

    # 2. PurgedTimeSeriesSplit
    print("\\n🔄 PurgedTimeSeriesSplit 테스트:")
    ptss = PurgedTimeSeriesSplit(n_splits=5, gap=20)

    for fold, (train_idx, test_idx) in enumerate(ptss.split(X), 1):
        print(f"Fold {fold}: 훈련={len(train_idx)}, 테스트={len(test_idx)}")
        validate_purged_cv_integrity(train_idx, test_idx, gap=20)

    # 3. CombinatorialPurgedCV
    print("\\n🔄 CombinatorialPurgedCV 테스트:")
    cpcv = CombinatorialPurgedCV(n_splits=8, n_test_groups=2, n_paths=5)

    for path, (train_idx, test_idx) in enumerate(cpcv.split(X), 1):
        print(f"Path {path}: 훈련={len(train_idx)}, 테스트={len(test_idx)}")
        validate_purged_cv_integrity(train_idx, test_idx, gap=int(0.01 * n_samples))
        if path >= 5:  # 처음 5개만 출력
            break

    print("\\n✅ Purged CV 시연 완료!")


if __name__ == "__main__":
    demonstrate_purged_cv()