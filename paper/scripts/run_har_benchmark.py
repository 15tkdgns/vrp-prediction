"""
HAR (Heterogeneous Autoregressive) 벤치마크 모델 실행
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime


class PurgedKFold:
    """Purged and Embargoed K-Fold Cross-Validation"""

    def __init__(self, n_splits=5, purge_length=5, embargo_length=5):
        self.n_splits = n_splits
        self.purge_length = purge_length
        self.embargo_length = embargo_length

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        for fold_idx in range(self.n_splits):
            # 검증 세트 범위
            val_start = fold_idx * fold_size
            val_end = (fold_idx + 1) * fold_size if fold_idx < self.n_splits - 1 else n_samples

            # 훈련 세트: 검증 세트 제외
            train_indices = np.concatenate([
                indices[:max(0, val_start - self.purge_length)],
                indices[min(n_samples, val_end + self.embargo_length):]
            ])

            # 검증 세트
            val_indices = indices[val_start:val_end]

            if len(train_indices) > 0 and len(val_indices) > 0:
                yield train_indices, val_indices

def create_har_model(returns, horizon=5):
    """
    HAR (Heterogeneous Autoregressive) 벤치마크 모델
    Ridge 모델과 동일한 타겟 생성 방식 사용
    """
    print("📊 HAR 벤치마크 모델 생성...")

    # HAR 특성: 일간(5일), 주간(5일), 월간(22일) 실현 변동성
    rv_daily = returns.rolling(5).std()  # 일간 (5일)
    rv_weekly = returns.rolling(5).std()  # 주간 (5일)
    rv_monthly = returns.rolling(22).std()  # 월간 (22일)

    # 타겟: t+1부터 t+5까지의 미래 변동성 (Ridge와 동일)
    vol_values = []
    for i in range(len(returns)):
        if i + horizon >= len(returns):
            vol_values.append(np.nan)
        else:
            # t+1 ~ t+5 미래 수익률의 표준편차
            future_returns = returns.iloc[i+1:i+1+horizon]
            if len(future_returns) == horizon:
                vol_values.append(future_returns.std())
            else:
                vol_values.append(np.nan)

    target_vol_5d = pd.Series(vol_values, index=returns.index)

    # HAR 모델 데이터 준비
    har_data = pd.DataFrame({
        'rv_daily': rv_daily,
        'rv_weekly': rv_weekly,
        'rv_monthly': rv_monthly,
        'target_vol_5d': target_vol_5d
    }).dropna()

    print(f"   HAR 데이터: {len(har_data)}개 관측치")

    if len(har_data) < 100:
        print(f"   ⚠️ HAR 데이터 부족")
        return None, None, None

    # HAR 회귀 모델
    X_har = har_data[['rv_daily', 'rv_weekly', 'rv_monthly']]
    y_har = har_data['target_vol_5d']

    har_model = Ridge(alpha=0.01)

    print(f"✅ HAR 모델 생성 완료 (3개 특성, horizon={horizon}일)")

    return har_model, X_har, y_har


def main():
    print("=" * 60)
    print("HAR 벤치마크 모델 실행")
    print("=" * 60)

    # 데이터 로드
    data_path = Path('/root/workspace/data/training/multi_modal_sp500_dataset.csv')
    print(f"\n📂 데이터 로드: {data_path}")

    data = pd.read_csv(data_path)
    print(f"   총 {len(data)}개 샘플")

    # HAR 모델 생성 (returns 컬럼 사용)
    har_model, X_har, y_har = create_har_model(data['returns'], horizon=5)

    if har_model is None or X_har is None or y_har is None:
        print("\n❌ HAR 모델 생성 실패")
        return

    # Purged K-Fold Cross-Validation
    print(f"\n🎯 Purged K-Fold Cross-Validation (5-fold)...")
    cv = PurgedKFold(n_splits=5, purge_length=5, embargo_length=5)

    cv_scores = []
    scaler = StandardScaler()

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_har), 1):
        # 분할
        X_train = X_har.iloc[train_idx]
        X_val = X_har.iloc[val_idx]
        y_train = y_har.iloc[train_idx]
        y_val = y_har.iloc[val_idx]

        # 스케일링
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # 훈련 및 예측
        model = Ridge(alpha=0.01)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)

        # 성능 평가
        fold_r2 = r2_score(y_val, y_pred)
        cv_scores.append(fold_r2)
        print(f"   Fold {fold_idx}: R² = {fold_r2:.6f}")

    # 평균 성능
    r2 = np.mean(cv_scores)
    r2_std = np.std(cv_scores)

    print(f"\n📊 평균 성능:")
    print(f"   R² Mean: {r2:.6f}")
    print(f"   R² Std:  {r2_std:.6f}")

    # 전체 데이터로 최종 훈련 (테스트용)
    split_point = int(len(X_har) * 0.8)
    X_train = X_har.iloc[:split_point]
    X_test = X_har.iloc[split_point:]
    y_train = y_har.iloc[:split_point]
    y_test = y_har.iloc[split_point:]

    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train)
    X_test_scaled = scaler_final.transform(X_test)

    har_model.fit(X_train_scaled, y_train)
    y_pred = har_model.predict(X_test_scaled)

    test_r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n📊 HAR 벤치마크 최종 성능:")
    print(f"   CV R² (Mean ± Std): {r2:.6f} ± {r2_std:.6f}")
    print(f"   Test R²:            {test_r2:.6f}")
    print(f"   Test MSE:           {mse:.6f}")
    print(f"   Test RMSE:          {rmse:.6f}")
    print(f"   Test MAE:           {mae:.6f}")

    # Ridge 모델과 비교
    ridge_perf_path = Path('/root/workspace/data/raw/model_performance.json')
    if ridge_perf_path.exists():
        with open(ridge_perf_path, 'r') as f:
            ridge_perf = json.load(f)

        ridge_r2 = ridge_perf['test_r2']
        improvement = ridge_r2 / r2 if r2 > 0 else float('inf')

        print(f"\n📈 Ridge 모델과 비교:")
        print(f"   HAR R²:    {r2:.6f}")
        print(f"   Ridge R²:  {ridge_r2:.6f}")
        print(f"   개선 배수: {improvement:.1f}x")

    # 결과 저장
    results = {
        "model_name": "HAR Benchmark",
        "model_type": "Heterogeneous Autoregressive",
        "target": "target_vol_5d",
        "cv_r2_mean": float(r2),
        "cv_r2_std": float(r2_std),
        "cv_fold_scores": [float(s) for s in cv_scores],
        "test_r2": float(test_r2),
        "test_mse": float(mse),
        "test_rmse": float(rmse),
        "test_mae": float(mae),
        "n_samples_total": int(len(X_har)),
        "n_samples_train": int(len(X_train)),
        "n_samples_test": int(len(X_test)),
        "n_features": int(X_har.shape[1]),
        "features": ['rv_daily', 'rv_weekly', 'rv_monthly'],
        "alpha": 0.01,
        "validation_method": "Purged K-Fold CV (5-fold, purge=5, embargo=5)",
        "timestamp": datetime.now().isoformat()
    }

    # JSON 저장
    output_path = Path('/root/workspace/data/raw/har_benchmark_performance.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ 결과 저장: {output_path}")

    # Ridge와의 비교 저장
    if ridge_perf_path.exists():
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "har_benchmark": {
                "r2": float(r2),
                "rmse": float(rmse),
                "mae": float(mae),
                "features": 3
            },
            "ridge_model": {
                "r2": float(ridge_r2),
                "rmse": float(ridge_perf['test_rmse']),
                "mae": float(ridge_perf['test_mae']),
                "features": int(ridge_perf['n_features'])
            },
            "improvement": {
                "r2_ratio": float(improvement),
                "r2_difference": float(ridge_r2 - r2),
                "conclusion": f"Ridge outperforms HAR by {improvement:.1f}x"
            }
        }

        comparison_path = Path('/root/workspace/data/raw/har_vs_ridge_comparison.json')
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)

        print(f"✅ 비교 결과 저장: {comparison_path}")

    print("\n" + "=" * 60)
    print("HAR 벤치마크 실행 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
