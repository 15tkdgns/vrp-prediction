#!/usr/bin/env python3
"""
ElasticNet 파라미터/피처 조합을 Purged K-Fold로 탐색하는 실험 스크립트.
모든 산출물은 experiments/elasticnet_grid/results/ 하위에 저장된다.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from src.models.correct_target_design import (
    PRIMARY_TARGET,
    PurgedKFold,
    create_correct_features,
    create_correct_targets,
    get_real_spy_data
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_args():
    parser = argparse.ArgumentParser(
        description="ElasticNet 그리드 실험 (Purged K-Fold 기반)"
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.05, 0.08, 0.10, 0.15],
        help="ElasticNet alpha 후보 (기본값: 0.05 0.08 0.10 0.15)",
    )
    parser.add_argument(
        "--l1-ratios",
        nargs="+",
        type=float,
        default=[0.5, 0.6, 0.7, 0.8],
        help="ElasticNet l1_ratio 후보 (기본값: 0.5 0.6 0.7 0.8)",
    )
    parser.add_argument(
        "--feature-variants",
        nargs="+",
        choices=["base", "extended"],
        default=["base", "extended"],
        help="실험할 피처 세트 (기본값: base extended)",
    )
    parser.add_argument(
        "--candidate-r2",
        type=float,
        default=0.30,
        help="후속 실험 대상으로 삼을 최소 CV R² (기본값: 0.30)",
    )
    parser.add_argument(
        "--candidate-std",
        type=float,
        default=0.18,
        help="후속 실험 대상으로 삼을 최대 CV 표준편차 (기본값: 0.18)",
    )
    return parser.parse_args()


def apply_feature_variant(features: pd.DataFrame, variant: str) -> pd.DataFrame:
    """간단한 파생 피처를 추가해 단계적 실험을 지원한다."""
    if variant == "extended":
        df = features.copy()
        df['volatility_5_sq'] = df['volatility_5'] ** 2
        df['volatility_10_over_5'] = df['volatility_10'] / (df['volatility_5'] + 1e-8)
        df['momentum_10_over_vol_10'] = df['momentum_10'] / (df['volatility_10'] + 1e-8)
        return df
    return features.copy()


def evaluate_config(X: pd.DataFrame, y: pd.Series, alpha: float, l1_ratio: float) -> dict:
    """주어진 하이퍼파라미터로 Purged K-Fold를 수행한다."""
    cv = PurgedKFold(n_splits=5, purge_length=5, embargo_length=5)
    fold_scores = []

    for train_idx, val_idx in cv.split(X):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X.iloc[train_idx])
        X_val = scaler.transform(X.iloc[val_idx])
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score = r2_score(y_val, preds)
        fold_scores.append(score)

    fold_scores = np.array(fold_scores)
    return {
        "cv_r2_mean": float(np.mean(fold_scores)),
        "cv_r2_std": float(np.std(fold_scores)),
        "cv_fold_scores": fold_scores.tolist(),
    }


def build_configs(alphas, l1_ratios, feature_variants):
    for variant in feature_variants:
        for alpha in alphas:
            for l1_ratio in l1_ratios:
                yield {"feature_variant": variant, "alpha": alpha, "l1_ratio": l1_ratio}


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data, is_real = get_real_spy_data()
    features = create_correct_features(data)
    targets = create_correct_targets(data)
    combined = pd.concat([features, targets], axis=1).dropna()
    target_series = combined[PRIMARY_TARGET]

    feature_store = {}
    for variant in args.feature_variants:
        variant_df = apply_feature_variant(features, variant)
        feature_store[variant] = variant_df.loc[combined.index]

    records = []
    for config in build_configs(args.alphas, args.l1_ratios, args.feature_variants):
        X = feature_store[config["feature_variant"]]
        scores = evaluate_config(X, target_series, config["alpha"], config["l1_ratio"])
        candidate_flag = (
            scores["cv_r2_mean"] >= args.candidate_r2
            or scores["cv_r2_std"] <= args.candidate_std
        )

        print(
            f"[{config['feature_variant']}] alpha={config['alpha']:.3f}, "
            f"l1_ratio={config['l1_ratio']:.2f} -> "
            f"CV R² {scores['cv_r2_mean']:.4f} ± {scores['cv_r2_std']:.4f} "
            f"{'(candidate)' if candidate_flag else ''}"
        )

        records.append(
            {
                **config,
                **scores,
                "n_features": int(X.shape[1]),
                "n_samples": int(X.shape[0]),
                "data_source": "real_spy" if is_real else "simulated",
                "candidate": candidate_flag,
            }
        )

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"elasticnet_grid_{timestamp}.json"
    csv_path = RESULTS_DIR / f"elasticnet_grid_{timestamp}.csv"

    json_path.write_text(json.dumps(records, indent=2))
    pd.DataFrame(records).to_csv(csv_path, index=False)

    print(f"\n✅ 결과 저장: {json_path}")
    print(f"✅ 결과 저장: {csv_path}")

    top_records = sorted(records, key=lambda r: r["cv_r2_mean"], reverse=True)[:5]
    print("\n🏆 상위 조합 (CV R² 기준)")
    for rec in top_records:
        print(
            f" - [{rec['feature_variant']}] alpha={rec['alpha']:.3f}, "
            f"l1_ratio={rec['l1_ratio']:.2f} | "
            f"CV R² {rec['cv_r2_mean']:.4f} ± {rec['cv_r2_std']:.4f}"
        )


if __name__ == "__main__":
    main()
