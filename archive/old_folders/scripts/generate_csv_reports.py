#!/usr/bin/env python3
"""
CSV Report Generator
실제 검증 데이터로부터 CSV 파일 생성
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np

def load_validation_data():
    """Load comprehensive validation results"""
    validation_path = Path('/root/workspace/data/validation/comprehensive_model_validation.json')
    with open(validation_path) as f:
        return json.load(f)

def generate_model_comparison_csv():
    """모델 비교 CSV 생성"""
    data = load_validation_data()
    models_data = data['models']

    # 모델별 메트릭 수집
    rows = []
    for model_name, metrics in models_data.items():
        row = {
            'Model': model_name,
            'CV_R2_Mean': round(metrics['cv_r2_mean'], 4),
            'CV_R2_Std': round(metrics['cv_r2_std'], 4),
            'CV_R2_Min': round(min(metrics['cv_fold_scores']), 4),
            'CV_R2_Max': round(max(metrics['cv_fold_scores']), 4),
            'Test_R2': round(metrics['test_r2'], 4),
            'Test_MSE': f"{metrics['test_mse']:.6e}",
            'Test_MAE': round(metrics['test_mae'], 6),
            'Test_RMSE': round(metrics['test_rmse'], 6),
            'N_Features': metrics['n_features'],
            'N_Samples': metrics['n_samples'],
            'CV_Test_Gap': round(metrics['cv_r2_mean'] - metrics['test_r2'], 4)
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # R2 성능 기준 정렬 (CV R2 기준)
    df = df.sort_values('CV_R2_Mean', ascending=False)

    # CSV 저장
    output_path = Path('/root/workspace/data/model_comparison.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"✅ Model comparison CSV created: {output_path}")
    print(f"   - {len(df)} models")
    print(f"   - {len(df.columns)} metrics per model")

    return df

def generate_fold_validation_csv():
    """Fold별 검증 결과 CSV 생성"""
    data = load_validation_data()
    models_data = data['models']

    rows = []
    for model_name, metrics in models_data.items():
        fold_scores = metrics['cv_fold_scores']
        for fold_idx, score in enumerate(fold_scores, 1):
            row = {
                'Model': model_name,
                'Fold': fold_idx,
                'CV_R2_Score': round(score, 4),
                'Above_Baseline': 'Yes' if score > 0.20 else 'No',
                'Above_Target': 'Yes' if score > 0.30 else 'No'
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # CSV 저장
    output_path = Path('/root/workspace/data/fold_validation_results.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"✅ Fold validation CSV created: {output_path}")
    print(f"   - {len(df)} fold results")
    print(f"   - {len(df['Model'].unique())} models")

    return df

def generate_performance_summary_csv():
    """성능 요약 CSV 생성 (발표용)"""
    data = load_validation_data()
    models_data = data['models']

    # 성능 등급 정의
    def get_performance_grade(r2):
        if r2 >= 0.30:
            return 'Success (≥0.30)'
        elif r2 >= 0.20:
            return 'Marginal (0.20-0.30)'
        elif r2 >= 0:
            return 'Weak (0-0.20)'
        else:
            return 'Failure (<0)'

    rows = []
    for model_name, metrics in models_data.items():
        cv_r2 = metrics['cv_r2_mean']
        test_r2 = metrics['test_r2']

        row = {
            'Model': model_name,
            'CV_R2': round(cv_r2, 4),
            'CV_Grade': get_performance_grade(cv_r2),
            'Test_R2': round(test_r2, 4),
            'Test_Grade': get_performance_grade(test_r2),
            'Stability': round(metrics['cv_r2_std'], 4),
            'Generalization': 'Good' if test_r2 > 0 else 'Poor',
            'Overfitting_Risk': 'High' if (cv_r2 - test_r2) > 0.25 else 'Low',
            'Recommended': 'Yes' if (cv_r2 >= 0.30 and test_r2 > 0) else 'No'
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # CV R2 기준 정렬
    df = df.sort_values('CV_R2', ascending=False)

    # CSV 저장
    output_path = Path('/root/workspace/data/performance_summary.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"✅ Performance summary CSV created: {output_path}")
    print(f"   - Recommended models: {df[df['Recommended'] == 'Yes']['Model'].tolist()}")

    return df

def generate_statistical_analysis_csv():
    """통계적 분석 CSV 생성"""
    data = load_validation_data()
    models_data = data['models']

    rows = []
    for model_name, metrics in models_data.items():
        fold_scores = metrics['cv_fold_scores']

        # 95% 신뢰구간 계산
        mean = np.mean(fold_scores)
        std = np.std(fold_scores, ddof=1)
        n = len(fold_scores)
        ci_margin = 1.96 * (std / np.sqrt(n))

        row = {
            'Model': model_name,
            'Mean': round(mean, 4),
            'Std': round(std, 4),
            'Min': round(min(fold_scores), 4),
            'Max': round(max(fold_scores), 4),
            'Range': round(max(fold_scores) - min(fold_scores), 4),
            'CI_95_Lower': round(mean - ci_margin, 4),
            'CI_95_Upper': round(mean + ci_margin, 4),
            'CV_Coefficient': round((std / mean * 100) if mean != 0 else 0, 2),
            'N_Folds': n
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values('Mean', ascending=False)

    # CSV 저장
    output_path = Path('/root/workspace/data/statistical_analysis.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"✅ Statistical analysis CSV created: {output_path}")
    print(f"   - 95% confidence intervals calculated")

    return df

def main():
    """모든 CSV 파일 생성"""
    print("="*60)
    print("CSV Report Generation - Real Validation Data")
    print("="*60)
    print()

    try:
        # 1. 모델 비교 CSV
        print("1. Generating model comparison CSV...")
        df_comparison = generate_model_comparison_csv()
        print()

        # 2. Fold 검증 결과 CSV
        print("2. Generating fold validation CSV...")
        df_folds = generate_fold_validation_csv()
        print()

        # 3. 성능 요약 CSV
        print("3. Generating performance summary CSV...")
        df_summary = generate_performance_summary_csv()
        print()

        # 4. 통계 분석 CSV
        print("4. Generating statistical analysis CSV...")
        df_stats = generate_statistical_analysis_csv()
        print()

        print("="*60)
        print("✅ All CSV files generated successfully!")
        print("="*60)
        print()
        print("Generated files:")
        print("  - data/model_comparison.csv")
        print("  - data/fold_validation_results.csv")
        print("  - data/performance_summary.csv")
        print("  - data/statistical_analysis.csv")
        print()

        # 간단한 요약 출력
        print("Key Findings:")
        print(f"  - Best CV R²: {df_comparison.iloc[0]['Model']} ({df_comparison.iloc[0]['CV_R2_Mean']})")
        print(f"  - Best Test R²: {df_comparison.loc[df_comparison['Test_R2'].idxmax()]['Model']} "
              f"({df_comparison.loc[df_comparison['Test_R2'].idxmax()]['Test_R2']})")
        print(f"  - Recommended: {', '.join(df_summary[df_summary['Recommended'] == 'Yes']['Model'].tolist())}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
