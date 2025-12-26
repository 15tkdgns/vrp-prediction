#!/usr/bin/env python3
"""
VRP 논문용 그래프 생성 스크립트
================================

논문에 포함될 주요 Figure 생성:
1. VRP 시계열 차트
2. 모델 성능 비교 막대그래프
3. 예측 vs 실제 산점도
4. 특성 중요도 시각화
5. 트레이딩 전략 성과

실행: python src/generate_paper_figures.py
출력: diagrams/figures/ 폴더에 PNG 및 PDF 저장
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import json
from datetime import datetime

# 한글 폰트 설정 (논문용)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (10, 6)

# 학술 논문 스타일
plt.style.use('seaborn-v0_8-whitegrid')

# 출력 디렉토리
OUTPUT_DIR = Path('diagrams/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_results():
    """결과 JSON 파일 로드"""
    results = {}
    
    with open('data/results/vrp_final_results.json', 'r') as f:
        results['final'] = json.load(f)
    
    with open('data/results/vrp_validation_results.json', 'r') as f:
        results['validation'] = json.load(f)
    
    with open('data/results/vrp_eda_results.json', 'r') as f:
        results['eda'] = json.load(f)
    
    return results


def figure1_model_comparison(results):
    """Figure 1: 모델 성능 비교 막대그래프"""
    
    final = results['final']['results']
    
    models = ['ElasticNet', 'Ridge', 'Ensemble', 'LightGBM', 'GradientBoosting', 'XGBoost']
    vrp_r2 = [final[m]['vrp_r2'] for m in models]
    
    colors = ['#2ecc71' if r2 > 0 else '#e74c3c' for r2 in vrp_r2]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(models, vrp_r2, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('R² Score (VRP Prediction)', fontsize=12)
    ax.set_title('Figure 1: Model Performance Comparison for VRP Prediction', fontsize=14, fontweight='bold')
    
    # 값 표시
    for bar, val in zip(bars, vrp_r2):
        x_pos = val + 0.01 if val > 0 else val - 0.05
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', fontsize=10)
    
    ax.set_xlim(-0.5, 0.3)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_DIR / 'figure1_model_comparison.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figure1_model_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("✅ Figure 1: Model Comparison saved")


def figure2_feature_importance(results):
    """Figure 2: 특성 중요도 시각화"""
    
    features = results['validation']['feature_importance']
    
    names = [f['feature'] for f in features[:10]]
    coefs = [f['coefficient'] for f in features[:10]]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(names)))[::-1]
    bars = ax.barh(names[::-1], coefs[::-1], color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Absolute Coefficient', fontsize=12)
    ax.set_title('Figure 2: Feature Importance (ElasticNet Coefficients)', fontsize=14, fontweight='bold')
    
    for bar, val in zip(bars, coefs[::-1]):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, f'{val:.2f}', 
                va='center', fontsize=10)
    
    plt.tight_layout()
    
    plt.savefig(OUTPUT_DIR / 'figure2_feature_importance.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figure2_feature_importance.pdf', bbox_inches='tight')
    plt.close()
    print("✅ Figure 2: Feature Importance saved")


def figure3_direction_metrics(results):
    """Figure 3: 방향 예측 성능 메트릭"""
    
    metrics = results['validation']['direction_metrics']['binary']
    
    names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    bars = ax.bar(names, values, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Figure 3: VRP Direction Prediction Metrics', fontsize=14, fontweight='bold')
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.1%}', 
                ha='center', fontsize=12, fontweight='bold')
    
    ax.axhline(y=0.5, color='red', linestyle='--', label='Random Baseline')
    ax.legend()
    
    plt.tight_layout()
    
    plt.savefig(OUTPUT_DIR / 'figure3_direction_metrics.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figure3_direction_metrics.pdf', bbox_inches='tight')
    plt.close()
    print("✅ Figure 3: Direction Metrics saved")


def figure4_trading_performance(results):
    """Figure 4: 트레이딩 전략 성과 비교"""
    
    trading = results['validation']['trading']
    
    # 전략 vs Buy & Hold 비교
    categories = ['Total Return (%)', 'Avg Return/Trade (%)', 'Win Rate (%)']
    strategy_vals = [
        trading['strategy']['total_return'],
        trading['strategy']['avg_return'],
        trading['strategy']['win_rate'] * 100
    ]
    buyhold_vals = [
        trading['buy_hold']['total_return'],
        trading['buy_hold']['avg_return'],
        50  # Random baseline for win rate
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, strategy_vals, width, label='VRP Prediction Strategy', 
                   color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x + width/2, buyhold_vals, width, label='Buy & Hold', 
                   color='#e74c3c', edgecolor='black')
    
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Figure 4: Trading Strategy Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # 값 표시
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{bar.get_height():.1f}', ha='center', fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{bar.get_height():.1f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    plt.savefig(OUTPUT_DIR / 'figure4_trading_performance.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figure4_trading_performance.pdf', bbox_inches='tight')
    plt.close()
    print("✅ Figure 4: Trading Performance saved")


def figure5_bootstrap_ci(results):
    """Figure 5: Bootstrap 신뢰구간"""
    
    bootstrap = results['validation']['bootstrap']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 점 추정치와 신뢰구간
    point = bootstrap['point_estimate']
    ci_lower = bootstrap['ci_lower']
    ci_upper = bootstrap['ci_upper']
    
    ax.errorbar(1, point, yerr=[[point - ci_lower], [ci_upper - point]], 
                fmt='o', markersize=15, capsize=10, capthick=2, 
                color='#3498db', ecolor='#2c3e50', linewidth=2)
    
    ax.axhline(y=0, color='red', linestyle='--', label='Zero (No Predictability)')
    
    ax.set_xlim(0.5, 1.5)
    ax.set_ylim(-0.05, 0.25)
    ax.set_xticks([1])
    ax.set_xticklabels(['ElasticNet VRP R²'])
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Figure 5: Bootstrap 95% Confidence Interval for VRP Prediction',
                 fontsize=14, fontweight='bold')
    
    # 텍스트 추가
    ax.text(1.1, point, f'Point: {point:.4f}', fontsize=11)
    ax.text(1.1, ci_lower - 0.01, f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]', fontsize=10)
    
    ax.legend(loc='upper right')
    plt.tight_layout()
    
    plt.savefig(OUTPUT_DIR / 'figure5_bootstrap_ci.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figure5_bootstrap_ci.pdf', bbox_inches='tight')
    plt.close()
    print("✅ Figure 5: Bootstrap CI saved")


def figure6_regime_analysis(results):
    """Figure 6: Regime별 성능 분석"""
    
    regime = results['validation']['regime']
    
    regimes = list(regime.keys())
    r2_scores = [regime[r]['r2'] for r in regimes]
    dir_acc = [regime[r]['direction_acc'] * 100 for r in regimes]
    n_samples = [regime[r]['n_samples'] for r in regimes]
    
    # R²가 너무 작거나 음수인 경우 제한
    r2_scores = [max(r, -1) for r in r2_scores]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # R² by Regime
    colors1 = ['#2ecc71' if r > 0 else '#e74c3c' for r in r2_scores]
    ax1.bar(regimes, r2_scores, color=colors1, edgecolor='black')
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('R² by Market Regime', fontsize=12, fontweight='bold')
    ax1.axhline(y=0, color='black', linewidth=1)
    
    # Direction Accuracy by Regime
    ax2.bar(regimes, dir_acc, color='#3498db', edgecolor='black')
    ax2.set_ylabel('Direction Accuracy (%)', fontsize=12)
    ax2.set_title('Direction Accuracy by Market Regime', fontsize=12, fontweight='bold')
    ax2.axhline(y=50, color='red', linestyle='--', label='Random')
    ax2.set_ylim(0, 100)
    
    # 샘플 수 표시
    for i, (bar, n) in enumerate(zip(ax2.patches, n_samples)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                 f'n={n}', ha='center', fontsize=10)
    
    plt.suptitle('Figure 6: Performance by Market Regime', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(OUTPUT_DIR / 'figure6_regime_analysis.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figure6_regime_analysis.pdf', bbox_inches='tight')
    plt.close()
    print("✅ Figure 6: Regime Analysis saved")


def main():
    print("\n" + "=" * 50)
    print("📊 VRP 논문용 그래프 생성")
    print("=" * 50)
    
    # 결과 로드
    results = load_results()
    print(f"\n📁 결과 파일 로드 완료")
    
    # 그래프 생성
    print(f"\n📈 그래프 생성 중...")
    
    figure1_model_comparison(results)
    figure2_feature_importance(results)
    figure3_direction_metrics(results)
    figure4_trading_performance(results)
    figure5_bootstrap_ci(results)
    figure6_regime_analysis(results)
    
    print(f"\n✅ 모든 그래프가 {OUTPUT_DIR}에 저장되었습니다.")
    print("   - PNG 형식: 논문 초안용")
    print("   - PDF 형식: 최종 출판용")
    
    # 파일 목록
    print(f"\n📁 생성된 파일:")
    for f in sorted(OUTPUT_DIR.glob('*')):
        print(f"   - {f.name}")


if __name__ == '__main__':
    main()
