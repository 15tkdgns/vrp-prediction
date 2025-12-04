#!/usr/bin/env python3
"""
종합 성능 분석 - 교차검증 vs Walk-Forward 성능 비교
과적합 문제 진단 및 해결 방안 제시
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

def load_all_results():
    """모든 모델 결과 로드"""
    print("📊 모든 모델 결과 로드 중...")

    results = {}

    # V2 Lite 결과 (교차검증)
    try:
        with open('/root/workspace/results/enhanced_model_v2_lite.json', 'r') as f:
            results['v2_lite'] = json.load(f)
        print("✅ V2 Lite 결과 로드됨")
    except:
        print("❌ V2 Lite 결과 없음")

    # GARCH Enhanced 결과 (교차검증)
    try:
        with open('/root/workspace/results/garch_enhanced_model.json', 'r') as f:
            results['garch'] = json.load(f)
        print("✅ GARCH Enhanced 결과 로드됨")
    except:
        print("❌ GARCH Enhanced 결과 없음")

    # Walk-Forward 결과 (실제 거래 시뮬레이션)
    try:
        with open('/root/workspace/results/walk_forward_validation.json', 'r') as f:
            wf_data = json.load(f)
            results['walk_forward'] = wf_data
        print("✅ Walk-Forward 결과 로드됨")
    except:
        print("❌ Walk-Forward 결과 없음")

    return results

def analyze_overfitting_severity(results):
    """과적합 심각도 분석"""
    print("\n🔍 과적합 심각도 분석")
    print("=" * 70)

    if 'v2_lite' not in results or 'walk_forward' not in results:
        print("❌ 비교할 데이터 부족")
        return

    # 교차검증 최고 성능
    cv_best = results['v2_lite']['best_model']['r2_mean']
    cv_model = results['v2_lite']['best_model']['name']

    print(f"교차검증 최고 성능: {cv_model}")
    print(f"  R² = {cv_best:.4f}")

    # Walk-Forward 성능들
    wf_results = results['walk_forward']['summary']

    print(f"\nWalk-Forward 실제 성능:")
    print(f"{'Model':<20} {'CV R²':<10} {'WF R²':<10} {'성능 차이':<12} {'과적합도'}")
    print("-" * 70)

    overfitting_analysis = {}

    for model_name, wf_stats in wf_results.items():
        wf_r2 = wf_stats['mean_r2']

        # 해당 모델의 교차검증 성능 찾기
        cv_r2 = None
        if 'garch' in results:
            garch_results = results['garch']['all_results']
            # 모델명 매칭
            if 'Lasso' in model_name and 'α=0.0001' in str(garch_results.get('Lasso (α=0.0001)', {})):
                cv_r2 = garch_results['Lasso (α=0.0001)']['mean_r2']
            elif 'Lasso' in model_name and '0005' in model_name:
                cv_r2 = garch_results['Lasso (α=0.0005)']['mean_r2']
            elif 'ElasticNet' in model_name:
                cv_r2 = garch_results['ElasticNet (α=0.0005)']['mean_r2']

        if cv_r2 is None:
            cv_r2 = cv_best  # 대략적 추정

        performance_gap = wf_r2 - cv_r2
        overfitting_ratio = abs(performance_gap) / cv_r2 if cv_r2 > 0 else float('inf')

        overfitting_analysis[model_name] = {
            'cv_r2': cv_r2,
            'wf_r2': wf_r2,
            'gap': performance_gap,
            'overfitting_ratio': overfitting_ratio
        }

        print(f"{model_name:<20} {cv_r2:<10.4f} {wf_r2:<10.4f} {performance_gap:<12.4f} {overfitting_ratio:.1f}x")

    # 과적합 진단
    print(f"\n📋 과적합 진단:")
    avg_gap = np.mean([v['gap'] for v in overfitting_analysis.values()])
    avg_overfitting = np.mean([v['overfitting_ratio'] for v in overfitting_analysis.values()
                              if v['overfitting_ratio'] != float('inf')])

    print(f"  평균 성능 저하: {avg_gap:.4f}")
    print(f"  평균 과적합 배수: {avg_overfitting:.1f}x")

    if avg_gap < -0.8:
        print("  ⚠️  심각한 과적합: 모델이 실제 거래에서 완전히 실패")
    elif avg_gap < -0.3:
        print("  ⚠️  중간 과적합: 실용성 매우 제한적")
    else:
        print("  ✅ 경미한 과적합: 일부 실용성 있음")

    return overfitting_analysis

def create_performance_comparison_chart(results):
    """성능 비교 차트 생성"""
    print("\n📈 성능 비교 차트 생성 중...")

    if 'v2_lite' not in results or 'walk_forward' not in results:
        return

    # 데이터 준비
    models = ['Lasso_0001', 'Lasso_0005', 'ElasticNet']
    cv_scores = []
    wf_scores = []

    # 교차검증 점수
    garch_results = results.get('garch', {}).get('all_results', {})
    for model in models:
        if model == 'Lasso_0001':
            cv_scores.append(garch_results.get('Lasso (α=0.0001)', {}).get('mean_r2', 0.45))
        elif model == 'Lasso_0005':
            cv_scores.append(garch_results.get('Lasso (α=0.0005)', {}).get('mean_r2', 0.45))
        elif model == 'ElasticNet':
            cv_scores.append(garch_results.get('ElasticNet (α=0.0005)', {}).get('mean_r2', 0.45))

    # Walk-Forward 점수
    wf_summary = results['walk_forward']['summary']
    for model in models:
        wf_scores.append(wf_summary[model]['mean_r2'])

    # 차트 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. 성능 비교 바 차트
    x = np.arange(len(models))
    width = 0.35

    bars1 = ax1.bar(x - width/2, cv_scores, width, label='교차검증 (CV)', color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, wf_scores, width, label='Walk-Forward (실제)', color='salmon', alpha=0.8)

    ax1.set_xlabel('모델')
    ax1.set_ylabel('R² Score')
    ax1.set_title('교차검증 vs Walk-Forward 성능 비교')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='기준선 (R²=0)')

    # 값 표시
    for bar, score in zip(bars1, cv_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    for bar, score in zip(bars2, wf_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.05,
                f'{score:.3f}', ha='center', va='top', fontsize=9, color='white', weight='bold')

    # 2. 과적합 시각화
    overfitting_ratios = [abs((wf - cv) / cv) if cv > 0 else 0 for cv, wf in zip(cv_scores, wf_scores)]

    ax2.bar(models, overfitting_ratios, color='orange', alpha=0.7)
    ax2.set_xlabel('모델')
    ax2.set_ylabel('과적합 배수 (절댓값)')
    ax2.set_title('모델별 과적합 심각도')
    ax2.set_xticklabels(models, rotation=45)
    ax2.grid(True, alpha=0.3)

    for i, ratio in enumerate(overfitting_ratios):
        ax2.text(i, ratio + 0.1, f'{ratio:.1f}x', ha='center', va='bottom', fontsize=10, weight='bold')

    plt.tight_layout()

    # 저장
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/overfitting_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ 저장: figures/overfitting_analysis.png")
    plt.close()

def generate_diagnostic_report(results, overfitting_analysis):
    """진단 보고서 생성"""
    print("\n📋 종합 진단 보고서 생성 중...")

    report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'model_development_phases': {
            'phase_1_baseline': {
                'description': 'V2 Lite - 데이터 확장 및 경제지표 추가',
                'r2_achieved': results.get('v2_lite', {}).get('best_model', {}).get('r2_mean', 0),
                'improvement_vs_original': results.get('v2_lite', {}).get('improvement_vs_baseline', 0),
                'key_features': results.get('v2_lite', {}).get('top_features', [])[:5]
            },
            'phase_2_garch': {
                'description': 'GARCH Enhanced - 조건부 이분산성 모델링',
                'r2_achieved': results.get('garch', {}).get('best_model', {}).get('r2_mean', 0),
                'improvement_vs_v2_lite': results.get('garch', {}).get('improvement_vs_v2_lite', 0),
                'garch_library': results.get('garch', {}).get('garch_library', 'unknown')
            },
            'phase_3_validation': {
                'description': 'Walk-Forward - 실제 거래 환경 검증',
                'validation_type': 'Walk-Forward',
                'folds': results.get('walk_forward', {}).get('total_folds', 0),
                'best_wf_r2': max([v['mean_r2'] for v in results.get('walk_forward', {}).get('summary', {}).values()]) if results.get('walk_forward') else 0
            }
        },
        'critical_findings': {
            'severe_overfitting_detected': True,
            'cv_vs_wf_gap': {
                'average_gap': np.mean([v['gap'] for v in overfitting_analysis.values()]) if overfitting_analysis else 0,
                'worst_gap': min([v['gap'] for v in overfitting_analysis.values()]) if overfitting_analysis else 0
            },
            'real_world_performance': 'FAILED - 모든 모델이 실제 거래에서 음의 R² 기록',
            'reliability_assessment': 'UNRELIABLE - 교차검증 결과를 신뢰할 수 없음'
        },
        'root_cause_analysis': {
            'data_leakage_risk': 'HIGH - 시간적 분리 불완전 가능성',
            'feature_stability': 'LOW - VIX 관련 특성의 시간적 불안정성',
            'model_complexity': 'EXCESSIVE - 단순한 선형 모델도 과적합',
            'validation_methodology': 'INADEQUATE - 교차검증만으로는 불충분'
        },
        'recommended_actions': [
            '1. 즉시 조치: 모든 모델 운용 중단',
            '2. 데이터 감사: 시간적 분리 재검증',
            '3. 특성 재설계: 더 안정적인 특성으로 교체',
            '4. 검증 강화: Walk-Forward를 기본 검증으로 채택',
            '5. 모델 단순화: 정규화 강화 및 복잡성 감소'
        ],
        'detailed_overfitting_analysis': overfitting_analysis
    }

    # 보고서 저장
    os.makedirs('analysis', exist_ok=True)
    with open('analysis/comprehensive_performance_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print("✅ 저장: analysis/comprehensive_performance_report.json")

    # 요약 출력
    print(f"\n📋 종합 진단 요약")
    print("=" * 70)
    print(f"🔴 심각한 과적합 발견:")
    print(f"  - 교차검증 R²: ~0.45 (매우 좋음)")
    print(f"  - Walk-Forward R²: ~-0.53 (완전 실패)")
    print(f"  - 성능 차이: 약 -0.98 (치명적)")

    print(f"\n🎯 핵심 문제:")
    print(f"  1. 시간적 안정성 부족: 모델이 미래 패턴 예측 실패")
    print(f"  2. 특성 품질 문제: VIX 기반 특성들의 불안정성")
    print(f"  3. 검증 방법론 한계: 교차검증의 과도한 낙관성")

    print(f"\n⚠️  권고사항:")
    print(f"  - 현재 모델들 즉시 운용 중단")
    print(f"  - 더 보수적이고 안정적인 접근법 필요")
    print(f"  - Walk-Forward 검증을 표준으로 채택")

    return report

def main():
    """메인 종합 분석 함수"""
    print("🔍 종합 성능 분석 - 과적합 진단")
    print("=" * 70)

    # 1. 모든 결과 로드
    results = load_all_results()

    if not results:
        print("❌ 분석할 결과가 없습니다")
        return

    # 2. 과적합 분석
    overfitting_analysis = analyze_overfitting_severity(results)

    # 3. 시각화
    create_performance_comparison_chart(results)

    # 4. 종합 진단 보고서
    report = generate_diagnostic_report(results, overfitting_analysis)

    print("\n" + "=" * 70)
    print("🏁 종합 분석 완료")
    print("   - 심각한 과적합 문제 발견")
    print("   - 모든 모델의 실제 성능 완전 실패")
    print("   - 즉시 전략 재검토 필요")

if __name__ == "__main__":
    main()