#!/usr/bin/env python3
"""
변수 상관계수 히트맵 생성
실제 데이터를 사용하여 25개 특성 + 타겟 변수 간 상관관계 시각화
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def load_spy_data():
    """SPY 데이터 로드 (2015-2024)"""
    print("📊 SPY 데이터 로드 중...")
    spy = yf.download('SPY', start='2015-01-01', end='2024-12-31', progress=False)
    spy['returns'] = spy['Close'].pct_change()

    # VIX 데이터 추가
    try:
        vix = yf.download('^VIX', start='2015-01-01', end='2024-12-31', progress=False)
        spy['vix'] = vix['Close'].reindex(spy.index, method='ffill')
    except:
        spy['vix'] = 20.0

    spy = spy.dropna()
    print(f"✅ 데이터 로드 완료: {len(spy)} 관측치")
    return spy

def create_volatility_features(data):
    """변동성 특성 생성 (25개)"""
    print("🔧 변동성 특성 생성 중...")

    features = pd.DataFrame(index=data.index)
    returns = data['returns']
    high = data['High']
    low = data['Low']
    prices = data['Close']

    # 1. 기본 변동성 (3개)
    for window in [5, 10, 20]:
        features[f'volatility_{window}'] = returns.rolling(window).std()
        features[f'realized_vol_{window}'] = features[f'volatility_{window}'] * np.sqrt(252)

    # 2. 지수 가중 변동성 (3개)
    for span in [5, 10, 20]:
        features[f'ewm_vol_{span}'] = returns.ewm(span=span).std()

    # 3. 래그 특성 (4개)
    for lag in [1, 2, 3, 5]:
        features[f'vol_lag_{lag}'] = features['volatility_5'].shift(lag)

    # 4. Garman-Klass 변동성 (2개)
    for window in [5, 10]:
        gk_vol = np.log(high / low) ** 2
        features[f'garman_klass_{window}'] = gk_vol.rolling(window).mean()

    # 5. 일중 변동성 (2개)
    for window in [5, 10]:
        intraday_range = (high - low) / prices
        features[f'intraday_vol_{window}'] = intraday_range.rolling(window).mean()

    # 6. VIX 특성 (4개 선택)
    if 'vix' in data.columns:
        vix = data['vix']
        features['vix_level'] = vix
        for window in [5, 20]:
            features[f'vix_ma_{window}'] = vix.rolling(window).mean()
        features[f'vix_std_{20}'] = vix.rolling(20).std()

    # 7. HAR 특성 (3개)
    features['rv_daily'] = features['volatility_5']
    features['rv_weekly'] = returns.rolling(5).std()
    features['rv_monthly'] = returns.rolling(22).std()

    print(f"✅ 특성 생성 완료: {len(features.columns)}개")
    return features

def create_target_volatility(data, horizon=5):
    """타겟 변동성 생성 (미래 t+1 ~ t+horizon)"""
    print(f"🎯 타겟 변동성 생성 중 (horizon={horizon})...")

    returns = data['returns']
    target = []

    for i in range(len(returns)):
        if i + horizon < len(returns):
            future_returns = returns.iloc[i+1:i+1+horizon]
            target.append(future_returns.std())
        else:
            target.append(np.nan)

    print(f"✅ 타겟 생성 완료")
    return pd.Series(target, index=data.index, name='target_vol_5d')

def select_top_25_features(features, target):
    """타겟과 상관계수 기반 상위 25개 특성 선택"""
    print("📊 상위 25개 특성 선택 중...")

    # 결합 후 NaN 제거
    combined = pd.concat([features, target], axis=1).dropna()

    # 타겟과의 상관계수 계산
    correlations = combined.corr()[target.name].drop(target.name)

    # 절대값 기준 상위 25개
    top_25 = correlations.abs().nlargest(25).index.tolist()

    print(f"✅ 선택 완료: {len(top_25)}개 특성")
    return top_25, combined

def create_correlation_heatmap(data, save_path):
    """상관계수 히트맵 생성"""
    print("📈 상관계수 히트맵 생성 중...")

    # 상관계수 행렬 계산
    corr_matrix = data.corr()

    # 그래프 크기 설정 (변수가 많으므로 큰 크기)
    fig, ax = plt.subplots(figsize=(20, 18))

    # 히트맵 생성
    sns.heatmap(
        corr_matrix,
        annot=False,  # 숫자 표시 안함 (너무 많아서)
        fmt='.2f',
        cmap='RdBu_r',  # Red-Blue reversed (양수=빨강, 음수=파랑)
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={
            'label': 'Correlation Coefficient',
            'shrink': 0.8
        },
        vmin=-1,
        vmax=1,
        ax=ax
    )

    # 제목 및 레이블
    plt.title(
        'Feature Correlation Heatmap - SPY Volatility Prediction\n'
        '25 Selected Features + Target Variable (target_vol_5d)',
        fontsize=16,
        fontweight='bold',
        pad=20
    )

    plt.xlabel('Features', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')

    # 축 레이블 회전
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    # 레이아웃 조정
    plt.tight_layout()

    # 저장
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 히트맵 저장 완료: {save_path}")

    plt.close()

def create_target_correlation_barplot(data, save_path):
    """타겟 변수와의 상관계수 막대 그래프"""
    print("📊 타겟 상관계수 막대 그래프 생성 중...")

    # 타겟과의 상관계수
    target_corr = data.corr()['target_vol_5d'].drop('target_vol_5d').sort_values(ascending=False)

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(14, 10))

    # 색상 설정 (양수=초록, 음수=빨강)
    colors = ['#2E7D32' if x > 0 else '#D32F2F' for x in target_corr.values]

    # 막대 그래프
    bars = ax.barh(range(len(target_corr)), target_corr.values, color=colors, alpha=0.7, edgecolor='black')

    # Y축 레이블
    ax.set_yticks(range(len(target_corr)))
    ax.set_yticklabels(target_corr.index, fontsize=10)

    # 축 레이블 및 제목
    ax.set_xlabel('Correlation with target_vol_5d', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.set_title(
        'Feature Correlation with Target Variable (target_vol_5d)\n'
        'Top 25 Features Ranked by Absolute Correlation',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    # 그리드
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.axvline(x=0, color='black', linewidth=1, linestyle='-')

    # 값 표시
    for i, (bar, val) in enumerate(zip(bars, target_corr.values)):
        ax.text(
            val + (0.01 if val > 0 else -0.01),
            i,
            f'{val:.3f}',
            va='center',
            ha='left' if val > 0 else 'right',
            fontsize=9,
            fontweight='bold'
        )

    # 레이아웃 조정
    plt.tight_layout()

    # 저장
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 막대 그래프 저장 완료: {save_path}")

    plt.close()

def create_clustered_heatmap(data, save_path):
    """클러스터링된 상관계수 히트맵 (유사한 변수끼리 그룹화)"""
    print("🔬 클러스터링 히트맵 생성 중...")

    # 상관계수 행렬
    corr_matrix = data.corr()

    # Clustermap 생성 (계층적 클러스터링)
    g = sns.clustermap(
        corr_matrix,
        method='complete',  # 완전 연결법
        metric='euclidean',  # 유클리드 거리
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        figsize=(20, 18),
        linewidths=0.5,
        cbar_kws={
            'label': 'Correlation Coefficient',
            'shrink': 0.8
        },
        dendrogram_ratio=0.1,
        cbar_pos=(0.02, 0.83, 0.03, 0.15)
    )

    # 제목
    g.fig.suptitle(
        'Hierarchical Clustered Correlation Heatmap\n'
        'Features Grouped by Similarity',
        fontsize=16,
        fontweight='bold',
        y=0.98
    )

    # 축 레이블 크기
    g.ax_heatmap.tick_params(labelsize=10)

    # 저장
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 클러스터링 히트맵 저장 완료: {save_path}")

    plt.close()

def save_correlation_csv(data, save_path):
    """상관계수 행렬을 CSV로 저장"""
    print("💾 상관계수 CSV 저장 중...")

    corr_matrix = data.corr()
    corr_matrix.to_csv(save_path, float_format='%.4f')

    print(f"✅ CSV 저장 완료: {save_path}")

def main():
    """메인 실행 함수"""
    print("="*70)
    print("Feature Correlation Heatmap Generation")
    print("="*70)
    print()

    # 1. 데이터 로드
    spy_data = load_spy_data()

    # 2. 특성 생성
    features = create_volatility_features(spy_data)

    # 3. 타겟 생성
    target = create_target_volatility(spy_data)

    # 4. 상위 25개 특성 선택
    top_25_features, combined_data = select_top_25_features(features, target)

    # 5. 선택된 특성 + 타겟 데이터프레임
    final_data = combined_data[top_25_features + ['target_vol_5d']]

    print(f"\n최종 데이터 shape: {final_data.shape}")
    print(f"특성 개수: {len(top_25_features)}")
    print(f"샘플 개수: {len(final_data)}")
    print()

    # 6. 출력 디렉토리 생성
    output_dir = Path('/root/workspace/paper/figures/correlation')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 7. 히트맵 생성
    print("="*70)
    print("Generating Visualizations")
    print("="*70)
    print()

    # 7-1. 기본 히트맵
    create_correlation_heatmap(
        final_data,
        output_dir / 'correlation_heatmap.png'
    )

    # 7-2. 타겟 상관계수 막대 그래프
    create_target_correlation_barplot(
        final_data,
        output_dir / 'target_correlation_barplot.png'
    )

    # 7-3. 클러스터링 히트맵
    create_clustered_heatmap(
        final_data,
        output_dir / 'correlation_heatmap_clustered.png'
    )

    # 8. CSV 저장
    save_correlation_csv(
        final_data,
        output_dir / 'correlation_matrix.csv'
    )

    # 9. 요약 통계
    print()
    print("="*70)
    print("Summary Statistics")
    print("="*70)
    print()

    target_corr = final_data.corr()['target_vol_5d'].drop('target_vol_5d')

    print(f"타겟과 상관계수 통계:")
    print(f"  - 평균: {target_corr.mean():.4f}")
    print(f"  - 표준편차: {target_corr.std():.4f}")
    print(f"  - 최대값: {target_corr.max():.4f} ({target_corr.idxmax()})")
    print(f"  - 최소값: {target_corr.min():.4f} ({target_corr.idxmin()})")
    print()

    print("상위 10개 특성 (타겟과의 상관계수):")
    for i, (feat, corr) in enumerate(target_corr.abs().nlargest(10).items(), 1):
        actual_corr = target_corr[feat]
        print(f"  {i:2d}. {feat:25s} : {actual_corr:+.4f}")

    print()
    print("="*70)
    print("✅ All visualizations completed!")
    print("="*70)
    print()
    print(f"출력 디렉토리: {output_dir}")
    print("생성된 파일:")
    print("  - correlation_heatmap.png            (기본 히트맵)")
    print("  - correlation_heatmap_clustered.png  (클러스터링 히트맵)")
    print("  - target_correlation_barplot.png     (타겟 상관계수 막대 그래프)")
    print("  - correlation_matrix.csv             (상관계수 행렬 CSV)")
    print()

    return final_data

if __name__ == '__main__':
    final_data = main()
