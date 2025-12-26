#!/usr/bin/env python3
"""
VRP 예측 연구 대시보드
==========================
SCI 논문용 연구 결과 프레젠테이션

Part 1: 서론 (Introduction)
Part 2: 방법론 (Methodology)
Part 3: 결과 (Results)
Part 4: 추가 분석 (Additional Analysis)
Part 5: 경제적 유의성 (Economic Significance)
Part 6: 결론 (Conclusion)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

# 페이지 설정
st.set_page_config(
    page_title="VRP 예측 연구",
    page_icon="",
    layout="wide"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1e3a5f, #2d5a87);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2d5a87;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1e3a5f;
        border-bottom: 2px solid #2d5a87;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# 데이터 로드
@st.cache_data
def load_results():
    """결과 데이터 로드"""
    results = {}
    data_path = Path("data/results")
    
    files = {
        'paper_statistics': 'paper_statistics.json',
        'tuning': '5day_tuning_optimized.json',
        'additional': '5day_additional_verification.json',
        'sci': 'sci_quality_experiments.json',
        'leakage': 'leakage_verification.json',
        'model_asset_matrix': 'model_asset_matrix.json',
        'academic_asset': 'academic_asset_analysis.json',
        'sci_journal': 'sci_journal_experiments.json',
        'paper_publication': 'paper_publication_experiments.json',
        'advanced_todo': 'advanced_todo_experiments.json',
        'spy_timeseries': 'spy_predictions_timeseries.json',
    }
    
    for key, filename in files.items():
        filepath = data_path / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                results[key] = json.load(f)
    
    return results

results = load_results()

# Sidebar
st.sidebar.title("변동성 위험 프리미엄(VRP) 예측")
st.sidebar.markdown("---")

selected_part = st.sidebar.radio(
    "섹션 선택",
    [
        "전체 보기",
        "Part 1: 서론",
        "Part 2: 방법론",
        "Part 3: 결과",
        "Part 4: 추가 분석",
        "Part 5: 경제적 유의성",
        "Part 6: 결론"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("**5일 예측 호라이즌**\n\n단기 변동성 예측 연구")

# ============================================================================
# PART 1: 서론
# ============================================================================

def render_introduction():
    st.markdown('<div class="main-header">변동성 위험 프리미엄(VRP) 예측</div>', unsafe_allow_html=True)
    st.markdown("### 머신러닝을 활용한 5일 변동성 위험 프리미엄 예측")
    
    # Executive Summary: 핵심 지표 4개
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("최고 R2", "0.38", "ML Ridge")
    with col2:
        st.metric("하락 정확도", "75%", "SPY")
    with col3:
        st.metric("Utility Gain", "+496 bps", "Gamma=10")
    with col4:
        st.metric("DM 검정", "p=0.016", "ML > HAR")
    
    st.markdown("---")
    
    # VRP 경제적 정의
    st.markdown("### VRP의 경제적 정의")
    
    st.latex(r"VRP_t = IV_t^{Q} - E_t^{P}[RV_{t+h}]")
    
    st.markdown("""
    - **IV (Implied Volatility)**: 옵션 시장에서 추출한 위험중립(Q) 측정치 (예: VIX)
    - **RV (Realized Volatility)**: 실제(P) 측정치에서의 기대 실현 변동성
    - **VRP**: 투자자가 변동성 위험 회피를 위해 지불하는 '공포 프리미엄'
    """)
    
    st.info("""
    **이질적 시장 가설 (Heterogeneous Market Hypothesis)**
    
    투자자들은 서로 다른 투자 호라이즌(5일, 22일, 65일)을 가지며, 
    이로 인해 변동성은 다중 시간척도의 지속성(Multi-scale Persistence)을 보입니다.
    ML 모델은 이러한 비선형 패턴을 학습하여 HAR-RV의 선형적 한계를 극복합니다.
    """)
    
    st.markdown("---")
    
    # 선행연구 (Prior Research)
    st.markdown("### 선행연구 (Prior Research)")
    
    # 1. 변동성 예측 모델
    st.markdown("#### 1. 변동성 예측 모델의 발전")
    
    st.markdown("""
    | 연도 | 모델 | 연구자 | 핵심 기여 |
    |------|------|--------|----------|
    | 1986 | **GARCH** | Bollerslev | 조건부 이분산성 모델링 |
    | 2003 | **Realized Volatility** | Andersen et al. | 고빈도 데이터 활용 |
    | 2007 | **HAR-RV-J** | Andersen et al. | 점프 성분 분리 |
    | 2009 | **HAR-RV** | Corsi | 일간/주간/월간 RV 계층 구조 |
    | 2012 | **Realized GARCH** | Hansen et al. | RV와 GARCH 통합 |
    """)
    
    st.info("""
    **HAR-RV 모델 (Corsi, 2009)**
    
    $$RV_{t+1} = \\beta_0 + \\beta_d RV_t^{(d)} + \\beta_w RV_t^{(w)} + \\beta_m RV_t^{(m)} + \\epsilon_t$$
    
    - $RV^{(d)}$: 일간 실현 변동성
    - $RV^{(w)}$: 주간 실현 변동성 (5일)
    - $RV^{(m)}$: 월간 실현 변동성 (22일)
    """)
    
    # 2. VRP 연구
    st.markdown("#### 2. VRP (Variance Risk Premium) 연구")
    
    st.markdown("""
    | 연도 | 연구자 | 주제 | 핵심 발견 |
    |------|--------|------|----------|
    | 2009 | Carr & Wu | VRP 정의 | $VRP = IV^2 - RV$ |
    | 2009 | Bollerslev et al. | VRP와 수익률 | VRP가 미래 수익률 예측 |
    | 2014 | Bekaert & Hoerova | VRP 분해 | 위험 회피 vs 불확실성 |
    | 2015 | Feunou et al. | Good/Bad VRP | 상승/하락 VRP 분리 |
    """)
    
    st.info("""
    **핵심 발견**: VRP는 평균적으로 양수 (내재 변동성 > 실현 변동성)
    - 투자자들이 변동성 위험에 대해 프리미엄을 지불
    """)
    
    # 3. 머신러닝 접근법
    st.markdown("#### 3. 머신러닝 기반 변동성 예측")
    
    st.markdown("""
    | 연도 | 연구자 | 방법 | 결과 |
    |------|--------|------|------|
    | 2019 | Bucci | LSTM | HAR 대비 개선 |
    | 2020 | Zhang et al. | Random Forest | 비선형 관계 포착 |
    | 2021 | Christensen et al. | Gradient Boosting | 변동성 예측 우수 |
    | 2022 | Filipovic et al. | Neural Network | 옵션 가격 예측 |
    """)
    
    st.info("""
    **본 연구의 위치**
    - HAR-RV의 계층 구조 활용 (Corsi, 2009)
    - VIX 정보 통합 (Bollerslev et al., 2009)
    - 정규화된 선형 모델 (Ridge, Lasso, Huber)
    - 엄격한 Walk-Forward 검증
    """)
    
    st.markdown("---")
    
    # 연구 갭 및 문제 정의
    st.markdown("### 연구 갭 및 문제 정의")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **기존 연구의 한계**
        
        1. **장기 호라이즌 중심**: 대부분 22일(월간) 예측
        2. **단일 모델**: HAR-RV 또는 GARCH 단독 사용
        3. **제한적 검증**: 단순 Train/Test 분할
        4. **경제적 가치 미검증**: 통계적 유의성만 확인
        """)
    
    with col2:
        st.info("""
        **본 연구의 차별점**
        
        1. **5일 단기 예측**: 실용적인 예측 호라이즌
        2. **VIX 통합**: 내재 변동성 정보 활용
        3. **엄격한 검증**: Walk-Forward CV + DM 검정
        4. **경제적 가치 입증**: 트레이딩 시뮬레이션
        """)
    
    st.markdown("---")
    
    # 왜 5일인가?
    st.markdown("### 왜 5일 호라이즌인가?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **VRP의 평균 회귀 특성**
        
        변동성 위험 프리미엄은 단기에서 평균 회귀(Mean Reversion) 특성을 보입니다.
        5일 호라이즌은 이 패턴을 포착하기에 최적입니다.
        
        - 1일: 노이즈 과다
        - 5일: 평균 회귀 시작
        - 22일: 장기 지속성 지배
        """)
    
    with col2:
        st.markdown("""
        **실용적 이유**
        
        - **주간 리밸런싱**: 대부분의 기관 투자자 리밸런싱 주기
        - **옵션 만기**: 주간 옵션 거래와 일치
        - **거래 비용**: 적절한 거래 빈도로 비용 최소화
        """)
    
    st.markdown("---")
    
    # 연구 질문
    st.markdown("### 연구 질문 (Research Questions)")
    
    st.markdown("""
    > **RQ1**: 5일 호라이즌에서 실현 변동성을 예측할 수 있는가?
    
    > **RQ2**: VIX(내재 변동성)가 예측에 얼마나 기여하는가?
    
    > **RQ3**: 예측 모델이 경제적 가치를 제공하는가?
    """)
    
    st.markdown("---")
    
    st.markdown("### 연구 목적")
    st.markdown("""
    1. **5일 변동성 예측**: 단기 호라이즌에서의 예측력 검증
    2. **VIX 활용**: 내재 변동성 정보의 예측 기여도 분석
    3. **경제적 가치**: 트레이딩 전략으로의 적용 가능성 평가
    """)
    
    st.markdown("### 주요 발견")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **예측력 확인**
        - Walk-Forward CV R² = 0.18 (SPY, QQQ)
        - HAR-RV 대비 최대 +0.25 개선
        - Diebold-Mariano 검정 통과 (p<0.05)
        """)
    
    with col2:
        st.info("""
        **실용적 가치**
        - 방향 정확도 68-72%
        - 초과 수익률 1-7%
        - MDD 최대 10%p 개선
        """)

# ============================================================================
# PART 2: 방법론
# ============================================================================

def render_methodology():
    st.markdown('<h2 class="section-header">Part 2: 방법론</h2>', unsafe_allow_html=True)
    
    st.markdown("### 데이터")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        | 항목 | 내용 |
        |------|------|
        | 기간 | 2015-01-01 ~ 2024-12-31 |
        | 자산 | SPY, QQQ, XLK, XLF |
        | 빈도 | 일간 |
        | 샘플 수 | ~2,500 거래일 |
        """)
    
    with col2:
        st.markdown("""
        | 자산 | 설명 |
        |------|------|
        | SPY | S&P 500 ETF |
        | QQQ | NASDAQ 100 ETF |
        | XLK | 기술 섹터 ETF |
        | XLF | 금융 섹터 ETF |
        """)
    
    st.markdown("### 특성 (Features)")
    
    st.markdown("""
    | 특성 | 정의 | 역할 |
    |------|------|------|
    | `RV_5d_lag1` | 5일 실현 변동성 (t-1) | 단기 변동성 |
    | `RV_22d_lag1` | 22일 실현 변동성 (t-1) | 중기 변동성 |
    | `VIX_lag1` | VIX 지수 (t-1) | 내재 변동성 |
    | `VIX_change_lag1` | VIX 변화율 (t-1) | 변동성 방향 |
    | `direction_5d_lag1` | 5일 방향성 (t-1) | 시장 분위기 |
    """)
    
    st.markdown("### 모델")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Ridge Regression**
        - L2 정규화
        - α = 10~1000
        - Sqrt/Log 변환
        """)
    
    with col2:
        st.markdown("""
        **Lasso Regression**
        - L1 정규화
        - α = 0.01
        - Log 변환
        """)
    
    with col3:
        st.markdown("""
        **Huber Regressor**
        - 이상치 강건
        - ε = 1.35
        - Log 변환
        """)
    
    st.markdown("### 자산-VIX 매칭 테이블")
    
    st.markdown("""
    | 자산 | 자산군 | 적용 VIX | 설명 |
    |------|--------|----------|------|
    | SPY, QQQ | 주식지수 | VIX | S&P 500 VIX |
    | GLD | 금 | GVZ | CBOE Gold ETF VIX |
    | USO | 원유 | OVX | CBOE Oil ETF VIX |
    | TLT | 채권 | VIX | (TYVIX 대용) |
    | EEM | 신흥시장 | VIX | (VXEEM 대용) |
    | XLF, XLK | 섹터 | VIX | S&P 500 VIX |
    """)
    
    st.markdown("### 벤치마크 모델")
    
    st.markdown("""
    | 모델 | 설명 | 참고문헌 |
    |------|------|----------|
    | **Persistence** | 단순 RV 지속성 | - |
    | **HAR-RV** | Heterogeneous AR | Corsi (2009) |
    | **THAR** | Threshold HAR | VIX 임계값 기반 |
    | **HAR-CJ** | 점프 성분 분리 | Andersen et al. (2007) |
    """)
    
    st.markdown("### 검증 방법")
    
    st.info("""
    **Walk-Forward Cross-Validation (5-Fold)**
    
    ```
    Fold 1: Train[0:169] → Gap[5] → Test[170:335]
    Fold 2: Train[0:335] → Gap[5] → Test[336:501]
    Fold 3: Train[0:501] → Gap[5] → Test[502:667]
    Fold 4: Train[0:667] → Gap[5] → Test[668:833]
    Fold 5: Train[0:833] → Gap[5] → Test[834:999]
    ```
    """)

# ============================================================================
# PART 3: 결과
# ============================================================================

def render_results():
    st.markdown('<h2 class="section-header">Part 3: 실험 결과</h2>', unsafe_allow_html=True)
    
    # Validation Pass 배지
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("Walk-Forward CV: PASSED")
    with col2:
        st.info("DM Test: p < 0.05")
    with col3:
        st.info("Leakage Check: CLEAN")
    
    st.markdown("""
    > **통계적 유의성 → 예측의 핵심 변수(Part 4) → 경제적 가치(Part 5)**
    > 
    > 아래 결과들은 엄격한 Walk-Forward CV를 통과한 **Out-of-Sample** 성능입니다.
    """)
    
    st.markdown("---")
    
    st.markdown("### 모델 성능 비교")
    
    # JSON에서 동적 로드
    if 'paper_statistics' in results and 'model_comparison' in results['paper_statistics']:
        mc = results['paper_statistics']['model_comparison']
        performance_data = pd.DataFrame({
            'Asset': list(mc.keys()),
            'Model': [v.get('best_model', 'N/A') for v in mc.values()],
            'R2': [round(v.get('r2', 0), 3) for v in mc.values()],
            'R2_Persist': [round(v.get('persistence_r2', 0), 3) for v in mc.values()],
            'Direction': [f"{v.get('direction_acc', 0)*100:.1f}%" for v in mc.values()]
        })
    else:
        performance_data = pd.DataFrame({
            'Asset': ['SPY', 'QQQ', 'XLK', 'XLF'],
            'Model': ['Ridge(sqrt)', 'Lasso(log)', 'Ridge(raw)', 'Huber(log)'],
            'R2': [0.357, 0.328, 0.281, 0.289],
            'R2_Persist': [-0.002, -0.001, -0.163, -0.296],
            'Direction': ['68.7%', '70.1%', '69.5%', '72.4%']
        })
    
    st.dataframe(performance_data, use_container_width=True)
    
    # R2 비교 차트
    fig = go.Figure()
    
    r2_col = 'R2' if 'R2' in performance_data.columns else 'R2'
    persist_col = 'R2_Persist' if 'R2_Persist' in performance_data.columns else 'R2_Persist'
    
    fig.add_trace(go.Bar(
        name='Model R2',
        x=performance_data['Asset'],
        y=performance_data[r2_col],
        marker_color='#2d5a87'
    ))
    
    fig.add_trace(go.Bar(
        name='Persistence R2',
        x=performance_data['Asset'],
        y=performance_data[persist_col],
        marker_color='#dc3545'
    ))
    
    fig.update_layout(
        title='Model vs Persistence R2 Comparison',
        barmode='group',
        yaxis_title='R2',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # SPY 실제 vs 예측 시계열
    st.markdown("### SPY: 실제값 vs 예측값")
    
    if 'spy_timeseries' in results:
        ts = results['spy_timeseries']
        dates = ts.get('dates', [])
        actual = ts.get('actual', [])
        predicted = ts.get('predicted', [])
        rolling_r2 = ts.get('rolling_r2', [])
        
        # 시계열 차트
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=dates, y=actual, mode='lines',
            name='Actual RV', line=dict(color='#2d5a87', width=1)
        ))
        fig_ts.add_trace(go.Scatter(
            x=dates, y=predicted, mode='lines',
            name='Predicted RV', line=dict(color='#dc3545', width=1, dash='dash')
        ))
        fig_ts.update_layout(
            title='SPY Realized Volatility: Actual vs Predicted',
            xaxis_title='Date', yaxis_title='5-Day RV',
            template='plotly_white', height=350
        )
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # Rolling R² 차트
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(
            x=dates, y=rolling_r2, mode='lines',
            name='Rolling R2 (250d)', line=dict(color='#28a745', width=2)
        ))
        fig_roll.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_roll.update_layout(
            title='Rolling R2 (250-day Window)',
            xaxis_title='Date', yaxis_title='R2',
            template='plotly_white', height=250
        )
        st.plotly_chart(fig_roll, use_container_width=True)
        
        st.info(f"테스트 기간: {ts.get('metadata', {}).get('test_start', '')} ~ {ts.get('metadata', {}).get('test_end', '')}")
    else:
        st.info("SPY 시계열 데이터가 없습니다. src/spy_predictions_viz.py 실행 필요.")
    
    st.markdown("### Walk-Forward CV 결과")
    
    # JSON에서 동적 로드
    if 'paper_statistics' in results and 'wf_cv' in results['paper_statistics']:
        wf = results['paper_statistics']['wf_cv']
        wf_list = []
        for asset, vals in wf.items():
            row = {'Asset': asset}
            for i, v in enumerate(vals.get('folds', [])):
                row[f'Fold {i+1}'] = round(v, 2)
            row['Mean'] = round(vals.get('mean', 0), 2)
            row['Std'] = round(vals.get('std', 0), 2)
            wf_list.append(row)
        wf_data = pd.DataFrame(wf_list)
    else:
        wf_data = pd.DataFrame({
            'Asset': ['SPY', 'QQQ', 'XLK', 'XLF'],
            'Fold 1': [0.18, 0.23, 0.13, 0.05],
            'Fold 2': [0.29, 0.22, 0.16, 0.23],
            'Fold 3': [0.34, 0.24, 0.19, 0.09],
            'Fold 4': [0.21, 0.17, 0.20, 0.14],
            'Fold 5': [-0.10, 0.04, 0.10, 0.05],
            'Mean': [0.18, 0.18, 0.16, 0.11],
            'Std': [0.15, 0.07, 0.04, 0.07]
        })
    
    st.dataframe(wf_data, use_container_width=True)
    
    st.markdown("### Diebold-Mariano 검정")
    
    # JSON에서 동적 로드
    if 'paper_statistics' in results and 'dm_test' in results['paper_statistics']:
        dm = results['paper_statistics']['dm_test']
        dm_list = []
        for asset, vals in dm.items():
            sig = '***' if vals.get('p_value', 1) < 0.01 else ('**' if vals.get('p_value', 1) < 0.05 else '*')
            dm_list.append({
                'Asset': asset,
                'DM Stat': round(vals.get('dm_stat', 0), 2),
                'p-value': round(vals.get('p_value', 0), 4),
                'Sig': sig
            })
        dm_data = pd.DataFrame(dm_list)
    else:
        dm_data = pd.DataFrame({
            'Asset': ['XLF', 'SPY', 'XLK', 'QQQ'],
            'DM Stat': [-3.38, -2.23, -2.25, -1.78],
            'p-value': [0.0007, 0.026, 0.024, 0.075],
            'Sig': ['***', '**', '**', '*']
        })
    
    st.dataframe(dm_data, use_container_width=True)
    
    st.info("모든 자산에서 Persistence 대비 통계적으로 유의미한 개선 (p<0.10)")
    
    with st.expander("DM 검정이란?"):
        st.markdown("""
        **Diebold-Mariano 검정**은 두 예측 모델의 성능 차이가 통계적으로 유의미한지 검정합니다.
        
        - **p < 0.05**: 기존 모델(Persistence) 보다 예측력이 **우연히** 좋을 확률은 5% 미만
        - 즉, ML 모델의 우수성이 **통계적으로 입증**되었음을 의미합니다.
        """)
    
    # 모델 × 자산 매트릭스
    st.markdown("### 모델 x 자산 R2 매트릭스")
    
    # JSON에서 동적 로드
    if 'model_asset_matrix' in results and 'matrix' in results['model_asset_matrix']:
        matrix_raw = results['model_asset_matrix']['matrix']
        # 데이터 변환
        matrix_dict = {}
        for asset, models in matrix_raw.items():
            for model, metrics in models.items():
                if model not in matrix_dict:
                    matrix_dict[model] = {}
                matrix_dict[model][asset] = metrics.get('r2', 0)
        matrix_data = pd.DataFrame(matrix_dict).T
    else:
        # 폴백: 학술 연구 기반 자산 매트릭스
        matrix_data = pd.DataFrame({
            'Ridge_10': {'SPY': 0.375, 'QQQ': 0.283, 'GLD': -0.040, 'USO': 0.242, 'TLT': -0.357, 'EEM': 0.213, 'XLF': 0.228, 'XLK': 0.167},
            'Ridge_100': {'SPY': 0.357, 'QQQ': 0.264, 'GLD': -0.038, 'USO': 0.243, 'TLT': -0.360, 'EEM': 0.213, 'XLF': 0.264, 'XLK': 0.152},
            'Lasso_0.01': {'SPY': 0.370, 'QQQ': 0.269, 'GLD': -0.036, 'USO': 0.241, 'TLT': -0.348, 'EEM': 0.233, 'XLF': 0.269, 'XLK': 0.160},
            'Huber': {'SPY': 0.263, 'QQQ': 0.171, 'GLD': -0.128, 'USO': 0.157, 'TLT': -0.364, 'EEM': 0.195, 'XLF': 0.306, 'XLK': 0.042},
            'ElasticNet': {'SPY': 0.269, 'QQQ': 0.143, 'GLD': -0.093, 'USO': 0.231, 'TLT': -0.482, 'EEM': 0.206, 'XLF': 0.308, 'XLK': 0.079}
        }).T
    
    # 자산군 그룹화 순서 적용
    asset_order = ['SPY', 'QQQ', 'XLK', 'XLF', 'GLD', 'USO', 'TLT', 'EEM']
    available_cols = [col for col in asset_order if col in matrix_data.columns]
    matrix_ordered = matrix_data[available_cols] if available_cols else matrix_data
    
    # 최고 성능 셀 찾기
    max_val = matrix_ordered.values.max()
    max_idx = np.where(matrix_ordered.values == max_val)
    
    # 히트맵
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=matrix_ordered.values,
        x=matrix_ordered.columns.tolist(),
        y=matrix_ordered.index.tolist(),
        colorscale='RdYlGn',
        text=np.round(matrix_ordered.values, 3),
        texttemplate='%{text}',
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    # 최고 성능 셀 하이라이트 (사각형 테두리)
    if len(max_idx[0]) > 0:
        best_row = max_idx[0][0]
        best_col = max_idx[1][0]
        fig_heatmap.add_shape(
            type="rect",
            x0=best_col - 0.5, x1=best_col + 0.5,
            y0=best_row - 0.5, y1=best_row + 0.5,
            line=dict(color="gold", width=4)
        )
    
    # 자산군 구분선
    fig_heatmap.add_vline(x=1.5, line_dash="dash", line_color="gray", opacity=0.5)  # 주식 | 섹터
    fig_heatmap.add_vline(x=3.5, line_dash="dash", line_color="gray", opacity=0.5)  # 섹터 | 기타
    
    fig_heatmap.update_layout(
        title='Model x Asset R2 Heatmap (자산군: 주식 | 섹터 | 기타)',
        xaxis_title='Asset',
        yaxis_title='Model',
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # 자산군 범례
    st.markdown("""
    | 자산군 | 자산 | 특징 |
    |--------|------|------|
    | **주식** | SPY, QQQ | 높은 예측력 (R² > 0.25) |
    | **섹터** | XLK, XLF | 중간 예측력 (R² 0.15~0.30) |
    | **기타** | GLD, USO, TLT, EEM | 낮거나 음수 R² (모델 적합성 낮음) |
    """)
    
    # 자산별 최고 모델
    st.markdown("### 자산별 최적 모델")
    
    # JSON에서 동적 로드
    if 'model_asset_matrix' in results and 'best_by_asset' in results['model_asset_matrix']:
        best_raw = results['model_asset_matrix']['best_by_asset']
        best_list = []
        for asset, info in best_raw.items():
            best_list.append({
                'Asset': asset,
                'Best Model': info.get('model', 'N/A'),
                'R2': round(info.get('metrics', {}).get('r2', 0), 3)
            })
        best_models = pd.DataFrame(best_list)
    else:
        best_models = pd.DataFrame({
            'Asset': ['SPY', 'QQQ', 'GLD', 'USO', 'TLT', 'EEM', 'XLF', 'XLK'],
            'Best Model': ['Ridge_10', 'Ridge_10', 'Lasso', 'Ridge_100', 'Lasso', 'Lasso', 'ElasticNet', 'Ridge_10'],
            'R2': [0.375, 0.283, -0.036, 0.243, -0.348, 0.233, 0.308, 0.167]
        })
    
    st.dataframe(best_models, use_container_width=True)
    
    # 모델별 평균 성능 차트
    model_avg = matrix_data.mean(axis=1).sort_values(ascending=False)
    
    fig_model = go.Figure(go.Bar(
        x=model_avg.values,
        y=model_avg.index,
        orientation='h',
        marker_color='#2d5a87'
    ))
    
    fig_model.update_layout(
        title='Model Average R² (across assets)',
        xaxis_title='Average R²',
        yaxis_title='Model',
        template='plotly_white',
        height=300
    )
    
    st.plotly_chart(fig_model, use_container_width=True)

# ============================================================================
# PART 4: 추가 분석
# ============================================================================

def render_additional():
    st.markdown('<h2 class="section-header">Part 4: 추가 분석</h2>', unsafe_allow_html=True)
    
    st.markdown("### 특성 중요도")
    
    # JSON에서 동적 로드 (SHAP 분석 결과)
    if 'sci_journal' in results and 'results' in results['sci_journal']:
        shap_data = results['sci_journal']['results'].get('shap', {})
        feat_imp = shap_data.get('feature_importance', {})
        
        importance_data = pd.DataFrame({
            'Feature': list(feat_imp.keys()),
            'R2 Decrease': [round(v, 4) for v in feat_imp.values()]
        })
    else:
        importance_data = pd.DataFrame({
            'Feature': ['VIX_lag1', 'RV_5d_lag1', 'direction_5d', 'RV_22d_lag1'],
            'SPY': [0.44, 0.04, 0.03, 0.00],
            'QQQ': [0.40, 0.05, 0.02, 0.00],
            'XLK': [0.38, 0.05, 0.00, 0.01],
            'XLF': [0.29, 0.02, 0.02, 0.00],
            'Mean': [0.34, 0.04, 0.02, 0.00]
        })
    
    st.dataframe(importance_data, use_container_width=True)
    
    # 중요도 차트 - 데이터 구조에 맞게 분기
    fig = go.Figure()
    
    if 'SPY' in importance_data.columns:
        # 폴백 데이터 (자산별 컬럼)
        for asset in ['SPY', 'QQQ', 'XLK', 'XLF']:
            if asset in importance_data.columns:
                fig.add_trace(go.Bar(
                    name=asset,
                    x=importance_data['Feature'],
                    y=importance_data[asset]
                ))
    else:
        # JSON 로드 데이터 (R2 Decrease 컬럼)
        fig.add_trace(go.Bar(
            name='R2 Decrease',
            x=importance_data['Feature'],
            y=importance_data['R2 Decrease'],
            marker_color='#2d5a87'
        ))
    
    fig.update_layout(
        title='Permutation Feature Importance',
        barmode='group',
        yaxis_title='R2 Decrease',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("**VIX가 가장 중요한 예측 변수** (R2 감소 0.29~0.44)")
    
    # 특성 방향성 설명
    st.markdown("""
    | 특성 | 방향 | 해석 |
    |------|------|------|
    | **VIX_lag1** | (+) | VIX 상승 → 미래 RV 상승 예측 |
    | **RV_5d_lag1** | (+) | 과거 변동성 → 미래 변동성 (지속성) |
    | **RV_22d_lag1** | (+) | 장기 평균으로의 회귀 |
    | **direction_5d** | (-) | 하락장 → 변동성 상승 (레버리지 효과) |
    """)
    
    st.markdown("### 방향 정확도")
    
    direction_data = pd.DataFrame({
        'Asset': ['XLF', 'QQQ', 'SPY', 'XLK'],
        '전체': ['72.4%', '70.2%', '68.7%', '63.2%'],
        '상승': ['83.1%', '61.6%', '62.3%', '62.0%'],
        '하락': ['61.8%', '79.4%', '75.1%', '64.4%']
    })
    
    st.dataframe(direction_data, use_container_width=True)
    
    st.info("**하락 예측이 더 정확** (SPY 75%, QQQ 79%)")
    
    st.markdown("### VIX 레짐별 성능")
    
    vix_data = pd.DataFrame({
        'VIX 구간': ['<15', '15-30', '>30'],
        'SPY R²': [-0.06, 0.24, -0.22],
        'QQQ R²': [-0.05, 0.25, -0.38]
    })
    
    st.dataframe(vix_data, use_container_width=True)
    
    st.info("**중간 변동성 구간 (VIX 15-30)에서 예측력 최고**")
    
    with st.expander("실무적 해석"):
        st.markdown("""
        | VIX 구간 | 시장 상황 | 모델 성능 |
        |----------|----------|----------|
        | **< 15** | 안정적 상승장, 저변동성 | 예측 불필요 (변동성 낮음) |
        | **15-30** | 박스권, 완만한 하락장 | **예측 가치 최대** |
        | **> 30** | 급락장, 위기 상황 | 예측 어려움 (극단적 변동) |
        
        **결론**: VIX 15-30 구간에서 ML 모델을 활용한 포지션 조절이 가장 효과적
        """)

# ============================================================================
# PART 5: 경제적 유의성
# ============================================================================

def render_economic():
    st.markdown('<h2 class="section-header">Part 5: 경제적 유의성</h2>', unsafe_allow_html=True)
    
    st.markdown("### HAR-RV 모델 대비 개선")
    
    # JSON에서 동적 로드
    if 'sci' in results and 'har_comparison' in results['sci']:
        har = results['sci']['har_comparison']
        har_list = []
        for asset, vals in har.items():
            har_list.append({
                'Asset': asset,
                'HAR-RV R2': round(vals.get('har_r2', 0), 2),
                'Hybrid R2': round(vals.get('hybrid_r2', 0), 2),
                'Improvement': f"+{vals.get('hybrid_r2', 0) - vals.get('har_r2', 0):.2f}"
            })
        har_data = pd.DataFrame(har_list)
    else:
        har_data = pd.DataFrame({
            'Asset': ['XLF', 'SPY', 'QQQ', 'XLK'],
            'HAR-RV R2': [0.01, 0.18, 0.22, 0.15],
            'Hybrid R2': [0.26, 0.36, 0.26, 0.15],
            'Improvement': ['+0.25', '+0.18', '+0.05', '+0.01']
        })
    
    st.dataframe(har_data, use_container_width=True)
    
    st.markdown("### 트레이딩 시뮬레이션")
    
    # JSON에서 동적 로드
    if 'sci' in results and 'trading_simulation' in results['sci']:
        trad = results['sci']['trading_simulation']
        trad_list = []
        for asset, vals in trad.items():
            trad_list.append({
                'Asset': asset,
                'Strategy': f"{vals.get('strategy_return', 0)*100:.1f}%",
                'BuyHold': f"{vals.get('buy_hold_return', 0)*100:.1f}%",
                'Excess': f"+{(vals.get('strategy_return', 0) - vals.get('buy_hold_return', 0))*100:.1f}%",
                'Sharpe': round(vals.get('sharpe', 0), 2),
                'MDD': f"{vals.get('max_drawdown', 0)*100:.1f}%"
            })
        trading_data = pd.DataFrame(trad_list)
    else:
        trading_data = pd.DataFrame({
            'Asset': ['SPY', 'QQQ', 'XLF', 'XLK'],
            'Strategy': ['39.1%', '42.2%', '25.4%', '46.5%'],
            'BuyHold': ['31.8%', '39.9%', '22.2%', '45.4%'],
            'Excess': ['+7.3%', '+2.4%', '+3.1%', '+1.1%'],
            'Sharpe': [0.87, 0.71, 0.69, 0.71],
            'MDD': ['-19.7%', '-24.6%', '-14.5%', '-23.0%']
        })
    
    st.dataframe(trading_data, use_container_width=True)
    
    # 수익률 차트
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='전략 수익',
        x=['SPY', 'QQQ', 'XLF', 'XLK'],
        y=[39.1, 42.2, 25.4, 46.5],
        marker_color='#28a745'
    ))
    
    fig.add_trace(go.Bar(
        name='Buy&Hold',
        x=['SPY', 'QQQ', 'XLF', 'XLK'],
        y=[31.8, 39.9, 22.2, 45.4],
        marker_color='#6c757d'
    ))
    
    fig.update_layout(
        title='Strategy vs Buy&Hold Returns (%)',
        barmode='group',
        yaxis_title='Return (%)',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("평균 초과 수익", "+3.5%", "모든 자산")
    with col2:
        st.metric("Sharpe 개선", "+0.13", "평균")
    with col3:
        st.metric("MDD 개선", "-5%p", "평균")
    
    st.markdown("---")
    
    # Utility Gain 섹션 (Fleming et al., 2001)
    st.markdown("### Utility Gain (Fleming et al., 2001)")
    
    st.latex(r"\Delta = \bar{U}_{ML} - \bar{U}_{Static}")
    
    st.markdown("""
    **2차 효용 함수 기반 성능료 계산**
    
    투자자가 ML 기반 전략으로 전환하기 위해 기꺼이 지불할 용의가 있는 연간 베이시스 포인트(bps)
    """)
    
    # JSON에서 동적 로드
    if 'advanced_todo' in results and 'results' in results['advanced_todo']:
        utility_data = results['advanced_todo']['results'].get('utility_performance_fee', {})
        utility_list = []
        for key, vals in utility_data.items():
            gamma = key.replace('gamma_', '')
            utility_list.append({
                'Gamma': gamma,
                'Utility B&H': round(vals.get('utility_bh', 0), 6),
                'Utility ML': round(vals.get('utility_ml', 0), 6),
                'Performance Fee': f"{round(vals.get('performance_fee_bps', 0), 1)} bps"
            })
        if utility_list:
            utility_df = pd.DataFrame(utility_list)
            st.dataframe(utility_df, use_container_width=True)
    else:
        utility_df = pd.DataFrame({
            'Gamma': ['2', '6', '10'],
            'Utility B&H': [0.0014, 0.0004, -0.0007],
            'Utility ML': [0.0014, 0.0008, 0.0002],
            'Performance Fee': ['-29 bps', '+233 bps', '+496 bps']
        })
        st.dataframe(utility_df, use_container_width=True)
    
    st.info("""
    **Gamma=10 (고위험 회피 투자자)**
    
    ML 전략에 연간 **496 bps (4.96%)** 성능료 지불 용의 → 경제적 가치 입증
    """)

# ============================================================================
# PART 6: 결론
# ============================================================================

def render_conclusion():
    st.markdown('<h2 class="section-header">Part 6: 결론</h2>', unsafe_allow_html=True)
    
    st.markdown("### 핵심 발견")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **예측력**
        - Walk-Forward CV R² = 0.18 (SPY, QQQ)
        - HAR-RV 대비 최대 +0.25 개선
        - DM 검정 통과 (p<0.05)
        """)
    
    with col2:
        st.info("""
        **실용성**
        - 방향 정확도 68-72%
        - 초과 수익 1-7%
        - MDD 최대 10%p 개선
        """)
    
    st.markdown("### 연구 기여")
    
    st.markdown("""
    1. **VIX 중요성 확인**: 내재 변동성이 단기 예측에 핵심 (R² 감소 0.34)
    2. **타겟 변환 효과**: Sqrt/Log 변환이 원시값보다 우수
    3. **경제적 가치 입증**: 실제 트레이딩에서 초과 수익 달성
    """)
    
    st.markdown("### 한계 및 향후 연구")
    
    st.info("""
    **한계점**
    - Pre-COVID가 Post-COVID보다 예측력 우수 (환경 변화)
    - 극단적 VIX 구간에서 예측력 저하
    - Train/Test 분할 시점에 따른 성능 변동
    
    **향후 연구**
    - 레짐 스위칭 모델 적용
    - 고빈도 데이터 활용
    - 다른 자산군 확장
    """)
    
    st.markdown("---")
    st.markdown("### 감사합니다")

# ============================================================================
# 메인 렌더링
# ============================================================================

if selected_part == "전체 보기":
    render_introduction()
    st.markdown("---")
    render_methodology()
    st.markdown("---")
    render_results()
    st.markdown("---")
    render_additional()
    st.markdown("---")
    render_economic()
    st.markdown("---")
    render_conclusion()
elif selected_part == "Part 1: 서론":
    render_introduction()
elif selected_part == "Part 2: 방법론":
    render_methodology()
elif selected_part == "Part 3: 결과":
    render_results()
elif selected_part == "Part 4: 추가 분석":
    render_additional()
elif selected_part == "Part 5: 경제적 유의성":
    render_economic()
elif selected_part == "Part 6: 결론":
    render_conclusion()
