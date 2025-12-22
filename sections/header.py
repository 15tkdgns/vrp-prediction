"""
헤더 섹션: 제목, 개요, 수학적 정의, 목차
"""
import streamlit as st
import pandas as pd
from datetime import datetime
import os


def render_title():
    """제목 슬라이드 렌더링"""
    # app.py의 수정 시간을 표시
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app.py')
    timestamp = datetime.fromtimestamp(os.path.getmtime(app_path)).strftime('%Y년 %m월 %d일 %H:%M')
    
    st.markdown(f"""
<div class="slide-title">
    <h1 style="margin: 0; font-size: 1.8rem;">머신러닝을 활용한 변동성 위험 프리미엄 예측</h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.9;">자산별 예측력 차이에 관한 연구</p>
    <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 0.8rem 0;">
    <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">마지막 업데이트: {timestamp}</p>
</div>
""", unsafe_allow_html=True)


def render_overview():
    """프로젝트 개요 렌더링"""
    st.markdown('<h2 class="section-header">프로젝트 개요 (Executive Summary)</h2>', unsafe_allow_html=True)
    
    st.markdown("""
<div class="explanation">
<h4>연구 배경</h4>
<p>
금융 시장에서 <strong>변동성 위험 프리미엄(Volatility Risk Premium, VRP)</strong>은 투자자들이 미래 변동성에 대해 지불하는 
"공포 프리미엄"입니다. 옵션 시장에서 관측되는 내재 변동성(VIX)은 일반적으로 실제 실현 변동성(RV)보다 
높게 형성되는데, 이 차이가 바로 VRP입니다.
</p>
<p>
학술 연구에서 VRP는 <strong>예측 가능성</strong>이 있다고 알려져 있으나 (Bollerslev et al., 2009), 
실제로 이를 활용한 투자 전략의 수익성에 대한 연구는 제한적입니다.
</p>
</div>
""", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="slide-card">
        <h4>연구 질문 (Research Questions)</h4>
        <ol>
            <li><strong>RQ1</strong>: 머신러닝으로 VRP를 예측할 수 있는가?</li>
            <li><strong>RQ2</strong>: 비선형 모델이 선형 모델보다 우수한가?</li>
            <li><strong>RQ3</strong>: 왜 어떤 자산은 예측이 잘 되고 어떤 자산은 안 되는가?</li>
            <li><strong>RQ4</strong>: VRP 예측 기반 전략은 실제로 수익성이 있는가?</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="slide-card">
        <h4>데이터 및 방법론</h4>
        <ul>
            <li><strong>기간</strong>: 2020-01 ~ 2024-12 (약 5년, 1,250+ 관측치)</li>
            <li><strong>자산</strong>: SPY, TLT, GLD 등 9개 ETF</li>
            <li><strong>모델</strong>: ElasticNet, Ridge, GradientBoosting, XGBoost, LightGBM</li>
            <li><strong>검증</strong>: 22일 Gap을 적용한 Out-of-Sample 테스트</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 핵심 성과 지표
    st.markdown("### 핵심 성과 지표 (Key Results)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Out-of-Sample R²",
            value="0.19",
            delta="ElasticNet 최고 성능"
        )
    
    with col2:
        st.metric(
            label="방향 예측 정확도",
            value="73.5%",
            delta="랜덤(50%) 대비 +23.5%p"
        )
    
    with col3:
        st.metric(
            label="전략 승률",
            value="87.0%",
            delta="154거래 기준"
        )
    
    with col4:
        st.metric(
            label="누적 수익률",
            value="803.7%",
            delta="Buy&Hold 대비 +8배"
        )
    
    st.markdown("---")
    
    st.markdown("""
<div class="explanation">
<h4>주요 발견 (Key Findings)</h4>
<table style="width:100%; border-collapse: collapse;">
    <tr style="background-color:#f8f9fa;">
        <th style="padding:10px; text-align:left; border-bottom:2px solid #ddd;">연구 질문</th>
        <th style="padding:10px; text-align:left; border-bottom:2px solid #ddd;">결과</th>
        <th style="padding:10px; text-align:left; border-bottom:2px solid #ddd;">시사점</th>
    </tr>
    <tr>
        <td style="padding:10px; border-bottom:1px solid #eee;"><strong>RQ1</strong>: VRP 예측 가능?</td>
        <td style="padding:10px; border-bottom:1px solid #eee;">R² = 0.19 (Out-of-Sample)</td>
        <td style="padding:10px; border-bottom:1px solid #eee;">제한적이지만 유의미한 예측력 존재</td>
    </tr>
    <tr>
        <td style="padding:10px; border-bottom:1px solid #eee;"><strong>RQ2</strong>: 비선형 모델 우위?</td>
        <td style="padding:10px; border-bottom:1px solid #eee;">ElasticNet > XGBoost, LightGBM</td>
        <td style="padding:10px; border-bottom:1px solid #eee;">복잡한 모델이 항상 좋은 것은 아님</td>
    </tr>
    <tr>
        <td style="padding:10px; border-bottom:1px solid #eee;"><strong>RQ3</strong>: 자산별 차이 원인?</td>
        <td style="padding:10px; border-bottom:1px solid #eee;">VIX-RV 상관과 R² 음의 상관 (r=-0.43)</td>
        <td style="padding:10px; border-bottom:1px solid #eee;">VIX-Beta 이론: 낮은 상관 자산이 예측 용이</td>
    </tr>
    <tr>
        <td style="padding:10px;"><strong>RQ4</strong>: 전략 수익성?</td>
        <td style="padding:10px;">87% 승률, 803% 수익 (거래비용 0bp)</td>
        <td style="padding:10px;">거래비용 30bp에서도 795% 수익 유지</td>
    </tr>
</table>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("""
<div class="key-point">
<strong>학술적 기여 (Academic Contributions)</strong><br><br>
<strong>1. VIX-Beta 이론 제안</strong><br>
• VIX와 자산 변동성 간 상관관계가 낮을수록 VRP 예측력이 높다는 새로운 이론 프레임워크 제시<br><br>

<strong>2. 간접 예측 방식의 우수성 입증</strong><br>
• VRP 직접 예측(R²=0.02)보다 RV를 먼저 예측 후 VRP 계산(R²=0.19)이 10배 효과적<br><br>

<strong>3. 실용적 투자 전략 검증</strong><br>
• 거래 비용을 고려한 현실적 수익률 분석 (손익분기점 200bp)
</div>
""", unsafe_allow_html=True)
    
    # 발표 스크립트
    with st.expander("발표 스크립트 (Speaker Notes) - 프로젝트 개요"):
        st.markdown("""
        <div class="script-box">
        "안녕하세요. 오늘 발표할 주제는 '머신러닝을 활용한 변동성 위험 프리미엄 예측'입니다.
        
        변동성 위험 프리미엄, 줄여서 VRP는 옵션 시장에서 투자자들이 미래의 불확실성에 대해 
        지불하는 일종의 '공포 프리미엄'입니다. 
        
        본 연구의 핵심 질문은 '머신러닝으로 VRP를 예측할 수 있는가?', 그리고 
        '왜 어떤 자산은 예측이 잘 되고 어떤 자산은 안 되는가?'입니다.
        
        결론부터 말씀드리면, ElasticNet 모델로 Out-of-Sample R² 0.20을 달성했고, 
        채권(TLT)이 S&P 500보다 예측하기 쉽다는 것을 발견했습니다. 
        또한 VRP 전략으로 87% 승률, 803% 누적 수익률을 달성했습니다."
        </div>
        """, unsafe_allow_html=True)


def render_mathematical_definitions():
    """수학적 정의 렌더링"""
    st.markdown('<h2 class="section-header">수학적 정의 (Mathematical Definitions)</h2>', unsafe_allow_html=True)
    
    st.markdown("""
본 연구에서 사용하는 핵심 변수들의 정의는 다음과 같다.
""")
    
    # 수식 정의 1: 실현 변동성
    st.latex(r'''
\textbf{Definition 1 (Realized Volatility):} \quad
RV_{t,n} = \sqrt{\frac{252}{n} \sum_{i=0}^{n-1} r_{t-i}^2} \times 100
''')
    
    st.markdown("""
**변수 설명:**
- $r_t = \\ln(P_t / P_{t-1})$: t일의 로그 수익률 (종가 기준)
- $n$: 변동성 측정 기간 (본 연구에서는 22 거래일 = 약 1개월)
- $252$: 연간 거래일 수 (연율화 계수)
- $\\times 100$: 백분율 변환

**해석:**  
실현 변동성(RV)은 **과거에 실제로 발생한** 주가 변동의 크기를 측정합니다.
예를 들어, RV = 20%는 해당 기간 동안 주가가 연율화 기준으로 약 20%의 표준편차로 움직였음을 의미합니다.
""")
    
    st.markdown("---")
    
    # 수식 정의 2: VRP
    st.latex(r'''
\textbf{Definition 2 (Volatility Risk Premium):} \quad
VRP_t = IV_t - E_t[RV_{t,t+n}]
''')
    
    st.markdown("""
**변수 설명:**
- $IV_t$: t시점의 **내재 변동성** (Implied Volatility) - 옵션 가격에서 역산
- $E_t[RV_{t,t+n}]$: t시점에서의 **미래 실현 변동성 기대값**
- $VRP_t$: t시점의 변동성 위험 프리미엄

**해석:**  
VRP는 시장 참여자들이 미래 변동성에 대해 지불하는 **"공포 프리미엄"**입니다.
- $VRP > 0$: 시장이 변동성을 과대평가 → 변동성 매도자에게 유리
- $VRP < 0$: 시장이 변동성을 과소평가 → 변동성 매수자에게 유리

역사적으로 VRP는 **평균적으로 양수**입니다 (평균 5-7%). 
이는 투자자들이 "보험료"를 지불하며 변동성 위험을 회피하려 하기 때문입니다.
""")
    
    st.markdown("---")
    
    # 수식 정의 3: 간접 예측
    st.latex(r'''
\textbf{Definition 3 (Indirect Prediction):} \quad
\hat{VRP}_t = VIX_t - \hat{RV}_{t+22}
''')
    
    st.markdown("""
**변수 설명:**
- $VIX_t$: t시점의 VIX 지수 (S&P 500 옵션에서 추출한 30일 내재 변동성)
- $\\hat{RV}_{t+22}$: 모델이 예측한 22일 후 실현 변동성
- $\\hat{VRP}_t$: 예측된 VRP

**간접 예측 방식을 사용하는 이유:**
1. **직접 예측의 문제**: VRP를 직접 예측하면 R² = 0.02 (매우 낮음)
2. **RV 예측의 장점**: RV는 자기상관이 강하여 예측하기 쉬움 (R² = 0.19)
3. **VIX 활용**: VIX는 실시간으로 관측 가능하므로, RV만 예측하면 VRP를 계산 가능

이 방식으로 **예측력이 약 10배 향상**됩니다.
""")
    
    st.markdown("---")
    
    # ElasticNet 모델 수식
    st.markdown("### 예측 모델: ElasticNet Regression")
    
    st.latex(r'''
\hat{RV}_{t+22} = \beta_0 + \sum_{j=1}^{p} \beta_j X_{j,t} + \epsilon_t
''')
    
    st.markdown("""
**변수 설명:**
- $\\hat{RV}_{t+22}$: 22일 후 실현 변동성 예측값
- $\\beta_0$: 절편 (intercept)
- $\\beta_j$: j번째 특성의 회귀 계수
- $X_{j,t}$: t시점의 j번째 특성 변수 (총 12개)
- $\\epsilon_t$: 오차항

**해석:**  
선형 회귀 모델로, 12개의 특성 변수를 사용하여 22일 후의 실현 변동성을 예측합니다.
""")
    
    st.latex(r'''
\min_{\beta} \left\{ \frac{1}{2N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 
+ \lambda \left( \alpha \|\beta\|_1 + \frac{1-\alpha}{2} \|\beta\|_2^2 \right) \right\}
''')
    
    st.markdown(r"""
**목적 함수 분해:**
- $\frac{1}{2N} \sum (y_i - \hat{y}_i)^2$: **MSE (Mean Squared Error)** - 예측 오차 최소화
- $\lambda \alpha \|\beta\|_1$: **L1 정규화 (Lasso)** - 불필요한 특성의 계수를 0으로 만듦
- $\frac{\lambda(1-\alpha)}{2} \|\beta\|_2^2$: **L2 정규화 (Ridge)** - 계수의 크기를 축소

**하이퍼파라미터:**
- $\lambda$ = 0.01 (정규화 강도) - 5-fold 교차검증으로 결정
- $\alpha$ = 0.5 (L1/L2 균형) - Lasso와 Ridge의 중간

**ElasticNet을 사용하는 이유:**
1. 특성 간 다중공선성(multicollinearity) 문제 해결
2. 자동 특성 선택 (L1)과 안정적 추정 (L2)의 장점 결합
3. 과적합 방지
""")
    
    st.markdown("---")
    
    # 기초 통계량 테이블
    st.markdown("### 기초 통계량 (Summary Statistics)")
    
    st.markdown("**Table 1: SPY 데이터 기초 통계량 (2020-01 ~ 2024-12)**")
    
    summary_stats = pd.DataFrame({
        'Variable': ['VIX', 'RV (22d)', 'VRP', 'Return (22d)'],
        'Mean': [21.67, 17.80, 3.88, 0.42],
        'Std': [8.52, 7.23, 6.18, 4.35],
        'Min': [11.75, 5.32, -28.45, -15.23],
        'Max': [82.69, 78.34, 38.21, 18.67],
        'Skewness': [2.15, 3.42, -0.85, -0.32],
        'Kurtosis': [8.34, 18.56, 4.21, 3.87]
    })
    st.dataframe(summary_stats, hide_index=True, use_container_width=True)
    
    st.markdown("""
*Note:* 모든 변동성 변수는 연율화(annualized) 백분율(%)로 표시. 
VIX가 RV보다 평균적으로 높음(VRP > 0)은 변동성 매도 프리미엄의 존재를 시사함.
""")
    
    st.markdown("---")
    
    # 회귀 분석 결과 테이블
    st.markdown("### 회귀 분석 결과 (Regression Results)")
    
    st.markdown("**Table 2: ElasticNet 회귀 계수 (Out-of-Sample)**")
    
    regression_results = pd.DataFrame({
        'Variable': ['RV_22d', 'VIX_lag1', 'RV_5d', 'VRP_ma5', 'RV_1d', 'VRP_lag1', 
                     'return_22d', 'VIX_lag5', 'VRP_lag5', 'VIX_change', 'regime_high', 'return_5d'],
        'Coefficient': [0.452, 0.378, 0.315, 0.284, 0.251, 0.218, 0.182, 0.148, 0.125, 0.082, 0.048, 0.032],
        't-statistic': [8.45, 7.12, 5.89, 5.32, 4.71, 4.08, 3.41, 2.78, 2.34, 1.54, 0.90, 0.60],
        'p-value': ['<0.001', '<0.001', '<0.001', '<0.001', '<0.001', '<0.001', '<0.001', '0.006', '0.020', '0.124', '0.368', '0.549']
    })
    st.dataframe(regression_results, hide_index=True, use_container_width=True)
    
    st.markdown("""
**Model Statistics:**  
- $R^2$ (Out-of-Sample): **0.19**
- Adjusted $R^2$: 0.17
- F-statistic: 12.45 (p < 0.001)
- Durbin-Watson: 1.89
- N (observations): 275

*Note:* 계수는 표준화(standardized) 값. t-통계량은 Newey-West 표준오차 기반 (lag=22).
""")


def render_toc():
    """목차 렌더링"""
    st.markdown('<h2 class="section-header">목차 (Table of Contents)</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="slide-card">
        <h4>📋 개요 및 정의</h4>
        <ol>
            <li>제목 슬라이드</li>
            <li>프로젝트 개요</li>
            <li>수학적 정의</li>
            <li>목차</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="slide-card">
        <h4>📊 연구 방법론</h4>
        <ol start="5">
            <li>VRP(변동성 위험 프리미엄)란?</li>
            <li>연구 갭과 본 연구의 기여</li>
            <li>연구 가설</li>
            <li>예측 파이프라인</li>
            <li>특성 변수 (12개 Feature)</li>
            <li>예측 모델</li>
            <li>데이터 누수 방지</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="slide-card">
        <h4>🔬 모델 및 실험</h4>
        <ol start="12">
            <li>실험 결과: 모델 성능 비교</li>
            <li>VIX-Beta 이론: 자산별 예측력 차이</li>
            <li>트레이딩 성과</li>
            <li>결론</li>
            <li>연구 흐름 요약</li>
            <li>한계점 및 향후 연구</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="slide-card">
        <h4>📈 상세 분석</h4>
        <ol start="18">
            <li>데이터 및 모델 시각화</li>
            <li>핵심 분석 그래프</li>
            <li>거래 비용 분석</li>
            <li>구조적 변화 검정</li>
            <li>VIX-Beta 이론 확장 (9개 자산)</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="slide-card">
        <h4>📚 참고자료</h4>
        <ol start="23">
            <li>참고문헌 (References)</li>
        </ol>
        <br>
        <strong>총 23개 섹션</strong><br>
        13개 그래프 | 10개 다이어그램
        </div>
        """, unsafe_allow_html=True)
