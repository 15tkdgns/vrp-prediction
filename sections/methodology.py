"""
연구 방법론 섹션: VRP 개념, 연구 갭, 가설, 파이프라인, 특성, 모델, 데이터 분할
"""
import streamlit as st
import pandas as pd
from utils.data_loader import load_image


def render_vrp_concept():
    """VRP 개념 설명"""
    st.markdown('<h2 class="section-header">1. VRP(변동성 위험 프리미엄)란?</h2>', unsafe_allow_html=True)
    
    st.markdown("""
<div class="explanation">
<h4>VRP의 정의</h4>
<p>
변동성 위험 프리미엄(Volatility Risk Premium, VRP)은 <strong>내재 변동성과 실현 변동성의 차이</strong>로 정의됩니다.
</p>
<ul>
    <li><strong>내재 변동성 (Implied Volatility, IV)</strong>: 옵션 가격에서 역산한 시장의 미래 변동성 예상치. 
    S&P 500의 경우 VIX 지수가 대표적입니다.</li>
    <li><strong>실현 변동성 (Realized Volatility, RV)</strong>: 과거 일정 기간 동안 실제로 발생한 가격 변동의 표준편차.
    일반적으로 22 거래일(약 1개월) 기준으로 계산합니다.</li>
</ul>
</div>
""", unsafe_allow_html=True)
    
    img = load_image("01_vrp_concept.png")
    if img:
        try:
            st.image(img)
        except Exception:
            st.info(" VRP 개념 다이어그램 (이미지 로딩 실패)")
    
    st.markdown("""
<div class="explanation">
<h4>VRP가 양수인 이유</h4>
<p>
통계적으로 VRP는 평균적으로 <strong>양수</strong>입니다. 이는 투자자들이 미래 변동성에 대해 
실제보다 더 높은 가격을 지불하기 때문입니다. 왜 그럴까요?
</p>
<ol>
    <li><strong>불확실성 회피</strong>: 인간은 불확실한 상황을 싫어하며, 보험료처럼 프리미엄을 지불합니다.</li>
    <li><strong>꼬리 위험(Tail Risk)</strong>: 극단적인 시장 폭락은 드물지만 발생 시 큰 손실을 야기합니다.</li>
    <li><strong>헤지 수요</strong>: 기관 투자자들은 포트폴리오 보호를 위해 옵션을 매수합니다.</li>
</ol>
</div>
""", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="hypothesis-card">
        <strong>VIX (내재 변동성)</strong><br>
        - CBOE에서 계산하는 S&P 500 옵션 기반 지수<br>
        - "공포 지수"라고도 불림<br>
        - 시장의 미래 30일 변동성 예상치
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="hypothesis-card">
        <strong>RV (실현 변동성)</strong><br>
        - 과거 가격 데이터에서 계산<br>
        - 실제로 발생한 변동성<br>
        - 일반적으로 VIX보다 낮음
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
<div class="key-point">
<strong>핵심 인사이트:</strong> VRP > 0이면 "공포 프리미엄"이 존재 → 변동성 매도 전략(옵션 매도)으로 수익 가능
</div>
""", unsafe_allow_html=True)
    
    with st.expander("발표 스크립트 - VRP 개념"):
        st.markdown("""
        <div class="script-box">
        "VRP는 간단히 말해서 '시장이 예상하는 변동성'에서 '실제 변동성'을 뺀 값입니다.
        
        VIX는 옵션 가격에서 역산한 값으로, 시장 참여자들이 앞으로 한 달간 얼마나 변동성이 
        클 것이라고 예상하는지를 보여줍니다. 반면 RV는 실제로 발생한 변동성입니다.
        
        흥미로운 점은, 평균적으로 VIX가 RV보다 높다는 것입니다. 
        즉, 시장은 항상 실제보다 더 불안해하고 있다는 뜻이죠. 
        이 '공포 프리미엄'을 수익화하는 것이 변동성 매도 전략의 핵심입니다."
        </div>
        """, unsafe_allow_html=True)


def render_research_gap():
    """연구 갭 설명"""
    st.markdown('<h2 class="section-header">2. 연구 갭과 본 연구의 기여</h2>', unsafe_allow_html=True)
    
    st.markdown("""
<div class="explanation">
<h4>기존 연구의 한계</h4>
<p>
VRP에 관한 기존 연구들은 주로 다음과 같은 특징을 가지고 있습니다:
</p>
<ul>
    <li><strong>S&P 500 중심</strong>: 대부분의 연구가 미국 주식 시장에만 집중</li>
    <li><strong>전통 통계 모델</strong>: HAR-RV, GARCH 등 시계열 모델 위주</li>
    <li><strong>자산 간 비교 부재</strong>: 왜 어떤 자산이 예측하기 쉬운지 설명 부족</li>
</ul>

<h4>본 연구의 기여</h4>
<ul>
    <li><strong>다중 자산 분석</strong>: SPY, GLD, EFA, EEM 등 4개 자산 비교</li>
    <li><strong>머신러닝 적용</strong>: MLP, Gradient Boosting 등 현대적 기법 활용</li>
    <li><strong>VIX-Beta 이론</strong>: 자산별 예측력 차이를 설명하는 새로운 프레임워크</li>
</ul>
</div>
""", unsafe_allow_html=True)
    
    img = load_image("02_research_gap.png")
    if img:
        try:
            st.image(img)
        except Exception:
            st.info(" 연구 갭 다이어그램 (이미지 로딩 실패)")
    
    with st.expander("발표 스크립트 - 연구 갭"):
        st.markdown("""
        <div class="script-box">
        "기존 VRP 연구들은 대부분 S&P 500에만 집중했고, HAR-RV나 GARCH 같은 전통적인 
        시계열 모델을 사용했습니다. 하지만 '왜 어떤 자산은 예측이 잘 되고 어떤 자산은 안 되는가'에 
        대한 답은 없었습니다.
        
        본 연구는 세 가지 측면에서 기여합니다:
        첫째, S&P 500뿐만 아니라 금, 선진국, 신흥국 ETF까지 분석했습니다.
        둘째, 신경망과 앙상블 모델 같은 최신 머신러닝 기법을 적용했습니다.
        셋째, 자산별 예측력 차이를 설명하는 'VIX-Beta 이론'을 새롭게 제시합니다."
        </div>
        """, unsafe_allow_html=True)


def render_hypothesis():
    """연구 가설"""
    st.markdown('<h2 class="section-header">3. 연구 가설</h2>', unsafe_allow_html=True)
    
    st.markdown("""
<div class="explanation">
<h4>본 연구의 세 가지 가설</h4>
<p>
본 연구는 다음 세 가지 핵심 가설을 검정합니다:
</p>
</div>
""", unsafe_allow_html=True)
    
    img = load_image("03_hypothesis.png")
    if img:
        try:
            st.image(img)
        except Exception:
            st.info(" 가설 다이어그램 (이미지 로딩 실패)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="slide-card" style="text-align: center; min-height: 200px;">
        <h4 style="color: #3498db;">H1: 모델 비교</h4>
        <p><strong>비선형 모델이 선형 모델보다 우수하다</strong></p>
        <hr>
        <p style="font-size: 0.9rem;">MLP, Gradient Boosting 같은 비선형 모델이 
        ElasticNet 같은 선형 모델보다 높은 R-squared를 달성할 것이다.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="slide-card" style="text-align: center; min-height: 200px;">
        <h4 style="color: #e74c3c;">H2: VIX-Beta</h4>
        <p><strong>VIX-RV 상관이 낮을수록 예측력이 높다</strong></p>
        <hr>
        <p style="font-size: 0.9rem;">VIX와 자산의 RV 간 상관관계가 낮은 자산일수록 
        VRP 예측 R-squared가 높을 것이다.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="slide-card" style="text-align: center; min-height: 200px;">
        <h4 style="color: #2ecc71;">H3: 트레이딩</h4>
        <p><strong>예측 기반 전략이 Buy & Hold를 능가</strong></p>
        <hr>
        <p style="font-size: 0.9rem;">VRP 예측 모델 기반 트레이딩 전략이 
        단순 보유 전략보다 높은 Sharpe Ratio를 달성할 것이다.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("발표 스크립트 - 연구 가설"):
        st.markdown("""
        <div class="script-box">
        "연구의 핵심 가설은 세 가지입니다.
        
        첫 번째 가설 H1은 '비선형 모델이 선형 모델보다 VRP 예측에 우수할 것이다'입니다.
        금융 데이터의 복잡한 패턴을 신경망이 더 잘 포착할 것으로 예상했습니다.
        
        두 번째 가설 H2는 본 연구의 핵심인 'VIX-Beta 가설'입니다.
        VIX가 S&P 500 기반이므로, S&P 500과 상관이 낮은 자산일수록 
        VIX가 해당 자산의 변동성을 잘못 반영하고, 이 '오차'를 예측할 수 있을 것입니다.
        
        세 번째 가설 H3은 '예측이 실제 수익으로 이어지는가'입니다.
        학술적 예측력이 실제 투자 성과로 연결되는지 검증합니다."
        </div>
        """, unsafe_allow_html=True)


def render_pipeline():
    """예측 파이프라인"""
    st.markdown('<h2 class="section-header">4. 예측 파이프라인</h2>', unsafe_allow_html=True)
    
    st.markdown("""
<div class="explanation">
<h4>전체 예측 프로세스</h4>
<p>
본 연구의 예측 파이프라인은 5단계로 구성됩니다. 각 단계에서 철저한 데이터 품질 관리와 
미래 정보 누수 방지를 적용했습니다.
</p>
</div>
""", unsafe_allow_html=True)
    
    img = load_image("04_pipeline.png")
    if img:
        try:
            st.image(img)
        except Exception:
            st.info(" 파이프라인 다이어그램 (이미지 로딩 실패)")
    
    st.markdown("""
<div class="explanation">
<h4>핵심 전략: RV 예측 후 VRP 계산</h4>
<p>
VRP를 직접 예측하는 대신, <strong>미래 RV를 먼저 예측</strong>한 후 VRP를 계산하는 전략을 사용했습니다:
</p>
<ol>
    <li><strong>Step 1</strong>: 12개 특성으로 22일 후 RV(실현 변동성) 예측</li>
    <li><strong>Step 2</strong>: VRP = 현재 VIX - 예측된 RV</li>
</ol>
<p>
이 방식의 장점은 RV가 VRP보다 안정적이고 예측하기 쉽다는 것입니다.
</p>
</div>
""", unsafe_allow_html=True)
    
    with st.expander("발표 스크립트 - 예측 파이프라인"):
        st.markdown("""
        <div class="script-box">
        "예측 파이프라인은 데이터 수집, 전처리, 특성 추출, 모델 학습, VRP 예측의 5단계입니다.
        
        중요한 전략적 결정은 VRP를 직접 예측하지 않고, 먼저 미래 RV를 예측한 후 
        'VRP = 현재 VIX - 예측된 RV'로 계산한다는 것입니다.
        
        왜냐하면 RV는 실제 가격 변동에서 계산되므로 더 안정적이고 예측하기 쉽습니다.
        반면 VRP는 VIX의 급변동에 영향을 받아 노이즈가 많습니다."
        </div>
        """, unsafe_allow_html=True)


def render_features():
    """특성 변수"""
    st.markdown('<h2 class="section-header">5. 특성 변수 (12개 Feature)</h2>', unsafe_allow_html=True)
    
    st.markdown("""
<div class="explanation">
<h4>특성 설계 원칙</h4>
<p>
총 12개의 특성 변수를 4개 카테고리로 분류했습니다. 모든 특성은 <strong>예측 시점(t) 이전</strong>의 
정보만 사용하여 미래 정보 누수를 방지했습니다.
</p>
</div>
""", unsafe_allow_html=True)
    
    img = load_image("05_features.png")
    if img:
        try:
            st.image(img)
        except Exception:
            st.info(" 특성 다이어그램 (이미지 로딩 실패)")
    
    features_df = pd.DataFrame({
        '카테고리': ['변동성']*3 + ['VIX']*3 + ['VRP']*3 + ['기타']*3,
        '특성': ['RV_1d', 'RV_5d', 'RV_22d', 'Vol_lag1', 'Vol_lag5', 'Vol_change',
                'VRP_lag1', 'VRP_lag5', 'VRP_ma5', 'regime_high', 'return_5d', 'return_22d'],
        '설명': ['1일 실현변동성', '5일 실현변동성', '22일 실현변동성', 
                '전일 VIX', '5일 전 VIX', 'VIX 일간 변화율',
                '전일 VRP', '5일 전 VRP', 'VRP 5일 이동평균', 
                '고변동성 구간 여부', '5일 누적수익률', '22일 누적수익률'],
        '계산식': ['|r_t| * sqrt(252) * 100', 'std(r_t-4:t) * sqrt(252) * 100', 
                  'std(r_t-21:t) * sqrt(252) * 100', 'VIX_t-1', 'VIX_t-5', 
                  '(VIX_t - VIX_t-1) / VIX_t-1', 'VRP_t-1', 'VRP_t-5', 
                  'mean(VRP_t-4:t)', 'I(VIX >= 25)', 'sum(r_t-4:t)', 'sum(r_t-21:t)']
    })
    
    st.dataframe(features_df, hide_index=True)
    
    st.markdown("""
<div class="warning-card">
<strong>주의:</strong> 모든 변동성은 연율화(annualized)하여 표시합니다. 
일간 변동성에 sqrt(252)를 곱하고 100을 곱해 퍼센트로 표현합니다.
</div>
""", unsafe_allow_html=True)


def render_models():
    """예측 모델"""
    st.markdown('<h2 class="section-header">6. 예측 모델</h2>', unsafe_allow_html=True)
    
    st.markdown("""
<div class="explanation">
<h4>사용된 모델</h4>
<p>
총 3가지 유형의 모델을 비교했습니다. 각 유형의 대표 모델은 다음과 같습니다:
</p>
</div>
""", unsafe_allow_html=True)
    
    img = load_image("06_mlp.png")
    if img:
        try:
            st.image(img)
        except Exception:
            st.info(" MLP 다이어그램 (이미지 로딩 실패)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="slide-card">
        <h4 style="color: #3498db;">선형 모델</h4>
        <p><strong>ElasticNet</strong></p>
        <ul>
            <li>L1 + L2 규제 조합</li>
            <li>해석 용이 (계수 직접 확인)</li>
            <li>과적합 방지</li>
            <li>빠른 학습</li>
        </ul>
        <p><em>장점: 안정적, 해석 가능</em><br>
        <em>단점: 비선형 관계 포착 불가</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="slide-card">
        <h4 style="color: #2ecc71;">트리 기반</h4>
        <p><strong>Gradient Boosting</strong></p>
        <ul>
            <li>순차적 트리 학습</li>
            <li>이전 트리의 오차 보정</li>
            <li>변수 중요도 제공</li>
            <li>XGBoost, LightGBM</li>
        </ul>
        <p><em>장점: 비선형 포착, 변수 선택</em><br>
        <em>단점: 과적합 위험</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="slide-card">
        <h4 style="color: #e74c3c;">신경망</h4>
        <p><strong>MLP (다층 퍼셉트론)</strong></p>
        <ul>
            <li>입력층 → 은닉층 → 출력층</li>
            <li>은닉층: 64 또는 (128, 64)</li>
            <li>활성화 함수: ReLU</li>
            <li>Dropout으로 정규화</li>
        </ul>
        <p><em>장점: 복잡한 패턴 학습</em><br>
        <em>단점: 해석 어려움, 데이터 필요</em></p>
        </div>
        """, unsafe_allow_html=True)


def render_data_split():
    """데이터 누수 방지"""
    st.markdown('<h2 class="section-header">7. 데이터 누수 방지</h2>', unsafe_allow_html=True)
    
    st.markdown("""
<div class="explanation">
<h4>데이터 누수란?</h4>
<p>
<strong>데이터 누수(Data Leakage)</strong>는 모델 학습 시 테스트 기간의 정보가 
의도치 않게 포함되어 예측 성능이 과대평가되는 현상입니다. 
금융 분야에서 특히 심각한 문제로, 백테스팅에서는 좋아 보이지만 실제 투자에서는 실패하는 
"과거에만 작동하는 전략"을 만들게 됩니다.
</p>
</div>
""", unsafe_allow_html=True)
    
    img = load_image("07_data_split.png")
    if img:
        try:
            st.image(img)
        except Exception:
            st.info(" 데이터 분할 다이어그램 (이미지 로딩 실패)")
    
    st.markdown("""
<div class="explanation">
<h4>22일 Gap의 중요성</h4>
<p>
본 연구에서 예측하는 타겟은 <strong>22일 후 실현변동성(RV_future)</strong>입니다.
이 값은 t+1일부터 t+22일까지의 가격 정보를 포함합니다.
</p>
<p>
따라서 학습 데이터의 마지막 날짜와 테스트 데이터의 첫 날짜 사이에 <strong>최소 22일</strong>의 
간격이 필요합니다. 이 간격이 없으면 학습 데이터의 타겟에 테스트 기간의 정보가 포함됩니다.
</p>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("""
<div class="result-card">
<strong>검증 결과:</strong><br>
- 22일 Gap 없이: R-squared = 0.67 (과적합)<br>
- 22일 Gap 적용: R-squared = 0.37 (현실적)<br>
- 무작위 타겟으로 학습: R-squared = -0.01 (정상, 예측 불가 확인)
</div>
""", unsafe_allow_html=True)
    
    with st.expander("발표 스크립트 - 데이터 누수 방지"):
        st.markdown("""
        <div class="script-box">
        "데이터 누수는 금융 머신러닝에서 가장 흔하고 치명적인 실수입니다.
        
        우리가 예측하려는 22일 후 실현변동성은 미래 22일간의 가격 정보를 포함합니다.
        따라서 학습 데이터와 테스트 데이터 사이에 22일의 간격을 두지 않으면,
        학습 데이터의 타겟에 테스트 기간의 정보가 섞이게 됩니다.
        
        실제로 Gap 없이 학습하면 R-squared가 0.67까지 올라가지만,
        이는 미래 정보가 포함된 가짜 성능입니다.
        22일 Gap을 적용하면 R-squared는 0.37로 현실적인 수준으로 내려옵니다."
        </div>
        """, unsafe_allow_html=True)


def render_all_methodology():
    """모든 방법론 섹션 렌더링"""
    render_vrp_concept()
    render_research_gap()
    render_hypothesis()
    render_pipeline()
    render_features()
    render_models()
    render_data_split()
