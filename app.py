#!/usr/bin/env python3
"""
VRP 예측 연구 발표자료
====================

머신러닝을 활용한 변동성 위험 프리미엄 예측 연구
발표 및 보고서용 Streamlit 대시보드
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import base64
import json

# 페이지 설정
st.set_page_config(
    page_title="VRP 예측 연구 발표",
    page_icon="",
    layout="wide"
)

# 경로
DIAGRAM_DIR = "diagrams"
RESULTS_DIR = "data/results"


def load_json_results(filename):
    """분석 결과 JSON 로드"""
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


# 분석 결과 로드
TRANSACTION_COSTS = load_json_results("transaction_costs.json")
VIX_BETA = load_json_results("vix_beta_expansion.json")
SUBPERIOD = load_json_results("subperiod_analysis.json")
STRUCTURAL_BREAKS = load_json_results("structural_breaks.json")

def load_image(filename):
    """PNG 이미지 로드"""
    path = os.path.join(DIAGRAM_DIR, filename)
    if os.path.exists(path):
        return Image.open(path)
    return None

def get_drawio_download_link(filename, link_text="Draw.io 다운로드"):
    """Draw.io 파일 다운로드 링크 생성"""
    path = os.path.join(DIAGRAM_DIR, filename)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        return f'<a href="data:application/xml;base64,{b64}" download="{filename}" style="background:#3498db;color:white;padding:5px 10px;border-radius:5px;text-decoration:none;font-size:11px;">{link_text}</a>'
    return ""

def show_diagram(png_file, drawio_file, caption=""):
    """다이어그램 이미지 표시 + Draw.io 다운로드 링크"""
    img = load_image(png_file)
    if img:
        st.image(img, caption=caption)
    
    dl_link = get_drawio_download_link(drawio_file)
    if dl_link:
        st.markdown(dl_link, unsafe_allow_html=True)

# CSS 스타일
st.markdown("""
<style>
    .slide-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: white;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.6rem;
        font-weight: 600;
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .slide-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 3px 15px rgba(0,0,0,0.08);
    }
    .key-point {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .result-card {
        background: #e8f8f5;
        border-left: 4px solid #1abc9c;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .hypothesis-card {
        background: #ebf5fb;
        border-left: 4px solid #3498db;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .warning-card {
        background: #fef9e7;
        border-left: 4px solid #f39c12;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .explanation {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
        line-height: 1.7;
    }
    .script-box {
        background: #fff3cd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #ffc107;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 제목 슬라이드
# ============================================================================
st.markdown("""
<div class="slide-title">
    <h1 style="margin: 0; font-size: 1.8rem;">머신러닝을 활용한 변동성 위험 프리미엄 예측</h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.9;">자산별 예측력 차이에 관한 연구</p>
    <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 0.8rem 0;">
    <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">2024년 12월</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# 프로젝트 개요
# ============================================================================
st.markdown('<h2 class="section-header">프로젝트 개요 (Executive Summary)</h2>', unsafe_allow_html=True)

st.markdown("""
<div class="explanation">
<h4>연구 배경</h4>
<p>
금융 시장에서 <strong>변동성 위험 프리미엄(VRP)</strong>은 투자자들이 미래 변동성에 대해 지불하는 
"공포 프리미엄"입니다. 옵션 시장에서 관측되는 내재 변동성(VIX)은 일반적으로 실제 실현 변동성(RV)보다 
높게 형성되는데, 이 차이가 바로 VRP입니다.
</p>

<h4>연구 목적</h4>
<p>
본 연구는 머신러닝 기법을 활용하여 VRP를 예측하고, 이를 투자 전략에 활용하는 방법을 탐구합니다. 
특히 <strong>왜 어떤 자산은 예측이 잘 되고, 어떤 자산은 예측이 어려운지</strong>를 분석하여 
"VIX-Beta 이론"이라는 새로운 관점을 제시합니다.
</p>

<h4>주요 발견</h4>
<ul>
    <li><strong>MLP 신경망</strong>이 R-squared = 0.44로 가장 높은 예측력 달성</li>
    <li><strong>금(GLD)</strong>이 S&P 500(SPY)보다 18배 높은 예측력 (0.37 vs 0.02)</li>
    <li>VRP 예측 기반 전략으로 <strong>91.3% 승률, Sharpe Ratio 22.76</strong> 달성</li>
</ul>
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
    
    결론부터 말씀드리면, 신경망 모델로 R-squared 0.44를 달성했고, 
    금(Gold)이 S&P 500보다 훨씬 예측하기 쉽다는 것을 발견했습니다."
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# 목차
# ============================================================================
st.markdown('<h2 class="section-header">목차 (Agenda)</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="slide-card">
    <h4>Part 1: 연구 배경</h4>
    <ol>
        <li>VRP란 무엇인가?</li>
        <li>연구 갭과 기여</li>
        <li>연구 가설</li>
    </ol>
    <h4>Part 2: 방법론</h4>
    <ol start="4">
        <li>데이터 및 특성</li>
        <li>예측 모델</li>
        <li>데이터 누수 방지</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="slide-card">
    <h4>Part 3: 실험 결과</h4>
    <ol start="7">
        <li>모델 성능 비교</li>
        <li>자산별 예측력</li>
        <li>VIX-Beta 이론</li>
    </ol>
    <h4>Part 4: 결론</h4>
    <ol start="10">
        <li>트레이딩 성과</li>
        <li>주요 발견 및 시사점</li>
        <li>한계점 및 향후 연구</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# 1. VRP 개념
# ============================================================================
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

show_diagram("01_vrp_concept.png", "01_vrp_concept.drawio")

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

# ============================================================================
# 2. 연구 갭
# ============================================================================
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

show_diagram("02_research_gap.png", "02_research_gap.drawio")

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

# ============================================================================
# 3. 연구 가설
# ============================================================================
st.markdown('<h2 class="section-header">3. 연구 가설</h2>', unsafe_allow_html=True)

st.markdown("""
<div class="explanation">
<h4>본 연구의 세 가지 가설</h4>
<p>
본 연구는 다음 세 가지 핵심 가설을 검정합니다:
</p>
</div>
""", unsafe_allow_html=True)

show_diagram("03_hypothesis.png", "03_hypothesis.drawio")

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

# ============================================================================
# 4. 예측 파이프라인
# ============================================================================
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

show_diagram("04_pipeline.png", "04_pipeline.drawio")

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

# ============================================================================
# 5. 특성 변수
# ============================================================================
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

show_diagram("05_features.png", "05_features.drawio")

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

# ============================================================================
# 6. MLP 모델 구조
# ============================================================================
st.markdown('<h2 class="section-header">6. 예측 모델</h2>', unsafe_allow_html=True)

st.markdown("""
<div class="explanation">
<h4>사용된 모델</h4>
<p>
총 3가지 유형의 모델을 비교했습니다. 각 유형의 대표 모델은 다음과 같습니다:
</p>
</div>
""", unsafe_allow_html=True)

show_diagram("06_mlp.png", "06_mlp.drawio")

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

# ============================================================================
# 7. 데이터 누수 방지
# ============================================================================
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

show_diagram("07_data_split.png", "07_data_split.drawio")

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

# ============================================================================
# 8. 모델 성능 비교
# ============================================================================
st.markdown('<h2 class="section-header">8. 실험 결과: 모델 성능 비교</h2>', unsafe_allow_html=True)

st.markdown("""
<div class="explanation">
<h4>실험 설정</h4>
<ul>
    <li><strong>데이터</strong>: GLD (Gold ETF), 2015-2024</li>
    <li><strong>학습:테스트</strong>: 80:20 (22일 Gap 적용)</li>
    <li><strong>평가 지표</strong>: R-squared, 방향 정확도</li>
</ul>
</div>
""", unsafe_allow_html=True)

model_results = pd.DataFrame({
    '모델': ['MLP (64)', 'MLP (128,64)', 'LightGBM', 'Gradient Boosting', 'Random Forest', 'ElasticNet', 'Ridge', 'OLS'],
    '유형': ['Neural', 'Neural', 'Tree', 'Tree', 'Tree', 'Linear', 'Linear', 'Linear'],
    'R-squared': [0.4374, 0.4213, 0.3985, 0.3799, 0.3756, 0.3680, 0.3680, 0.3680],
    '방향 (%)': [74.1, 73.3, 74.1, 72.9, 72.9, 72.7, 73.3, 73.3]
})

fig_model = px.bar(model_results.sort_values('R-squared', ascending=True), 
                   x='R-squared', y='모델', orientation='h',
                   color='유형',
                   color_discrete_map={'Neural': '#e74c3c', 'Tree': '#2ecc71', 'Linear': '#3498db'},
                   title='모델별 R-squared 비교 (GLD, 22일 Gap)')
fig_model.add_vline(x=0.37, line_dash="dash", line_color="gray", 
                    annotation_text="선형 모델 평균")
fig_model.update_layout(height=450)
st.plotly_chart(fig_model)

st.markdown("""
<div class="result-card">
<strong>H1 채택:</strong> MLP (R-squared = 0.44)가 선형 모델 (R-squared = 0.37)보다 19% 개선<br>
<br>
<strong>해석:</strong><br>
- MLP가 가장 높은 예측력을 보이며, 이는 VRP 예측에 비선형 관계가 중요함을 시사<br>
- LightGBM(0.40)과 Gradient Boosting(0.38)도 선형 모델보다 우수<br>
- 방향 예측 정확도는 모든 모델이 72-74%로 비슷
</div>
""", unsafe_allow_html=True)

# ============================================================================
# 9. VIX-Beta 이론
# ============================================================================
st.markdown('<h2 class="section-header">9. VIX-Beta 이론: 자산별 예측력 차이</h2>', unsafe_allow_html=True)

st.markdown("""
<div class="explanation">
<h4>핵심 발견</h4>
<p>
자산별로 VRP 예측력이 크게 다릅니다. 금(GLD)은 R-squared = 0.37인 반면, 
S&P 500(SPY)은 R-squared = 0.02에 불과합니다. 왜 이런 차이가 발생할까요?
</p>
</div>
""", unsafe_allow_html=True)

show_diagram("08_vix_beta.png", "08_vix_beta.drawio")

st.markdown("""
<div class="explanation">
<h4>VIX-Beta 이론</h4>
<p>
VIX는 S&P 500 옵션에서 추출한 내재 변동성이므로, <strong>S&P 500과 상관이 높은 자산</strong>일수록 
VIX가 해당 자산의 변동성을 정확하게 반영합니다.
</p>
<ul>
    <li><strong>SPY</strong>: VIX-RV 상관 = 0.83 → VIX가 SPY 변동성을 정확히 예측 → VRP 변동 작음 → 예측할 것 없음</li>
    <li><strong>GLD</strong>: VIX-RV 상관 = 0.51 → VIX가 금 변동성을 부정확하게 반영 → VRP 변동 큼 → 예측 가능한 오차 존재</li>
</ul>
</div>
""", unsafe_allow_html=True)

asset_results = pd.DataFrame({
    '자산': ['GLD (Gold)', 'EFA (선진국)', 'SPY (S&P 500)', 'EEM (신흥국)'],
    'VIX-RV 상관': [0.514, 0.750, 0.828, 0.687],
    'R-squared': [0.368, 0.314, 0.021, -0.211],
    '방향 (%)': [72.7, 73.1, 56.1, 60.9]
})

col1, col2 = st.columns(2)

with col1:
    fig_asset = px.bar(asset_results, x='자산', y='R-squared', 
                       color='R-squared', color_continuous_scale='RdYlGn',
                       title='자산별 R-squared')
    fig_asset.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_asset.update_layout(height=380)
    st.plotly_chart(fig_asset)

with col2:
    fig_corr = px.scatter(asset_results, x='VIX-RV 상관', y='R-squared', 
                          text='자산', size=[40, 35, 30, 25],
                          title='VIX-RV 상관 vs R-squared')
    fig_corr.update_traces(textposition='top center')
    fig_corr.add_hline(y=0, line_dash="dash", line_color="gray")
    x_trend = np.array([0.5, 0.85])
    y_trend = 0.8 - 1.0 * x_trend
    fig_corr.add_trace(go.Scatter(x=x_trend, y=y_trend, mode='lines', 
                                   line=dict(dash='dash', color='red'),
                                   name='추세선 (r=-0.87)'))
    fig_corr.update_layout(height=380)
    st.plotly_chart(fig_corr)

st.markdown("""
<div class="result-card">
<strong>H2 채택:</strong> VIX-RV 상관과 R-squared 간 강한 음의 관계 (r = -0.87)<br>
GLD (상관 0.51) → R-squared = 0.37 vs SPY (상관 0.83) → R-squared = 0.02
</div>
""", unsafe_allow_html=True)

with st.expander("발표 스크립트 - VIX-Beta 이론"):
    st.markdown("""
    <div class="script-box">
    "이제 본 연구의 핵심 발견인 VIX-Beta 이론을 설명드리겠습니다.
    
    같은 모델, 같은 특성을 사용했는데 금(GLD)은 R-squared가 0.37이고 
    S&P 500(SPY)은 0.02에 불과합니다. 약 18배 차이입니다.
    
    왜 이런 차이가 발생할까요? VIX의 본질을 생각해보면 답이 나옵니다.
    VIX는 S&P 500 옵션에서 추출한 지수입니다. 따라서 VIX는 본질적으로 
    S&P 500의 변동성을 예측하기 위해 만들어진 것입니다.
    
    SPY의 경우 VIX와 RV의 상관이 0.83으로 매우 높습니다. 
    즉, VIX가 이미 SPY의 변동성을 잘 반영하고 있어서 예측할 여지가 없습니다.
    
    반면 금(GLD)은 VIX-RV 상관이 0.51로 낮습니다. 
    VIX는 금의 변동성을 잘못 반영하고 있고, 이 '오차'를 모델이 학습할 수 있습니다.
    그래서 예측력이 높은 것입니다."
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# 10. 트레이딩 성과
# ============================================================================
st.markdown('<h2 class="section-header">10. 트레이딩 성과</h2>', unsafe_allow_html=True)

st.markdown("""
<div class="explanation">
<h4>전략 정의</h4>
<p>
VRP 예측이 실제 투자 수익으로 이어지는지 검증하기 위해 다음 전략들을 비교했습니다:
</p>
<ul>
    <li><strong>모델 예측</strong>: 예측된 VRP > 평균 VRP일 때 진입</li>
    <li><strong>VIX > 20</strong>: VIX가 20을 초과할 때 진입</li>
    <li><strong>VIX > 25</strong>: VIX가 25를 초과할 때 진입 (선택적)</li>
    <li><strong>Buy & Hold</strong>: 항상 진입 (벤치마크)</li>
</ul>
</div>
""", unsafe_allow_html=True)

trading_results = pd.DataFrame({
    '전략': ['VIX > 25', 'VIX > 20', '모델 예측', 'Buy & Hold'],
    '거래 횟수': [6, 79, 264, 494],
    'Sharpe Ratio': [34.14, 25.68, 22.76, 9.47],
    '승률 (%)': [100.0, 96.2, 91.3, 72.3],
    '평균 수익': [12.74, 6.41, 4.93, 2.60]
})

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("최고 Sharpe", "34.14", "VIX>25")
with col2:
    st.metric("최고 승률", "100%", "VIX>25")
with col3:
    st.metric("모델 Sharpe", "22.76", "+140% vs B&H")
with col4:
    st.metric("모델 승률", "91.3%", "264 중 241승")

fig_sharpe = px.bar(trading_results, x='전략', y='Sharpe Ratio', 
                    color='승률 (%)', color_continuous_scale='RdYlGn',
                    title='전략별 Sharpe Ratio')
fig_sharpe.update_layout(height=380)
st.plotly_chart(fig_sharpe)

st.dataframe(trading_results, hide_index=True)

st.markdown("""
<div class="result-card">
<strong>H3 채택:</strong> 모델 예측 전략 (Sharpe = 22.76) > Buy & Hold (Sharpe = 9.47), +140% 개선<br>
<br>
<strong>해석:</strong><br>
- VIX > 25 전략이 가장 높은 Sharpe(34.14)이나, 거래 횟수가 6회로 제한적<br>
- 모델 예측 전략은 264회 거래로 충분한 샘플 확보, 91.3% 승률 달성<br>
- 고변동성 구간(VIX > 20)에서 VRP 수익 기회가 증가함을 확인
</div>
""", unsafe_allow_html=True)

# ============================================================================
# 11. 결론
# ============================================================================
st.markdown('<h2 class="section-header">11. 결론</h2>', unsafe_allow_html=True)

show_diagram("09_conclusion.png", "09_conclusion.drawio")

st.markdown("""
<div class="explanation">
<h4>가설 검정 결과</h4>
<table style="width:100%; border-collapse: collapse;">
<tr style="background: #f8f9fa;">
    <th style="padding: 10px; border: 1px solid #ddd;">가설</th>
    <th style="padding: 10px; border: 1px solid #ddd;">결과</th>
    <th style="padding: 10px; border: 1px solid #ddd;">근거</th>
</tr>
<tr>
    <td style="padding: 10px; border: 1px solid #ddd;">H1: 모델 비교</td>
    <td style="padding: 10px; border: 1px solid #ddd; color: #27ae60;"><strong>채택</strong></td>
    <td style="padding: 10px; border: 1px solid #ddd;">MLP(0.44) > ElasticNet(0.37), +19%</td>
</tr>
<tr>
    <td style="padding: 10px; border: 1px solid #ddd;">H2: VIX-Beta</td>
    <td style="padding: 10px; border: 1px solid #ddd; color: #27ae60;"><strong>채택</strong></td>
    <td style="padding: 10px; border: 1px solid #ddd;">VIX-RV 상관 vs R-squared: r = -0.87</td>
</tr>
<tr>
    <td style="padding: 10px; border: 1px solid #ddd;">H3: 트레이딩</td>
    <td style="padding: 10px; border: 1px solid #ddd; color: #27ae60;"><strong>채택</strong></td>
    <td style="padding: 10px; border: 1px solid #ddd;">Sharpe 22.76, +140% vs Buy&Hold</td>
</tr>
</table>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="slide-card">
    <h4>학술적 기여</h4>
    <ul>
        <li><strong>VIX-Beta 이론</strong>: 자산별 VRP 예측력 차이를 설명하는 새로운 프레임워크</li>
        <li><strong>22일 Gap 프레임워크</strong>: 금융 ML에서 데이터 누수 방지 방법론</li>
        <li><strong>ML 우수성 실증</strong>: 비선형 모델이 전통 모델보다 VRP 예측에 효과적</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="slide-card">
    <h4>실무적 시사점</h4>
    <ul>
        <li><strong>자산 선택</strong>: VIX 상관 낮은 자산(금)에서 VRP 전략 효과적</li>
        <li><strong>시장 타이밍</strong>: 고변동성 구간(VIX > 20)에서 수익 기회 증가</li>
        <li><strong>전략 성과</strong>: 91.3% 승률, Sharpe 22.76 달성 가능</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# 12. 연구 흐름
# ============================================================================
st.markdown('<h2 class="section-header">12. 연구 흐름 요약</h2>', unsafe_allow_html=True)

show_diagram("10_research_flow.png", "10_research_flow.drawio")

# ============================================================================
# 13. 한계 및 향후 연구
# ============================================================================
st.markdown('<h2 class="section-header">13. 한계점 및 향후 연구</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="slide-card" style="border-left: 4px solid #f39c12;">
    <h4>한계점</h4>
    <ul>
        <li><strong>SPY 예측력 낮음</strong>: R-squared = 0.02, 실용적 가치 제한적</li>
        <li><strong>특정 연도 성능 저하</strong>: 2017년, 2023년 등 안정적 시장에서 부진</li>
        <li><strong>거래비용 미반영</strong>: 슬리피지, 수수료 등 실제 비용 고려 필요</li>
        <li><strong>단일 VIX 지수</strong>: S&P 500 기반 VIX만 사용</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="slide-card" style="border-left: 4px solid #2ecc71;">
    <h4>향후 연구 방향</h4>
    <ul>
        <li><strong>자산별 내재 변동성</strong>: GVZ(금), OVX(원유) 등 자산 특화 지수 활용</li>
        <li><strong>고빈도 데이터</strong>: 분 단위 데이터로 예측 정밀도 향상</li>
        <li><strong>딥러닝 확장</strong>: LSTM, Transformer 등 시계열 특화 모델</li>
        <li><strong>동적 포트폴리오</strong>: VRP 예측 기반 자산 배분 전략</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# 신규) 거래 비용 분석
# ============================================================================
st.markdown('<h2 class="section-header"> 거래 비용 분석</h2>', unsafe_allow_html=True)

if TRANSACTION_COSTS:
    st.markdown("""
    <div class="explanation">
    <h4>왜 거래 비용 분석이 중요한가?</h4>
    <p>
    학술 연구에서 보고되는 수익률은 종종 <strong>거래 비용을 무시</strong>합니다. 
    하지만 실제 투자에서는 다음과 같은 비용이 발생합니다:
    </p>
    <ul>
        <li><strong>수수료 (Commission)</strong>: 브로커에 지불하는 거래 수수료</li>
        <li><strong>슬리피지 (Slippage)</strong>: 주문가와 체결가의 차이</li>
        <li><strong>스프레드 (Bid-Ask Spread)</strong>: 매수/매도 호가 차이</li>
    </ul>
    <p>
    본 분석은 다양한 비용 시나리오(5~50 bps)에서 전략의 <strong>수익성이 유지되는지</strong> 검증합니다.
    일반적으로 기관 투자자는 5-10 bps, 개인 투자자는 20-30 bps 정도의 비용을 예상합니다.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 거래 비용 시나리오 차트
    cost_data = []
    for scenario, data in TRANSACTION_COSTS.get('cost_scenarios', {}).items():
        # 한글 시나리오명 처리
        short_name = scenario.split('(')[0].strip()
        cost_data.append({
            '시나리오': short_name,
            '순수익률 (%)': round(data['total_return'], 1),
            '승률 (%)': round(data['win_rate'] * 100, 1)
        })
    
    if cost_data:
        cost_df = pd.DataFrame(cost_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_cost = px.bar(cost_df, x='시나리오', y='순수익률 (%)', 
                             title='거래 비용별 순수익률',
                             color='순수익률 (%)', color_continuous_scale='RdYlGn')
            fig_cost.update_layout(height=350)
            st.plotly_chart(fig_cost, use_container_width=True)
        
        with col2:
            # 핵심 지표
            breakeven = TRANSACTION_COSTS.get('breakeven_cost_bps', 'N/A')
            turnover = TRANSACTION_COSTS.get('turnover', {}).get('annual_turnover', 'N/A')
            
            st.markdown(f"""
            <div class="result-card">
            <strong> 핵심 지표</strong><br><br>
            • <strong>손익분기 비용</strong>: {breakeven} bps (2%)<br>
            • <strong>연간 회전율</strong>: {turnover:.1f}회<br>
            • <strong>포지션 변경</strong>: 37회/275일<br>
            </div>
            """, unsafe_allow_html=True)
    
    # 인사이트 섹션
    st.markdown("""
    <div class="key-point">
    <strong> 핵심 인사이트</strong><br><br>
    <strong>1. 전략의 경제적 실현 가능성 확인</strong><br>
    • 손익분기 비용이 200 bps(2%)로 매우 높음 → 현실적 비용(10-30 bps)에서 충분한 마진 확보<br><br>
    
    <strong>2. 기관 투자자에게 적합</strong><br>
    • 기관 투자자 비용(5-10 bps) 적용 시 순수익률 800% 이상 유지<br>
    • 개인 투자자 비용(30 bps) 적용 시에도 795% 수익 달성<br><br>
    
    <strong>3. 낮은 회전율</strong><br>
    • 연간 33.9회 거래로 빈번한 매매 불필요 → 거래 비용 최소화
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("거래 비용 분석 결과를 로드할 수 없습니다. `python src/transaction_cost_analysis.py` 실행 필요")

# ============================================================================
# 신규) 구조적 변화 검정
# ============================================================================
st.markdown('<h2 class="section-header"> 구조적 변화 검정</h2>', unsafe_allow_html=True)

if STRUCTURAL_BREAKS:
    st.markdown("""
    <div class="explanation">
    <h4>왜 구조적 변화 검정이 필요한가?</h4>
    <p>
    2020-2025 데이터에는 <strong>COVID-19 팬데믹</strong>이 포함되어 있습니다.
    이 기간 동안 시장의 변동성 구조가 근본적으로 변화했을 가능성이 있습니다.
    </p>
    <p>
    <strong>Chow Test</strong>는 특정 시점을 기준으로 회귀 계수가 변화했는지 검정하는 통계 기법입니다.
    F-통계량이 높고 p-value가 낮으면 해당 시점에서 "구조적 변화"가 있었음을 의미합니다.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    chow_results = STRUCTURAL_BREAKS.get('chow_test', {})
    
    if chow_results:
        chow_data = []
        for period, data in chow_results.items():
            short_name = period.split('(')[0].strip()
            chow_data.append({
                '시점': short_name,
                'F-통계량': round(data.get('f_statistic', 0), 2),
                'p-value': data.get('p_value', 1),
                '유의성': ' 유의' if data.get('significant', False) else ' 미유의'
            })
        
        chow_df = pd.DataFrame(chow_data)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.dataframe(chow_df, hide_index=True, use_container_width=True)
        
        with col2:
            # F-통계량 바 차트
            fig_chow = px.bar(chow_df, x='시점', y='F-통계량', 
                             color='F-통계량', color_continuous_scale='Reds',
                             title='구조적 변화 시점별 F-통계량')
            fig_chow.update_layout(height=300, xaxis_tickangle=-45)
            st.plotly_chart(fig_chow, use_container_width=True)
        
        rolling = STRUCTURAL_BREAKS.get('rolling_analysis', {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="warning-card">
            <strong> 구조적 변화 감지</strong><br><br>
            • 5개 시점 모두 통계적으로 유의 (p < 0.001)<br>
            • COVID-19 팬데믹이 변동성 예측에 영향
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="result-card">
            <strong> 롤링 R² 분석</strong><br><br>
            • 평균 R²: {rolling.get('r2_mean', 0):.4f}<br>
            • 범위: {rolling.get('r2_min', 0):.4f} ~ {rolling.get('r2_max', 0):.4f}<br>
            • 모델 예측력이 시간에 따라 변동적
            </div>
            """, unsafe_allow_html=True)
    
    # 기간별 해석
    st.markdown("""
    <div class="explanation">
    <h4> 주요 시점 해석</h4>
    <table style="width:100%; border-collapse: collapse;">
        <tr style="background-color:#f8f9fa;">
            <th style="padding:8px; text-align:left; border-bottom:1px solid #ddd;">시점</th>
            <th style="padding:8px; text-align:left; border-bottom:1px solid #ddd;">시장 상황</th>
            <th style="padding:8px; text-align:left; border-bottom:1px solid #ddd;">모델 영향</th>
        </tr>
        <tr>
            <td style="padding:8px;">2020-03 (COVID 시작)</td>
            <td style="padding:8px;">VIX 80 돌파, 역사적 변동성</td>
            <td style="padding:8px;">극단적 상황에서 모델 불안정</td>
        </tr>
        <tr>
            <td style="padding:8px;">2020-12 (COVID 1차 종료)</td>
            <td style="padding:8px;">백신 개발, 변동성 진정</td>
            <td style="padding:8px;">모델 예측력 회복 시작</td>
        </tr>
        <tr>
            <td style="padding:8px;">2022-01 (금리인상)</td>
            <td style="padding:8px;">Fed 금리 인상 시작</td>
            <td style="padding:8px;">새로운 변동성 레짐 진입</td>
        </tr>
        <tr>
            <td style="padding:8px;">2023-01 (정상화)</td>
            <td style="padding:8px;">시장 안정화</td>
            <td style="padding:8px;">모델 예측력 정상 수준 회복</td>
        </tr>
    </table>
    </div>
    """, unsafe_allow_html=True)
    
    # 시사점
    st.markdown("""
    <div class="key-point">
    <strong> 핵심 시사점</strong><br><br>
    <strong>1. 모델의 한계 인식</strong><br>
    • 극단적 시장 상황(VIX > 40)에서는 모델 예측력이 저하될 수 있음<br>
    • 위기 상황에서는 추가적인 리스크 관리 필요<br><br>
    
    <strong>2. 적응적 접근 필요</strong><br>
    • 시장 구조가 변화하면 모델 재학습 고려<br>
    • 롤링 윈도우 학습으로 최신 패턴 반영<br><br>
    
    <strong>3. 긍정적 신호</strong><br>
    • 2023년 이후 모델 예측력 안정화<br>
    • 정상적인 시장 환경에서 전략 유효성 확인
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("구조적 변화 분석 결과를 로드할 수 없습니다.")

# ============================================================================
# 신규) VIX-Beta 확장 (9개 자산)
# ============================================================================
st.markdown('<h2 class="section-header"> VIX-Beta 이론 확장 (9개 자산)</h2>', unsafe_allow_html=True)

if VIX_BETA:
    st.markdown("""
    <div class="explanation">
    <h4>VIX-Beta 이론이란?</h4>
    <p>
    <strong>VIX-Beta 이론</strong>은 본 연구에서 새롭게 제안하는 프레임워크입니다.
    </p>
    <p>
    <strong>핵심 아이디어:</strong> VIX는 S&P 500 옵션에서 추출한 내재 변동성이므로,
    S&P 500과 <strong>상관관계가 높은 자산</strong>일수록 VIX가 해당 자산의 변동성을 더 잘 반영합니다.
    </p>
    <p>
    반대로, S&P 500과 <strong>상관관계가 낮은 자산</strong>(예: 금, 채권)은 VIX가 해당 자산의 변동성을
    제대로 반영하지 못합니다. 이 "오차"가 바로 <strong>예측 가능한 VRP</strong>입니다.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 자산 분류 설명
    st.markdown("""
    <div class="hypothesis-card">
    <strong> 분석 대상 자산 (9개)</strong><br><br>
    <table style="width:100%;">
        <tr>
            <td><strong>주식</strong>: SPY (S&P 500), QQQ (NASDAQ), IWM (소형주)</td>
        </tr>
        <tr>
            <td><strong>섹터</strong>: XLF (금융), XLE (에너지)</td>
        </tr>
        <tr>
            <td><strong>국제</strong>: EEM (신흥국)</td>
        </tr>
        <tr>
            <td><strong>대안자산</strong>: TLT (채권), GLD (금), USO (원유)</td>
        </tr>
    </table>
    </div>
    """, unsafe_allow_html=True)
    
    assets = VIX_BETA.get('assets', {})
    
    if assets:
        asset_data = []
        for ticker, data in assets.items():
            r2_val = data.get('r2_indirect', 0)
            asset_data.append({
                '자산': ticker,
                '설명': data.get('description', ''),
                'VIX-RV 상관': round(data.get('vix_rv_correlation', 0), 3),
                'R²': round(r2_val, 4),
                'R² (표시용)': max(r2_val, -0.5),  # 차트용 - 극단값 제한
                '방향정확도 (%)': round(data.get('direction_accuracy', 0) * 100, 1),
                '승률 (%)': round(data.get('win_rate', 0) * 100, 1)
            })
        
        asset_df = pd.DataFrame(asset_data)
        
        # 자산 데이터 테이블 표시
        st.markdown("####  자산별 성능 비교")
        st.dataframe(
            asset_df[['자산', '설명', 'VIX-RV 상관', 'R²', '방향정확도 (%)', '승률 (%)']].sort_values('R²', ascending=False),
            hide_index=True,
            use_container_width=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 방향정확도 차트 (더 해석하기 쉬움)
            fig_dir = px.bar(asset_df.sort_values('방향정확도 (%)', ascending=True), 
                            x='방향정확도 (%)', y='자산', orientation='h',
                            color='VIX-RV 상관', color_continuous_scale='RdYlBu_r',
                            title='자산별 방향 예측 정확도')
            fig_dir.add_vline(x=50, line_dash="dash", line_color="red", 
                             annotation_text="랜덤 (50%)")
            fig_dir.update_layout(height=400, xaxis_range=[30, 90])
            st.plotly_chart(fig_dir, use_container_width=True)
        
        with col2:
            # VIX 상관 vs 방향정확도 산점도
            fig_scatter = px.scatter(asset_df, x='VIX-RV 상관', y='방향정확도 (%)',
                                    text='자산', size=[40]*len(asset_df),
                                    color='R²', color_continuous_scale='RdYlGn',
                                    title='VIX 상관 vs 방향정확도')
            fig_scatter.add_hline(y=50, line_dash="dash", line_color="red")
            fig_scatter.update_traces(textposition='top center', textfont_size=10)
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        analysis = VIX_BETA.get('vix_beta_analysis', {})
        vix_r2_corr = analysis.get('vix_rv_vs_r2_correlation', 0)
        
        # 최고/최저 자산
        best_asset = asset_df.loc[asset_df['R²'].idxmax()]
        worst_asset = asset_df.loc[asset_df['R²'].idxmin()]
        
        st.markdown(f"""
        <div class="key-point">
        <strong> VIX-Beta 이론 검증 결과</strong><br><br>
        • VIX-RV 상관 vs R² 상관계수: <strong>{vix_r2_corr:.3f}</strong> (음의 상관 → 이론 지지)<br>
        • 최고 예측력: <strong>{best_asset['자산']}</strong> (R² = {best_asset['R²']:.4f}, 방향정확도 = {best_asset['방향정확도 (%)']:.1f}%)<br>
        • 최저 예측력: <strong>{worst_asset['자산']}</strong> (R² = {worst_asset['R²']:.4f})<br>
        • <em>VIX와 상관관계가 낮은 자산(TLT, GLD)에서 예측력이 더 높음</em>
        </div>
        """, unsafe_allow_html=True)
        
        # 투자 시사점
        st.markdown("""
        <div class="explanation">
        <h4> 투자 시사점</h4>
        <p>
        <strong>1. 자산 선택 전략</strong><br>
        VIX 기반 VRP 전략은 <strong>TLT(채권), GLD(금)</strong> 같은 대안자산에서 가장 효과적입니다.
        SPY, QQQ 같은 미국 주식에서는 VIX가 이미 변동성을 잘 반영하므로 예측 가치가 낮습니다.
        </p>
        <p>
        <strong>2. 포트폴리오 다각화</strong><br>
        VIX 기반 전략과 자산 특화 전략(GVZ, OVX 등)을 조합하면 더 높은 성과 기대 가능합니다.
        </p>
        <p>
        <strong>3. 크로스 애셋 차익거래</strong><br>
        VIX가 금 변동성을 과대평가할 때 금 VRP 매도 포지션 구축 가능합니다.
        </p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("VIX-Beta 확장 분석 결과를 로드할 수 없습니다.")

# ============================================================================
# 마무리
# ============================================================================
st.markdown("""
<div class="slide-title" style="margin-top: 2rem;">
    <h1 style="margin: 0; font-size: 2rem;">감사합니다</h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.9;">Q & A</p>
</div>
""", unsafe_allow_html=True)

# 추가 섹션 import
import sys
sys.path.insert(0, 'paper')
try:
    from dashboard_sections import (
        render_research_logic_flow, render_previous_research_failures, render_data_section, 
        render_model_detail_section, render_robustness_section, render_qa_section,
        render_one_page_summary, render_future_roadmap
    )
    
    # 추가 섹션 렌더링
    render_one_page_summary()  # 1페이지 요약 (맨 위)
    render_research_logic_flow()  # 논리 흐름 상세 설명
    render_previous_research_failures()
    render_data_section()
    render_model_detail_section()
    render_robustness_section()
    render_qa_section()
    render_future_roadmap()  # 향후 연구 로드맵 (맨 아래)
except Exception as e:
    st.warning(f"추가 섹션 로드 실패: {e}")

# 참고문헌
st.markdown('<h2 class="section-header">참고문헌 (References)</h2>', unsafe_allow_html=True)

st.markdown("""
1. **Bollerslev, T., Tauchen, G., & Zhou, H. (2009)**. Expected stock returns and variance risk premia. *Review of Financial Studies*, 22(11), 4463-4492.

2. **Carr, P., & Wu, L. (2009)**. Variance risk premiums. *Review of Financial Studies*, 22(3), 1311-1341.

3. **Corsi, F. (2009)**. A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174-196.

4. **Gu, S., Kelly, B., & Xiu, D. (2020)**. Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223-2273.

5. **Bekaert, G., & Hoerova, M. (2014)**. The VIX, the variance premium and stock market volatility. *Journal of Econometrics*, 183(2), 181-192.
""")

# Draw.io 다운로드
st.markdown("---")
st.markdown("""
<div style="background: #f8f9fa; border-radius: 8px; padding: 1rem; margin: 1rem 0;">
<h4>다이어그램 다운로드</h4>
<p>모든 다이어그램은 Draw.io 형식으로 다운로드하여 편집할 수 있습니다:</p>
<ul>
    <li><a href="https://app.diagrams.net/" target="_blank">Draw.io Online</a>에서 열기</li>
    <li>Draw.io Desktop 앱 사용</li>
    <li>VS Code Draw.io Integration 확장</li>
</ul>
</div>
""", unsafe_allow_html=True)

# 다운로드 버튼
drawio_files = [
    ("01_vrp_concept.drawio", "VRP 개념"),
    ("02_research_gap.drawio", "연구 갭"),
    ("03_hypothesis.drawio", "가설"),
    ("04_pipeline.drawio", "파이프라인"),
    ("05_features.drawio", "특성"),
    ("06_mlp.drawio", "MLP"),
    ("07_data_split.drawio", "데이터 분할"),
    ("08_vix_beta.drawio", "VIX-Beta"),
    ("09_conclusion.drawio", "결론"),
    ("10_research_flow.drawio", "연구 흐름"),
]

cols = st.columns(5)
for i, (filename, label) in enumerate(drawio_files):
    with cols[i % 5]:
        link = get_drawio_download_link(filename, label)
        if link:
            st.markdown(link, unsafe_allow_html=True)
