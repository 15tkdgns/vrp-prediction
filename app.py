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
from datetime import datetime
import os
from config.constants import COLORS, COLOR_SCHEMES, MODEL_COLORS, CATEGORY_COLORS, ASSET_COLORS
from utils.data_loader import load_spy_data, load_json_results, load_image, load_all_results
from utils.styles import get_css_styles

# 페이지 설정
st.set_page_config(
    page_title="VRP 예측 연구 발표",
    page_icon="📊",
    layout="wide"
)

# Sidebar 네비게이션
st.sidebar.title("📊 VRP 예측 연구")
st.sidebar.markdown("---")

selected_section = st.sidebar.radio(
    "섹션 선택",
    [
        "전체 보기",
        "개요 및 정의",
        "연구 방법론",
        "실험 결과",
        "상세 분석",
        "참고문헌"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.85rem; opacity: 0.7;">
<strong>총 23개 섹션</strong><br>
13개 그래프 | 10개 다이어그램
</div>
""", unsafe_allow_html=True)

# 분석 결과 로드
results = load_all_results()
TRANSACTION_COSTS = results['TRANSACTION_COSTS']
VIX_BETA = results['VIX_BETA']
SUBPERIOD = results['SUBPERIOD']
STRUCTURAL_BREAKS = results['STRUCTURAL_BREAKS']


# CSS 스타일 적용
st.markdown(get_css_styles(), unsafe_allow_html=True)

# Import section renderers
from sections.header import render_title, render_overview, render_mathematical_definitions, render_toc

# ============================================================================
# 제목 슬라이드
# ============================================================================
render_title()

# ============================================================================
# 프로젝트 개요
# ============================================================================
render_overview()

# ============================================================================
# 수학적 정의
# ============================================================================
render_mathematical_definitions()

# ============================================================================
# 목차
# ============================================================================
render_toc()

# ============================================================================
# 연구 방법론 섹션 (VRP 개념, 연구 갭, 가설, 파이프라인, 특성, 모델, 데이터 분할)
# ============================================================================
from sections.methodology import render_all_methodology
render_all_methodology()

# ============================================================================
# 실험 결과 섹션 (모델 성능, VIX-Beta, 트레이딩, 결론, 연구흐름, 한계점)
# ============================================================================
from sections.results import render_all_results
render_all_results()

# ============================================================================
# 상세 분석 섹션 (시각화, 핵심 그래프, 거래비용, 구조적 변화, VIX-Beta 확장)
# ============================================================================
from sections.analysis import render_all_analysis
render_all_analysis(TRANSACTION_COSTS, STRUCTURAL_BREAKS, VIX_BETA)

# ============================================================================
# 마무리 및 참고문헌
# ============================================================================
from sections.references import render_closing, render_references
render_closing()
render_references()



