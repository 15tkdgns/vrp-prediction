#!/usr/bin/env python3
"""
VRP 예측 연구 발표자료
====================

머신러닝을 활용한 변동성 위험 프리미엄 예측 연구
발표 및 보고서용 Streamlit 대시보드

재배열된 순서 (2024-12-22):
Part 1: Introduction (서론) - 1~7
Part 2: Methodology (방법론) - 8~12  
Part 3: Model Analysis (모델 분석) - 13~17
Part 4: Economic Value (경제적 검증) - 18~20
Part 5: Conclusion (결론) - 21~25
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
        "Part 1: 서론",
        "Part 2: 방법론",
        "Part 3: 모델 분석",
        "Part 4: 경제적 검증",
        "Part 5: 결론"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.85rem; opacity: 0.7;">
<strong>총 25개 섹션</strong><br>
15개 그래프 | 10개 다이어그램
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

# ============================================================================
# 섹션 모듈 Import
# ============================================================================
from sections.header import render_title, render_overview, render_mathematical_definitions, render_toc
from sections.methodology import (render_vrp_concept, render_research_gap, render_hypothesis, 
                                   render_pipeline, render_features, render_models, render_data_split)
from sections.results import (render_model_performance, render_vix_beta, 
                               render_trading_performance, render_conclusion, 
                               render_research_flow, render_limitations)
from sections.analysis import (render_data_visualization, render_core_graphs,
                                render_transaction_costs, render_structural_breaks, 
                                render_vix_beta_expansion)
from sections.model_explainability import render_model_explainability
from sections.references import render_closing, render_references

# ============================================================================
# PART 1: Introduction (서론) - 섹션 1~7
# ============================================================================
st.markdown("---")
st.markdown("## 📘 Part 1: Introduction (서론)")

# 1. 제목 슬라이드
render_title()

# 2. 목차
render_toc()

# 3. 프로젝트 개요
render_overview()

# 4. VRP 개념
render_vrp_concept()

# 5. 수학적 정의
render_mathematical_definitions()

# 6. 연구 갭
render_research_gap()

# 7. 연구 가설
render_hypothesis()

# ============================================================================
# PART 2: Methodology (방법론) - 섹션 8~12
# ============================================================================
st.markdown("---")
st.markdown("## 📗 Part 2: Methodology (방법론)")

# 8. 예측 파이프라인
render_pipeline()

# 9. 데이터 시각화
render_data_visualization()

# 10. 특성 변수
render_features()

# 11. 데이터 분할
render_data_split()

# 12. 모델 선택
render_models()

# ============================================================================
# PART 3: Model Analysis (모델 분석) - 섹션 13~17
# ============================================================================
st.markdown("---")
st.markdown("## 📙 Part 3: Model Analysis (모델 분석)")

# 13. 모델 성능
render_model_performance()

# 14. 모델 설명가능성 (NEW!)
render_model_explainability()

# 15. 핵심 분석 그래프
render_core_graphs()

# 16. VIX-Beta 이론
render_vix_beta()

# 17. VIX-Beta 확장
render_vix_beta_expansion(VIX_BETA)

# ============================================================================
# PART 4: Economic Value (경제적 검증) - 섹션 18~20
# ============================================================================
st.markdown("---")
st.markdown("## 📕 Part 4: Economic Value (경제적 검증)")

# 18. 트레이딩 성과
render_trading_performance()

# 19. 거래 비용 분석
render_transaction_costs(TRANSACTION_COSTS)

# 20. 구조적 변화 검정
render_structural_breaks(STRUCTURAL_BREAKS)

# ============================================================================
# PART 5: Conclusion (결론) - 섹션 21~25
# ============================================================================
st.markdown("---")
st.markdown("## 📓 Part 5: Conclusion (결론)")

# 21. 연구 흐름 요약
render_research_flow()

# 22. 결론
render_conclusion()

# 23. 한계점 및 향후 연구
render_limitations()

# 24. 참고문헌
render_references()

# 25. 마무리 슬라이드
render_closing()
