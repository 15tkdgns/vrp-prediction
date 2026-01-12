"""
Streamlit Dashboard - Refactored Main App
VIX-RV Basis 변동성 예측 연구

Phase 2 Refactoring: 모듈화된 탭 구조
"""
import streamlit as st
import sys
sys.path.insert(0, '.')

# 페이지 설정
st.set_page_config(
    page_title="CAVB 변동성 예측",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #f8f9fa;
        border-left: 4px solid #48bb78;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# 메인 타이틀
st.title("📊 VIX-RV Basis 기반 Cross-Asset 변동성 예측")
st.markdown("**5일 선행 예측 | ElasticNet 모델 | HAR-RV 벤치마크**")
st.markdown("---")

# 탭 생성
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 연구 개요",
    "🔬 방법론",
    "📈 결과",
    "✓ 검증",
    "📚 참고문헌"
])

with tab1:
    from dashboard.tabs.tab_overview import render_overview
    render_overview()

with tab2:
    from dashboard.tabs.tab_methodology import render_methodology
    render_methodology()

with tab3:
    from dashboard.tabs.tab_results import render_results
    render_results()

with tab4:
    from dashboard.tabs.tab_validation import render_validation
    render_validation()

with tab5:
    from dashboard.tabs.tab_references import render_references
    render_references()

# Footer
st.markdown("---")
st.markdown("""
<p style="text-align:center; color:#999; font-size:0.9rem;">
    CAVB 예측 연구 (5일 예측 기간) | 데이터: Yahoo Finance (2015-2025) | 
    모델: ElasticNet | 검증: 3-Way Split + 6-fold Leakage Test
</p>
""", unsafe_allow_html=True)
