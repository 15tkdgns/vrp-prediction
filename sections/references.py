"""
참고문헌 섹션
"""
import streamlit as st


def render_closing():
    """마무리 슬라이드 렌더링"""
    st.markdown("""
<div class="slide-title" style="margin-top: 2rem;">
    <h1 style="margin: 0; font-size: 2rem;">감사합니다</h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.9;">Q & A</p>
</div>
""", unsafe_allow_html=True)


def render_references():
    """참고문헌 렌더링"""
    st.markdown('<h2 class="section-header">참고문헌 (References)</h2>', unsafe_allow_html=True)
    
    st.markdown("""
1. **Bollerslev, T., Tauchen, G., & Zhou, H. (2009)**. Expected stock returns and variance risk premia. *Review of Financial Studies*, 22(11), 4463-4492.

2. **Carr, P., & Wu, L. (2009)**. Variance risk premiums. *Review of Financial Studies*, 22(3), 1311-1341.

3. **Corsi, F. (2009)**. A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174-196.

4. **Gu, S., Kelly, B., & Xiu, D. (2020)**. Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223-2273.

5. **Bekaert, G., & Hoerova, M. (2014)**. The VIX, the variance premium and stock market volatility. *Journal of Econometrics*, 183(2), 181-192.
""")
