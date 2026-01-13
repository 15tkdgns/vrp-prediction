import streamlit as st
import pandas as pd

def render_references():
    """선행연구 및 연구 포지셔닝 탭 렌더링"""
    
    st.header("📚 선행연구 및 연구 포지셔닝 (Prior Work & Positioning)")
    
    st.markdown("""
    본 섹션에서는 VIX·VRP 및 관련 자산가격 결정 모형에 관한 **실존 선행연구**를 요약하고, 
    본 연구가 어떤 **방법론적·실증적 틈새(Research Gap)**를 겨냥하는지 개념적으로 정리합니다.
    """)
    
    st.warning("⚠️ **주의**: 원문 논문과 대조할 수 없는 특정 수치(R², t-stat 등)는 배제하였으며, 핵심 논리와 본 연구의 차별점을 기술하는 데 집중하였습니다.")
    
    st.markdown("---")

    # ==========================================
    # 1. Branco et al. (2024)
    # ==========================================
    st.subheader("1. Forecasting Realized Volatility: Linear Models vs. Alternatives")
    
    with st.expander("📄 **Branco et al. (2024) – Empirical Finance Evidence**", expanded=True):
        st.markdown("""
        **실존 논문**: 
        > Branco, R., Rubesam, A., & Zevallos, M. (2024).  
        > "Forecasting realized volatility: Does anything beat linear models?"  
        > *Journal of Empirical Finance*, 78, 101524.
        
        **핵심 내용**:
        1. 주식지수 등의 실현 변동성을 여러 예측 모형(HAR, 머신러닝 등)으로 비교 분석함.
        2. **단순한 선형 모형이 여전히 강력한 벤치마크**임을 실증적으로 보고.
        3. 다양한 확장형 모델이나 복잡한 비선형 모델이 항상 유의미한 표본 외(out-of-sample) 성능 개선을 보장하지 않음을 강조.
        """)
        
    with st.container():
        st.success("""
        **🎯 본 연구의 위치 (Positioning)**
        
        우리 연구는 **VIX 및 VRP 관련 변수(예: VIX–RV 차이)를 포함한 확장형 선형 모형(ElasticNet 계열)**을 이용하여 5일 선행 RV를 예측합니다.
        
        Branco et al. (2024)이 "단순 선형 모형의 강점"을 문서화한 것과 맥락을 같이 하되, 우리는 다음과 같은 구체적인 확장을 탐색합니다:
        - **VRP 관련 Feature 설계**: 단순 RV 예측을 넘어 VIX와의 괴리(Basis) 정보 활용.
        - **정규화(Regularization)**: ElasticNet을 통한 변수 선택으로 과적합 방지.
        - **다자산(Cross-asset)**: 단일 자산이 아닌 5개 자산군에 대한 일반화 가능성 탐색.
        """)

    st.markdown("---")

    # ==========================================
    # 2. VRP Components
    # ==========================================
    st.subheader("2. Variance Risk Premium Components")

    with st.expander("📄 **관련 실존 연구 (Londono & Xu / Prokopczuk et al.)**"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Londono & Xu (2019, IFDP Notes)**
            - **제목**: "Variance Risk Premium Components and International Stock Return Predictability"
            - **내용**: VRP의 구성요소가 국제 주식 수익률 예측에 유의미한 정보를 제공함을 시사함.
            """)
        with col2:
            st.markdown("""
            **Prokopczuk et al. (2017, JBF)**
            - **제목**: "Variance risk in commodity markets"
            - **내용**: 원자재 시장에서의 VRP 측정 및 시계열 특성 분석.
            """)

    with st.container():
        st.info("""
        **🎯 본 연구와의 개념적 차별점**
        
        기존 연구들은 주로 월간/분기 빈도에서 주식 수익률(Return) 예측에 VRP를 활용하였습니다. 반면 본 연구는:
        
        1. **일간 빈도(Daily Frequency)**에서 **변동성(RV) 자체**를 예측하는 데 초점을 둡니다.
        2. VRP를 복잡하게 분해하기보다, **지속(Persistent) vs 단기(Transitory)** 성분으로 나누는 단순 필터링(이동평균 등) 방식을 적용하여 실무적 효용성을 검증합니다.
        """)

    st.markdown("---")

    # ==========================================
    # 3. Bali et al. (2023)
    # ==========================================
    st.subheader("3. Machine Learning for Option Return Predictability")

    with st.expander("📄 **Bali et al. (2023) – Machine Learning and Big Data**", expanded=True):
        st.markdown("""
        **실존 논문**: 
        > Bali, T. G., Beckmeyer, H., Moerke, M., & Weigert, F. (2023).  
        > "Option Return Predictability with Machine Learning and Big Data"  
        > *The Review of Financial Studies*, 36(9), 3548–3602.
        
        **핵심 내용**:
        1. 수백만 건 이상의 대규모 옵션 데이터를 사용.
        2. 심층 비선형 ML 모형이 선형 모형을 **옵션 수익률 예측**에서 지속적으로 능가함을 입증.
        3. 옵션 수익률과 같은 고도의 비선형성/노이즈 환경에서는 ML이 경제적으로 유의미한 성과를 냄.
        """)

    with st.container():
        st.error("""
        **🎯 본 연구와의 연결 및 대조**
        
        - Bali et al.은 **초대형 데이터셋(Big Data)**과 **비선형 타겟(Option Returns)** 환경에서의 ML 우위를 보여줍니다.
        - 본 연구는 상대적으로 **중간 규모(Moderate Data)**의 일간 패널 데이터와, **선형성이 강한 RV–VIX 관계**를 다룹니다.
        
        👉 **시사점**: "데이터 규모와 비선형성 정도에 따라 ML과 선형 모형의 우위가 달라진다"는 조건부 해석을 지지하며, 이 환경에서는 **ElasticNet과 같은 정규화된 선형 모형**이 더 효율적일 수 있음을 시사합니다.
        """)

    st.markdown("---")

    # ==========================================
    # 4. Term Structure & Good/Bad Volatility
    # ==========================================
    st.subheader("4. Additional Theoretical Frameworks")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Term Structure of Risk")
        st.markdown("""
        **Hollstein et al. (2019, JFM)**:
        - 개별 자산의 체계적·비체계적 위험의 만기 구조 분석.
        
        **본 연구의 단순화 전략**:
        - 정교한 파생상품 가격 대신, **VIX의 변화율과 이동평균** 등을 사용하여 '기간 구조 유사 정보'를 근사(Proxy)합니다.
        - 이는 데이터 가용성이 제한적인 실무 환경에서의 적용 가능성을 높입니다.
        """)

    with col2:
        st.markdown("#### Good/Bad Environment")
        st.markdown("""
        **Bekaert & Engstrom (2017, JPE)**:
        - 경제의 'Good' 상태와 'Bad' 상태에서 위험 회피도가 비대칭적으로 작동함.
        
        **본 연구의 응용**:
        - 단순히 총 변동성을 보는 것보다, **상승장(Good Vol)**과 **하락장(Bad Vol)**을 구분합니다.
        - 특히 **신흥국 시장(EEM)** 등 위험 회피 성향이 강한 자산군에서 이러한 비대칭적 정보가 유효한지 검증합니다.
        """)

    st.markdown("---")

    # ==========================================
    # Summary Table
    # ==========================================
    st.subheader("5. 요약: 검증된 범위 내에서의 Research Gap")

    gap_data = pd.DataFrame({
        "선행 연구 (Reference)": [
            "Branco et al. (2024)", 
            "Prokopczuk / Londono", 
            "Bali et al. (2023)", 
            "Bekaert & Engstrom (2017)"
        ],
        "주요 내용 및 기여": [
            "RV 예측에서 선형 모형의 강건성 입증",
            "VRP 특성 및 구성요소 분석 (Return 예측)",
            "빅데이터/옵션 수익률에서 ML 우위 입증",
            "Good/Bad 환경의 비대칭적 위험 모형"
        ],
        "본 연구의 차별점 (Gap)": [
            "VIX-RV Basis(CAVB) 변수 도입 및 Cross-asset 확장",
            "일간 빈도의 RV 예측으로 타겟 변경, 단순화된 필터링 적용",
            "중간 규모 데이터/RV 예측에서 선형(ElasticNet) 효율성 재확인",
            "비대칭성 이론을 자산별 RV 예측 변수(Feature)로 실증 적용"
        ]
    })
    
    st.table(gap_data)
