import streamlit as st
import pandas as pd

def render_references():
    """선행연구 및 연구 포지셔닝 탭 렌더링"""
    
    st.header("📚 선행연구 및 레퍼런스")

    tab1, tab2 = st.tabs(["선행연구 (Prior Work)", "레퍼런스 (Bibliography)"])

    with tab1:
        st.markdown("""
        본 섹션에서는 VIX·VRP 및 관련 자산가격 결정 모형에 관한 **실존 선행연구**를 심층 분석하고, 
        본 연구가 어떤 **방법론적·실증적 틈새(Research Gap)**를 겨냥하는지 상세히 기술합니다.
        """)
        
        st.info("💡 각 연구의 핵심 내용과 본 연구와의 차별점(Positioning)을 중심으로 정리했습니다.")
        
        st.markdown("---")

        # ==========================================
        # 1. Branco et al. (2024)
        # ==========================================
        st.subheader("1. Branco et al. (2024) – Forecasting Realized Volatility")
        
        with st.expander("📄 **연구 개요**", expanded=True):
            st.markdown("""
            ### 1.1 기본정보 및 연구 질문
            - **논문**: Branco, R. R., Rubesam, A., & Zevallos, M. (2024). *"Forecasting realized volatility: Does anything beat linear models?"* Journal of Empirical Finance.
            - **링크**: https://doi.org/10.1016/j.jempfin.2024.101524
            - **연구 질문**:
                - HAR-RV 같은 단순 선형 모형을 체계적으로 능가하는 대안이 존재하는가?
                - 비선형 ML 모형이 선형 모형 대비 통계적·경제적으로 얼마나 개선을 제공하는가?

            ### 1.2 핵심 결과
            | 항목 | 결과 개요 |
            | :--- | :--- |
            | **ML vs 선형** | **비선형 ML이 선형 모형을 체계적으로 능가한다는 증거 없음** |
            | **변동성 구간** | ML은 고변동성에서 과소, 저변동성에서 과대 예측하는 경향 |
            | **경제적 가치** | 월간 RV 예측 이용 전략에서는 **단순·저차원 선형 모형이 더 유리** |
            """)
        
        with st.expander("🔬 **상세 방법론 보기**"):
            st.markdown("""
            ### 데이터 및 샘플
            - **자산 범위**: S&P 500, EuroStoxx, Nikkei, Shanghai 등 **10개 글로벌 주식시장 지수**
            - **표본 기간**: 2000년~2021년 (약 20년)
            - **RV 계산**: 고빈도 수익률(5분/10분)로부터 realized variance 계산
              - 여러 Horizon: 일간 RV, 5일 평균 RV, 22일 평균 RV
            
            ### 예측 변수(Features)
            
            **Baseline (HAR-RV 구조)**:
            - Daily RV, Weekly RV (5일 평균), Monthly RV (22일 평균)
            
            **Extended Features (HAR-X)**:
            - **Implied Volatility**: 지수 옵션으로부터 추출 (1M, 3M 만기 IV)
            - **주식 수익률**: Daily/Weekly/Monthly 수익률
            - **거시·심리 변수**: Fed Funds rate, Term spread, Option skewness 등
            
            ### 모형군
            
            **선형 모형**:
            - HAR-RV: $RV_{t+h} = \\beta_0 + \\beta_1 RV_t + \\beta_2 \\overline{RV}_{5d} + \\beta_3 \\overline{RV}_{22d} + \\varepsilon$
            - Lasso, Ridge, ElasticNet (정규화 모형, CV로 λ 선택)
            
            **비선형 ML 모형**:
            - Random Forest (트리 100~500개, 깊이 5~20)
            - Gradient Boosting (learning rate 0.01~0.1)
            - Neural Network (은닉층 1~3개, 각 50~200 노드, ReLU 활성화)
            
            ### 평가 전략
            - **Rolling Window Cross-Validation**: 학습 5년 → 테스트 1개월, 롤링
            - **평가 지표**:
              - 통계적: MSE, RMSE, MAE, Out-of-sample R²
              - 경제적: Volatility-targeting 포트폴리오의 Sharpe ratio
            - **Diebold-Mariano Test**: 모형 간 예측 성능 차이 검정
            """)

        st.success("""
        **🎯 본 연구의 차별점 (Positioning)**
        
        - Branco et al.은 "ML이 항상 낫다"는 통념에 의문을 제기하며 **선형 모형의 강건성**을 입증했습니다.
        - 본 연구는 이 결과를 지지하며, **VRP 관련 Feature(VIX-RV 괴리)와 ElasticNet**의 조합을 통해 "선형 모형 + 특화 변수"의 효율성을 RV-VIX 맥락에서 재검증합니다.
        """)

        st.markdown("---")

        # ==========================================
        # 2. VRP Components
        # ==========================================
        st.subheader("2. Londono & Xu / Prokopczuk et al. – VRP와 국제/원자재 시장")

        with st.expander("📄 **연구 개요**", expanded=False):
            st.markdown("""
            ### 2.1 기본정보
            - **Londono & Xu (2019)**: *"Variance Risk Premium Components..."* (Federal Reserve IFDP)
            - **Prokopczuk et al. (2017)**: *"Variance risk in commodity markets"* (Journal of Banking & Finance)
            
            ### 2.2 핵심 결과
            | 연구 | 타겟 | 주요 결과 |
            | :--- | :--- | :--- |
            | **Londono & Xu** | 국제 주식수익 | VRP 구성요소가 국제 주식 수익률 예측에 유의미 |
            | **Prokopczuk** | 원자재 VRP | VRP가 크게 음(-)이며 Sharpe Ratio 높음, 타 자산군과 독립적 |
            """)
        
        with st.expander("🔬 **상세 방법론 보기**"):
            st.markdown("""
            ### VRP 정의
            
            $$VRP_t(\\tau) = IV^2_t(\\tau) - \\mathbb{E}_t[RV_{t,t+\\tau}]$$
            
            - $IV_t(\\tau)$: 만기 τ 옵션의 risk-neutral 기대 변동성
            - $\\mathbb{E}_t[RV]$: 물리적 확률 기반 실현 변동성 기대값
            
            ### Londono & Xu – VRP Components
            
            **데이터**:
            - 국가: 미국, 선진국, 신흥국 대표 지수
            - 빈도: **월간** (월말 측정 → 다음달 수익률 예측)
            - 기간: 1990년대~2010년대
            
            **VRP 분해**:
            - 만기별: 단기(1M) vs 장기(12M) VRP
            - 방향별: Downside/Upside VRP (OTM Put/Call 옵션 분리)
            
            **예측 회귀**:
            $$R_{i,t+1} = \\alpha + \\beta' VRP\_Components_t + \\gamma' Controls_t + \\varepsilon$$
            
            ### Prokopczuk et al. – Commodity Markets
            
            **데이터**: 21개 원자재 선물/옵션 (1990~2015, 월간)
            
            **VRP 측정 – Synthetic Variance Swaps**:
            - Variance swap rate 계산 (Britten-Jones & Neuberger 공식)
            - Put/Call 옵션 적분으로 risk-neutral 분산 추출
            
            **분석**:
            1. 평균 VRP, Sharpe ratio 계산
            2. 주식/채권 요인과의 독립성 검정 (회귀 분석)
            """)

        st.info("""
        **🎯 본 연구의 차별점 (Positioning)**
        
        - 선행연구들은 주로 **월간 수익률(Return)** 예측에 초점을 두었습니다.
        - 본 연구는 **일간 RV(변동성)** 자체를 예측하는 데 VRP(VIX-RV 차이)를 활용하며, 복잡한 분해 대신 **간단한 이동평균 필터링(지속 vs 단기)**을 적용하여 실무적 효용성을 검증합니다.
        """)

        st.markdown("---")

        # ==========================================
        # 3. Bali et al. (2023)
        # ==========================================
        st.subheader("3. Bali et al. (2023) – Option Return Predictability with ML")

        with st.expander("📄 **연구 개요**", expanded=True):
            st.markdown("""
            ### 3.1 기본정보 및 연구 질문
            - **논문**: Bali, T. G., et al. (2023). *"Option Return Predictability with Machine Learning and Big Data."* RFS.
            - **링크**: https://doi.org/10.1093/rfs/hhad017
            - **연구 질문**: 옵션/주식 특성 기반 **ML 모형**이 선형 모형보다 **옵션 수익률**을 더 잘 예측하는가?

            ### 3.2 핵심 결과
            | 항목 | 결과 개요 |
            | :--- | :--- |
            | **예측력** | **비선형 ML 모형**이 선형 모형 대비 옵션 수익률 예측력 우수 |
            | **경제적 가치** | ML 기반 롱-숏 옵션 전략에서 유의한 초과수익(Alpha) 발생 |
            """)
        
        with st.expander("🔬 **상세 방법론 보기**"):
            st.markdown("""
            ### 데이터
            - **표본 규모**: **1,200만 건 이상**의 옵션-월 관측치 (1996~2020)
            - **자산**: 미국 개별 주식 옵션 (S&P 500 구성 종목 중 높은 유동성 종목)
            - **필터링**: 극단적 deep OTM 제외, 거래량 부족·만기 1주 미만 제외
            
            ### 타겟 변수 – 옵션 수익률 (Delta-Hedged)
            
            $$Return_{t+1} = \\frac{V_{t+1}^{delta-hedged} - V_t}{V_t}$$
            
            - 옵션 가격 변화 - 델타 헷지 수익 (주식 공매)
            - 보유 기간: 주로 1개월
            
            ### 예측 변수 (70~120개 Features)
            
            **1) 옵션 기반 특성 (~30개)**:
            - **Greeks**: Delta, Gamma, Vega, Theta, Rho
            - **IV 관련**: Implied volatility 수준/변화, IV 비대칭(skew)
            - **Moneyness**: $M = \\ln(K/S)$ (행사가/현물가)
            - **거래량·유동성**: 일일 거래량, Bid-ask spread
            
            **2) 주식 기반 특성 (~25개)**:
            - **수익률**: 과거 1M/3M/12M 수익률, 평균 회귀 지표
            - **변동성**: 20일/60일 historical volatility
            - **기본 특성**: 시가총액, Book-to-market, 수익성
            
            **3) 시장·거시 변수 (~15개)**:
            - S&P 500 수익률, VIX 수준/변화
            - 금리 스프레드 (3M vs 10Y Treasury)
            - Put/Call ratio (시장 심리)
            
            ### 모형
            
            **선형 모형**:
            - 다중 회귀 (OLS), Lasso/Ridge
            
            **비선형 ML 모형**:
            - **Random Forest**: 트리 100~500개, 깊이 10~30
            - **Gradient Boosting** (XGBoost): Learning rate 0.01~0.1, 깊이 3~7
            - **Neural Network**:
              - 구조: Input (70~120) → Hidden 1 (256~512, ReLU, Dropout) → Hidden 2 (128~256) → Output (1)
              - 최적화: Adam, L2 regularization, Early stopping
            
            ### 평가
            - **Rolling Window**: 학습 2~3년 → 테스트 1개월
            - **하이퍼파라미터**: 5-fold CV, Grid/Random search
            - **평가 지표**:
              - 예측 성능: Pearson correlation, Out-of-sample R²
              - 경제적 가치: 예측 상위 10% 롱/하위 10% 숏, Sharpe ratio (거래비용 반영)
            """)

        st.error("""
        **🎯 본 연구의 차별점 (Positioning)**
        
        - Bali et al.은 **빅데이터 + 고도의 비선형 타겟(옵션 수익률)** 환경에서 ML의 우위를 입증했습니다.
        - 반면 본 연구는 **중간 규모 데이터 + 선형성이 강한 타겟(RV)** 환경입니다. 
        - 이는 "데이터 특성에 따라 최적 모형이 다르다"는 점을 시사하며, 우리 환경에서는 **ElasticNet과 같은 정규화된 선형 모형**이 더 적합할 수 있음을 보여주는 대조군(Counterpart) 역할을 합니다.
        """)

        st.markdown("---")

        # ==========================================
        # 4. Hollstein et al. (2019)
        # ==========================================
        st.subheader("4. Hollstein et al. (2019) – Term Structure of Risk")

        with st.expander("📄 **연구 개요**", expanded=False):
            st.markdown("""
            ### 4.1 기본정보
            - **논문**: Hollstein, F., et al. (2019). *"The term structure of systematic and idiosyncratic risk."* JFM.
            - **링크**: https://centaur.reading.ac.uk/81271
            - **질문**: 체계적/비체계적 위험의 만기 구조가 시장 상태에 따라 어떻게 변하는가?
            """)
        
        with st.expander("🔬 **상세 방법론 보기**"):
            st.markdown("""
            ### 위험 분해 (Risk Decomposition)
            
            자산 수익률 회귀:
            $$R_{i,t} = \\alpha_i + \\beta_i(t) R_{M,t} + \\varepsilon_{i,t}$$
            
            - **체계적 위험**: $\\sigma_{Sys}^2(\\tau) = \\beta^2(\\tau) \\cdot \\sigma_M^2(\\tau)$
            - **비체계적 위험**: $\\sigma_{Idio}^2(\\tau) = Var(\\varepsilon(\\tau))$
            
            ### Term Structure 구성
            
            만기별 벡터: $TS = [\\sigma(1M), \\sigma(3M), \\sigma(6M), \\sigma(12M)]$
            
            **요인 추출**:
            - **Level**: $\\frac{1}{k}\\sum \\sigma(\\tau)$ (평균)
            - **Slope**: $\\sigma(12M) - \\sigma(1M)$ (기울기)
            - **Curvature**: $2\\sigma(3M) - \\sigma(1M) - \\sigma(6M)$ (곡률)
            
            ### 레짐 분석
            - Low volatility: VIX < 15
            - Normal: VIX 15~25
            - High stress: VIX > 25
            
            → 각 상태별 Level/Slope/Curvature 비교
            """)

        st.warning("""
        **🎯 본 연구의 차별점 (Positioning)**
        
        - 본 연구는 복잡한 파생상품 만기 구조를 모델링하는 대신, **VIX의 변화율, 이동평균** 등을 사용하여 '기간 구조 정보'를 **단순화(Proxy)**하여 활용합니다.
        - 이는 데이터 제약이 있는 환경에서의 실무적 적용 가능성을 높이는 접근입니다.
        """)

        st.markdown("---")

        # ==========================================
        # 5. Bekaert & Engstrom (2017)
        # ==========================================
        st.subheader("5. Bekaert & Engstrom (2017) – Good/Bad Environment")

        with st.expander("📄 **연구 개요**", expanded=False):
            st.markdown("""
            ### 5.1 기본정보
            - **논문**: Bekaert, G., & Engstrom, E. (2017). *"Asset Return Dynamics under Habits and Bad Environment–Good Environment Fundamentals."* JPE.
            - **링크**: https://doi.org/10.1086/691450
            - **질문**: Habit과 Good/Bad 환경이 자산 수익률 동학과 위험 프리미엄에 미치는 영향은?
            """)
        
        with st.expander("🔬 **상세 방법론 보기**"):
            st.markdown("""
            ### 이론 모형 구조
            
            **상태 변수**:
            - $s_t \\in \\{Good, Bad\\}$: 2-state Markov process
            - $h_t$: 소비 습관 (Habit stock), $h_{t+1} = \\rho h_t + (1-\\rho) C_t$
            
            **소비 동학** (상태 의존):
            $$\\ln(C_{t+1}/C_t) = g_s + \\sigma_s \\epsilon_{t+1}$$
            
            - Good: $g_G, \\sigma_G$ (낮은 변동성)
            - Bad: $g_B < g_G, \\sigma_B > \\sigma_G$ (높은 변동성)
            
            **효용 함수** (Habit-based):
            $$U(C_t, h_t) = \\frac{(C_t - h_t)^{1-\\gamma}}{1-\\gamma}$$
            
            - 상대적 위험회피도: $RRA_t = \\gamma \\cdot S_t / (C_t - h_t)$
            - 소비가 습관에 근접할수록 RRA 급증
            
            ### 위험 프리미엄
            
            $$Risk\\ Premium_t = \\gamma \\cdot Cov[r_t, \\Delta c_t] + \\lambda(s_t) \\cdot \\gamma$$
            
            - 첫 번째 항: 소비 공분산 (전통 CCAPM)
            - 두 번째 항: 상태 의존 위험회피도 ($\\lambda(Bad) >> \\lambda(Good)$)
            
            ### 추정
            - GMM (Generalized Method of Moments)
            - Bayesian MCMC
            - 2단계 추정: 상태 필터링 → 파라미터 추정
            """)

        st.success("""
        **🎯 본 연구의 차별점 (Positioning)**
        
        - "상승장(Good Vol) vs 하락장(Bad Vol)"을 분리하여 RV 예측 변수로 사용하는 본 연구의 아이디어에 대한 **이론적 근거**를 제공합니다.
        - 특히 신흥국(EEM) 등 위험 회피 성향이 강한 시장에서의 비대칭적 반응을 해석하는 틀로 활용합니다.
        """)

    with tab2:
        st.markdown("## 📑 참고문헌 목록 (Bibliography)")
        st.markdown("""
        ### 주요 인용 논문

        1.  **Bali, T. G., Beckmeyer, H., Moerke, M., & Weigert, F. (2023)**. Option Return Predictability with Machine Learning and Big Data. *The Review of Financial Studies*, 36(9), 3548–3602.
        2.  **Bekaert, G., & Engstrom, E. (2017)**. Asset Return Dynamics under Habits and Bad Environment–Good Environment Fundamentals. *Journal of Political Economy*, 125(3), 713–760.
        3.  **Branco, R. R., Rubesam, A., & Zevallos, M. (2024)**. Forecasting realized volatility: Does anything beat linear models? *Journal of Empirical Finance*, 78, 101524.
        4.  **Hollstein, F., Prokopczuk, M., & Wese Simen, C. (2019)**. The term structure of systematic and idiosyncratic risk. *Journal of Futures Markets*.
        5.  **Londono, J. M., & Xu, N. R. (2019)**. Variance Risk Premium Components and International Stock Return Predictability. *International Finance Discussion Papers*, Federal Reserve.
        6.  **Prokopczuk, M., Symeonidis, L., & Wese Simen, C. (2017)**. Variance risk in commodity markets. *Journal of Banking & Finance*, 81, 136–149.
        """)
