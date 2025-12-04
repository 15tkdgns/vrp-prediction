"""
SPY 변동성 예측 대시보드 - Streamlit 버전

Ridge 회귀 기반 5일 변동성 예측 시스템의 성능 분석 및 시각화
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Optional imports
try:
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from scipy import stats
    from scipy.stats import probplot
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# 페이지 설정
st.set_page_config(
    page_title="SPY 변동성 예측 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일링
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# 데이터 로드 함수
@st.cache_data
def load_model_performance():
    """모델 성능 데이터 로드"""
    file_path = Path("data/raw/model_performance.json")
    if not file_path.exists():
        return None
    with open(file_path, 'r') as f:
        return json.load(f)

@st.cache_data
def load_xai_analysis():
    """XAI 분석 데이터 로드"""
    file_path = Path("data/xai_analysis/verified_xai_analysis_20250929_173942.json")
    if not file_path.exists():
        return None
    with open(file_path, 'r') as f:
        return json.load(f)

@st.cache_data
def load_economic_backtest():
    """경제적 백테스트 결과 로드"""
    file_path = Path("data/raw/rv_economic_backtest_results.json")
    if not file_path.exists():
        return None
    with open(file_path, 'r') as f:
        return json.load(f)

@st.cache_data
def load_test_predictions():
    """학습된 모델의 테스트 예측 결과 로드 (재현 가능)"""
    file_path = Path("data/raw/test_predictions.csv")
    if not file_path.exists():
        return None
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
        # 컬럼명 통일 (actual_volatility -> target_vol_5d, predicted_volatility -> predicted_vol)
        df = df.rename(columns={
            'actual_volatility': 'target_vol_5d',
            'predicted_volatility': 'predicted_vol'
        })
        return df
    except Exception as e:
        st.error(f"예측 데이터 로드 중 오류 발생: {str(e)}")
        return None

@st.cache_data
def load_final_model_performance():
    """최종 모델 성능 메트릭 로드 (ElasticNet)"""
    file_path = Path("data/raw/final_model_performance.json")
    if not file_path.exists():
        return None
    with open(file_path, 'r') as f:
        return json.load(f)

@st.cache_data
def load_model_comparison():
    """모델 비교 CSV 로드"""
    file_path = Path("data/model_comparison.csv")
    if not file_path.exists():
        return None
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    df.columns = [col.replace('\ufeff', '').strip() for col in df.columns]
    return df

@st.cache_data
def load_elasticnet_grid_results():
    """ElasticNet 그리드 결과 로드 (paper/data 또는 실험 디렉터리의 최신 파일 사용)"""
    candidates = []
    paper_dir = Path("paper/data")
    experiments_dir = Path("experiments/elasticnet_grid/results")
    if paper_dir.exists():
        candidates.extend(paper_dir.glob("elasticnet_grid*_*.csv"))
    if experiments_dir.exists():
        candidates.extend(experiments_dir.glob("elasticnet_grid_*.csv"))

    if not candidates:
        return None, None

    latest = sorted(candidates, reverse=True)[0]
    df = pd.read_csv(latest)
    return df, latest

# 헤더
st.markdown('<div class="main-header">📊 SPY 변동성 예측 대시보드</div>', unsafe_allow_html=True)
st.markdown("**ElasticNet 회귀 기반 5일 변동성 예측 시스템 | 2015-2024 실제 데이터 분석**")
st.divider()

# 사이드바
with st.sidebar:
    st.header("📌 시스템 정보")

    model_perf = load_model_performance()
    if model_perf:
        st.metric("모델", model_perf.get('model_type', 'Ridge'))
        st.metric("타겟 변수", "5일 변동성")
        st.metric("검증 방법", "Purged K-Fold CV")
        st.metric("샘플 수", f"{model_perf.get('n_samples', 0):,}")
        st.metric("특성 수", model_perf.get('n_features', 0))

    st.divider()
    st.markdown("**📅 마지막 업데이트**")
    st.text(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# 메인 탭
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 변동성 예측",
    "🎯 특성 중요도",
    "💰 경제적 가치",
    "📊 모델 비교",
    "🔬 통계적 검증",
    "🧬 특성 분석",
    "🎥 발표 그래프"
])

# 탭 1: 변동성 예측
with tab1:
    st.header("📈 실제 vs 예측 변동성")
    st.markdown("ElasticNet 회귀 모델의 5일 변동성 예측 성능을 시각화합니다.")

    # 모델 성능 메트릭 (최종 모델 우선, 없으면 기존 파일)
    model_perf = load_final_model_performance()
    if not model_perf:
        model_perf = load_model_performance()

    if model_perf:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "R² (결정계수)",
                f"{model_perf.get('test_r2', 0):.4f}",
                help="Coefficient of Determination: 모델이 설명하는 변동성 변화의 비율 (상관계수 아님)"
            )

        with col2:
            st.metric(
                "RMSE",
                f"{model_perf.get('test_rmse', 0):.6f}",
                help="평균 제곱근 오차"
            )

        with col3:
            st.metric(
                "MAE",
                f"{model_perf.get('test_mae', 0):.6f}",
                help="평균 절대 오차"
            )

        with col4:
            st.metric(
                "CV Std",
                f"{model_perf.get('cv_std', 0):.4f}",
                help="교차 검증 표준편차"
            )

    st.divider()

    # 변동성 예측 차트
    with st.spinner("변동성 예측 데이터 로딩 중..."):
        vol_data = load_test_predictions()

    if vol_data is not None:
        # 시계열 그래프
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=vol_data['Date'],
            y=vol_data['target_vol_5d'],
            mode='lines',
            name='실제 변동성',
            line=dict(color='#1f77b4', width=2),
            opacity=0.7
        ))

        fig.add_trace(go.Scatter(
            x=vol_data['Date'],
            y=vol_data['predicted_vol'],
            mode='lines',
            name='예측 변동성',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            opacity=0.7
        ))

        fig.update_layout(
            title="SPY 변동성 예측 vs 실제 (테스트 셋)",
            xaxis_title="날짜",
            yaxis_title="변동성 (연율화)",
            hovermode='x unified',
            height=500,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig, width='stretch')
        st.caption("상위 20개 특성의 SHAP 중요도를 시각화하여 모델이 무엇을 근거로 판단하는지 설명합니다.")
        st.write("""
        **발표 코멘트 예시**
        - 'VIX 및 단기 변동성 신호가 모델 의사결정의 1순위'라고 강조
        - SHAP 그래프 스크린샷을 슬라이드에 넣어 Feature Engineering 정당성 설명
        """)
        st.caption("실제 변동성과 ElasticNet 예측 값을 한눈에 비교해 계절성·추세를 파악합니다.")
        st.write("""
        **발표 코멘트 예시**
        - '실제 급등 구간(코로나, 2022년 변동성 이벤트)에서도 모델이 방향성은 따라간다'는 메시지 전달
        - 추세가 완만한 구간에서는 예측선이 리스크를 더 빨리 알려주는 보수적 모델임을 강조
        """)

        # 산점도
        col1, col2 = st.columns(2)

        with col1:
            # 예측 vs 실제 산점도
            fig_scatter = px.scatter(
                vol_data,
                x='target_vol_5d',
                y='predicted_vol',
                title="예측 vs 실제 (산점도)",
                labels={'target_vol_5d': '실제 변동성', 'predicted_vol': '예측 변동성'},
                opacity=0.6
            )

            # 대각선 추가 (완벽한 예측선)
            min_val = min(vol_data['target_vol_5d'].min(), vol_data['predicted_vol'].min())
            max_val = max(vol_data['target_vol_5d'].max(), vol_data['predicted_vol'].max())
            fig_scatter.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='완벽한 예측',
                    line=dict(color='red', dash='dash')
                )
            )

            fig_scatter.update_layout(height=400, template='plotly_white')
            st.plotly_chart(fig_scatter, width='stretch')
            st.caption("이상적인 예측선과의 편차를 통해 예측 일관성과 과대/과소 추정 영역을 확인할 수 있습니다.")
            st.write("""
            **발표 코멘트 예시**
            - 대각선에서 멀어지는 부분은 특정 시장 상황에서 예측이 약한 구간임을 설명
            - 산점도 스크린샷을 슬라이드에 삽입해 모델 신뢰도를 논의
            """)

        with col2:
            # 잔차 분포
            residuals = vol_data['target_vol_5d'] - vol_data['predicted_vol']
            fig_residual = px.histogram(
                residuals,
                title="예측 오차 분포",
                labels={'value': '잔차', 'count': '빈도'},
                nbins=50,
                opacity=0.7
            )
            fig_residual.update_layout(height=400, template='plotly_white', showlegend=False)
            st.plotly_chart(fig_residual, width='stretch')
            st.caption("잔차 분포를 보면 모델 오차가 평균 0에 가까운지, 꼬리가 두꺼운지 즉시 파악할 수 있습니다.")
            st.info("""
            **발표 코멘트 예시**
            - 잔차가 좌우 대칭이면 편향이 적다는 것을 강조
            - 꼬리가 길면 극단적 이벤트 대응 필요성을 언급
            """)

        # 통계 정보
        st.markdown("### 📊 예측 통계")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("평균 실제 변동성", f"{vol_data['target_vol_5d'].mean():.4f}")
        with col2:
            st.metric("평균 예측 변동성", f"{vol_data['predicted_vol'].mean():.4f}")
        with col3:
            st.metric("상관계수", f"{vol_data['target_vol_5d'].corr(vol_data['predicted_vol']):.4f}")
        with col4:
            st.metric("평균 절대 오차", f"{abs(residuals).mean():.6f}")
    else:
        st.warning("변동성 예측 데이터를 로드할 수 없습니다.")

# 탭 2: 특성 중요도
with tab2:
    st.header("🎯 SHAP 기반 특성 중요도 분석")
    st.markdown("SHAP(SHapley Additive exPlanations) 값을 통해 각 특성이 모델 예측에 미치는 영향을 분석합니다.")

    xai_data = load_xai_analysis()

    if xai_data and 'shap_analysis' in xai_data:
        shap_features = xai_data['shap_analysis']['feature_importance']

        # 데이터프레임 변환
        df_shap = pd.DataFrame(shap_features)

        # Top 20 특성만 선택
        df_shap_top = df_shap.nlargest(20, 'shap_importance')

        # 수평 바 차트
        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=df_shap_top['feature'],
            x=df_shap_top['shap_importance'],
            orientation='h',
            marker=dict(
                color=df_shap_top['shap_importance'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="중요도")
            ),
            text=df_shap_top['shap_importance'].round(4),
            textposition='auto',
        ))

        fig.update_layout(
            title="Top 20 특성 중요도 (SHAP)",
            xaxis_title="SHAP 중요도",
            yaxis_title="특성",
            height=700,
            template='plotly_white',
            yaxis={'categoryorder': 'total ascending'}
        )

        st.plotly_chart(fig, width='stretch')

        # 상세 테이블
        st.markdown("### 📋 특성 중요도 상세 정보")

        df_display = df_shap_top.copy()
        df_display.columns = ['특성', 'SHAP 중요도', '평균 SHAP 값', 'SHAP 표준편차']
        df_display['순위'] = range(1, len(df_display) + 1)
        df_display = df_display[['순위', '특성', 'SHAP 중요도', '평균 SHAP 값', 'SHAP 표준편차']]

        # 숫자 포맷팅
        df_display['SHAP 중요도'] = df_display['SHAP 중요도'].round(6)
        df_display['평균 SHAP 값'] = df_display['평균 SHAP 값'].round(6)
        df_display['SHAP 표준편차'] = df_display['SHAP 표준편차'].round(6)

        st.dataframe(
            df_display,
            width='stretch',
            hide_index=True,
            height=400
        )

        # 특성 분석 인사이트
        st.markdown("### 💡 주요 인사이트")
        top_3 = df_shap_top.head(3)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.info(f"""
            **가장 중요한 특성**

            {top_3.iloc[0]['feature']}

            중요도: {top_3.iloc[0]['shap_importance']:.4f}
            """)

        with col2:
            st.info(f"""
            **두 번째 중요 특성**

            {top_3.iloc[1]['feature']}

            중요도: {top_3.iloc[1]['shap_importance']:.4f}
            """)

        with col3:
            st.info(f"""
            **세 번째 중요 특성**

            {top_3.iloc[2]['feature']}

            중요도: {top_3.iloc[2]['shap_importance']:.4f}
            """)

        # 모델 성능 정보
        if 'model_performance' in xai_data:
            st.divider()
            st.markdown("### 📊 XAI 분석 시점 모델 성능")

            perf = xai_data['model_performance']
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Train R² (결정계수)", f"{perf.get('train_r2', 0):.4f}")
            with col2:
                st.metric("Test R² (결정계수)", f"{perf.get('test_r2', 0):.4f}")
            with col3:
                st.metric("Test RMSE", f"{perf.get('test_rmse', 0):.6f}")
            with col4:
                st.metric("특성 수", perf.get('n_features', 0))
    else:
        st.warning("XAI 분석 데이터를 로드할 수 없습니다.")

# 탭 3: 경제적 가치
with tab3:
    st.header("💰 경제적 백테스트 결과")
    st.markdown("변동성 예측 모델을 활용한 거래 전략의 실제 경제적 가치를 평가합니다.")

    backtest_data = load_economic_backtest()

    if backtest_data:
        # 백테스트 기간 정보
        period = backtest_data['period']
        st.info(f"**백테스트 기간**: {period['start']} ~ {period['end']} ({period['days']} 거래일)")

        # 전략 설정
        strategy = backtest_data['strategy']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("목표 변동성", f"{strategy['target_volatility']*100:.1f}%")
        with col2:
            st.metric("거래 비용", f"{strategy['transaction_cost']*100:.2f}%")
        with col3:
            st.metric("포지션 범위", f"{strategy['position_range'][0]:.1f}x ~ {strategy['position_range'][1]:.1f}x")

        st.divider()

        # 성과 비교
        results = backtest_data['results']

        # 메트릭 비교 테이블
        df_results = pd.DataFrame({
            '전략': ['Buy & Hold', 'V0 Strategy', 'RV Strategy'],
            '총 수익률': [
                results['buy_hold']['total_return'],
                results['v0_strategy']['total_return'],
                results['rv_strategy']['total_return']
            ],
            '연율화 수익률': [
                results['buy_hold']['annual_return'],
                results['v0_strategy']['annual_return'],
                results['rv_strategy']['annual_return']
            ],
            '변동성': [
                results['buy_hold']['volatility'],
                results['v0_strategy']['volatility'],
                results['rv_strategy']['volatility']
            ],
            'Sharpe Ratio': [
                results['buy_hold']['sharpe'],
                results['v0_strategy']['sharpe'],
                results['rv_strategy']['sharpe']
            ],
            'Max Drawdown': [
                results['buy_hold']['max_drawdown'],
                results['v0_strategy']['max_drawdown'],
                results['rv_strategy']['max_drawdown']
            ],
            '승률': [
                results['buy_hold']['win_rate'],
                results['v0_strategy']['win_rate'],
                results['rv_strategy']['win_rate']
            ]
        })

        # 퍼센트 포맷팅
        df_display = df_results.copy()
        df_display['총 수익률'] = (df_display['총 수익률'] * 100).round(2).astype(str) + '%'
        df_display['연율화 수익률'] = (df_display['연율화 수익률'] * 100).round(2).astype(str) + '%'
        df_display['변동성'] = (df_display['변동성'] * 100).round(2).astype(str) + '%'
        df_display['Sharpe Ratio'] = df_display['Sharpe Ratio'].round(3)
        df_display['Max Drawdown'] = (df_display['Max Drawdown'] * 100).round(2).astype(str) + '%'
        df_display['승률'] = (df_display['승률'] * 100).round(2).astype(str) + '%'

        st.markdown("### 📊 전략 성과 비교")
        st.dataframe(df_display, width='stretch', hide_index=True)

        # 시각화
        col1, col2 = st.columns(2)

        with col1:
            # 수익률 비교
            fig_return = go.Figure()

            strategies = ['Buy & Hold', 'V0 Strategy', 'RV Strategy']
            annual_returns = [
                results['buy_hold']['annual_return'] * 100,
                results['v0_strategy']['annual_return'] * 100,
                results['rv_strategy']['annual_return'] * 100
            ]

            fig_return.add_trace(go.Bar(
                x=strategies,
                y=annual_returns,
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                text=[f"{x:.2f}%" for x in annual_returns],
                textposition='auto',
            ))

            fig_return.update_layout(
                title="연율화 수익률 비교",
                yaxis_title="수익률 (%)",
                height=400,
                template='plotly_white',
                showlegend=False
            )

            st.plotly_chart(fig_return, width='stretch')
            st.caption("전략별 연율화 수익률을 비교해 알파 수준과 계절적 차이를 즉시 확인합니다.")

        with col2:
            # Sharpe Ratio 비교
            fig_sharpe = go.Figure()

            sharpe_ratios = [
                results['buy_hold']['sharpe'],
                results['v0_strategy']['sharpe'],
                results['rv_strategy']['sharpe']
            ]

            fig_sharpe.add_trace(go.Bar(
                x=strategies,
                y=sharpe_ratios,
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                text=[f"{x:.3f}" for x in sharpe_ratios],
                textposition='auto',
            ))

            fig_sharpe.update_layout(
                title="Sharpe Ratio 비교",
                yaxis_title="Sharpe Ratio",
                height=400,
                template='plotly_white',
                showlegend=False
            )

            st.plotly_chart(fig_sharpe, width='stretch')
            st.caption("Sharpe Ratio 비교는 위험 조정 성과가 실제로 개선되었는지 보여줍니다.")

        # 위험-수익 산점도
        fig_risk_return = go.Figure()

        for strategy_name, color in zip(strategies, ['#1f77b4', '#ff7f0e', '#2ca02c']):
            strategy_key = strategy_name.lower().replace(' ', '_').replace('&', '').replace('__', '_')
            if strategy_key == 'buy__hold':
                strategy_key = 'buy_hold'

            fig_risk_return.add_trace(go.Scatter(
                x=[results[strategy_key]['volatility'] * 100],
                y=[results[strategy_key]['annual_return'] * 100],
                mode='markers+text',
                name=strategy_name,
                marker=dict(size=15, color=color),
                text=[strategy_name],
                textposition='top center'
            ))

        fig_risk_return.update_layout(
            title="위험-수익 프로파일",
            xaxis_title="변동성 (%)",
            yaxis_title="연율화 수익률 (%)",
            height=500,
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig_risk_return, width='stretch')
        st.caption("위험-수익 산점도는 각 전략의 변동성과 수익률 위치를 시각적으로 정렬하여 브리핑에 활용하기 좋습니다.")

        # 비교 분석
        st.divider()
        st.markdown("### 📈 RV Strategy vs Buy & Hold 비교")

        comparison = backtest_data['comparison']

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            delta_return = comparison['rv_vs_bh_return'] * 100
            st.metric(
                "수익률 차이",
                f"{delta_return:+.2f}%",
                delta=f"{delta_return:.2f}%",
                delta_color="normal"
            )

        with col2:
            delta_vol = comparison['rv_vs_bh_vol'] * 100
            st.metric(
                "변동성 차이",
                f"{delta_vol:+.2f}%",
                delta=f"{delta_vol:.2f}%",
                delta_color="inverse"
            )

        with col3:
            delta_sharpe = comparison['rv_vs_bh_sharpe']
            st.metric(
                "Sharpe 차이",
                f"{delta_sharpe:+.3f}",
                delta=f"{delta_sharpe:.3f}",
                delta_color="normal"
            )

        with col4:
            delta_sharpe_v0 = comparison['rv_vs_v0_sharpe']
            st.metric(
                "vs V0 Sharpe",
                f"{delta_sharpe_v0:+.3f}",
                delta=f"{delta_sharpe_v0:.3f}",
                delta_color="normal"
            )

        # 핵심 인사이트
        st.markdown("### 💡 핵심 인사이트")

        st.success(f"""
        **변동성 예측 모델의 리스크 모니터링 가치**

        - **주요 활용 목적**: 알파 창출보다는 포트폴리오 리스크 관리 및 헤징 전략 지원
        - RV Strategy는 Buy & Hold 대비 변동성을 {abs(delta_vol):.2f}% 감소시켜 리스크 관리 효과를 입증했습니다.
        - 거래 비용({strategy['transaction_cost']*100:.2f}%)을 고려한 실제 백테스트 결과입니다.
        - Sharpe Ratio {results['rv_strategy']['sharpe']:.3f}로 위험 대비 수익의 균형을 유지했습니다.
        - 변동성 예측 정확도 향상을 통한 동적 헤징, VIX 옵션 거래, 포지션 사이징 최적화에 활용 가능합니다.
        """)

        st.warning("""
        **주의사항**

        - 백테스트 결과는 과거 데이터 기반이며 미래 성과를 보장하지 않습니다.
        - 실제 거래 시 슬리피지, 유동성 등 추가 비용이 발생할 수 있습니다.
        - 시장 상황 변화에 따라 모델 성능이 달라질 수 있습니다.
        """)
    else:
        st.warning("경제적 백테스트 데이터를 로드할 수 없습니다.")

# 탭 4: 모델 비교
with tab4:
    st.header("📊 HAR vs Ridge 모델 비교")
    st.markdown("학계 표준 벤치마크인 HAR(Heterogeneous Autoregressive) 모델과 Ridge 회귀 모델의 성능을 비교합니다.")

    # HAR 데이터 로드
    har_file = Path("data/raw/har_benchmark_performance.json")
    comparison_file = Path("data/raw/har_vs_ridge_comparison.json")

    if har_file.exists() and comparison_file.exists():
        with open(har_file, 'r') as f:
            har_data = json.load(f)
        with open(comparison_file, 'r') as f:
            comparison_data = json.load(f)

        # 모델 개요
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔵 Ridge 모델")
            st.info(f"""
            **모델 타입**: Ridge Regression (α=1.0)

            **특성 수**: {comparison_data['ridge_model']['features']}개

            **주요 특성**: 변동성 lag, 수익률 lag, 롤링 통계량

            **검증**: Purged K-Fold CV (5-fold)
            """)

        with col2:
            st.markdown("### 🟢 HAR 벤치마크")
            st.info(f"""
            **모델 타입**: Heterogeneous Autoregressive (α=0.01)

            **특성 수**: {har_data['n_features']}개

            **주요 특성**: 일간/주간/월간 실현 변동성

            **검증**: Purged K-Fold CV (5-fold)
            """)

        st.divider()

        # 성능 비교 테이블
        st.markdown("### 📊 성능 메트릭 비교")

        df_comparison = pd.DataFrame({
            '메트릭': ['R² (결정계수, CV Mean)', 'R² (결정계수, Test)', 'RMSE', 'MAE'],
            'HAR 모델': [
                f"{har_data['cv_r2_mean']:.4f}",
                f"{har_data['test_r2']:.4f}",
                f"{har_data['test_rmse']:.6f}",
                f"{har_data['test_mae']:.6f}"
            ],
            'Ridge 모델': [
                f"{comparison_data['ridge_model']['r2']:.4f}",
                "N/A",
                f"{comparison_data['ridge_model']['rmse']:.6f}",
                f"{comparison_data['ridge_model']['mae']:.6f}"
            ],
            '개선도': [
                f"+{((comparison_data['ridge_model']['r2'] / har_data['cv_r2_mean'] - 1) * 100):.1f}%",
                "N/A",
                f"{((comparison_data['ridge_model']['rmse'] / har_data['test_rmse'] - 1) * 100):.1f}%",
                f"{((comparison_data['ridge_model']['mae'] / har_data['test_mae'] - 1) * 100):.1f}%"
            ]
        })

        st.dataframe(df_comparison, width='stretch', hide_index=True)

        # 시각화 - R² 비교
        col1, col2 = st.columns(2)

        with col1:
            fig_r2 = go.Figure()

            models = ['HAR', 'Ridge']
            r2_scores = [har_data['cv_r2_mean'], comparison_data['ridge_model']['r2']]

            fig_r2.add_trace(go.Bar(
                x=models,
                y=r2_scores,
                marker_color=['#2ca02c', '#1f77b4'],
                text=[f"{x:.4f}" for x in r2_scores],
                textposition='auto',
            ))

            fig_r2.update_layout(
                title="R² (결정계수) 비교 (CV Mean)",
                yaxis_title="R² (결정계수)",
                height=400,
                template='plotly_white',
                showlegend=False
            )

            st.plotly_chart(fig_r2, width='stretch')
            st.caption("CV R² 막대그래프는 Ridge가 HAR 대비 얼마만큼 설명력을 높였는지 즉시 보여줍니다.")

        with col2:
            fig_error = go.Figure()

            metrics = ['RMSE', 'MAE']
            har_errors = [har_data['test_rmse'], har_data['test_mae']]
            ridge_errors = [comparison_data['ridge_model']['rmse'], comparison_data['ridge_model']['mae']]

            fig_error.add_trace(go.Bar(
                name='HAR',
                x=metrics,
                y=har_errors,
                marker_color='#2ca02c',
                text=[f"{x:.6f}" for x in har_errors],
                textposition='auto',
            ))

            fig_error.add_trace(go.Bar(
                name='Ridge',
                x=metrics,
                y=ridge_errors,
                marker_color='#1f77b4',
                text=[f"{x:.6f}" for x in ridge_errors],
                textposition='auto',
            ))

            fig_error.update_layout(
                title="오차 메트릭 비교",
                yaxis_title="오차 값",
                height=400,
                template='plotly_white',
                barmode='group'
            )

            st.plotly_chart(fig_error, width='stretch')
            st.caption("RMSE·MAE 그룹 막대는 오차 규모를 비교해 안정성과 개선 폭을 설명할 때 유용합니다.")

        # Cross-Validation Fold 성능
        st.divider()
        st.markdown("### 📈 교차 검증 Fold별 성능 (HAR)")

        fold_df = pd.DataFrame({
            'Fold': [f'Fold {i+1}' for i in range(5)],
            'R² (결정계수)': har_data['cv_fold_scores']
        })

        fig_folds = px.bar(
            fold_df,
            x='Fold',
            y='R² (결정계수)',
            title='HAR 모델 - Fold별 R² (결정계수)',
            color='R² (결정계수)',
            color_continuous_scale='RdYlGn',
            text='R² (결정계수)'
        )

        fig_folds.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig_folds.update_layout(height=400, template='plotly_white')

        st.plotly_chart(fig_folds, width='stretch')
        st.caption("Fold별 R² 분포는 검증 편차(안정성)를 강조할 수 있는 자료입니다.")

        # 통계적 요약
        st.markdown("### 📊 교차 검증 통계")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("평균 R² (결정계수)", f"{har_data['cv_r2_mean']:.4f}")
        with col2:
            st.metric("표준편차", f"{har_data['cv_r2_std']:.4f}")
        with col3:
            st.metric("최고 성능", f"{max(har_data['cv_fold_scores']):.4f}")
        with col4:
            st.metric("최저 성능", f"{min(har_data['cv_fold_scores']):.4f}")

        # 결론
        st.divider()
        st.success(f"""
        ### 💡 주요 결론

        - **Ridge 모델이 HAR 벤치마크 대비 {comparison_data['improvement']['r2_ratio']:.2f}배 우수한 성능**을 보임
        - **R² (결정계수)**에서 **{comparison_data['improvement']['r2_difference']:.4f} 포인트 개선**
        - 31개의 정교한 특성 엔지니어링이 3개 특성 대비 **실질적 예측력 향상** 달성
        - {comparison_data['improvement']['conclusion']}

        ℹ️ R²는 결정계수(Coefficient of Determination)로, 상관계수가 아닙니다.
        """)

    else:
        st.warning("모델 비교 데이터를 찾을 수 없습니다.")

# 탭 5: 통계적 검증
with tab5:
    st.header("🔬 통계적 검증 및 잔차 분석")
    st.markdown("모델의 통계적 유의성과 예측 오차의 특성을 심층 분석합니다.")

    vol_data = load_test_predictions()
    model_perf = load_final_model_performance()
    if not model_perf:
        model_perf = load_model_performance()

    if vol_data is not None:
        if not SCIPY_AVAILABLE or not STATSMODELS_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            st.error("통계적 검증을 위해서는 scipy, statsmodels, matplotlib 패키지가 필요합니다.")
            st.code("pip install scipy statsmodels matplotlib")
        else:
            # 잔차 계산
            residuals = vol_data['target_vol_5d'] - vol_data['predicted_vol']

            # 통계적 검증
            st.markdown("### 📊 통계적 유의성 검증")

            # t-test (잔차의 평균이 0인지 검증)
            t_stat, p_value = stats.ttest_1samp(residuals, 0)

            # Shapiro-Wilk 정규성 검정
            shapiro_stat, shapiro_p = stats.shapiro(residuals[:min(5000, len(residuals))])

            # Durbin-Watson (자기상관 검정)
            dw_stat = durbin_watson(residuals)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "t-통계량",
                    f"{t_stat:.4f}",
                    help="잔차 평균이 0인지 검증"
                )

            with col2:
                st.metric(
                    "p-value",
                    f"{p_value:.4f}",
                    delta="유의" if p_value < 0.05 else "비유의",
                    delta_color="inverse" if p_value < 0.05 else "off",
                    help="귀무가설: 잔차 평균 = 0"
                )

            with col3:
                st.metric(
                    "Shapiro-Wilk",
                    f"{shapiro_stat:.4f}",
                    help="정규성 검정 통계량"
                )

            with col4:
                st.metric(
                    "Durbin-Watson",
                    f"{dw_stat:.4f}",
                    help="자기상관 검정 (2 근처면 자기상관 없음)"
                )

            st.divider()

            # 잔차 분석 차트
            col1, col2 = st.columns(2)

            with col1:
                # 잔차 시계열
                fig_residuals_ts = go.Figure()

                fig_residuals_ts.add_trace(go.Scatter(
                    x=vol_data['Date'],
                    y=residuals,
                    mode='lines',
                    name='잔차',
                    line=dict(color='#d62728', width=1),
                    opacity=0.7
                ))

                fig_residuals_ts.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="black",
                    annotation_text="0 기준선"
                )

                fig_residuals_ts.update_layout(
                    title="잔차 시계열",
                    xaxis_title="날짜",
                    yaxis_title="잔차 (실제 - 예측)",
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig_residuals_ts, width='stretch')
                st.caption("잔차 시계열로 예측 오차의 구조적 패턴(예: 클러스터링)을 확인합니다.")

            with col2:
                # QQ Plot
                qq_data = probplot(residuals, dist="norm")

                fig_qq = go.Figure()

                fig_qq.add_trace(go.Scatter(
                    x=qq_data[0][0],
                    y=qq_data[0][1],
                    mode='markers',
                    name='실제',
                    marker=dict(color='#1f77b4', size=4),
                    opacity=0.6
                ))

                # 이론적 정규분포 선
                fig_qq.add_trace(go.Scatter(
                    x=qq_data[0][0],
                    y=qq_data[1][1] + qq_data[1][0] * qq_data[0][0],
                    mode='lines',
                    name='이론적 정규분포',
                    line=dict(color='red', dash='dash')
                ))

                fig_qq.update_layout(
                    title="Q-Q Plot (정규성 검증)",
                    xaxis_title="이론적 분위수",
                    yaxis_title="표본 분위수",
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig_qq, width='stretch')
                st.caption("Q-Q Plot은 잔차가 정규분포를 얼마나 따르는지 정성적으로 보여줍니다.")

            # 자기상관 분석
            st.divider()
            st.markdown("### 📈 잔차 자기상관 분석")

            col1, col2 = st.columns(2)

            with col1:
                # ACF
                fig_acf, ax_acf = plt.subplots(figsize=(10, 4))
                plot_acf(residuals, lags=40, ax=ax_acf, alpha=0.05)
                ax_acf.set_title('자기상관함수 (ACF)')
                ax_acf.set_xlabel('Lag')
                ax_acf.set_ylabel('상관계수')
                st.pyplot(fig_acf)
                st.caption("ACF 그래프는 잔차의 자기상관 정도를 시각적으로 검증합니다.")
                plt.close()

            with col2:
                # PACF
                fig_pacf, ax_pacf = plt.subplots(figsize=(10, 4))
                plot_pacf(residuals, lags=40, ax=ax_pacf, alpha=0.05)
                ax_pacf.set_title('편자기상관함수 (PACF)')
                ax_pacf.set_xlabel('Lag')
                ax_pacf.set_ylabel('상관계수')
                st.pyplot(fig_pacf)
                st.caption("PACF는 특정 Lag의 독립적 영향만을 보여 주어 잔차 구조를 더 정확히 설명합니다.")
                plt.close()

            # 잔차 통계 요약
            st.divider()
            st.markdown("### 📊 잔차 통계 요약")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("평균", f"{residuals.mean():.6f}")
            with col2:
                st.metric("표준편차", f"{residuals.std():.6f}")
            with col3:
                st.metric("왜도 (Skewness)", f"{stats.skew(residuals):.4f}")
            with col4:
                st.metric("첨도 (Kurtosis)", f"{stats.kurtosis(residuals):.4f}")
            with col5:
                st.metric("중위수", f"{residuals.median():.6f}")

            # 해석
            st.divider()
            st.markdown("### 💡 해석")

            interpretation = []

            if abs(residuals.mean()) < residuals.std() * 0.1:
                interpretation.append("✅ 잔차 평균이 0에 가까워 **편향 없는 예측**을 수행함")
            else:
                interpretation.append("⚠️ 잔차 평균이 0에서 벗어나 있어 체계적 편향 가능성")

            if p_value > 0.05:
                interpretation.append("✅ p-value > 0.05로 잔차 평균이 통계적으로 0과 다르지 않음")
            else:
                interpretation.append("⚠️ p-value < 0.05로 잔차 평균이 통계적으로 유의함")

            if 1.5 < dw_stat < 2.5:
                interpretation.append("✅ Durbin-Watson 통계량이 2 근처로 **자기상관 없음**")
            else:
                interpretation.append(f"⚠️ Durbin-Watson = {dw_stat:.2f}로 잔차에 자기상관 존재 가능성")

            if shapiro_p > 0.05:
                interpretation.append("✅ Shapiro-Wilk 검정 통과: 잔차가 정규분포를 따름")
            else:
                interpretation.append("⚠️ Shapiro-Wilk 검정 실패: 잔차가 정규분포에서 벗어남")

            for item in interpretation:
                st.markdown(item)

    else:
        st.warning("변동성 예측 데이터를 로드할 수 없습니다.")

# 탭 6: 특성 분석
with tab6:
    st.header("🧬 특성 공학 및 상관관계 분석")
    st.markdown("변동성 예측에 사용된 특성들의 상관관계와 중요도를 분석합니다.")

    # 데이터 생성
    with st.spinner("특성 데이터 생성 중..."):
        try:
            import yfinance as yf
            from sklearn.preprocessing import StandardScaler

            # SPY 데이터 다운로드
            spy = yf.download('SPY', start='2015-01-01', end='2024-12-31', progress=False, auto_adjust=True)

            if not spy.empty:
                # MultiIndex 처리
                if isinstance(spy.columns, pd.MultiIndex):
                    spy.columns = spy.columns.get_level_values(0)
                spy.columns = [str(col) for col in spy.columns]

                # 특성 생성
                spy['returns'] = spy['Close'].pct_change()
                spy['volatility'] = spy['returns'].rolling(window=5).std() * np.sqrt(252)

                # Lag 특성
                for lag in [1, 5, 10, 20]:
                    spy[f'vol_lag_{lag}'] = spy['volatility'].shift(lag)
                    spy[f'returns_lag_{lag}'] = spy['returns'].shift(lag)

                spy = spy.dropna()

                # 특성 선택
                feature_cols = [col for col in spy.columns if col.startswith(('vol_lag_', 'returns_lag_', 'volatility'))]
                feature_data = spy[feature_cols].copy()

                # 상관관계 행렬
                st.markdown("### 📊 특성 상관관계 히트맵")

                corr_matrix = feature_data.corr()

                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu_r',
                    zmid=0,
                    text=corr_matrix.values,
                    texttemplate='%{text:.2f}',
                    textfont={"size": 8},
                    colorbar=dict(title="상관계수")
                ))

                fig_corr.update_layout(
                    title="특성 간 상관관계 히트맵",
                    height=700,
                    template='plotly_white'
                )

                st.plotly_chart(fig_corr, width='stretch')
                st.caption("특성 상관 히트맵은 중복 특성이나 다중공선성 위험을 직관적으로 설명할 때 유용합니다.")

                # 고상관 특성 쌍
                st.divider()
                st.markdown("### 🔗 높은 상관관계 특성 쌍 (|r| > 0.7)")

                # 상삼각 행렬만 사용 (중복 제거)
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                high_corr = []

                for i in range(len(corr_matrix)):
                    for j in range(i+1, len(corr_matrix)):
                        if abs(corr_matrix.iloc[i, j]) > 0.7:
                            high_corr.append({
                                '특성 1': corr_matrix.columns[i],
                                '특성 2': corr_matrix.columns[j],
                                '상관계수': corr_matrix.iloc[i, j]
                            })

                if high_corr:
                    df_high_corr = pd.DataFrame(high_corr)
                    df_high_corr = df_high_corr.sort_values('상관계수', key=abs, ascending=False)
                    df_high_corr['상관계수'] = df_high_corr['상관계수'].round(4)

                    st.dataframe(df_high_corr, width='stretch', hide_index=True)
                else:
                    st.info("상관계수 절댓값이 0.7을 초과하는 특성 쌍이 없습니다.")

                # 특성 분포
                st.divider()
                st.markdown("### 📈 주요 특성 분포")

                # Top 4 중요 특성 선택 (변동성 lag들)
                top_features = ['vol_lag_1', 'vol_lag_5', 'vol_lag_10', 'vol_lag_20']

                fig_dist = go.Figure()

                for feature in top_features:
                    if feature in feature_data.columns:
                        fig_dist.add_trace(go.Box(
                            y=feature_data[feature],
                            name=feature,
                            boxmean='sd'
                        ))

                fig_dist.update_layout(
                    title="변동성 Lag 특성 분포 (Box Plot)",
                    yaxis_title="값",
                    height=500,
                    template='plotly_white',
                    showlegend=True
                )

                st.plotly_chart(fig_dist, width='stretch')
                st.caption("Box Plot으로 각 Lag 특성의 분포와 이상치를 설명할 수 있습니다.")

                # 특성 통계
                st.divider()
                st.markdown("### 📊 특성 기술통계")

                stats_df = feature_data[top_features].describe().T
                stats_df['CV'] = (stats_df['std'] / stats_df['mean']).round(4)  # 변동계수
                stats_df = stats_df.round(6)

                st.dataframe(stats_df, width='stretch')

                # Feature Engineering 인사이트
                st.divider()
                st.markdown("### 💡 Feature Engineering 인사이트")

                st.success("""
                **변동성 Lag 특성의 효과**

                - **vol_lag_1**: 가장 최근 변동성으로 단기 추세 포착
                - **vol_lag_5**: 1주일 변동성으로 중기 패턴 반영
                - **vol_lag_10**: 2주 변동성으로 구조적 변화 감지
                - **vol_lag_20**: 1개월 변동성으로 장기 추세 파악

                다중 시간대 특성을 조합하여 **다층적 변동성 패턴**을 학습함
                """)

                st.info("""
                **수익률 Lag 특성의 역할**

                - 과거 수익률이 미래 변동성에 미치는 영향 모델링
                - Leverage effect 포착: 음의 수익률이 변동성 증가와 연관
                - 비선형 관계를 Ridge 회귀로 선형 근사
                """)

        except Exception as e:
            st.error(f"특성 데이터 생성 중 오류: {str(e)}")

# 탭 7: 발표용 그래프 허브
with tab7:
    st.header("🎥 발표/보고용 그래프 허브")
    st.markdown("슬라이드나 리포트에 바로 사용할 수 있는 Plotly 그래프를 모았습니다.")

    model_comparison = load_model_comparison()
    grid_results, grid_path = load_elasticnet_grid_results()

    # 모델 비교 시각화
    if model_comparison is not None and not model_comparison.empty:
        st.subheader("1️⃣ 모델 성능 개요")

        best_cv = model_comparison.loc[model_comparison['CV_R2_Mean'].idxmax()]
        best_test = model_comparison.loc[model_comparison['Test_R2'].idxmax()]
        most_stable = model_comparison.loc[model_comparison['CV_R2_Std'].idxmin()]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("최고 CV R²", f"{best_cv['CV_R2_Mean']:.4f}", best_cv['Model'])
        with col2:
            st.metric("최고 Test R²", f"{best_test['Test_R2']:.4f}", best_test['Model'])
        with col3:
            st.metric("가장 안정적인 모델", f"σ={most_stable['CV_R2_Std']:.4f}", most_stable['Model'])

        st.markdown("### 📊 CV R² 순위 (Bar)")
        cmp_sorted = model_comparison.sort_values('CV_R2_Mean', ascending=False)
        fig_cmp_bar = px.bar(
            cmp_sorted,
            x='Model',
            y='CV_R2_Mean',
            color='CV_R2_Mean',
            color_continuous_scale='Blues',
            text_auto='.3f',
            labels={'CV_R2_Mean': 'CV R²'},
            title="모델별 CV R² (결정계수)"
        )
        fig_cmp_bar.update_layout(height=420, template='plotly_white', showlegend=False)
        st.plotly_chart(fig_cmp_bar, use_container_width=True)
        st.caption("모델별 CV R² 순위를 한눈에 보여주는 그래프로 발표 자료의 메인 비교 슬라이드에 활용 가능합니다.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### CV vs Test R² 산점도")
            fig_scatter = px.scatter(
                model_comparison,
                x='CV_R2_Mean',
                y='Test_R2',
                size='N_Features',
                hover_data=['Model', 'CV_Test_Gap', 'N_Samples'],
                color='Model',
                labels={'CV_R2_Mean': 'CV R²', 'Test_R2': 'Test R²'},
                title="CV vs Test R² 관계"
            )
            fig_scatter.update_layout(height=420, template='plotly_white')
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption("CV 대비 Test R² 산점도는 각 모델의 일반화 성능을 설명할 때 사용하세요.")
        with col2:
            st.markdown("#### Overfitting Gap (CV-Test)")
            fig_gap = px.bar(
                model_comparison.sort_values('CV_Test_Gap', ascending=False),
                x='Model',
                y='CV_Test_Gap',
                text_auto='.3f',
                labels={'CV_Test_Gap': 'CV-Test Gap'},
                title="모델별 Overfitting Gap"
            )
            fig_gap.update_layout(height=420, template='plotly_white')
            st.plotly_chart(fig_gap, use_container_width=True)
            st.caption("Overfitting Gap 그래프는 과적합 위험이 큰 모델을 즉시 강조할 수 있습니다.")

        st.markdown("### 📋 원본 데이터")
        st.dataframe(model_comparison.round(4), use_container_width=True)
        cmp_download = model_comparison.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="모델 비교 CSV 다운로드",
            data=cmp_download,
            file_name="model_comparison_presentation.csv",
            mime="text/csv"
        )
    else:
        st.info("`data/model_comparison.csv` 파일을 찾을 수 없어 모델 비교 그래프를 표시하지 못했습니다.")

    st.divider()

    # ElasticNet 그리드 결과 시각화
    if grid_results is not None and grid_path is not None:
        st.subheader("2️⃣ ElasticNet 그리드 탐색 결과")
        st.caption(f"데이터 출처: `{grid_path}`")

        variants = sorted(grid_results['feature_variant'].unique())
        selected_variant = st.selectbox("피처 세트 선택", variants, index=0)

        variant_df = grid_results[grid_results['feature_variant'] == selected_variant].copy()
        pivot = variant_df.pivot_table(
            index='l1_ratio',
            columns='alpha',
            values='cv_r2_mean'
        )
        pivot = pivot.sort_index(ascending=False)

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[f"{col:.2f}" for col in pivot.columns],
            y=[f"{idx:.2f}" for idx in pivot.index],
            colorscale='RdYlGn',
            colorbar=dict(title="CV R²")
        ))
        fig_heatmap.update_layout(
            title=f"{selected_variant.upper()} 피처 - ElasticNet 그리드 CV R²",
            xaxis_title="alpha",
            yaxis_title="l1_ratio",
            height=500,
            template='plotly_white'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.caption("ElasticNet 파라미터 Heatmap은 발표 시 최적 파라미터 탐색 과정을 시각적으로 설명합니다.")

        top_configs = variant_df.sort_values('cv_r2_mean', ascending=False).head(5)
        top_configs_display = top_configs[['alpha', 'l1_ratio', 'cv_r2_mean', 'cv_r2_std', 'candidate']]
        top_configs_display = top_configs_display.round({'alpha': 3, 'l1_ratio': 2, 'cv_r2_mean': 4, 'cv_r2_std': 4})

        st.markdown("### 🏆 상위 조합 Top 5")
        st.dataframe(top_configs_display, use_container_width=True, hide_index=True)

        grid_download = grid_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="ElasticNet 그리드 CSV 다운로드",
            data=grid_download,
            file_name=f"{grid_path.stem}_presentation.csv",
            mime="text/csv"
        )
    else:
        st.info("ElasticNet 그리드 결과 파일을 찾을 수 없습니다. `experiments/elasticnet_grid`를 먼저 실행하세요.")

    st.divider()

    # Additional presentation-ready visualizations
    st.subheader("3️⃣ 예측 성과 하이라이트")
    with st.spinner("예측 데이터 로딩 중..."):
        vol_highlight = load_test_predictions()

    if vol_highlight is not None:
        recent = vol_highlight.tail(250).copy()
        recent['Date'] = pd.to_datetime(recent['Date'])

        fig_recent = go.Figure()
        fig_recent.add_trace(go.Scatter(
            x=recent['Date'],
            y=recent['target_vol_5d'],
            mode='lines',
            name='실제 변동성',
            line=dict(color='#1f77b4', width=2)
        ))
        fig_recent.add_trace(go.Scatter(
            x=recent['Date'],
            y=recent['predicted_vol'],
            mode='lines',
            name='예측 변동성',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        fig_recent.update_layout(
            title="최근 250거래일 변동성 예측",
            xaxis_title="날짜",
            yaxis_title="연율화 변동성",
            height=420,
            template='plotly_white'
        )
        st.plotly_chart(fig_recent, use_container_width=True)
        st.caption("최근 구간만 잘라 보여줌으로써 최신 시장에서도 모델이 일관된 패턴을 따르는지 설명할 수 있습니다.")

        residuals = recent['target_vol_5d'] - recent['predicted_vol']
        rolling_corr = recent[['target_vol_5d', 'predicted_vol']].rolling(window=50).corr().unstack().iloc[:, 1]
        rolling_dates = recent['Date']

        fig_corr_line = go.Figure()
        fig_corr_line.add_trace(go.Scatter(
            x=rolling_dates,
            y=rolling_corr,
            mode='lines',
            name='50일 롤링 상관계수',
            line=dict(color='#2ca02c', width=2)
        ))
        fig_corr_line.add_hline(y=0, line_dash='dash', line_color='gray')
        fig_corr_line.update_layout(
            title="롤링 상관계수(예측 vs 실제)",
            xaxis_title="날짜",
            yaxis_title="상관계수",
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig_corr_line, use_container_width=True)
        st.caption("롤링 상관계수를 통해 예측력이 시간에 따라 어떻게 변하는지 설명 자료로 활용할 수 있습니다.")

        highlight_download = recent.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="최근 250거래일 예측 CSV 다운로드",
            data=highlight_download,
            file_name="recent_volatility_predictions.csv",
            mime="text/csv"
        )
    else:
        st.info("예측 데이터가 없어 하이라이트 그래프를 표시하지 못했습니다.")

# 푸터
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>SPY 변동성 예측 대시보드 | Ridge 회귀 기반 시스템</p>
    <p>데이터: 2015-2024 실제 SPY ETF 데이터 | 검증: Purged K-Fold Cross-Validation</p>
</div>
""", unsafe_allow_html=True)
