# VRP 예측 연구

> 머신러닝을 활용한 변동성 위험 프리미엄(VRP) 예측 연구

---

## 프로젝트 개요

**VRP (Volatility Risk Premium)**은 옵션 시장의 내재 변동성(VIX)과 실현 변동성(RV)의 차이입니다. 본 연구는 머신러닝을 활용하여 VRP를 예측하고 투자 전략에 활용합니다.

### 핵심 성과
- **MLP 신경망**: R² = 0.44 (최고 성능)
- **금(GLD)**: SPY 대비 18배 높은 예측력 (0.37 vs 0.02)
- **트레이딩**: 91.3% 승률, Sharpe Ratio 22.76

---

## 빠른 시작

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 대시보드 실행
```bash
streamlit run app.py
# http://localhost:8501
```

---

## 프로젝트 구조

```
vrp-prediction/
├── app.py                  # Streamlit 대시보드 (발표용)
├── requirements.txt        # Python 의존성
├── src/                    # 연구 스크립트
│   ├── vrp_prediction_research.py
│   ├── vrp_eda.py
│   ├── vrp_validation.py
│   ├── vrp_improvement.py
│   └── vrp_max_performance.py
├── data/results/           # 실험 결과 JSON
├── diagrams/               # 발표 다이어그램 (PNG, DrawIO)
├── docs/                   # 연구 문서
└── references/             # 참고 자료
```

---

## 핵심 개념

### VRP 정의
```
VRP = IV (내재 변동성) - RV (실현 변동성)
    = VIX - 실제 변동성
```

### VIX-Beta 이론 (핵심 발견)
VIX와 자산의 RV 상관관계가 낮을수록 VRP 예측력이 높습니다:

| 자산 | VIX-RV 상관 | R² |
|------|------------|-----|
| GLD (금) | 0.51 | **0.37** |
| SPY (S&P 500) | 0.83 | 0.02 |

---

## 기술 스택

- **Python**: 3.9+
- **ML**: scikit-learn, XGBoost, LightGBM
- **대시보드**: Streamlit, Plotly
- **데이터**: pandas, numpy, yfinance

---

## 라이선스

MIT License
