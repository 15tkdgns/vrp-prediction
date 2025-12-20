# VRP 예측 연구

> 머신러닝을 활용한 변동성 위험 프리미엄(VRP) 예측 연구

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## 프로젝트 개요

**VRP (Volatility Risk Premium)**은 옵션 시장의 내재 변동성(VIX)과 실현 변동성(RV)의 차이입니다. 본 연구는 머신러닝을 활용하여 VRP를 예측하고 투자 전략에 활용합니다.

### 핵심 성과
| 지표 | 결과 |
|------|------|
| **VRP 예측 R²** | 0.19 (ElasticNet) |
| **방향 예측 정확도** | 73.5% |
| **트레이딩 승률** | 77.7% |
| **거래당 초과 수익** | +3.09% |

### VIX-Beta 이론 (핵심 발견)
VIX와 자산의 RV 상관관계가 낮을수록 VRP 예측력이 높습니다:

| 자산 | VIX-RV 상관 | R² |
|------|------------|-----|
| GLD (금) | 0.51 | **0.37** |
| SPY (S&P 500) | 0.83 | 0.02 |

---

## 빠른 시작

### 1. 환경 설정
```bash
# 저장소 클론
git clone https://github.com/15tkdgns/vrp-prediction.git
cd vrp-prediction

# 의존성 설치
pip install -r requirements.txt
```

### 2. 연구 재현

```bash
# 1단계: VRP 예측 모델 실행
python src/vrp_prediction_research.py

# 2단계: 검증 (Bootstrap, Regime별, 트레이딩 시뮬레이션)
python src/vrp_validation.py

# 3단계: 탐색적 분석
python src/vrp_eda.py

# 4단계: 모델 개선 실험
python src/vrp_improvement.py

# 5단계: 최대 성능 도달 실험
python src/vrp_max_performance.py
```

### 3. 대시보드 실행 (발표용)
```bash
streamlit run app.py
# 브라우저에서 http://localhost:8501 접속
```

---

## 프로젝트 구조

```
vrp-prediction/
├── app.py                      # Streamlit 대시보드 (발표용)
├── requirements.txt            # Python 의존성 (버전 고정)
├── LICENSE                     # MIT 라이선스
├── src/                        # 연구 스크립트
│   ├── vrp_prediction_research.py  # 핵심 VRP 예측 모델
│   ├── vrp_validation.py           # 강건성 검증
│   ├── vrp_eda.py                  # 탐색적 분석
│   ├── vrp_improvement.py          # 모델 개선
│   └── vrp_max_performance.py      # 최대 성능 실험
├── data/
│   ├── raw/                    # 원본 데이터
│   │   └── spy_data_2020_2025.csv
│   └── results/                # 실험 결과 JSON
├── diagrams/                   # 발표 다이어그램 (PNG, DrawIO)
├── docs/                       # 연구 문서
│   ├── vrp_prediction_paper.md # 논문 초안
│   ├── PROJECT_INSIGHTS.md     # 프로젝트 인사이트
│   └── DATA_INTEGRITY_GUIDE.md # 데이터 무결성 가이드
└── references/                 # 참고 자료
    └── references.bib          # BibTeX 참고문헌
```

---

## 핵심 개념

### VRP 정의
```
VRP = IV (내재 변동성) - RV (실현 변동성)
    = VIX - 실제 변동성 (22일)
```

### 연구 방법론
- **데이터**: SPY ETF, VIX (2020-2025, 1,375 거래일)
- **모델**: ElasticNet, Ridge, GradientBoosting, XGBoost, LightGBM
- **검증**: Bootstrap 95% CI, Regime별 분석, 연도별 안정성

---

## 기술 스택

- **Python**: 3.9+
- **ML**: scikit-learn, XGBoost, LightGBM
- **대시보드**: Streamlit, Plotly
- **데이터**: pandas, numpy, yfinance

---

## 인용

이 연구를 인용하려면:

```bibtex
@misc{vrp_prediction_2025,
  author = {VRP Prediction Research},
  title = {Machine Learning Approach to Volatility Risk Premium Prediction},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/15tkdgns/vrp-prediction}
}
```

---

## 라이선스

[MIT License](LICENSE)
