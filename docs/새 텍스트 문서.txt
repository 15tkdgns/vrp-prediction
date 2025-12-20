# VRP 예측 연구 종합 요약서

> **머신러닝을 활용한 변동성 위험 프리미엄(VRP) 예측 연구**
> 
> GitHub: https://github.com/15tkdgns/vrp-prediction
> 작성일: 2025-12-20

---

## 한 줄 요약

**ElasticNet 선형 모델로 VRP를 R²=0.19, 방향 정확도 73.5%로 예측 가능하며, 이를 활용한 트레이딩 전략은 77.7% 승률과 거래당 +3.09% 초과 수익을 달성**

---

## 1. 연구 개요

### 1.1 연구 질문
> "VRP를 예측할 수 있는가? 예측이 가능하다면 어떤 정보가 가장 유용한가?"

### 1.2 VRP 정의
```
VRP = VIX (내재 변동성) - RV (실현 변동성)
    = 옵션 시장의 "공포 프리미엄"
```

### 1.3 데이터
| 항목 | 값 |
|------|-----|
| 자산 | SPY ETF |
| 기간 | 2020-2025 (5년) |
| 표본 | 1,375 거래일 |
| 소스 | Yahoo Finance, CBOE VIX |

---

## 2. 핵심 결과

### 2.1 모델 성능 (VRP 예측 R²)
```
ElasticNet     ████████████████████  0.191 ✅ 최고
Ridge          ███████████████████   0.191
HAR-RV         ████████              0.103
LightGBM       ▓▓▓▓                 -0.097 (과적합)
GradientBoost  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓       -1.404 (과적합)
XGBoost        ▓▓▓▓▓▓▓              -0.393 (과적합)
```

### 2.2 통계적 유의성 (Diebold-Mariano Test)
| 비교 | p-value | 결과 |
|------|---------|------|
| ElasticNet vs Ridge | 0.032** | ElasticNet 우수 |
| ElasticNet vs LightGBM | <0.001*** | ElasticNet 우수 |
| ElasticNet vs HAR-RV | 0.633 | 차이 없음 |

### 2.3 방향 예측 성능
| 지표 | 값 |
|------|-----|
| 정확도 | 71.3% |
| 정밀도 | 74.8% |
| 재현율 | 84.4% |
| F1 Score | 0.79 |

### 2.4 트레이딩 성과
| 지표 | 예측 전략 | Buy & Hold | 차이 |
|------|----------|-----------|------|
| 승률 | **77.7%** | - | - |
| 총 수익 | 697.27% | 100.75% | **+596%** |
| 거래당 수익 | 3.45% | 0.37% | **+3.09%** |

---

## 3. 핵심 발견 5가지

1. **VRP 예측 가능** 
   - R² = 0.13~0.19, 방향 정확도 71~74%

2. **선형 모델이 최적** 
   - ElasticNet > Ridge > HAR-RV >> 트리 모델
   - 복잡한 모델은 과적합

3. **RV 예측 후 VRP 계산 방식** 
   - VRP 직접 예측보다 효과적: `VRP_pred = VIX - RV_pred`

4. **VIX 래그가 가장 강력한 예측자** 
   - VIX_lag1, VIX_lag5 > RV_22d > VRP 래그

5. **트레이딩 가치 확인** 
   - 예측 기반 전략이 Buy & Hold 대비 우수

---

## 4. 특성 중요도 (Top 5)

| 순위 | 특성 | 계수 |
|------|------|------|
| 1 | VIX_lag1 (전일 VIX) | 5.77 |
| 2 | VIX_lag5 (5일 전 VIX) | 5.47 |
| 3 | RV_22d (22일 실현변동성) | 4.25 |
| 4 | VRP_lag5 (5일 전 VRP) | 2.36 |
| 5 | VRP_ma5 (VRP 5일 이동평균) | 1.88 |

---

## 5. VIX-Beta 이론 (핵심 발견)

VIX와 자산의 RV 상관관계가 낮을수록 VRP 예측력이 높음:

| 자산 | VIX-RV 상관 | R² |
|------|------------|-----|
| **GLD (금)** | 0.51 | **0.37** |
| SPY (S&P 500) | 0.83 | 0.02 |

**해석**: VIX는 S&P 500 옵션 기반 → SPY에 이미 최적화 → 다른 자산에서 예측 여지 더 큼

---

## 6. 연구 한계

1. **데이터 기간**: COVID-19 포함 (특수 시장 환경)
2. **단일 자산**: SPY만 분석 (일반화 한계)
3. **넓은 신뢰구간**: 95% CI [0.07, 0.20]
4. **거래 비용 미반영**: 슬리피지, 수수료 제외

---

## 7. 프로젝트 구조

```
vrp-prediction/
├── app.py                      # Streamlit 대시보드 (발표용)
├── src/
│   ├── vrp_prediction_research.py  # 핵심 VRP 예측 모델
│   ├── vrp_validation.py           # 강건성 검증
│   ├── statistical_tests.py        # 통계적 유의성 검정
│   ├── generate_paper_figures.py   # 논문 그래프 생성
│   └── vrp_eda.py                  # 탐색적 분석
├── data/
│   ├── raw/spy_data_2020_2025.csv  # 원본 데이터
│   └── results/*.json              # 실험 결과
├── diagrams/figures/               # 논문 그래프 (6개)
├── docs/
│   ├── vrp_prediction_paper.md     # 논문 본문
│   └── PROJECT_INSIGHTS.md         # 프로젝트 인사이트
└── references/references.bib       # BibTeX 참고문헌 (15편)
```

---

## 8. 실행 명령어

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 핵심 모델 실행
python src/vrp_prediction_research.py

# 3. 검증 실행
python src/vrp_validation.py

# 4. 통계적 유의성 검정
python src/statistical_tests.py

# 5. 논문 그래프 생성
python src/generate_paper_figures.py

# 6. 대시보드 실행
streamlit run app.py
```

---

## 9. 참고문헌 (핵심 3편)

1. **Bollerslev et al. (2009)** - VRP와 주식 수익률 관계
2. **Corsi (2009)** - HAR-RV 모델 (벤치마크)
3. **Bekaert & Hoerova (2014)** - VIX와 분산 프리미엄

---

## 10. 인용

```bibtex
@misc{vrp_prediction_2025,
  author = {VRP Prediction Research},
  title = {Machine Learning Approach to VRP Prediction},
  year = {2025},
  url = {https://github.com/15tkdgns/vrp-prediction}
}
```

---

**라이선스**: MIT License
