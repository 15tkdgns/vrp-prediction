# SPY 변동성 예측 Streamlit 대시보드

## 개요

Ridge 회귀 기반 5일 변동성 예측 시스템의 성능을 시각화하는 인터랙티브 대시보드입니다.

## 실행 방법

### 기본 실행
```bash
streamlit run app.py
```

### PYTHONPATH 포함 실행
```bash
PYTHONPATH=/root/workspace streamlit run app.py
```

### 포트 지정 실행
```bash
streamlit run app.py --server.port 8501
```

### 백그라운드 실행
```bash
PYTHONPATH=/root/workspace streamlit run app.py --server.port 8501 --server.headless true &
```

## 접속 주소

- **Local**: http://localhost:8501
- **Network**: http://172.20.79.76:8501
- **External**: http://124.195.230.58:8501

## 대시보드 구조

### 📈 Tab 1: 변동성 예측
- **실제 vs 예측 시계열 차트**: SPY ETF의 실제 변동성과 Ridge 모델 예측 비교
- **산점도**: 예측 정확도 시각화 (완벽한 예측선 포함)
- **잔차 분포**: 예측 오차의 히스토그램
- **주요 메트릭**: R² Score, RMSE, MAE, CV Std
- **통계 정보**: 평균 변동성, 상관계수, 평균 절대 오차

### 🎯 Tab 2: 특성 중요도
- **SHAP 기반 Top 20 특성**: 수평 바 차트로 중요도 시각화
- **특성 상세 테이블**: 순위, SHAP 중요도, 평균/표준편차
- **주요 인사이트**: Top 3 특성 강조 표시
- **모델 성능**: XAI 분석 시점의 Train/Test R², RMSE

### 💰 Tab 3: 경제적 가치
- **백테스트 기간 정보**: 2015-2024 실제 거래 데이터
- **전략 설정**: 목표 변동성, 거래 비용, 포지션 범위
- **성과 비교 테이블**: Buy & Hold, V0 Strategy, RV Strategy 비교
- **시각화**:
  - 연율화 수익률 비교 (바 차트)
  - Sharpe Ratio 비교 (바 차트)
  - 위험-수익 프로파일 (산점도)
- **비교 분석**: RV Strategy vs Buy & Hold 차이 (수익률, 변동성, Sharpe)
- **핵심 인사이트**: 경제적 가치 요약 및 주의사항

## 기술 스택

### Backend
- **Python 3.13**
- **Streamlit 1.51.0**: 웹 인터페이스
- **Pandas 2.3.3**: 데이터 처리
- **NumPy 2.3.4**: 수치 계산
- **scikit-learn 1.7.2**: 모델링
- **yfinance 0.2.66**: 실제 SPY 데이터

### Frontend
- **Plotly 6.4.0**: 인터랙티브 차트
- **Plotly Express**: 빠른 시각화
- **Bootstrap (Streamlit 내장)**: 반응형 레이아웃

### 데이터 소스
- `data/raw/model_performance.json`: 모델 성능 메트릭
- `data/xai_analysis/verified_xai_analysis_*.json`: SHAP 분석 결과
- `data/raw/rv_economic_backtest_results.json`: 백테스트 결과
- **실시간 SPY 데이터**: yfinance API를 통한 다운로드

## 주요 기능

### 1. 데이터 캐싱
```python
@st.cache_data
def load_model_performance():
    # 자동 캐싱으로 빠른 로딩
```

### 2. 실시간 데이터 생성
- yfinance를 통한 SPY 데이터 다운로드
- 변동성 계산 (5일 실현 변동성)
- Ridge 모델 학습 및 예측
- 결과 자동 캐싱

### 3. 인터랙티브 시각화
- **줌/팬**: 차트 확대/축소 및 이동
- **호버 정보**: 마우스 오버 시 상세 데이터 표시
- **레전드 토글**: 범례 클릭으로 데이터 시리즈 표시/숨김
- **통합 호버**: 시계열 차트에서 x축 기준 모든 데이터 표시

### 4. 반응형 레이아웃
- **Wide 모드**: 화면 공간 최대 활용
- **컬럼 레이아웃**: 메트릭과 차트를 효율적으로 배치
- **탭 구조**: 정보를 논리적으로 분리

## 개선 사항

### 이전 대시보드 (HTML/JavaScript)
- ❌ 정적 차트, 제한된 인터랙션
- ❌ 데이터 업데이트 시 JavaScript 코드 수정 필요
- ❌ 복잡한 레이아웃 관리
- ❌ 서버 설정 필요 (http-server, serve 등)

### 현재 대시보드 (Streamlit)
- ✅ Plotly 기반 완전한 인터랙티브 차트
- ✅ Python 코드만으로 전체 시스템 관리
- ✅ 자동 데이터 캐싱 및 로딩
- ✅ 깔끔한 3-탭 구조로 정보 분리
- ✅ 명확한 색상 구분과 레이블링
- ✅ Streamlit 내장 서버로 간편한 실행
- ✅ 실시간 데이터 자동 생성

## 성능 최적화

### 캐싱 전략
```python
# 모델 성능 데이터는 한 번만 로드
@st.cache_data
def load_model_performance():
    ...

# 변동성 예측은 계산이 필요하지만 결과는 캐싱
@st.cache_data
def load_volatility_predictions():
    ...
```

### 효율적인 데이터 로딩
- JSON 파일은 메모리에 캐싱
- SPY 데이터는 yfinance로 한 번만 다운로드
- 모델 학습/예측 결과 캐싱

## 문제 해결

### yfinance 모듈 없음
```bash
pip3 install yfinance
```

### scikit-learn 모듈 없음
```bash
pip3 install scikit-learn
```

### Streamlit 설치
```bash
pip3 install streamlit plotly
```

### 포트 충돌
```bash
# 다른 포트 사용
streamlit run app.py --server.port 8502
```

### 프로세스 종료
```bash
# Streamlit 프로세스 찾기
ps aux | grep streamlit

# 강제 종료
pkill -f "streamlit run"
```

## 데이터 무결성

### 완전한 시간적 분리
- **특성**: ≤ t (과거 데이터만 사용)
- **타겟**: ≥ t+1 (미래 변동성 예측)
- **Zero Overlap**: 데이터 누출 제로

### 검증 방법
- **Purged K-Fold CV**: 5-fold, purge_length=5, embargo_length=5
- **실제 데이터**: SPY ETF 2015-2024 (시뮬레이션 데이터 없음)
- **벤치마크 비교**: HAR 모델 (학계 표준)
- **경제적 백테스트**: 거래 비용 포함

## 모델 성능

| 메트릭 | 값 |
|--------|------|
| **R² Score** | 0.3113 |
| **RMSE** | 0.005305 |
| **MAE** | 0.003323 |
| **CV Std** | 0.1975 |
| **샘플 수** | 2,460 |
| **특성 수** | 31 |

## 경제적 가치

| 메트릭 | RV Strategy | Buy & Hold | 차이 |
|--------|-------------|------------|------|
| **연율화 수익률** | 12.58% | 11.61% | +0.97% |
| **변동성** | 20.32% | 17.77% | +2.56% |
| **Sharpe Ratio** | 0.520 | 0.541 | -0.020 |
| **Max Drawdown** | -32.92% | -35.75% | +2.83% |

**핵심 가치**: 변동성 예측을 통한 리스크 관리 및 동적 포지션 조절

## 시스템 요구사항

### 최소 요구사항
- Python 3.8+
- 2GB RAM
- 인터넷 연결 (yfinance 데이터 다운로드)

### 권장 요구사항
- Python 3.13
- 4GB RAM
- SSD 스토리지

## 향후 개선 계획

1. **실시간 데이터 업데이트**: 자동 새로고침 기능
2. **모델 비교**: 여러 모델 성능 동시 비교
3. **커스터마이징**: 사용자 정의 백테스트 파라미터
4. **다운로드 기능**: 차트 및 데이터 다운로드
5. **다크 모드**: 테마 전환 기능
6. **모바일 최적화**: 반응형 디자인 개선

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 문의

문제가 발생하거나 개선 제안이 있으시면 이슈를 등록해주세요.
