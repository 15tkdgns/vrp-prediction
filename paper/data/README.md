# Paper Data Tables

## 📊 CSV / JSON Files

### 1. model_performance_comparison.csv
**모든 모델의 성능 종합 비교**
- 9개 모델 (HAR, Ridge, LSTM, TFT, Lasso, ElasticNet, RF, GARCH)
- 변동성 예측 vs 수익률 예측
- CV R², Test R², MAE, RMSE, Feature Count
- 상태: Stable / Unstable / Failed / Overfitting

### 2. key_findings_summary.csv
**논문의 핵심 발견사항 18개**
- Ridge 성능 vs HAR 불안정성
- EMH 실증 (수익률 예측 실패)
- 자기상관 분석 (변동성 0.931 vs 수익률 -0.117)
- 과적합 경고 (CV R² > 0.45)
- 최적 피처 수 (25-40개)

### 3. validation_method_comparison.csv
**검증 방법 비교 (6가지)**
- CV Only, Purged K-Fold, Walk-Forward, TimeSeriesSplit, BlockedCV, Standard K-Fold
- 5개 차원: Reliability, Conservatism, Leak Prevention, Real-world Accuracy, Speed
- 금융 ML 권장사항

### 4. economic_backtest_results.csv
**경제적 가치 실증**
- Ridge 전략 vs Buy & Hold 벤치마크
- 연수익률, 변동성, Sharpe Ratio, Max Drawdown
- 핵심: 변동성 0.8% 감소 (리스크 관리)

### 5. model_performance_paper.json
- Streamlit/README에서 사용하는 ElasticNet 메트릭의 논문 버전
- `data/raw/model_performance.json`에서 직접 복사

### 6. comprehensive_model_validation_paper.json
- Purged K-Fold 결과 전체를 포함
- `data/validation/comprehensive_model_validation.json`과 동일 (논문 제출용 스냅샷)

### 7. rv_economic_backtest_results.json
- 경제적 백테스트 세부 지표 (JSON)
- Appendix에서 표/텍스트로 전환 가능

### 8. elasticnet_grid_real_spy_20251129_073237.{json,csv}
- ElasticNet 파라미터/피처 그리드 실험 결과 (실제 SPY 데이터)
- `experiments/elasticnet_grid/run_elasticnet_grid.py` 출력

## 📁 Usage

**논문 작성:**
```latex
\begin{table}
\centering
\csvautotabular{model_performance_comparison.csv}
\caption{Model Performance Comparison}
\end{table}
```

**Excel/Spreadsheet:**
- 모든 CSV 파일은 UTF-8 인코딩
- Excel에서 바로 열기 가능
- Pivot Table 생성 가능

**Python Analysis:**
```python
import pandas as pd
df = pd.read_csv('paper/data/model_performance_comparison.csv')
print(df[df['Status'] == 'Stable'])
```

## 🔄 Regeneration

모든 데이터는 다음 소스에서 추출:
- `/root/workspace/data/raw/model_performance.json`
- `/root/workspace/data/raw/har_benchmark_performance.json`
- `/root/workspace/data/raw/lstm_model_performance.json`
- `/root/workspace/data/raw/tft_model_performance.json`
- `/root/workspace/paper/scripts/create_paper_figures.py` (Figure data)

## 📋 Quick Summary

| File | Rows | Purpose |
|------|------|---------|
| model_performance_comparison.csv | 9 | All model metrics |
| key_findings_summary.csv | 18 | Core paper findings |
| validation_method_comparison.csv | 6 | Validation methods |
| economic_backtest_results.csv | 3 | Economic value |
| elasticnet_grid_real_spy_20251129_073237.csv | 32 | ElasticNet grid (real SPY) |

**추가 JSON 스냅샷:**  
`model_performance_paper.json`, `comprehensive_model_validation_paper.json`, `rv_economic_backtest_results.json`, `elasticnet_grid_real_spy_20251129_073237.json`

**Total: 68 data points across CSV + 4 JSON snapshots**
