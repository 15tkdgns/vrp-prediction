# Paper Figures

## 📊 Directory Structure

### main_results/ (4 files)
**핵심 실험 결과** - 논문의 메인 주장을 뒷받침하는 그래프

- **Figure 1: Model Performance Comparison**
  - HAR Benchmark vs Ridge vs Complex Models
  - CV R² vs Walk-Forward R² 비교
  - 핵심 발견: Ridge 안정적, HAR 불안정, 복잡 모델 과적합

- **Figure 2: Return Prediction Failure**
  - Ridge vs LSTM vs TFT 수익률 예측 성능
  - 핵심 발견: 모든 모델 R² ≈ 0 (EMH 실증)

### analysis/ (4 files)
**분석 및 인사이트** - 왜 이런 결과가 나왔는지 설명

- **Figure 3: Autocorrelation Analysis**
  - 변동성 vs 수익률 자기상관 비교
  - 핵심 발견: 변동성(0.93) vs 수익률(-0.12)
  - 예측 가능성의 근본 원인

- **Figure 5: Feature Count Analysis**
  - 피처 수 vs 성능 관계
  - 핵심 발견: 3개(HAR) → 31개(Ridge) → 50개+(과적합)
  - 골디락스 존: 25-40개

### methodology/ (4 files)
**방법론 검증** - 검증 방법의 중요성 입증

- **Figure 4: Validation Method Comparison**
  - CV Only vs Purged K-Fold vs Walk-Forward
  - 핵심 발견: Purged K-Fold 필수

- **Figure 6: CV Threshold Analysis**
  - CV R² 임계값 분석
  - 핵심 발견: CV R² > 0.45 = 과적합 경고

## 📁 File Formats

각 Figure는 2개 포맷 제공:
- **PNG** (300 DPI) - 프레젠테이션, 웹
- **PDF** (Vector) - 논문 출판용

## 🔄 Regeneration

```bash
cd /root/workspace
python paper/scripts/create_paper_figures.py
```

All figures will be regenerated in their respective directories.

## 📋 Figure Summary

| Figure | Category | Key Finding |
|--------|----------|-------------|
| Figure 1 | Main Results | Ridge 안정적, HAR 불안정 |
| Figure 2 | Main Results | 수익률 예측 불가 (R²≈0) |
| Figure 3 | Analysis | 변동성 자기상관 0.93 |
| Figure 4 | Methodology | Purged K-Fold 필수 |
| Figure 5 | Analysis | 최적 피처: 25-40개 |
| Figure 6 | Methodology | CV R²>0.45 과적합 |
