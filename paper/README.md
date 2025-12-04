# Paper: Volatility Prediction in Financial Markets

**Ridge Regression Approach with Temporal Purging**

## 📄 Files

### Main Paper
- `PAPER_STRUCTURE.md` - Complete paper structure (9 sections)
- `PAPER_ABSTRACT.md` - Abstract (~250 words)
- `PAPER_INTRODUCTION.md` - Introduction (~1,450 words)
- `PAPER_REFERENCES.bib` - BibTeX references (30+ papers)
- `PAPER_SUBMISSION_STATUS.md` - Submission status

### Figures (figures/)
- Figure 1: Model Performance Comparison (CV vs WF)
- Figure 2: Return Prediction Failure
- Figure 3: Autocorrelation Analysis
- Figure 4: Validation Method Comparison
- Figure 5: Feature Count Analysis
- Figure 6: CV Threshold Analysis

### Scripts (scripts/)
- `run_har_benchmark.py` - HAR benchmark execution
- `create_paper_figures.py` - Generate all 6 figures
- `create_paper_structure.py` - Paper structure generator

## 🎯 Key Results

| Model | CV R² | Test R² | Status |
|-------|-------|---------|--------|
| **Ridge** | **0.303** | N/A | **Success** |
| HAR Benchmark | 0.215 | -0.047 | Unstable |
| Complex Models | 0.45+ | Negative | Overfitting |

**Improvement:** Ridge outperforms HAR by 1.41x

## 📊 Data Sources

All data files in `/root/workspace/data/raw/`:
- `har_benchmark_performance.json`
- `model_performance.json`
- `har_vs_ridge_comparison.json`

## 🔄 Reproducibility

```bash
# Regenerate HAR benchmark
python scripts/run_har_benchmark.py

# Regenerate all figures
python scripts/create_paper_figures.py
```

## 📚 Citation

```bibtex
@article{volatility2025,
  title={Volatility Prediction in Financial Markets: A Ridge Regression Approach with Temporal Purging},
  year={2025},
  note={Ridge R² = 0.303, HAR benchmark R² = 0.215}
}
```
