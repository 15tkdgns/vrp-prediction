"""
Financial Forecasting System - Advanced Time Series Modeling

근본적 접근법 변경:
- 가격 예측 → 로그 수익률 예측 (통계적 정상성)
- 단순 모델 → 고급 계량경제/딥러닝 모델
- 부정확한 검증 → 시계열 안전 검증
- MSE 최적화 → 금융 성과 지표 최적화

주요 구성요소:
- Statistical Foundation: 로그 수익률, ADF 검정, 정상성 확보
- Econometric Models: ARIMA-GARCH, 변동성 군집 모델링
- Deep Learning: LSTM, Transformer, TFT
- Uncertainty Modeling: MDN, 확률 분포 예측
- Alternative Data: FRED, FinBERT, HMM 레짐 감지
- Financial Metrics: 샤프, 소르티노, MDD 최적화
"""

from .core import (
    LogReturnProcessor,
    TimeSeriesSafeValidator,
    FinancialMetrics
)

from .models import (
    ARIMAGARCHModel,
    TemporalFusionTransformer,
    MixtureDensityNetwork,
    UncertaintyQuantifier
)

from .features import (
    AlternativeDataIntegrator,
    MarketRegimeDetector,
    SentimentAnalyzer
)

from .validation import (
    WalkForwardValidator,
    FinancialBacktester,
    RiskMetricsCalculator
)

__version__ = "1.0.0"
__author__ = "Financial ML Research Team"

__all__ = [
    # Core Components
    "LogReturnProcessor",
    "TimeSeriesSafeValidator",
    "FinancialMetrics",

    # Advanced Models
    "ARIMAGARCHModel",
    "TemporalFusionTransformer",
    "MixtureDensityNetwork",
    "UncertaintyQuantifier",

    # Feature Engineering
    "AlternativeDataIntegrator",
    "MarketRegimeDetector",
    "SentimentAnalyzer",

    # Validation & Risk
    "WalkForwardValidator",
    "FinancialBacktester",
    "RiskMetricsCalculator"
]