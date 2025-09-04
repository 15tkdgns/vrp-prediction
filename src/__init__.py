"""
AI Stock Prediction System - Source Package
S&P500 주식 이벤트 탐지 및 예측을 위한 AI 시스템
"""

__version__ = "1.0.0"
__author__ = "AI Stock Prediction Team"
__description__ = "AI-powered S&P500 stock event detection and prediction system"

# 패키지 레벨 임포트 (선택적)
try:
    from .utils.system_orchestrator import SystemOrchestrator
    from .models.model_training import SP500EventDetectionModel
    from .core.data_collection_pipeline import SP500DataCollector
    
    __all__ = [
        'SystemOrchestrator', 
        'SP500EventDetectionModel', 
        'SP500DataCollector',
        '__version__',
        '__author__',
        '__description__'
    ]
except ImportError:
    # 의존성이 없을 경우 패키지만 정의
    __all__ = ['__version__', '__author__', '__description__']
