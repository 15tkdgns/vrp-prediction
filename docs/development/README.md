# 🚀 개발자 가이드

SP500 변동성 예측 시스템 개발을 위한 가이드 및 참고 자료입니다.

## 📋 개발 환경 설정

### 1. 필수 요구사항
- **Python**: 3.10 이상
- **Node.js**: 16.x 이상
- **메모리**: 최소 8GB RAM
- **디스크**: 최소 5GB 여유 공간

### 2. 의존성 설치
```bash
# Python 의존성
pip install -r requirements/base.txt
pip install -r requirements/dev.txt

# Node.js 의존성 (대시보드용)
cd dashboard && npm install
```

### 3. 개발 도구 설정
```bash
# 코드 포맷팅
pip install black ruff
black .
ruff .

# 테스트 도구
pip install pytest pytest-cov
pytest tests/ --cov=src
```

## 🏗️ 코드 구조 가이드

### 모듈 구조
```
src/
├── core/                    # 핵심 시스템
│   ├── unified_config.py   # 설정 관리
│   ├── logger.py           # 로깅
│   └── exceptions/         # 예외 처리
├── models/                 # ML 모델
├── validation/             # 검증 시스템
├── evaluation/             # 성능 평가
├── advanced_learning/      # 고급 학습
├── data/                   # 데이터 처리
└── features/               # 특성 엔지니어링
```

### 코딩 컨벤션

#### 1. 네이밍 규칙
```python
# 클래스: PascalCase
class UnifiedEnsembleSystem:
    pass

# 함수/변수: snake_case
def calculate_model_performance():
    model_accuracy = 0.85

# 상수: UPPER_SNAKE_CASE
MAX_ITERATIONS = 1000
```

#### 2. 문서화
```python
def predict_volatility(data: pd.DataFrame) -> np.ndarray:
    """
    다음날 변동성을 예측합니다.

    Args:
        data: 입력 데이터 (특성 포함)

    Returns:
        예측된 변동성 배열

    Raises:
        ValueError: 입력 데이터가 유효하지 않은 경우
    """
    pass
```

#### 3. 타입 힌트
```python
from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np

def process_features(
    data: pd.DataFrame,
    feature_columns: List[str],
    config: Optional[Dict[str, Union[str, int]]] = None
) -> pd.DataFrame:
    pass
```

## 🧪 테스트 가이드

### 테스트 구조
```
tests/
├── conftest.py              # 테스트 설정
├── test_unified_systems.py  # 통합 시스템 테스트
├── unit/                   # 단위 테스트
│   └── test_*.py
└── integration/            # 통합 테스트
    └── test_*.py
```

### 테스트 작성 예시
```python
import pytest
import pandas as pd
from src.models.unified_ensemble import UnifiedEnsembleSystem

class TestUnifiedEnsemble:
    def test_ensemble_creation(self):
        """앙상블 생성 테스트"""
        system = UnifiedEnsembleSystem()
        assert system is not None

    def test_model_training(self, sample_data):
        """모델 훈련 테스트"""
        system = UnifiedEnsembleSystem()
        X, y = sample_data

        system.add_ensemble("test", "voting")
        result = system.train_all_ensembles(X, y)

        assert result is True
        assert len(system.ensembles) > 0

@pytest.fixture
def sample_data():
    """테스트용 샘플 데이터"""
    X = pd.DataFrame(np.random.random((100, 5)))
    y = pd.Series(np.random.random(100))
    return X, y
```

## 🔄 개발 워크플로우

### 1. 새 기능 개발
```bash
# 새 브랜치 생성
git checkout -b feature/new-feature

# 개발 및 테스트
pytest tests/
black .
ruff .

# 커밋 및 푸시
git add .
git commit -m "feat: Add new feature"
git push origin feature/new-feature
```

### 2. 코드 리뷰 체크리스트
- [ ] 코딩 컨벤션 준수
- [ ] 테스트 추가/업데이트
- [ ] 문서화 완료
- [ ] 성능 영향 없음
- [ ] 데이터 누출 없음

### 3. 배포 전 체크리스트
- [ ] 모든 테스트 통과
- [ ] 코드 품질 검사 통과
- [ ] 문서 업데이트
- [ ] 성능 벤치마크 확인

## ⚙️ 설정 관리

### YAML 설정 구조
```yaml
# config/default.yaml
data:
  symbol: "SPY"
  start_date: "2020-01-01"

models:
  ensemble:
    use_gpu: true
    cv_folds: 5

validation:
  method: "purged_time_series_split"
  test_size: 0.2
```

### 환경별 설정
```python
from src.core.unified_config import get_config

# 개발 환경
config = get_config('development')

# 운영 환경
config = get_config('production')

# 설정 값 접근
symbol = config.get('data.symbol')
use_gpu = config.get('models.ensemble.use_gpu')
```

## 🚫 금지 사항

### 1. 하드코딩 금지
```python
# ❌ 금지
accuracy = 0.892  # 하드코딩된 성능 값

# ✅ 권장
accuracy = calculate_accuracy(y_true, y_pred)
```

### 2. 데이터 누출 방지
```python
# ❌ 금지 - 미래 정보 사용
features['future_price'] = df['Close'].shift(-1)

# ✅ 권장 - 과거 정보만 사용
features['price_lag1'] = df['Close'].shift(1)
```

### 3. Random 사용 금지
```python
# ❌ 금지
price = 450 + random.random() * 10

# ✅ 권장
price = fetch_real_stock_price(symbol, date)
```

## 🔧 디버깅 팁

### 1. 로깅 활용
```python
from src.core.logger import get_logger

logger = get_logger(__name__)

def complex_function():
    logger.info("Starting complex calculation")
    try:
        result = perform_calculation()
        logger.info(f"Calculation completed: {result}")
        return result
    except Exception as e:
        logger.error(f"Calculation failed: {e}")
        raise
```

### 2. 성능 프로파일링
```python
import cProfile
import pstats

# 성능 프로파일링
profiler = cProfile.Profile()
profiler.enable()

# 측정할 코드
run_model_training()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(10)
```

### 3. 메모리 사용량 모니터링
```python
import psutil
import tracemalloc

# 메모리 추적 시작
tracemalloc.start()

# 코드 실행
process_large_dataset()

# 메모리 사용량 확인
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
```

## 📞 도움말

### 문제 해결
1. **테스트 실패**: `pytest tests/ -v --tb=short`
2. **타입 오류**: `mypy src/`
3. **성능 이슈**: 프로파일링 도구 사용

### 리소스
- [프로젝트 위키](../../README.md)
- [API 문서](../api/)
- [이슈 트래커](https://github.com/issues)

---

*개발 중 문제가 있으면 [이슈](https://github.com/issues)를 등록하거나 개발팀에 문의하세요.*