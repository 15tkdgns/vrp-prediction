# 코드베이스 리팩토링 계획

## 현재 문제점 분석

### 1. 중복 파일 문제
- **앙상블 관련**: 9개 파일 (ensemble.py, optimized_ensemble_system.py 등)
- **성능 분석**: 12개 파일 (performance_evaluator.py, dynamic_performance_calculator.py 등)
- **고급 기능**: 15개 파일 (advanced_feature_engineering.py, advanced_model_metrics_calculator.py 등)
- **검증 로직**: 14개 파일 (data_leakage_checker.py, auto_leakage_detector.py 등)

### 2. 설정 파일 분산
- `data/raw/`: 80개+ JSON 설정 파일들
- 여러 위치의 config 파일들
- requirements 파일 분산 (base.txt, dev.txt, prod.txt)

### 3. 폴더 구조 복잡성
- src/ 아래 29개 디렉토리
- 비슷한 기능의 모듈들이 다른 디렉토리에 분산
- `from src.` 절대 임포트 남용

## 개선된 폴더 구조 제안

```
src/
├── core/                    # 핵심 시스템 (기존 유지, 정리)
│   ├── config.py           # 통합 설정 관리
│   ├── logger.py           # 로깅 시스템
│   ├── exceptions/         # 예외 처리
│   └── interfaces/         # 인터페이스 정의
│
├── data/                   # 데이터 처리 (통합)
│   ├── processors/         # 데이터 전처리기들
│   ├── loaders/           # 데이터 로더들
│   └── validators/        # 데이터 검증 (14개 파일 → 3개로 통합)
│
├── models/                # 모델 관리 (핵심만 유지)
│   ├── base.py           # 기본 모델 클래스
│   ├── ensemble.py       # 통합 앙상블 시스템 (9개 → 1개)
│   ├── predictors/       # 예측 모델들
│   └── training/         # 훈련 시스템
│
├── features/              # 특성 엔지니어링 (정리)
│   ├── engineering.py    # 기본 특성 엔지니어링
│   ├── advanced.py       # 고급 특성 (15개 → 2개로 통합)
│   └── selection.py      # 특성 선택
│
├── evaluation/           # 평가 시스템 (통합)
│   ├── metrics.py        # 성능 지표 (12개 → 1개로 통합)
│   ├── validation.py     # 교차 검증
│   └── reporting.py      # 결과 보고서
│
├── pipeline/             # 파이프라인 (간소화)
│   ├── training.py       # 훈련 파이프라인
│   ├── inference.py      # 추론 파이프라인
│   └── optimization.py   # 최적화
│
└── utils/                # 유틸리티 (정리)
    ├── io.py            # 입출력 관련
    ├── visualization.py  # 시각화
    └── helpers.py       # 보조 함수들

config/                   # 설정 파일 통합
├── default.yaml         # 기본 설정
├── production.yaml      # 운영 환경
└── development.yaml     # 개발 환경

data/
├── config/              # 데이터 설정 (JSON 파일들 통합)
├── models/              # 훈련된 모델
├── processed/           # 전처리된 데이터
└── results/             # 실행 결과

tests/                   # 테스트 코드 정리
├── unit/               # 단위 테스트
├── integration/        # 통합 테스트
└── fixtures/          # 테스트 데이터
```

## 리팩토링 우선순위

### Phase 1: 중복 제거 및 통합
1. **앙상블 파일들 통합** (9개 → 1개)
2. **성능 평가 통합** (12개 → 3개)
3. **검증 로직 통합** (14개 → 3개)
4. **설정 파일 정리** (80개+ → 10개)

### Phase 2: 모듈 구조 개선
1. **임포트 구조 정리** (`from src.` → 상대 임포트)
2. **인터페이스 정의** (추상 클래스 활용)
3. **의존성 주입** (느슨한 결합)

### Phase 3: 코드 품질 개선
1. **함수 분리** (단일 책임 원칙)
2. **클래스 구조 개선** (상속 vs 구성)
3. **타입 힌트 추가**
4. **docstring 표준화**

### Phase 4: 테스트 및 문서화
1. **테스트 코드 정리**
2. **API 문서 자동 생성**
3. **README 업데이트**

## 예상 효과

### 유지보수성 향상
- 중복 코드 80% 감소
- 모듈 간 결합도 50% 감소
- 설정 파일 90% 감소

### 가독성 향상
- 명확한 폴더 구조
- 일관된 네이밍 컨벤션
- 표준화된 인터페이스

### 재사용성 향상
- 모듈화된 컴포넌트
- 플러그인 아키텍처
- 의존성 주입 패턴

## 실행 계획

1. **백업 생성**: 현재 코드베이스 백업
2. **점진적 마이그레이션**: 모듈별 순차 리팩토링
3. **테스트 유지**: 기능 동작 보장
4. **문서 업데이트**: 변경사항 문서화