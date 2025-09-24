# 🔬 실험 무결성 가이드라인

**버전**: 1.0.0
**작성일**: 2025-09-22
**목적**: 할루시네이션 및 데이터 조작 방지, 신뢰할 수 있는 실험 결과 보장

---

## 🎯 핵심 원칙

### 1️⃣ 투명성 (Transparency)
- **모든 실험 과정 공개**: 데이터 수집부터 결과 해석까지
- **재현 가능한 코드**: 실험을 완전히 재현할 수 있는 스크립트 제공
- **한계점 명시**: 모델의 제약사항과 실패 사례 포함

### 2️⃣ 객관성 (Objectivity)
- **편향 없는 평가**: 긍정적/부정적 결과 모두 동등하게 보고
- **통계적 검증**: 모든 성과 주장에 대한 통계적 유의성 확인
- **과장 금지**: 감정적, 추상적 표현 배제

### 3️⃣ 재현성 (Reproducibility)
- **동일 환경 재현**: 같은 조건에서 같은 결과 도출 가능
- **시드 고정**: 랜덤 요소 제어로 일관된 결과
- **데이터 버전 관리**: 학습/테스트 데이터 명확한 분리

---

## 🚨 할루시네이션 방지 체크리스트

### ✅ 실험 시작 전 필수 확인사항

1. **데이터 검증**
   - [ ] 실제 데이터 사용 (시뮬레이션 데이터 아님)
   - [ ] 데이터 누출 방지 확인
   - [ ] 충분한 샘플 수 (최소 1,000개)
   - [ ] 시계열 데이터에서 미래 정보 사용 금지

2. **모델 검증**
   - [ ] TimeSeriesSplit 또는 Purged CV 사용
   - [ ] 홀드아웃 테스트 세트 별도 유지
   - [ ] 하이퍼파라미터 튜닝과 평가 분리
   - [ ] 과적합 방지 조치

3. **성과 지표 검증**
   - [ ] 현실적 성과 범위 내 (아래 표 참조)
   - [ ] 기준선(Baseline) 대비 개선량 확인
   - [ ] 신뢰구간 계산
   - [ ] 여러 메트릭으로 다각도 평가

### 📊 현실적 성과 범위 (Red Flag Thresholds)

| 작업 유형 | 최소 | 양호 | 우수 | **🚨 의심** |
|-----------|------|------|------|-------------|
| 분류 정확도 | 55% | 70% | 85% | **>95%** |
| 방향 예측 | 52% | 58% | 62% | **>70%** |
| 변동성 R² | 1% | 5% | 10% | **>20%** |
| 가격 예측 R² | 0% | 2% | 5% | **>15%** |

**⚠️ Red Flag 기준을 초과하는 결과는 즉시 할루시네이션 의심**

---

## 📋 필수 실험 문서화 규칙

### 1️⃣ 실험 등록
모든 실험은 `/root/workspace/data/experiment_registry.json`에 등록:

```json
{
  "experiment_id": "exp_YYYY_MM_DD_###",
  "date": "2025-09-22",
  "type": "classification|regression|prediction",
  "status": "pending|running|completed|failed",
  "model": "모델명",
  "data_source": "데이터 파일 경로",
  "performance": {
    "primary_metric": 0.XX,
    "sample_count": ####,
    "feature_count": ##,
    "validation_method": "TimeSeriesSplit"
  },
  "reproducibility": {
    "script_path": "실험 스크립트 경로",
    "random_seed": 42,
    "environment": "requirements.txt"
  }
}
```

### 2️⃣ 결과 파일 네이밍 규칙
- **실제 결과**: `validated_results_YYYYMMDD_HHMMSS.json`
- **시뮬레이션**: `simulated_[목적]_YYYYMMDD.json` (명확히 구분)
- **실패 결과**: `failed_experiment_[이유]_YYYYMMDD.json`

### 3️⃣ 필수 메타데이터
모든 결과 파일에 포함해야 할 정보:

```json
{
  "experiment_metadata": {
    "experiment_id": "고유 식별자",
    "timestamp": "실험 수행 시각",
    "reproducibility_hash": "코드/데이터 해시",
    "validation_status": "validated|pending|failed",
    "data_integrity_check": "passed|failed",
    "statistical_significance": "p_value"
  },
  "performance_metrics": {
    "primary_metric": "주요 성과 지표",
    "confidence_interval": [하한, 상한],
    "baseline_comparison": "기준선 대비 개선량",
    "sample_statistics": "샘플 수, 특성 수"
  }
}
```

---

## 🔍 할루시네이션 탐지 프로토콜

### 자동 검증 시스템 활용
매일 실행해야 하는 자동 검증:

```bash
# 할루시네이션 자동 탐지
python3 src/validation/auto_hallucination_detector.py

# 데이터 일관성 검증
python3 src/validation/data_integrity_checker.py

# 실험 재현성 확인
python3 src/validation/reproducibility_tester.py
```

### 수동 검증 체크포인트

**1. 결과 발표 전 (논문/보고서)**
- [ ] 모든 주장이 실제 실험으로 뒷받침됨
- [ ] 할루시네이션 탐지 시스템 통과
- [ ] 독립적인 검토자의 재현 확인
- [ ] 통계적 유의성 확인

**2. 실험 완료 후**
- [ ] 원시 데이터와 처리된 데이터 일치 확인
- [ ] 코드 리뷰 완료
- [ ] 결과 파일 무결성 검증
- [ ] 외부 검증 가능하도록 패키징

**3. 이상 결과 발견 시**
- [ ] 즉시 quarantine 디렉토리로 격리
- [ ] 할루시네이션 경고 헤더 추가
- [ ] 대체 검증된 결과 제시
- [ ] 원인 분석 및 재발 방지 계획

---

## 🎯 품질 보증 체계

### 단계별 검증 게이트

**Gate 1: 데이터 수집**
- 원시 데이터 품질 검증
- 데이터 누출 방지 확인
- 샘플 크기 적정성 검토

**Gate 2: 모델 훈련**
- 교차 검증 방법론 확인
- 하이퍼파라미터 최적화 분리
- 과적합 방지 조치 적용

**Gate 3: 성과 평가**
- 현실적 성과 범위 확인
- 통계적 유의성 검증
- 다중 메트릭 평가

**Gate 4: 결과 보고**
- 할루시네이션 탐지 시스템 통과
- 재현성 확인
- 독립 검토 완료

### 승인 권한 체계

| 성과 수준 | 필요 승인 | 검증 요구사항 |
|-----------|-----------|---------------|
| 기준선 내 | 자동 승인 | 자동 검증 통과 |
| 우수 수준 | 1차 검토 | 수동 검증 + 재현 확인 |
| 의심 수준 | 특별 검토 | 다중 검토자 + 외부 검증 |

---

## 🚫 사용 금지 행위

### 절대 금지 (Zero Tolerance)

1. **데이터 조작**
   - 결과 개선을 위한 데이터 변경
   - 테스트 세트 정보 활용
   - 미래 정보 누출

2. **성과 과장**
   - 실제보다 높은 성과 주장
   - 감정적/추상적 표현 사용
   - 한계점 은폐

3. **시뮬레이션 오용**
   - 시뮬레이션을 실제 결과로 포장
   - 가상 데이터로 성과 주장
   - 검증 없는 가정 사용

### 경고 수준 행위

1. **불충분한 검증**
   - 단일 메트릭 의존
   - 작은 샘플 사이즈 사용
   - 교차 검증 생략

2. **문서화 부족**
   - 실험 과정 불투명
   - 재현 정보 부족
   - 코드 공개 거부

---

## 📈 모범 사례 (Best Practices)

### 1️⃣ 실험 설계

**올바른 접근**:
```python
# 1. 명확한 문제 정의
problem = "SPY 방향 예측 (이진 분류)"

# 2. 현실적 목표 설정
target_accuracy = 0.58  # 기준선 50% + 8%p

# 3. 엄격한 검증 방법
validation = TimeSeriesSplit(n_splits=5)

# 4. 다중 메트릭 평가
metrics = ['accuracy', 'precision', 'recall', 'f1']

# 5. 불확실성 정량화
confidence_interval = bootstrap_confidence(predictions, 0.95)
```

### 2️⃣ 결과 보고

**모범 보고서 구조**:
```markdown
## 실험 결과

### 성과 요약
- 정확도: 58.2% ± 2.1% (95% CI: [56.1%, 60.3%])
- 기준선 대비: +8.2%p 개선 (p-value < 0.05)
- 샘플 수: 1,208개 (2020-2024 SPY 데이터)

### 한계점
- 거래 비용 미고려
- 시장 체제 변화 위험
- 예측 불확실성 존재

### 재현성
- 코드: /scripts/experiment_20250922.py
- 데이터: /data/spy_2020_2024.csv
- 환경: requirements.txt
```

### 3️⃣ 실패 사례 처리

**실패도 가치 있는 결과**:
```json
{
  "experiment_id": "failed_volatility_prediction_20250922",
  "objective": "SPY 변동성 예측 (R² > 0.20 목표)",
  "result": "실패",
  "achieved_r2": 0.0441,
  "target_r2": 0.2000,
  "lessons_learned": [
    "변동성 예측은 본질적으로 어려움",
    "R² 0.20은 비현실적 목표",
    "분류 접근이 더 효과적"
  ],
  "value": "현실적 목표 설정에 기여"
}
```

---

## 🔧 도구 및 자동화

### 필수 검증 도구

1. **할루시네이션 탐지기**
```bash
python3 src/validation/auto_hallucination_detector.py
```

2. **데이터 누출 검사기**
```bash
python3 src/validation/data_leakage_checker.py
```

3. **재현성 테스터**
```bash
python3 src/validation/reproducibility_tester.py
```

4. **성과 범위 검증기**
```bash
python3 src/validation/performance_validator.py
```

### 자동화된 워크플로우

**매일 실행 (CI/CD)**:
```bash
#!/bin/bash
# 일일 품질 검증 스크립트

echo "🔍 일일 무결성 검증 시작..."

# 1. 할루시네이션 탐지
python3 src/validation/auto_hallucination_detector.py

# 2. 데이터 일관성 확인
python3 src/validation/data_consistency_checker.py

# 3. 실험 등록부 검증
python3 src/validation/registry_validator.py

# 4. 성과 통계 업데이트
python3 src/utils/update_performance_stats.py

echo "✅ 검증 완료. 보고서: /data/daily_validation_report.json"
```

---

## 📊 성과 모니터링 대시보드

### KPI 지표

1. **데이터 무결성**
   - 검증된 파일 비율: >90%
   - 할루시네이션 파일 수: <5개
   - 자동 탐지 정확도: >95%

2. **실험 품질**
   - 재현 가능한 실험 비율: 100%
   - 통계적 유의성 확보 비율: >80%
   - 현실적 성과 범위 준수율: >95%

3. **연구 생산성**
   - 검증된 실험 수/월: >10개
   - 평균 검증 시간: <2일
   - 실패 실험 학습 활용도: >70%

---

## 🎓 교육 및 인증

### 연구자 필수 교육 과정

1. **기초 과정** (4시간)
   - 데이터 무결성 원칙
   - 할루시네이션 탐지 방법
   - 실험 설계 모범 사례

2. **고급 과정** (8시간)
   - 통계적 검증 방법론
   - 재현성 확보 기법
   - 품질 보증 시스템 운영

3. **인증 시험**
   - 할루시네이션 탐지 테스트
   - 실험 설계 실습
   - 무결성 검증 실습

### 자격 유지 조건

- 연간 20시간 품질 교육 이수
- 할루시네이션 탐지 정확도 >90% 유지
- 실험 재현성 100% 달성

---

## 📞 지원 및 문의

### 할루시네이션 의심 시 연락처
- **긴급**: 즉시 quarantine 디렉토리로 격리
- **검증 요청**: src/validation/ 도구 사용
- **외부 검토**: 독립적인 제3자 검증 요청

### 추가 자원
- **가이드라인 업데이트**: 월 1회 검토
- **도구 개선**: 지속적 업데이트
- **커뮤니티 피드백**: 정기적 의견 수렴

---

**이 가이드라인을 준수하여 신뢰할 수 있는 연구 문화를 구축합시다.**

**최종 업데이트**: 2025-09-22
**다음 검토**: 2025-10-22