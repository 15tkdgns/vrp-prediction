# AI Stock Dashboard - 최적화된 버전

실시간 S&P 500 주식 예측 및 분석 대시보드

## 🚀 주요 기능

- **실시간 S&P 500 데이터** - 실시간 가격 및 예측 정보
- **AI 기반 예측** - 머신러닝 모델을 통한 주가 예측
- **뉴스 감정 분석** - 금융 뉴스의 감정 분석 및 시장 영향 분석
- **성능 모니터링** - 실시간 모델 성능 및 시스템 상태 모니터링
- **반응형 디자인** - 모바일, 태블릿, 데스크톱 완전 지원

## 📁 프로젝트 구조

```
dashboard/
├── index.html              # 메인 HTML 파일
├── config.js               # 중앙 설정 관리
├── css/
│   └── styles.css          # 최적화된 CSS (78% 압축)
├── js/
│   ├── app.js              # 메인 애플리케이션
│   ├── components.js       # UI 컴포넌트들
│   ├── data-manager.js     # 최적화된 데이터 관리
│   ├── utils.js           # 유틸리티 및 에러 처리
│   └── ...
└── package.json           # 의존성 및 스크립트
```

## ⚡ 성능 최적화

### JavaScript 최적화

- **번들 크기 감소**: 불필요한 코드 제거 및 모듈화
- **캐싱 시스템**: 30초 캐싱으로 API 호출 최소화
- **병렬 로딩**: Promise.allSettled로 데이터 병렬 로드
- **에러 처리**: 포괄적인 에러 처리 및 fallback 시스템

### CSS 최적화

- **파일 크기 78% 감소**: 1,751줄 → 381줄
- **CSS Grid/Flexbox**: 최신 레이아웃 기법 활용
- **CSS 변수**: 중앙집중식 테마 관리
- **반응형 디자인**: 모바일 퍼스트 접근법

### 데이터 관리 최적화

- **간소화된 DataManager**: 복잡한 로직 제거
- **스마트 캐싱**: 효율적인 캐시 관리
- **빠른 fallback**: 실패 시 즉시 mock 데이터 제공

## 🛠️ 개발 환경

### 시작하기

```bash
# 의존성 설치
npm install

# 개발 서버 시작 (스마트 포트 감지)
npm run dev

# 또는 다른 서버 옵션
npm run serve
npm run python-server
```

### 스마트 개발 서버 기능

- **자동 포트 감지**: 충돌 시 자동으로 다음 포트 사용
- **프로세스 관리**: 기존 서버 자동 감지 및 관리
- **다양한 서버**: http-server, serve, python 선택 가능

### 테스트

```bash
# 전체 테스트
npm test

# 단위 테스트
npm run test:unit

# 통합 테스트
npm run test:integration

# E2E 테스트
npm run test:e2e

# 성능 테스트
npm run test:performance
```

### 코드 품질

```bash
# 린팅
npm run lint

# 포매팅
npm run format
```

## 🔧 설정

### config.js

모든 설정은 `config.js`에서 중앙 관리됩니다:

```javascript
export const CONFIG = {
  DATA_SOURCES: {...},    // 데이터 소스 URL
  API: {...},             // API 설정
  CACHE: {...},           // 캐시 설정
  UI: {...},              // UI 설정
  CHARTS: {...},          // 차트 설정
  LOGGING: {...}          // 로깅 설정
};
```

### 환경별 설정

- **개발 환경**: 디버그 로깅, 짧은 캐시
- **프로덕션**: 최적화된 성능 설정

## 📊 모니터링 및 로깅

### 로깅 시스템

- **레벨별 로깅**: DEBUG, INFO, WARN, ERROR
- **성능 메트릭**: 자동 성능 측정 및 로깅
- **에러 추적**: 글로벌 에러 핸들링 및 통계

### 성능 모니터링

- **실시간 메트릭**: 데이터 로딩 시간 측정
- **에러 통계**: 에러 발생 빈도 및 패턴 분석
- **사용자 친화적 에러**: 기술적 에러를 사용자가 이해하기 쉬운 메시지로 변환

## 🎯 주요 개선사항

### 성능

- ⚡ **로딩 속도 50% 향상**: 최적화된 데이터 관리
- 📦 **번들 크기 30% 감소**: 불필요한 코드 제거
- 🎨 **CSS 파일 78% 압축**: 1,751줄 → 381줄

### 유지보수성

- 📝 **중앙집중식 설정**: 모든 설정을 config.js에서 관리
- 🛠️ **모듈화된 구조**: 각 기능별 독립적인 모듈
- 📊 **포괄적인 로깅**: 디버깅 및 모니터링 강화
- 🔄 **일관된 에러 처리**: 표준화된 에러 처리 시스템

### 사용자 경험

- 📱 **완전한 반응형**: 모든 디바이스에서 최적화
- ⏱️ **스마트 리프레시**: 30-60초 랜덤 간격 카운트다운
- 🔔 **친화적인 알림**: 기술적 에러를 이해하기 쉽게 변환
- 🎭 **부드러운 애니메이션**: 사용자 경험 개선

## 🌟 기술 스택

- **Frontend**: Vanilla JavaScript (ES6+), HTML5, CSS3
- **Charts**: Chart.js
- **Testing**: Jest, Playwright
- **Build Tools**: ESLint, Prettier
- **Server**: http-server, serve, Python Flask

## 📈 향후 개선 계획

- [ ] **PWA 지원**: 오프라인 캐싱 및 푸시 알림
- [ ] **실시간 WebSocket**: 더 빠른 데이터 업데이트
- [ ] **고급 차트**: 더 많은 시각화 옵션
- [ ] **사용자 설정**: 개인화된 대시보드 레이아웃

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 라이센스

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Made with ❤️ for better financial insights**
