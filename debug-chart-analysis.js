// 차트 데이터 문제 분석 스크립트

console.log("🔍 주요 종목 실시간 가격 & 예측 위젯 문제 분석");
console.log("==================================================");

// 1. 실제 데이터 형태 (realtime_results.json에서)
const realData = {
    "AAPL": {
        "current_price": 258.52,
        "predicted_price": 252.56,
        "change_percent": -2.31
    },
    "MSFT": {
        "current_price": 225.22, 
        "predicted_price": 227.01,
        "change_percent": 0.8
    }
};

console.log("📊 1. 실제 데이터 구조:");
console.log("   - 현재가: 단일 값 (예: $258.52)");
console.log("   - 예측가: 단일 값 (예: $252.56)");
console.log("   - 시계열 데이터: 없음");

// 2. 차트에서 시도하는 것
console.log("\n📈 2. 차트가 시도하는 것:");
console.log("   - 실제 데이터: 30일 시계열 생성 (시뮬레이션)");
console.log("   - 예측 데이터: 30일 시계열 생성 (시뮬레이션)");
console.log("   - 날짜 범위: 2025-07-22 ~ 2025-08-21 (23 영업일)");

// 3. 문제점 식별
console.log("\n❌ 3. 발견된 문제들:");

console.log("\n   A. 데이터 불일치 문제:");
console.log("      - 실제 데이터: 시뮬레이션된 과거 30일");
console.log("      - 예측 데이터: 시뮬레이션된 예측 30일");  
console.log("      - 결과: 서로 다른 기준으로 생성된 데이터");

console.log("\n   B. Y축 스케일 문제:");
console.log("      - 현재: 현재가 기준 ±8% 고정");
console.log("      - AAPL 예시: $258.52 ± 8% = $238 ~ $279");
console.log("      - 실제 변동: -2.31% vs 시뮬레이션 변동: 랜덤");

console.log("\n   C. 차트 오버플로우 문제:");
console.log("      - CSS: overflow: hidden 설정됨");
console.log("      - 차트 높이: 120px");
console.log("      - 문제: 차트 레전드/레이블이 영역 초과할 수 있음");

// 4. Y축 범위 계산 예시
function calculateYAxisRange(currentPrice, fixedRange = 0.08) {
    const yMin = currentPrice * (1 - fixedRange);
    const yMax = currentPrice * (1 + fixedRange);
    return { yMin, yMax, range: yMax - yMin };
}

console.log("\n📐 4. Y축 범위 분석:");
Object.entries(realData).forEach(([ticker, data]) => {
    const range = calculateYAxisRange(data.current_price);
    console.log(`   ${ticker}:`);
    console.log(`     현재가: $${data.current_price}`);
    console.log(`     Y축 범위: $${range.yMin.toFixed(2)} ~ $${range.yMax.toFixed(2)}`);
    console.log(`     실제 변동: ${data.change_percent}%`);
});

console.log("\n💡 5. 해결 필요 사항:");
console.log("   ✓ 실제/예측 데이터 일관성 확보");
console.log("   ✓ Y축 범위 동적 조정");
console.log("   ✓ 차트 레이아웃 최적화");
console.log("   ✓ 범위 초과 방지");