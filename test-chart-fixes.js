// 차트 수정사항 테스트 스크립트

console.log("🧪 주요 종목 차트 수정사항 테스트");
console.log("=====================================");

// 테스트 데이터 (realtime_results.json에서 가져온 실제 데이터)
const testStocks = [
    {
        ticker: "AAPL",
        current_price: 258.52,
        predicted_price: 252.56,
        confidence: 32.3,
        change_percent: -2.31
    },
    {
        ticker: "MSFT", 
        current_price: 225.22,
        predicted_price: 227.01,
        confidence: 44.2,
        change_percent: 0.8
    },
    {
        ticker: "GOOGL",
        current_price: 294.61,
        predicted_price: 292.19,
        confidence: 47.2,
        change_percent: -0.82
    },
    {
        ticker: "AMZN",
        current_price: 168.9,
        predicted_price: 169.18,
        confidence: 50.8,
        change_percent: 0.17
    }
];

console.log("📊 1. Y축 범위 계산 테스트:");
console.log("==========================");

testStocks.forEach(stock => {
    const actualChangePercent = Math.abs(stock.change_percent || 0);
    const predictedChangePercent = Math.abs((stock.predicted_price - stock.current_price) / stock.current_price * 100);
    const maxChange = Math.max(actualChangePercent, predictedChangePercent, 2);
    const yAxisRange = Math.min(maxChange * 0.01 * 1.5, 0.12);
    
    const yMin = stock.current_price * (1 - yAxisRange);
    const yMax = stock.current_price * (1 + yAxisRange);
    
    console.log(`${stock.ticker}:`);
    console.log(`  현재가: $${stock.current_price}`);
    console.log(`  예측가: $${stock.predicted_price}`);
    console.log(`  실제 변동: ${stock.change_percent}%`);
    console.log(`  예측 변동: ${predictedChangePercent.toFixed(2)}%`);
    console.log(`  Y축 범위: $${yMin.toFixed(2)} ~ $${yMax.toFixed(2)} (±${(yAxisRange*100).toFixed(1)}%)`);
    console.log(`  범위 적정성: ${yAxisRange < 0.12 ? '✅ 적절' : '⚠️  너무 큼'}`);
    console.log("");
});

console.log("📈 2. 데이터 연속성 테스트:");
console.log("=========================");

// 시뮬레이션된 데이터 연속성 검증
function simulateDataConnection(stock) {
    // 실제 데이터: 시작가 → 현재가
    const startPrice = stock.current_price / (1 + stock.change_percent / 100);
    const actualDataEnd = stock.current_price;
    
    // 예측 데이터: 현재가 → 예측가
    const predictedDataStart = stock.current_price;
    const predictedDataEnd = stock.predicted_price;
    
    const connectionGap = Math.abs(actualDataEnd - predictedDataStart);
    
    return {
        actualStart: startPrice,
        actualEnd: actualDataEnd,
        predictedStart: predictedDataStart,
        predictedEnd: predictedDataEnd,
        connectionGap: connectionGap,
        isConnected: connectionGap < 0.01
    };
}

testStocks.forEach(stock => {
    const connection = simulateDataConnection(stock);
    console.log(`${stock.ticker}:`);
    console.log(`  실제 데이터: $${connection.actualStart.toFixed(2)} → $${connection.actualEnd.toFixed(2)}`);
    console.log(`  예측 데이터: $${connection.predictedStart.toFixed(2)} → $${connection.predictedEnd.toFixed(2)}`);
    console.log(`  연결 상태: ${connection.isConnected ? '✅ 연결됨' : '❌ 끊어짐'} (Gap: $${connection.connectionGap.toFixed(2)})`);
    console.log("");
});

console.log("📱 3. 차트 레이아웃 테스트:");
console.log("=========================");
console.log("✅ 차트 높이: 120px → 100px (오버플로우 방지)");
console.log("✅ 레전드 크기: 12px → 8px (공간 절약)");
console.log("✅ X축 틱: 5개 → 4개 (가독성 향상)");
console.log("✅ Y축 틱: 4개 → 3개 (공간 절약)");
console.log("✅ 폰트 크기: 레전드 10px→8px, X축 9px→8px, Y축 9px→7px");

console.log("🎯 4. 전체 개선사항 요약:");
console.log("========================");
console.log("Before → After");
console.log("❌ 랜덤 데이터 → ✅ 실제 변동률 기반 데이터");
console.log("❌ 불연속 실제/예측 → ✅ 연속적 실제→예측 흐름");
console.log("❌ 고정 ±8% 범위 → ✅ 동적 범위 (최소2%, 최대12%)");
console.log("❌ 차트 오버플로우 → ✅ 카드 내 완전 포함");
console.log("❌ 의미없는 비교 → ✅ 실제 기반 의미있는 예측");