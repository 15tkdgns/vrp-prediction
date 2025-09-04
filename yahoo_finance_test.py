#!/usr/bin/env python3
import yfinance as yf
import requests
from datetime import datetime
import time

print('🔍 Yahoo Finance API 상세 진단...')

# 1. 네트워크 연결 테스트
print('\n1. 📡 Yahoo Finance 서버 연결 테스트:')
try:
    response = requests.get('https://finance.yahoo.com', timeout=10)
    print(f'✅ Yahoo Finance 웹사이트 접근: {response.status_code}')
except Exception as e:
    print(f'❌ Yahoo Finance 웹사이트 접근 실패: {e}')

# 2. yfinance API 응답 시간 테스트
print('\n2. ⏱️ API 응답 시간 테스트:')
symbols = ['AAPL', 'MSFT', 'GOOGL']
for symbol in symbols:
    start_time = time.time()
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='1d')
        end_time = time.time()
        response_time = end_time - start_time
        
        if not hist.empty:
            price = hist['Close'].iloc[-1]
            print(f'✅ {symbol}: ${price:.2f} (응답시간: {response_time:.2f}초)')
        else:
            print(f'⚠️ {symbol}: 빈 데이터 (응답시간: {response_time:.2f}초)')
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        print(f'❌ {symbol}: 오류 - {str(e)[:50]}... (응답시간: {response_time:.2f}초)')

# 3. S&P 500 상세 정보
print('\n3. 📊 S&P 500 상세 정보:')
try:
    spy = yf.Ticker('^GSPC')
    hist = spy.history(period='5d')
    print(f'✅ 최근 5일 데이터 수: {len(hist)}개')
    if not hist.empty:
        latest = hist.iloc[-1]
        print(f'✅ 최신 종가: ${latest["Close"]:.2f}')
        print(f'✅ 최신 거래량: {latest["Volume"]:,.0f}')
        print(f'✅ 데이터 날짜: {hist.index[-1]}')
    
    # 실시간 정보 시도
    try:
        info = spy.info
        if info:
            print(f'✅ 실시간 정보 획득 가능')
            if 'regularMarketPrice' in info:
                print(f'✅ 실시간 가격: ${info["regularMarketPrice"]:.2f}')
        else:
            print('⚠️ 실시간 정보 없음')
    except Exception as e:
        print(f'⚠️ 실시간 정보 오류: {e}')
        
except Exception as e:
    print(f'❌ S&P 500 정보 오류: {e}')

# 4. API 제한/레이트 리미트 테스트
print('\n4. 🚦 API 레이트 리미트 테스트:')
try:
    print('연속 요청 테스트 중...')
    for i in range(5):
        start_time = time.time()
        ticker = yf.Ticker('AAPL')
        hist = ticker.history(period='1d')
        end_time = time.time()
        print(f'요청 {i+1}: {end_time - start_time:.2f}초')
        time.sleep(0.5)  # 0.5초 대기
except Exception as e:
    print(f'❌ 레이트 리미트 테스트 오류: {e}')

# 5. 시장 상태 확인
print('\n5. 🏪 시장 상태 확인:')
try:
    now = datetime.now()
    print(f'현재 시간: {now}')
    
    # 시장 시간 확인 (미국 동부 시간 기준)
    from datetime import timezone, timedelta
    et = timezone(timedelta(hours=-5))  # EST (동계) 기준
    et_now = now.astimezone(et)
    print(f'미국 동부 시간: {et_now}')
    
    weekday = et_now.weekday()  # 0=월요일, 6=일요일
    hour = et_now.hour
    
    if weekday < 5 and 9 <= hour <= 16:  # 평일 9:30AM-4PM EST
        print('✅ 시장 개장 시간')
    else:
        print('⚠️ 시장 폐장 시간 - 데이터가 오래될 수 있음')
        
except Exception as e:
    print(f'❌ 시장 상태 확인 오류: {e}')

print('\n🔚 Yahoo Finance API 진단 완료')