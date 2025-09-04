#!/usr/bin/env python3
import yfinance as yf
import requests
from datetime import datetime, timezone, timedelta
import json

def check_stock_prices():
    print('🔍 실제 주가 vs 대시보드 표시 주가 검증...')
    
    # 현재 시간 정보
    now = datetime.now()
    et_tz = timezone(timedelta(hours=-5))  # EST 타임존  
    et_now = now.astimezone(et_tz)
    
    print(f'\n📅 현재 시간:')
    print(f'로컬 시간: {now.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'미국 동부 시간: {et_now.strftime("%Y-%m-%d %H:%M:%S")}')
    
    # 시장 상태 확인
    weekday = et_now.weekday()  # 0=월요일, 6=일요일
    hour = et_now.hour
    is_market_open = weekday < 5 and 9 <= hour <= 16
    
    print(f'🏪 시장 상태: {"개장" if is_market_open else "폐장"}')
    if not is_market_open:
        print('⚠️ 시장 폐장 시간 - 표시되는 가격은 마지막 거래가격입니다')
    
    # 주요 종목 확인
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    print(f'\n📊 Yahoo Finance에서 가져온 주식 데이터:')
    
    api_prices = {}
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            
            # 최근 거래일 데이터
            hist = ticker.history(period='2d')
            if not hist.empty:
                latest_price = float(hist['Close'].iloc[-1])
                latest_date = hist.index[-1].strftime('%Y-%m-%d')
                volume = int(hist['Volume'].iloc[-1])
                
                api_prices[symbol] = {
                    'price': latest_price,
                    'date': latest_date,
                    'volume': volume
                }
                
                print(f'{symbol}: ${latest_price:.2f} (날짜: {latest_date}, 거래량: {volume:,})')
            else:
                print(f'{symbol}: ❌ 역사 데이터 없음')
                
        except Exception as e:
            print(f'{symbol}: ❌ 오류 - {str(e)[:50]}...')
    
    # 대시보드 API에서 가져온 데이터와 비교
    print(f'\n🌐 대시보드 API 데이터와 비교:')
    try:
        response = requests.get('http://localhost:8091/api/stocks/live', timeout=10)
        if response.ok:
            dashboard_data = response.json()
            
            for prediction in dashboard_data.get('predictions', []):
                symbol = prediction['symbol']
                dashboard_price = prediction['current_price']
                
                if symbol in api_prices:
                    api_price = api_prices[symbol]['price']
                    difference = abs(dashboard_price - api_price)
                    percentage_diff = (difference / api_price) * 100 if api_price > 0 else 0
                    
                    status = '✅' if percentage_diff < 1 else ('⚠️' if percentage_diff < 5 else '❌')
                    
                    print(f'{symbol}:')
                    print(f'  대시보드: ${dashboard_price:.2f}')
                    print(f'  Yahoo Finance: ${api_price:.2f}')
                    print(f'  차이: ${difference:.2f} ({percentage_diff:.1f}%) {status}')
                else:
                    print(f'{symbol}: 비교 데이터 없음')
        else:
            print(f'❌ 대시보드 API 호출 실패: {response.status_code}')
            
    except Exception as e:
        print(f'❌ 대시보드 API 연결 실패: {e}')
    
    # 실제 현재가와 비교 (외부 소스)
    print(f'\n🌍 실제 현재가 확인 (외부 검증):')
    print('참고: https://finance.yahoo.com 에서 직접 확인하시거나')
    print('     https://www.google.com/finance 에서 비교해보세요')
    
    # 데이터 신선도 확인
    if api_prices:
        latest_dates = [data['date'] for data in api_prices.values()]
        oldest_date = min(latest_dates)
        newest_date = max(latest_dates)
        
        print(f'\n📅 데이터 신선도:')
        print(f'가장 오래된 데이터: {oldest_date}')
        print(f'가장 최신 데이터: {newest_date}')
        
        # 오늘 날짜와 비교
        today = now.strftime('%Y-%m-%d')
        if newest_date < today:
            days_old = (now.date() - datetime.strptime(newest_date, '%Y-%m-%d').date()).days
            print(f'⚠️ 데이터가 {days_old}일 이전 것입니다')
        else:
            print('✅ 데이터가 최신입니다')
    
    print(f'\n🔚 주가 검증 완료')

if __name__ == '__main__':
    check_stock_prices()