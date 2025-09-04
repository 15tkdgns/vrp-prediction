import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from textblob import TextBlob
import logging
import time
from .config_manager import get_config_manager
from ..utils.yfinance_manager import get_yfinance_manager


class APIManager:
    def __init__(self):
        # ConfigManager를 통한 안전한 설정 로드
        self.config_manager = get_config_manager()
        
        self.apis = {
            "news": {
                "primary": "yahoo_rss",
                "secondary": "free_news_api", 
                "backup": "web_scraping",
            },
            "market_data": {"primary": "yfinance", "secondary": "alpha_vantage_free"},
            "sp500_data": {
                "ALPHA_VANTAGE": {
                    "base_url": "https://www.alphavantage.co/query",
                    "api_key": self.config_manager.get_api_key("ALPHA_VANTAGE"),
                },
                "TWELVE_DATA": {
                    "base_url": "https://api.twelvedata.com",
                    "api_key": self.config_manager.get_api_key("TWELVE_DATA"),
                },
                "FINNHUB": {
                    "base_url": "https://finnhub.io/api/v1",
                    "api_key": self.config_manager.get_api_key("FINNHUB"),
                },
                "MARKETAUX": {
                    "base_url": "https://api.marketaux.com/v1",
                    "api_key": self.config_manager.get_api_key("MARKETAUX"),
                },
                "POLYGON": {
                    "base_url": "https://api.polygon.io",
                    "api_key": self.config_manager.get_api_key("POLYGON"),
                },
                "FMP": {
                    "base_url": "https://financialmodelingprep.com/api/v3",
                    "api_key": self.config_manager.get_api_key("FMP"),
                },
                "IEX_CLOUD": {
                    "base_url": "https://cloud.iexapis.com/stable",
                    "api_key": self.config_manager.get_api_key("IEX_CLOUD"),
                },
                "NEWS_API": {
                    "base_url": "https://newsapi.org/v2",
                    "api_key": self.config_manager.get_api_key("NEWS_API"),
                },
            },
        }

        self.logger = logging.getLogger(__name__)
        
        # 사용 가능한 서비스 로깅
        available_services = self.config_manager.get_available_services()
        self.logger.info(f"🔑 사용 가능한 API 서비스: {', '.join(available_services)}")
        
        # API 요청 제한 설정
        self.rate_limit = self.config_manager.get_system_config('api_rate_limit', 60)
        self.last_request_time = 0

    def _respect_rate_limit(self):
        """API 요청 제한 준수"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        min_interval = 60.0 / self.rate_limit  # 분당 요청 수 기반 최소 간격
        
        if time_since_last_request < min_interval:
            sleep_time = min_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def get_news_data_marketaux(self, ticker, limit=10):
        """Marketaux API를 통한 뉴스 데이터 수집 - 안전한 오류 처리"""
        try:
            api_key = self.apis["sp500_data"]["MARKETAUX"]["api_key"]
            if not api_key:
                self.logger.warning("Marketaux API 키를 사용할 수 없습니다")
                return []
            
            # 요청 제한 준수
            self._respect_rate_limit()
            
            url = f"{self.apis['sp500_data']['MARKETAUX']['base_url']}/news/all"
            params = {
                'symbols': ticker,
                'filter_entities': 'true',
                'language': 'en',
                'api_token': api_key
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()  # HTTP 상태 에러를 예외로 발생
            
            data = response.json()
            
            # 응답 구조 검증
            if not isinstance(data, dict):
                raise ValueError(f"예상된 dict 응답이 아닙니다: {type(data)}")
            
            if data.get('meta', {}).get('found', 0) == 0:
                self.logger.info(f"Marketaux: {ticker}에 대한 뉴스를 찾을 수 없습니다")
                return []
                
            # 안전한 데이터 처리
            news_data = []
            for article in data.get('data', [])[:limit]:
                if not isinstance(article, dict):
                    continue
                    
                processed_article = self._process_news_article(article, ticker)
                if processed_article:
                    news_data.append(processed_article)
                    
            return news_data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Marketaux API 요청 실패: {e}")
            return []
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            self.logger.error(f"Marketaux 데이터 처리 실패: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Marketaux 뉴스 수집 중 예기치 못한 오류: {e}")
            return []
    
    def _process_news_article(self, article, ticker):
        """뉴스 기사 데이터를 안전하게 처리"""
        try:
            title = article.get("title", "")
            description = article.get("description", "")
            
            if not title and not description:
                return None
            
            full_text = f"{title} {description}".strip()
            
            # 감성 분석
            blob = TextBlob(full_text)
            sentiment = blob.sentiment.polarity
            
            # 감성 라벨 결정
            if sentiment > 0.1:
                sentiment_label = "positive"
            elif sentiment < -0.1:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
                
            return {
                "ticker": ticker,
                "title": title,
                "description": description,
                "url": article.get("url", ""),
                "publishedAt": article.get("published_at", ""),
                "source": article.get("source", {}).get("name", "Unknown") if isinstance(article.get("source"), dict) else "Unknown",
                "sentiment_label": sentiment_label,
                "sentiment_score": abs(sentiment),
                "polarity": sentiment,
                "text_length": len(full_text),
            }
            
        except Exception as e:
            self.logger.warning(f"기사 처리 실패: {e}")
            return None

    def get_news_data_yahoo_rss(self, ticker, limit=10):
        """Yahoo Finance RSS 뉴스 데이터 수집"""
        try:
            import feedparser

            # Yahoo Finance RSS URL
            rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"

            feed = feedparser.parse(rss_url)
            news_data = []

            for entry in feed.entries[:limit]:
                # 감성 분석 (TextBlob 사용)
                title = entry.title
                summary = entry.summary if hasattr(entry, "summary") else ""
                full_text = f"{title} {summary}"

                blob = TextBlob(full_text)
                sentiment = blob.sentiment.polarity

                # 감성 라벨 변환
                if sentiment > 0.1:
                    sentiment_label = "positive"
                elif sentiment < -0.1:
                    sentiment_label = "negative"
                else:
                    sentiment_label = "neutral"

                news_data.append(
                    {
                        "ticker": ticker,
                        "title": title,
                        "description": summary,
                        "url": entry.link,
                        "sentiment_label": sentiment_label,
                        "sentiment_score": abs(sentiment),
                        "polarity": sentiment,
                        "text_length": len(full_text),
                    }
                )

            return news_data

        except Exception as e:
            self.logger.error(f"Yahoo RSS 뉴스 수집 실패: {e}")
            return []

    def get_news_data_free_api(self, ticker, limit=10):
        """무료 뉴스 API 사용"""
        try:
            # NewsData.io 무료 API (일일 200회 제한)
            url = f"https://newsdata.io/api/1/news?apikey=FREE&q={ticker}&language=en&category=business"

            response = requests.get(url)
            data = response.json()

            if data.get("status") == "success":
                news_data = []

                for article in data.get("results", [])[:limit]:
                    # 감성 분석
                    title = article.get("title", "")
                    description = article.get("description", "")
                    full_text = f"{title} {description}"

                    blob = TextBlob(full_text)
                    sentiment = blob.sentiment.polarity

                    news_data.append(
                        {
                            "ticker": ticker,
                            "title": title,
                            "description": description,
                            "url": article.get("link", ""),
                            "publishedAt": article.get("pubDate", ""),
                            "source": article.get("source_id", "Unknown"),
                            "sentiment_label": (
                                "positive"
                                if sentiment > 0.1
                                else "negative" if sentiment < -0.1 else "neutral"
                            ),
                            "sentiment_score": abs(sentiment),
                            "polarity": sentiment,
                            "text_length": len(full_text),
                        }
                    )

                return news_data

        except Exception as e:
            self.logger.error(f"무료 뉴스 API 수집 실패: {e}")

        return []

    def get_news_data_web_scraping(self, ticker, limit=5):
        """웹 스크래핑 백업 방법"""
        try:
            from bs4 import BeautifulSoup

            # Google News 검색
            url = f"https://news.google.com/search?q={ticker}&hl=en-US&gl=US&ceid=US:en"

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, "html.parser")

            news_data = []
            articles = soup.find_all("article")[:limit]

            for article in articles:
                try:
                    title_elem = article.find("h3")
                    title = title_elem.get_text() if title_elem else "No title"

                    # 감성 분석
                    blob = TextBlob(title)
                    sentiment = blob.sentiment.polarity

                    news_data.append(
                        {
                            "ticker": ticker,
                            "title": title,
                            "description": title,  # 제목만 사용
                            "url": "",
                            "publishedAt": datetime.now().isoformat(),
                            "source": "Google News",
                            "sentiment_label": (
                                "positive"
                                if sentiment > 0.1
                                else "negative" if sentiment < -0.1 else "neutral"
                            ),
                            "sentiment_score": abs(sentiment),
                            "polarity": sentiment,
                            "text_length": len(title),
                        }
                    )

                except Exception:
                    continue

            return news_data

        except Exception as e:
            self.logger.error(f"웹 스크래핑 실패: {e}")
            return []

    def get_market_data_yfinance(self, ticker, period="1d", interval="1m"):
        """YFinance를 통한 시장 데이터 수집 (새로운 YFinanceManager 사용)"""
        try:
            yf_manager = get_yfinance_manager()
            
            # YFinanceManager를 통한 데이터 수집
            result = yf_manager.get_stock_history(ticker, period=period, interval=interval)
            
            if result['success']:
                # 성공한 경우 DataFrame으로 변환
                df = pd.DataFrame(result['data'])
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                
                self.logger.info(f"✅ YFinance를 통해 {ticker} 데이터 수집 성공 ({len(df)} 레코드)")
                return df
            else:
                # 실패한 경우 상세한 에러 정보 로깅
                error_msg = result.get('message', 'Unknown error')
                self.logger.error(f"❌ YFinance 데이터 수집 실패 ({ticker}): {error_msg}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ YFinance 데이터 수집 중 예외 발생 ({ticker}): {e}")
            return None

    def get_market_data_alpha_vantage_free(self, ticker):
        """Alpha Vantage 무료 API"""
        try:
            # 무료 API 키 (제한적)
            api_key = "demo"  # 실제로는 회원가입 필요
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={api_key}"

            response = requests.get(url)
            data = response.json()

            if "Global Quote" in data:
                quote = data["Global Quote"]

                # DataFrame 형태로 변환
                df_data = {
                    "Open": [float(quote["02. open"])],
                    "High": [float(quote["03. high"])],
                    "Low": [float(quote["04. low"])],
                    "Close": [float(quote["05. price"])],
                    "Volume": [int(quote["06. volume"])],
                }

                df = pd.DataFrame(df_data)
                df.index = [datetime.now()]

                return df

        except Exception as e:
            self.logger.error(f"Alpha Vantage 데이터 수집 실패: {e}")

        return None

    def get_news_data(self, ticker, limit=10):
        """뉴스 데이터 수집 (투명한 폴백 방식)"""
        self.logger.info(f"📰 뉴스 데이터 수집 시작: {ticker} (limit={limit})")
        
        # 1차: Marketaux API
        self.logger.debug(f"1차 시도: Marketaux API를 통한 {ticker} 뉴스 수집")
        news_data = self.get_news_data_marketaux(ticker, limit)
        if news_data:
            self.logger.info(f"✅ Marketaux API를 통해 {ticker} 뉴스 {len(news_data)}개 수집 완료")
            return news_data
        
        # 2차: Yahoo RSS
        self.logger.debug(f"2차 시도: Yahoo RSS를 통한 {ticker} 뉴스 수집")
        news_data = self.get_news_data_yahoo_rss(ticker, limit)
        if news_data:
            self.logger.info(f"✅ Yahoo RSS를 통해 {ticker} 뉴스 {len(news_data)}개 수집 완료")
            return news_data
        
        # 3차: 무료 API
        self.logger.debug(f"3차 시도: 무료 API를 통한 {ticker} 뉴스 수집")
        news_data = self.get_news_data_free_api(ticker, limit)
        if news_data:
            self.logger.info(f"✅ 무료 API를 통해 {ticker} 뉴스 {len(news_data)}개 수집 완료")
            return news_data
        
        # 4차: 웹 스크래핑
        self.logger.debug(f"4차 시도: 웹 스크래핑을 통한 {ticker} 뉴스 수집")
        news_data = self.get_news_data_web_scraping(ticker, limit)
        if news_data:
            self.logger.info(f"✅ 웹 스크래핑을 통해 {ticker} 뉴스 {len(news_data)}개 수집 완료")
            return news_data
        
        # 모든 방법 실패
        self.logger.error(f"❌ 모든 방법을 통한 {ticker} 뉴스 수집 실패")
        self.logger.error(f"   - Marketaux API: 실패 (API 키 확인 필요)")
        self.logger.error(f"   - Yahoo RSS: 실패 (네트워크 또는 RSS 피드 문제)")
        self.logger.error(f"   - 무료 API: 실패 (일일 한도 초과 가능)")
        self.logger.error(f"   - 웹 스크래핑: 실패 (사이트 접근 제한 가능)")
        self.logger.error(f"💡 권장사항: API 키 설정 확인 또는 네트워크 연결 상태 점검")
        
        return []

    def get_market_data_polygon(
        self, ticker, multiplier=1, timespan="day", from_date=None, to_date=None
    ):
        """Polygon.io API를 통한 시장 데이터 수집"""
        try:
            api_key = self.apis["sp500_data"]["POLYGON"]["api_key"]
            base_url = self.apis["sp500_data"]["POLYGON"]["base_url"]

            if from_date is None:
                from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            if to_date is None:
                to_date = datetime.now().strftime("%Y-%m-%d")

            url = f"{base_url}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"

            self.logger.info(
                f"Polygon.io API request URL for {ticker}: {url}"
            )  # Log Polygon.io URL
            response = requests.get(url)
            if not response.ok:
                self.logger.error(
                    f"Polygon.io API request failed with status {response.status_code}: {response.text}"
                )
                return None

            data = response.json()
            self.logger.info(
                f"Polygon.io API parsed JSON data for {ticker}: {data}"
            )  # Log Polygon.io JSON data

            if data.get("status") == "OK" and data.get("resultsCount", 0) > 0:
                df_data = []
                for result in data["results"]:
                    df_data.append(
                        {
                            "Date": datetime.fromtimestamp(result["t"] / 1000).strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "Open": result["o"],
                            "High": result["h"],
                            "Low": result["l"],
                            "Close": result["c"],
                            "Volume": result["v"],
                        }
                    )
                df = pd.DataFrame(df_data)
                df["Date"] = pd.to_datetime(df["Date"])
                df.reset_index(inplace=True)  # 인덱스를 컬럼으로 변환
                return df
        except Exception as e:
            self.logger.error(f"Polygon.io 데이터 수집 실패: {e}")
        return None

    def get_market_data(self, ticker, period="1d", interval="1m"):
        """시장 데이터 수집 (투명한 폴백 방식)"""
        self.logger.info(f"📊 시장 데이터 수집 시작: {ticker} (period={period}, interval={interval})")
        
        # 1차: YFinance (새로운 매니저 사용)
        self.logger.debug(f"1차 시도: YFinance를 통한 {ticker} 데이터 수집")
        data = self.get_market_data_yfinance(ticker, period, interval)
        if data is not None:
            self.logger.info(f"✅ YFinance를 통해 {ticker} 데이터 수집 완료")
            return data
        
        # 2차: Polygon.io
        self.logger.debug(f"2차 시도: Polygon.io를 통한 {ticker} 데이터 수집")
        data = self.get_market_data_polygon(ticker)
        if data is not None:
            self.logger.info(f"✅ Polygon.io를 통해 {ticker} 데이터 수집 완료")
            return data
        
        # 3차: Alpha Vantage
        self.logger.debug(f"3차 시도: Alpha Vantage를 통한 {ticker} 데이터 수집")
        data = self.get_market_data_alpha_vantage_free(ticker)
        if data is not None:
            self.logger.info(f"✅ Alpha Vantage를 통해 {ticker} 데이터 수집 완료")
            return data
        
        # 모든 방법 실패
        self.logger.error(f"❌ 모든 API를 통한 {ticker} 데이터 수집 실패")
        self.logger.error(f"   - YFinance: 실패")
        self.logger.error(f"   - Polygon.io: 실패")
        self.logger.error(f"   - Alpha Vantage: 실패")
        self.logger.error(f"💡 권장사항: API 키 설정 확인 또는 네트워크 연결 상태 점검")
        
        return None


# 의존성 설치를 위한 추가 요구사항
additional_requirements = """
feedparser>=6.0.0
beautifulsoup4>=4.11.0
requests>=2.28.0
"""

if __name__ == "__main__":
    api_manager = APIManager()

    # 테스트
    print("API 테스트 시작...")

    # 뉴스 데이터 테스트
    news = api_manager.get_news_data("AAPL", 5)
    print(f"뉴스 데이터: {len(news)}개")

    # 시장 데이터 테스트
    market = api_manager.get_market_data("AAPL")
    print(f"시장 데이터: {market is not None}")

    print("API 테스트 완료")
