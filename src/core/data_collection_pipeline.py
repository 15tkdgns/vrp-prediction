import pandas as pd
from datetime import datetime, timedelta
import os
import numpy as np
from textblob import TextBlob
import ta
from transformers import pipeline
import logging

from tqdm import tqdm
from src.core.api_config import APIManager
from src.utils.yfinance_manager import get_yfinance_manager

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SP500DataCollector:
    """
    S&P500 주식 관련 데이터를 수집, 처리, 가공하여 모델 훈련용 데이터셋을 생성하는 클래스.

    주요 기능:
    1. S&P500 종목 티커 수집
    2. 개별 종목의 주가 데이터 및 거래량 수집 (yfinance)
    3. 뉴스 데이터 수집 및 감성 분석 (NewsAPI, HuggingFace Transformers)
    4. 기술적 지표 계산 (TA-Lib)
    5. 이벤트 라벨링 (가격, 거래량, 변동성 기반)
    6. 최종 훈련용 데이터셋 생성 및 저장
    """

    def __init__(self, data_dir="data/raw"):
        """
        SP500DataCollector 인스턴스를 초기화합니다.

        Args:
            data_dir (str): 수집된 원본 데이터를 저장할 디렉토리 경로.
        """
        self.data_dir = data_dir
        self.sp500_tickers = []
        # 금융 텍스트에 특화된 FinBERT 모델 로드
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis", model="ProsusAI/finbert"
        )
        self.api_manager = APIManager()  # APIManager 인스턴스 생성

        # 데이터 저장 디렉토리 생성
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def get_sp500_tickers(self):
        """
        S&P500 구성 종목의 최신 티커 목록을 웹에서 가져와 CSV 파일로 저장합니다.
        """
        url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
        try:
            df = pd.read_csv(url)
            self.sp500_tickers = df["Symbol"].tolist()
            df.to_csv(f"{self.data_dir}/sp500_constituents.csv", index=False)
            logging.info(f"S&P500 티커 {len(self.sp500_tickers)}개 수집 완료.")
        except Exception as e:
            logging.error(f"S&P500 티커 수집 실패: {e}")


    def collect_stock_data(self, period="1y", num_tickers=10):
        """
        지정된 기간 동안의 주가 및 거래량 데이터를 수집합니다.
        실패한 경우 명확한 오류 메시지를 로깅합니다.

        Args:
            period (str): 수집할 데이터 기간 (e.g., '1y', '6mo').
            num_tickers (int): 수집할 티커의 수 (테스트 목적으로 사용).
        """
        if not self.sp500_tickers:
            self.get_sp500_tickers()

        # YFinanceManager 인스턴스 가져오기
        yf_manager = get_yfinance_manager()
        
        # 테스트를 위해 일부 티커만 사용
        tickers_to_fetch = self.sp500_tickers[:num_tickers]
        successful_collections = 0
        failed_collections = 0

        for ticker in tqdm(tickers_to_fetch, desc="Collecting stock data"):
            try:
                # 새로운 YFinanceManager를 통한 데이터 수집
                result = yf_manager.get_stock_history(ticker, period=period)
                
                if result['success']:
                    # 성공적으로 데이터를 가져온 경우
                    hist_data = result['data']
                    hist_df = pd.DataFrame(hist_data)
                    
                    # Date 컬럼이 문자열인 경우 datetime으로 변환
                    if 'Date' in hist_df.columns:
                        hist_df['Date'] = pd.to_datetime(hist_df['Date'])
                    
                    hist_df.to_csv(f"{self.data_dir}/stock_{ticker}.csv", index=False)
                    logging.info(f"✅ {ticker} 주가 데이터 저장 완료 ({len(hist_data)}개 레코드)")
                    successful_collections += 1
                    
                else:
                    # 실패한 경우 - 모의 데이터 생성하지 않고 명확한 오류 로깅
                    error_msg = result.get('message', 'Unknown error')
                    logging.error(f"❌ {ticker} 주가 데이터 수집 실패: {error_msg}")
                    failed_collections += 1
                    
                    # 실패한 티커를 별도 파일에 기록
                    self._log_failed_collection(ticker, 'stock_data', error_msg)
                    
            except Exception as e:
                logging.error(f"❌ {ticker} 데이터 수집 중 예외 발생: {e}")
                failed_collections += 1
                self._log_failed_collection(ticker, 'stock_data', str(e))
        
        # 수집 결과 요약
        total_attempted = len(tickers_to_fetch)
        success_rate = (successful_collections / total_attempted) * 100 if total_attempted > 0 else 0
        
        logging.info(f"📊 주가 데이터 수집 완료: {successful_collections}/{total_attempted} 성공 ({success_rate:.1f}%)")
        
        if failed_collections > 0:
            logging.warning(f"⚠️ {failed_collections}개 티커 데이터 수집 실패. 실패 목록은 failed_collections.log 참조")

    def calculate_technical_indicators(self, df):
        """
        주가 데이터프레임에 다양한 기술적 지표를 계산하여 추가합니다.
        """
        df_ti = df.copy()
        # 이동평균, RSI, MACD 등 기본 지표 추가
        df_ti["sma_20"] = ta.trend.sma_indicator(df_ti["Close"], window=20)
        df_ti["sma_50"] = ta.trend.sma_indicator(df_ti["Close"], window=50)
        df_ti["rsi"] = ta.momentum.rsi(df_ti["Close"], window=14)
        df_ti["macd"] = ta.trend.macd_diff(df_ti["Close"])
        # 볼린저 밴드
        df_ti["bb_upper"] = ta.volatility.bollinger_hband(df_ti["Close"])
        df_ti["bb_lower"] = ta.volatility.bollinger_lband(df_ti["Close"])
        # 변동성 및 거래량 관련 지표
        df_ti["atr"] = ta.volatility.average_true_range(
            df_ti["High"], df_ti["Low"], df_ti["Close"]
        )
        df_ti["volatility"] = df_ti["Close"].rolling(window=20).std()
        df_ti["obv"] = ta.volume.on_balance_volume(df_ti["Close"], df_ti["Volume"])

        return df_ti

    def collect_news_and_sentiment(self, num_tickers=5):
        """
        APIManager를 사용하여 뉴스 기사를 수집하고, FinBERT와 TextBlob으로 감성 분석을 수행합니다.
        실패한 경우 명확한 오류 메시지를 로깅합니다.
        """
        all_news = []
        tickers_to_fetch = self.sp500_tickers[:num_tickers]

        for ticker in tqdm(tickers_to_fetch, desc="Collecting news data"):
            articles = []
            try:
                # APIManager를 통해 뉴스 데이터 수집
                articles = self.api_manager.get_news_data(ticker)
                if not articles:  # If API returns empty or fails
                    logging.warning(f"⚠️ {ticker} 뉴스 데이터를 수집하지 못했습니다.")
                    self._log_failed_collection(ticker, 'news_data', 'No articles returned from API')
                    continue  # 다음 티커로 넘어가기
            except Exception as e:
                logging.error(f"❌ {ticker} 뉴스 처리 중 오류: {e}")
                self._log_failed_collection(ticker, 'news_data', str(e))
                continue  # 다음 티커로 넘어가기

            for article in articles:
                title = article.get("title", "")
                description = article.get("description", "")
                full_text = f"{title}. {description}"
                if not full_text.strip() or full_text == ". ":  # ". "인 경우도 필터링
                    continue

                # FinBERT 분석
                finbert_sentiment = self.sentiment_analyzer(full_text[:512])[0]
                # TextBlob 분석
                blob = TextBlob(full_text)

                all_news.append(
                    {
                        "ticker": ticker,
                        "publishedAt": article.get("publishedAt"),
                        "date": (
                            datetime.fromisoformat(
                                article.get("publishedAt").replace("Z", "+00:00")
                            ).date()
                            if article.get("publishedAt")
                            else None
                        ),
                        "title": title,
                        "finbert_label": finbert_sentiment["label"],
                        "finbert_score": finbert_sentiment["score"],
                        "textblob_polarity": blob.sentiment.polarity,
                    }
                )
            logging.info(f"{ticker} 뉴스 {len(articles)}개 수집 및 분석 완료.")

        news_df = pd.DataFrame(all_news)
        news_df.to_csv(f"{self.data_dir}/news_sentiment_data.csv", index=False)

    def _log_failed_collection(self, ticker, data_type, error_message):
        """
        실패한 데이터 수집을 별도 로그 파일에 기록합니다.
        """
        failed_log_path = os.path.join(self.data_dir, "failed_collections.log")
        timestamp = datetime.now().isoformat()
        
        with open(failed_log_path, "a", encoding="utf-8") as f:
            f.write(f"{timestamp} | {data_type} | {ticker} | {error_message}\n")
        
        logging.debug(f"실패한 수집 기록됨: {ticker} ({data_type})")

    def create_training_dataset(self, num_tickers=10):
        """
        수집된 모든 데이터를 통합하고 가공하여 최종 훈련용 데이터셋을 생성합니다.
        이 과정에는 기술적 지표 추가, 뉴스 감성 데이터 병합, 이벤트 라벨링이 포함됩니다.
        """
        all_features = []
        all_labels = []

        tickers_to_process = self.sp500_tickers[:num_tickers]

        # 뉴스 데이터 로드
        try:
            news_df = pd.read_csv(f"{self.data_dir}/news_sentiment_data.csv")
            news_df["publishedAt"] = pd.to_datetime(news_df["publishedAt"]).dt.date
        except FileNotFoundError:
            news_df = pd.DataFrame()

        for ticker in tqdm(tickers_to_process):
            stock_file_path = f"{self.data_dir}/stock_{ticker}.csv"
            if not os.path.exists(stock_file_path):
                logging.warning(
                    f"{ticker}의 주가 데이터 파일을 찾을 수 없습니다. 이 티커는 건너뜁니다."
                )
                continue

            try:
                # 주가 데이터 로드 및 기술적 지표 계산
                stock_df = pd.read_csv(stock_file_path, parse_dates=["Date"])
                logging.info(
                    f"Columns read from CSV for {ticker}: {stock_df.columns.tolist()}"
                )

                if stock_df.empty:
                    logging.warning(
                        f"{ticker}의 주가 데이터가 비어 있습니다. 이 티커는 건너뜁니다."
                    )
                    continue

                stock_df["Date"] = pd.to_datetime(
                    stock_df["Date"], utc=True
                ).dt.tz_localize(None)
                stock_df_ti = self.calculate_technical_indicators(stock_df)
                stock_df_ti["date_key"] = stock_df_ti["Date"].dt.date

                # 뉴스 데이터와 병합
                if not news_df.empty:
                    ticker_news = news_df[news_df["ticker"] == ticker]
                    daily_sentiment = (
                        ticker_news.groupby("publishedAt")
                        .agg(
                            news_sentiment=("finbert_score", "mean"),
                            news_polarity=("textblob_polarity", "mean"),
                            news_count=("title", "count"),
                        )
                        .reset_index()
                    )
                    daily_sentiment.rename(
                        columns={"publishedAt": "date_key"}, inplace=True
                    )
                    stock_df_ti = pd.merge(
                        stock_df_ti, daily_sentiment, on="date_key", how="left"
                    )

                # 이벤트 라벨 생성
                stock_df_ti["price_change"] = stock_df_ti["Close"].pct_change()
                stock_df_ti["volume_change"] = stock_df_ti["Volume"].pct_change()
                stock_df_ti["unusual_volume"] = (
                    stock_df_ti["Volume"]
                    > stock_df_ti["Volume"].rolling(window=20).mean() * 2
                ).astype(int)
                stock_df_ti["price_spike"] = (
                    abs(stock_df_ti["price_change"]) > 0.05
                ).astype(int)

                # 주요 이벤트 정의: 가격 스파이크 또는 이례적 거래량 발생 시
                stock_df_ti["major_event"] = (
                    (stock_df_ti["price_spike"] == 1)
                    | (stock_df_ti["unusual_volume"] == 1)
                ).astype(int)

                # 데이터 정리
                stock_df_ti.fillna(0, inplace=True)
                stock_df_ti["ticker"] = ticker

                # 특성과 라벨 분리
                feature_cols = [
                    "ticker",
                    "Date",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                    "sma_20",
                    "sma_50",
                    "rsi",
                    "macd",
                    "bb_upper",
                    "bb_lower",
                    "atr",
                    "volatility",
                    "obv",
                    "price_change",
                    "volume_change",
                    "unusual_volume",
                    "price_spike",
                    "news_sentiment",
                    "news_polarity",
                    "news_count",
                ]
                label_cols = [
                    "ticker",
                    "Date",
                    "major_event",
                    "price_spike",
                    "unusual_volume",
                ]

                # news 관련 컬럼이 없는 경우를 대비
                for col in ["news_sentiment", "news_polarity", "news_count"]:
                    if col not in stock_df_ti:
                        stock_df_ti[col] = 0

                all_features.append(stock_df_ti[feature_cols])
                all_labels.append(stock_df_ti[label_cols])

            except Exception as e:
                logging.error(f"{ticker} 데이터셋 생성 중 오류: {e}")

        # 최종 데이터프레임 생성 및 저장
        if all_features:
            features_df = pd.concat(all_features, ignore_index=True)
            labels_df = pd.concat(all_labels, ignore_index=True)

            features_df.to_csv(f"{self.data_dir}/training_features.csv", index=False)
            labels_df.to_csv(f"{self.data_dir}/event_labels.csv", index=False)
            logging.info("최종 훈련용 특성 및 라벨 파일 생성 완료.")
        else:
            logging.warning(
                "수집된 데이터가 없어 훈련용 특성 및 라벨 파일을 생성하지 않습니다."
            )


if __name__ == "__main__":
    # --- 데이터 수집 파이프라인 실행 ---
    collector = SP500DataCollector()

    # 1. S&P500 티커 목록 가져오기
    collector.get_sp500_tickers()

    # 2. 주가 데이터 수집
    collector.collect_stock_data(num_tickers=10)  # 모든 종목 수집

    # 3. 뉴스 데이터 수집 및 감성 분석
    collector.collect_news_and_sentiment(num_tickers=10)

    # 4. 모든 데이터를 통합하여 최종 훈련 데이터셋 생성
    collector.create_training_dataset(num_tickers=10)
