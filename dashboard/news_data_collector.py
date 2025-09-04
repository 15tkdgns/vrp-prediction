#!/usr/bin/env python3
"""
뉴스 데이터 수집 파이프라인
- News API를 통한 실시간 뉴스 수집
- SPY/S&P 500 관련 뉴스 필터링
- 감정 분석을 위한 전처리
"""

import os
import json
import requests
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import logging
from dataclasses import dataclass

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    title: str
    description: str
    content: str
    url: str
    published_at: datetime
    source: str
    spy_relevance: float = 0.0
    market_impact_potential: float = 0.0

class NewsDataCollector:
    """
    다중 소스 뉴스 데이터 수집기
    """
    
    def __init__(self):
        # API 키들 (환경변수에서 로드)
        self.news_api_key = os.getenv('NEWS_API_KEY', 'your_news_api_key_here')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_KEY', 'your_alpha_vantage_key')
        
        # SPY/시장 관련 키워드
        self.spy_keywords = [
            'S&P 500', 'SPY ETF', 'S&P500', 'SP500',
            'broad market', 'large cap', 'market index',
            'equity market', 'stock market index'
        ]
        
        self.market_keywords = [
            'Federal Reserve', 'Fed', 'interest rates', 'inflation',
            'GDP', 'unemployment', 'earnings', 'economic growth',
            'recession', 'bull market', 'bear market',
            'market volatility', 'VIX', 'market sentiment'
        ]
        
        # 감정 관련 키워드
        self.positive_keywords = [
            'rally', 'surge', 'bullish', 'growth', 'strong', 'beat',
            'exceed', 'optimistic', 'positive', 'gains', 'rise', 'up'
        ]
        
        self.negative_keywords = [
            'crash', 'drop', 'fall', 'bearish', 'recession', 'weak',
            'miss', 'decline', 'pessimistic', 'negative', 'losses', 'down'
        ]
        
        # 뉴스 소스 신뢰도 (0-1)
        self.source_credibility = {
            'reuters': 0.95,
            'bloomberg': 0.95,
            'wsj': 0.90,
            'cnbc': 0.85,
            'marketwatch': 0.80,
            'yahoo-finance': 0.75,
            'seeking-alpha': 0.70,
            'motley-fool': 0.65
        }
    
    def calculate_spy_relevance(self, title: str, description: str) -> float:
        """
        뉴스의 SPY 관련도 계산 (0-1)
        """
        text = f"{title} {description}".lower()
        
        # 직접 관련 키워드 가중치
        spy_score = 0.0
        for keyword in self.spy_keywords:
            if keyword.lower() in text:
                spy_score += 0.3  # 직접 언급 시 높은 점수
        
        # 시장 관련 키워드 가중치
        for keyword in self.market_keywords:
            if keyword.lower() in text:
                spy_score += 0.1  # 간접 관련
        
        return min(spy_score, 1.0)  # 최대 1.0으로 제한
    
    def calculate_market_impact_potential(self, title: str, description: str, source: str) -> float:
        """
        시장 영향도 잠재력 계산 (0-1)
        """
        text = f"{title} {description}".lower()
        
        # 기본 소스 신뢰도
        impact_score = self.source_credibility.get(source.lower(), 0.5)
        
        # 강한 감정 표현 가중치
        strong_sentiment = 0.0
        for keyword in self.positive_keywords + self.negative_keywords:
            if keyword in text:
                strong_sentiment += 0.1
        
        # Fed, 경제지표 관련 추가 가중치
        high_impact_terms = ['federal reserve', 'fed rate', 'inflation data', 
                           'gdp', 'unemployment rate', 'earnings report']
        for term in high_impact_terms:
            if term in text:
                impact_score += 0.2
        
        return min(impact_score + strong_sentiment, 1.0)
    
    async def collect_news_api_data(self, date: datetime) -> List[NewsArticle]:
        """
        News API에서 뉴스 수집
        """
        logger.info(f"News API에서 {date.date()} 뉴스 수집 중...")
        
        # 키워드 쿼리 구성
        query = 'S&P 500 OR SPY OR "stock market" OR "Federal Reserve" OR "interest rates"'
        
        url = 'https://newsapi.org/v2/everything'
        params = {
            'q': query,
            'from': date.strftime('%Y-%m-%d'),
            'to': date.strftime('%Y-%m-%d'),
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': 100,
            'apiKey': self.news_api_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        articles = []\n                        \n                        for item in data.get('articles', []):\n                            if not item.get('title') or not item.get('description'):\n                                continue\n                            \n                            # SPY 관련도 계산\n                            spy_relevance = self.calculate_spy_relevance(\n                                item['title'], item['description']\n                            )\n                            \n                            # 관련도가 낮으면 스킵\n                            if spy_relevance < 0.2:\n                                continue\n                            \n                            # 시장 영향도 계산\n                            market_impact = self.calculate_market_impact_potential(\n                                item['title'], item['description'], \n                                item['source']['name']\n                            )\n                            \n                            article = NewsArticle(\n                                title=item['title'],\n                                description=item['description'] or '',\n                                content=item['content'] or '',\n                                url=item['url'],\n                                published_at=datetime.fromisoformat(item['publishedAt'].replace('Z', '+00:00')),\n                                source=item['source']['name'],\n                                spy_relevance=spy_relevance,\n                                market_impact_potential=market_impact\n                            )\n                            articles.append(article)\n                        \n                        logger.info(f\"News API에서 {len(articles)}개 관련 뉴스 수집\")\n                        return articles\n                    \n                    else:\n                        logger.error(f\"News API 오류: {response.status}\")\n                        return []\n                        \n        except Exception as e:\n            logger.error(f\"News API 수집 실패: {str(e)}\")\n            return []\n    \n    async def collect_alpha_vantage_news(self, date: datetime) -> List[NewsArticle]:\n        \"\"\"\n        Alpha Vantage News & Sentiment API에서 뉴스 수집\n        \"\"\"\n        logger.info(f\"Alpha Vantage에서 {date.date()} 뉴스 수집 중...\")\n        \n        url = 'https://www.alphavantage.co/query'\n        params = {\n            'function': 'NEWS_SENTIMENT',\n            'tickers': 'SPY',\n            'time_from': date.strftime('%Y%m%dT0000'),\n            'time_to': date.strftime('%Y%m%dT2359'),\n            'limit': 50,\n            'apikey': self.alpha_vantage_key\n        }\n        \n        try:\n            async with aiohttp.ClientSession() as session:\n                async with session.get(url, params=params) as response:\n                    if response.status == 200:\n                        data = await response.json()\n                        articles = []\n                        \n                        for item in data.get('feed', []):\n                            if not item.get('title') or not item.get('summary'):\n                                continue\n                            \n                            # SPY 관련도는 이미 SPY 티커로 필터링됨\n                            spy_relevance = 0.8  # 높은 관련도\n                            \n                            article = NewsArticle(\n                                title=item['title'],\n                                description=item['summary'],\n                                content=item.get('summary', ''),\n                                url=item['url'],\n                                published_at=datetime.fromisoformat(item['time_published'][:8] + 'T' + item['time_published'][9:]),\n                                source=item.get('source', 'Alpha Vantage'),\n                                spy_relevance=spy_relevance,\n                                market_impact_potential=float(item.get('overall_sentiment_score', 0)) if item.get('overall_sentiment_score') else 0.5\n                            )\n                            articles.append(article)\n                        \n                        logger.info(f\"Alpha Vantage에서 {len(articles)}개 뉴스 수집\")\n                        return articles\n                    \n                    else:\n                        logger.error(f\"Alpha Vantage API 오류: {response.status}\")\n                        return []\n                        \n        except Exception as e:\n            logger.error(f\"Alpha Vantage 수집 실패: {str(e)}\")\n            return []\n    \n    def filter_and_deduplicate(self, articles: List[NewsArticle]) -> List[NewsArticle]:\n        \"\"\"\n        뉴스 필터링 및 중복 제거\n        \"\"\"\n        # 제목 기반 중복 제거\n        seen_titles = set()\n        unique_articles = []\n        \n        for article in articles:\n            title_key = article.title.lower().strip()\n            if title_key not in seen_titles:\n                seen_titles.add(title_key)\n                unique_articles.append(article)\n        \n        # 관련도 및 영향도 기준 필터링\n        filtered_articles = [\n            article for article in unique_articles\n            if article.spy_relevance >= 0.3 and article.market_impact_potential >= 0.4\n        ]\n        \n        # 관련도 순으로 정렬\n        filtered_articles.sort(key=lambda x: (x.spy_relevance, x.market_impact_potential), reverse=True)\n        \n        return filtered_articles[:20]  # 상위 20개만 선택\n    \n    async def collect_daily_news(self, date: datetime = None) -> List[NewsArticle]:\n        \"\"\"\n        특정 날짜의 뉴스 수집 (메인 함수)\n        \"\"\"\n        if date is None:\n            date = datetime.now() - timedelta(days=1)  # 전날 뉴스\n        \n        logger.info(f\"📰 {date.date()} 뉴스 수집 시작\")\n        \n        # 병렬로 여러 소스에서 수집\n        tasks = [\n            self.collect_news_api_data(date),\n            self.collect_alpha_vantage_news(date)\n        ]\n        \n        results = await asyncio.gather(*tasks, return_exceptions=True)\n        \n        # 결과 통합\n        all_articles = []\n        for result in results:\n            if isinstance(result, list):\n                all_articles.extend(result)\n            else:\n                logger.error(f\"뉴스 수집 오류: {str(result)}\")\n        \n        # 필터링 및 중복 제거\n        filtered_articles = self.filter_and_deduplicate(all_articles)\n        \n        logger.info(f\"✅ 최종 {len(filtered_articles)}개 뉴스 선별 완료\")\n        return filtered_articles\n    \n    def save_news_data(self, articles: List[NewsArticle], date: datetime):\n        \"\"\"\n        수집된 뉴스 데이터를 JSON 파일로 저장\n        \"\"\"\n        # 데이터 직렬화\n        news_data = {\n            'date': date.strftime('%Y-%m-%d'),\n            'collection_time': datetime.now().isoformat(),\n            'total_articles': len(articles),\n            'articles': []\n        }\n        \n        for article in articles:\n            news_data['articles'].append({\n                'title': article.title,\n                'description': article.description,\n                'content': article.content[:500],  # 처음 500자만\n                'url': article.url,\n                'published_at': article.published_at.isoformat(),\n                'source': article.source,\n                'spy_relevance': article.spy_relevance,\n                'market_impact_potential': article.market_impact_potential\n            })\n        \n        # 파일 저장\n        filename = f\"data/raw/news_data_{date.strftime('%Y%m%d')}.json\"\n        os.makedirs(os.path.dirname(filename), exist_ok=True)\n        \n        with open(filename, 'w', encoding='utf-8') as f:\n            json.dump(news_data, f, indent=2, ensure_ascii=False)\n        \n        logger.info(f\"💾 뉴스 데이터 저장: {filename}\")\n    \n    async def run_daily_collection(self):\n        \"\"\"\n        일일 뉴스 수집 실행\n        \"\"\"\n        try:\n            # 전날 뉴스 수집 (시장 마감 후)\n            yesterday = datetime.now() - timedelta(days=1)\n            articles = await self.collect_daily_news(yesterday)\n            \n            if articles:\n                self.save_news_data(articles, yesterday)\n                \n                # 통계 출력\n                avg_relevance = sum(a.spy_relevance for a in articles) / len(articles)\n                avg_impact = sum(a.market_impact_potential for a in articles) / len(articles)\n                \n                logger.info(f\"📊 수집 통계:\")\n                logger.info(f\"   - 평균 SPY 관련도: {avg_relevance:.2f}\")\n                logger.info(f\"   - 평균 시장 영향도: {avg_impact:.2f}\")\n                logger.info(f\"   - 상위 소스: {[a.source for a in articles[:3]]}\")\n            \n            else:\n                logger.warning(\"수집된 뉴스가 없습니다.\")\n                \n        except Exception as e:\n            logger.error(f\"일일 뉴스 수집 실패: {str(e)}\")\n\ndef main():\n    \"\"\"\n    뉴스 수집 실행\n    \"\"\"\n    collector = NewsDataCollector()\n    \n    # 비동기 실행\n    asyncio.run(collector.run_daily_collection())\n\nif __name__ == \"__main__\":\n    main()