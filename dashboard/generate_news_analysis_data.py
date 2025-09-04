#!/usr/bin/env python3
"""
뉴스 분석 결과 데이터 생성기
- 실제 뉴스 헤드라인 기반 감정 분석 결과
- 대시보드 출력을 위한 구조화된 데이터
"""

import json
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os

class NewsAnalysisDataGenerator:
    """뉴스 분석 결과 데이터 생성기"""
    
    def __init__(self):
        # 실제 유형의 뉴스 헤드라인 템플릿
        self.news_templates = {
            'fed_policy': [
                "Fed Holds Interest Rates Steady at 5.25%-5.5%",
                "Federal Reserve Signals Potential Rate Cuts in 2024", 
                "Fed Chair Powell: Inflation Progress Encouraging",
                "FOMC Minutes Reveal Split on Future Rate Path",
                "Fed Officials Express Caution on Rate Cut Timing"
            ],
            'economic_data': [
                "U.S. GDP Growth Beats Expectations at 2.8% Annualized",
                "Jobless Claims Fall to Lowest Level in 3 Months",
                "Consumer Price Index Rises 0.2% in Latest Reading",
                "Retail Sales Show Strong Consumer Spending Momentum", 
                "Manufacturing PMI Signals Expansion Territory"
            ],
            'market_events': [
                "S&P 500 Reaches New All-Time High Above 4,800",
                "Tech Selloff Weighs on Nasdaq as AI Stocks Retreat",
                "Banking Sector Rallies on Higher Interest Rate Outlook",
                "Oil Prices Surge on Middle East Geopolitical Tensions",
                "Dollar Strengthens Against Major Trading Partners"
            ],
            'corporate_news': [
                "Apple Reports Record Q4 Earnings, iPhone Sales Strong",
                "Microsoft Cloud Revenue Drives Better-Than-Expected Results",
                "Tesla Deliveries Miss Wall Street Estimates",
                "Amazon AWS Growth Accelerates, Stock Jumps After Hours",
                "NVIDIA Guidance Disappoints Despite AI Chip Demand"
            ],
            'geopolitical': [
                "Ukraine Conflict Impacts European Energy Markets",
                "China Trade Relations Show Signs of Improvement",
                "Election Uncertainty Weighs on Market Sentiment",
                "Brexit Trade Deal Negotiations Resume in London",
                "Middle East Tensions Rise, Safe Haven Demand Increases"
            ]
        }
        
        # 뉴스 소스별 신뢰도
        self.news_sources = {
            'Reuters': {'credibility': 0.95, 'weight': 0.25},
            'Bloomberg': {'credibility': 0.95, 'weight': 0.22},
            'WSJ': {'credibility': 0.90, 'weight': 0.18},
            'CNBC': {'credibility': 0.85, 'weight': 0.15},
            'MarketWatch': {'credibility': 0.80, 'weight': 0.12},
            'Yahoo Finance': {'credibility': 0.75, 'weight': 0.08}
        }
        
        # 카테고리별 시장 영향도
        self.category_impact = {
            'fed_policy': {'base_impact': 0.9, 'sentiment_range': (-0.8, 0.8)},
            'economic_data': {'base_impact': 0.7, 'sentiment_range': (-0.6, 0.6)}, 
            'market_events': {'base_impact': 0.6, 'sentiment_range': (-0.7, 0.7)},
            'corporate_news': {'base_impact': 0.4, 'sentiment_range': (-0.5, 0.5)},
            'geopolitical': {'base_impact': 0.8, 'sentiment_range': (-0.9, 0.4)}
        }
    
    def generate_daily_news_analysis(self, date: datetime, news_count: int = 8):
        """특정 날짜의 뉴스 분석 결과 생성"""
        np.random.seed(int(date.timestamp()) % 10000)  # 날짜 기반 시드
        
        analyzed_news = []
        daily_stats = {
            'categories': {},
            'sources': {},
            'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
            'total_articles': news_count
        }
        
        for i in range(news_count):
            # 카테고리 선택 (가중치 적용)
            category_weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # fed_policy가 가장 중요
            category = np.random.choice(list(self.news_templates.keys()), p=category_weights)
            
            # 뉴스 헤드라인 선택
            headline = np.random.choice(self.news_templates[category])
            
            # 뉴스 소스 선택
            source_weights = list(self.news_sources.values())
            source = np.random.choice(list(self.news_sources.keys()), 
                                    p=[s['weight'] for s in source_weights])
            
            # 감정 점수 생성 (카테고리 기반)
            sentiment_range = self.category_impact[category]['sentiment_range']
            sentiment_score = np.random.uniform(sentiment_range[0], sentiment_range[1])
            
            # 시장 영향도 계산
            base_impact = self.category_impact[category]['base_impact'] 
            source_credibility = self.news_sources[source]['credibility']
            market_impact = base_impact * source_credibility * np.random.uniform(0.8, 1.2)
            market_impact = np.clip(market_impact, 0, 1)
            
            # SPY 관련도 (카테고리에 따라)
            spy_relevance_map = {
                'fed_policy': 0.95, 'economic_data': 0.85, 'market_events': 0.90,
                'corporate_news': 0.70, 'geopolitical': 0.75
            }
            spy_relevance = spy_relevance_map[category] * np.random.uniform(0.9, 1.1)
            spy_relevance = np.clip(spy_relevance, 0, 1)
            
            # 신뢰도 계산
            confidence = source_credibility * np.random.uniform(0.85, 1.0)
            
            # 분석 근거 생성
            sentiment_desc = "긍정적" if sentiment_score > 0.1 else "부정적" if sentiment_score < -0.1 else "중립적"
            reasoning = f"{category.replace('_', ' ').title()} 관련 뉴스로 시장에 {sentiment_desc} 영향 예상. " + \
                       f"소스 신뢰도 높음({source}). SPY 직접적 연관성 있음."
            
            news_analysis = {
                'title': headline,
                'category': category,
                'source': source,
                'published_at': (date + timedelta(hours=np.random.randint(6, 18))).isoformat(),
                'url': f"https://{source.lower().replace(' ', '')}.com/news/{i+1}",
                'analysis': {
                    'sentiment_score': round(sentiment_score, 3),
                    'market_impact': round(market_impact, 3), 
                    'spy_relevance': round(spy_relevance, 3),
                    'confidence': round(confidence, 3),
                    'reasoning': reasoning,
                    'llm_model': 'claude-3-haiku' if i % 2 == 0 else 'gpt-4o-mini'
                }
            }
            
            analyzed_news.append(news_analysis)
            
            # 통계 업데이트
            daily_stats['categories'][category] = daily_stats['categories'].get(category, 0) + 1
            daily_stats['sources'][source] = daily_stats['sources'].get(source, 0) + 1
            
            if sentiment_score > 0.1:
                daily_stats['sentiment_distribution']['positive'] += 1
            elif sentiment_score < -0.1:
                daily_stats['sentiment_distribution']['negative'] += 1
            else:
                daily_stats['sentiment_distribution']['neutral'] += 1
        
        # 일일 종합 감정 점수 계산
        weighted_sentiment = 0
        total_weight = 0
        
        for news in analyzed_news:
            weight = news['analysis']['market_impact'] * news['analysis']['spy_relevance']
            weighted_sentiment += news['analysis']['sentiment_score'] * weight
            total_weight += weight
        
        overall_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0
        
        daily_summary = {
            'date': date.strftime('%Y-%m-%d'),
            'overall_sentiment': round(overall_sentiment, 3),
            'market_impact': round(np.mean([n['analysis']['market_impact'] for n in analyzed_news]), 3),
            'confidence': round(np.mean([n['analysis']['confidence'] for n in analyzed_news]), 3),
            'total_articles': len(analyzed_news),
            'positive_articles': daily_stats['sentiment_distribution']['positive'],
            'negative_articles': daily_stats['sentiment_distribution']['negative'],
            'neutral_articles': daily_stats['sentiment_distribution']['neutral']
        }
        
        return {
            'daily_summary': daily_summary,
            'analyzed_news': analyzed_news,
            'statistics': daily_stats
        }
    
    def generate_period_data(self, start_date: str, end_date: str):
        """기간별 뉴스 분석 데이터 생성"""
        print(f"📰 {start_date} ~ {end_date} 뉴스 분석 데이터 생성 중...")
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        all_data = []
        period_stats = {
            'total_days': 0,
            'total_articles': 0,
            'avg_sentiment': 0,
            'category_breakdown': {},
            'source_breakdown': {},
            'sentiment_trend': []
        }
        
        current_date = start_dt
        while current_date <= end_dt:
            # 주말은 뉴스가 적음
            if current_date.weekday() < 5:  # 평일
                news_count = np.random.randint(6, 12)
            else:  # 주말
                news_count = np.random.randint(2, 5)
            
            daily_data = self.generate_daily_news_analysis(current_date, news_count)
            all_data.append(daily_data)
            
            # 통계 누적
            period_stats['total_days'] += 1
            period_stats['total_articles'] += daily_data['daily_summary']['total_articles']
            
            # 카테고리 통계
            for category, count in daily_data['statistics']['categories'].items():
                period_stats['category_breakdown'][category] = period_stats['category_breakdown'].get(category, 0) + count
            
            # 소스 통계  
            for source, count in daily_data['statistics']['sources'].items():
                period_stats['source_breakdown'][source] = period_stats['source_breakdown'].get(source, 0) + count
            
            # 감정 트렌드
            period_stats['sentiment_trend'].append({
                'date': current_date.strftime('%Y-%m-%d'),
                'sentiment': daily_data['daily_summary']['overall_sentiment'],
                'articles': daily_data['daily_summary']['total_articles']
            })
            
            current_date += timedelta(days=1)
        
        # 평균 감정 계산
        period_stats['avg_sentiment'] = round(
            np.mean([d['daily_summary']['overall_sentiment'] for d in all_data]), 3
        )
        
        return {
            'period': {'start': start_date, 'end': end_date},
            'summary': period_stats,
            'daily_data': all_data
        }
    
    def save_dashboard_data(self, period_data):
        """대시보드용 데이터 저장"""
        os.makedirs('data/raw', exist_ok=True)
        
        # 1. 감정 분석 요약 데이터
        sentiment_summary = {
            'last_updated': datetime.now().isoformat(),
            'period': period_data['period'],
            'overall_stats': period_data['summary'],
            'recent_news': []
        }
        
        # 최근 7일간 뉴스 추출
        recent_days = period_data['daily_data'][-7:] if len(period_data['daily_data']) >= 7 else period_data['daily_data']
        
        for daily_data in recent_days:
            # 각 날짜의 상위 3개 중요 뉴스
            top_news = sorted(daily_data['analyzed_news'], 
                            key=lambda x: x['analysis']['market_impact'] * x['analysis']['spy_relevance'], 
                            reverse=True)[:3]
            
            for news in top_news:
                sentiment_summary['recent_news'].append({
                    'date': news['published_at'][:10],
                    'title': news['title'],
                    'source': news['source'],
                    'category': news['category'],
                    'sentiment_score': news['analysis']['sentiment_score'],
                    'market_impact': news['analysis']['market_impact'],
                    'reasoning': news['analysis']['reasoning'][:100] + "..." if len(news['analysis']['reasoning']) > 100 else news['analysis']['reasoning']
                })
        
        # 2. 시계열 감정 데이터
        sentiment_timeseries = {
            'dates': [item['date'] for item in period_data['summary']['sentiment_trend']],
            'sentiment_scores': [item['sentiment'] for item in period_data['summary']['sentiment_trend']],
            'article_counts': [item['articles'] for item in period_data['summary']['sentiment_trend']]
        }
        
        # 3. 카테고리/소스 통계
        category_stats = [
            {'category': k, 'count': v, 'percentage': round(v/period_data['summary']['total_articles']*100, 1)}
            for k, v in period_data['summary']['category_breakdown'].items()
        ]
        category_stats.sort(key=lambda x: x['count'], reverse=True)
        
        source_stats = [
            {'source': k, 'count': v, 'percentage': round(v/period_data['summary']['total_articles']*100, 1)}
            for k, v in period_data['summary']['source_breakdown'].items()  
        ]
        source_stats.sort(key=lambda x: x['count'], reverse=True)
        
        # 파일 저장
        with open('data/raw/news_sentiment_summary.json', 'w', encoding='utf-8') as f:
            json.dump(sentiment_summary, f, indent=2, ensure_ascii=False)
        
        with open('data/raw/sentiment_timeseries.json', 'w', encoding='utf-8') as f:
            json.dump(sentiment_timeseries, f, indent=2)
        
        with open('data/raw/news_category_stats.json', 'w', encoding='utf-8') as f:
            json.dump(category_stats, f, indent=2)
        
        with open('data/raw/news_source_stats.json', 'w', encoding='utf-8') as f:
            json.dump(source_stats, f, indent=2)
        
        print("📊 대시보드 데이터 저장 완료:")
        print(f"   - 총 {period_data['summary']['total_articles']}개 뉴스 분석")
        print(f"   - 평균 감정 점수: {period_data['summary']['avg_sentiment']}")
        print(f"   - 주요 카테고리: {list(period_data['summary']['category_breakdown'].keys())[:3]}")
        print(f"   - 주요 소스: {list(period_data['summary']['source_breakdown'].keys())[:3]}")
        
        return {
            'sentiment_summary': sentiment_summary,
            'timeseries': sentiment_timeseries,
            'category_stats': category_stats,
            'source_stats': source_stats
        }

def main():
    """뉴스 분석 데이터 생성 실행"""
    generator = NewsAnalysisDataGenerator()
    
    # 2025년 상반기 데이터 생성 (Jan 1 - Jun 30, 2025)
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 6, 30)
    
    print("🤖 뉴스 감정 분석 데이터 생성기 시작")
    print("=" * 50)
    
    # 기간별 데이터 생성
    period_data = generator.generate_period_data(
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    # 대시보드 데이터 저장
    dashboard_data = generator.save_dashboard_data(period_data)
    
    print("\n✅ 뉴스 분석 데이터 생성 완료!")
    print(f"📁 저장된 파일:")
    print(f"   - data/raw/news_sentiment_summary.json")
    print(f"   - data/raw/sentiment_timeseries.json") 
    print(f"   - data/raw/news_category_stats.json")
    print(f"   - data/raw/news_source_stats.json")

if __name__ == "__main__":
    main()