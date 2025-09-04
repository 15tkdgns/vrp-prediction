#!/usr/bin/env python3
"""
LLM 기반 뉴스 감정 분석기
- Claude와 GPT를 활용한 다층 감정 분석
- SPY/시장 영향도 평가
- 배치 처리 및 비용 최적화
"""

import os
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
import anthropic
import openai
from news_data_collector import NewsArticle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SentimentAnalysis:
    """감정 분석 결과"""
    sentiment_score: float  # -1.0 to +1.0
    market_impact: float    # 0.0 to 1.0
    spy_relevance: float    # 0.0 to 1.0
    confidence: float       # 0.0 to 1.0
    reasoning: str
    analysis_time: datetime
    llm_model: str

class LLMSentimentAnalyzer:
    """
    다중 LLM 기반 감정 분석 시스템
    """
    
    def __init__(self):
        # API 키 로드
        self.anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        self.openai_key = os.getenv('OPENAI_API_KEY')
        
        # API 클라이언트 초기화
        if self.anthropic_key:
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_key)
        else:
            self.anthropic_client = None
            
        if self.openai_key:
            openai.api_key = self.openai_key
            self.openai_client = openai
        else:
            self.openai_client = None
        
        # 모델 설정
        self.models = {
            'claude': {
                'model': 'claude-3-haiku-20240307',
                'cost_per_token': 0.00025,  # 입력 토큰당 비용
                'max_tokens': 4000,
                'temperature': 0.3
            },
            'gpt': {
                'model': 'gpt-4o-mini',
                'cost_per_token': 0.00015,
                'max_tokens': 4000, 
                'temperature': 0.3
            }
        }
        
        # 프롬프트 템플릿
        self.sentiment_prompt = self._create_sentiment_prompt()
        
    def _create_sentiment_prompt(self) -> str:
        """감정 분석 프롬프트 생성"""
        return """
다음 뉴스 기사를 분석하여 S&P 500 지수(SPY ETF)에 대한 영향을 정확히 평가해주세요.

뉴스 제목: {title}
뉴스 요약: {description}
발행일: {date}
뉴스 소스: {source}

다음 기준으로 신중하게 분석해주세요:

1. 감정 점수 (-1.0 ~ +1.0):
   - +1.0: 매우 긍정적 (강한 상승 요인)
   - 0.0: 중립적 (영향 없음)
   - -1.0: 매우 부정적 (강한 하락 요인)

2. 시장 영향도 (0.0 ~ 1.0):
   - 1.0: 큰 영향 (시장 전체를 움직일 수 있음)
   - 0.5: 보통 영향 (일부 반응 예상)
   - 0.0: 영향 없음 (시장 무관심)

3. SPY 관련도 (0.0 ~ 1.0):
   - 1.0: 직접 관련 (S&P 500 전체에 영향)
   - 0.5: 간접 관련 (주요 섹터에 영향)
   - 0.0: 무관 (SPY에 영향 없음)

4. 신뢰도 (0.0 ~ 1.0):
   - 분석의 확실성 정도
   - 뉴스의 명확성과 신뢰성 고려

중요 고려사항:
- Fed 금리, 인플레이션, GDP 등 거시경제 지표는 높은 영향도
- 개별 기업 뉴스는 해당 기업이 SPY에서 차지하는 비중 고려
- 지정학적 리스크, 전쟁, 팬데믹 등은 높은 영향도
- 단순 루머나 추측성 기사는 낮은 신뢰도

JSON 형태로만 응답해주세요:
{{
    "sentiment_score": float,
    "market_impact": float, 
    "spy_relevance": float,
    "confidence": float,
    "reasoning": "분석 근거를 2-3문장으로 설명"
}}
"""
    
    async def analyze_with_claude(self, article: NewsArticle) -> Optional[SentimentAnalysis]:
        """Claude를 사용한 감정 분석"""
        if not self.anthropic_client:
            logger.warning("Claude API 키가 없습니다.")
            return None
        
        try:
            # 프롬프트 구성
            prompt = self.sentiment_prompt.format(
                title=article.title,
                description=article.description,
                date=article.published_at.strftime('%Y-%m-%d'),
                source=article.source
            )
            
            # Claude API 호출
            response = self.anthropic_client.messages.create(
                model=self.models['claude']['model'],
                max_tokens=self.models['claude']['max_tokens'],
                temperature=self.models['claude']['temperature'],
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # 응답 파싱
            content = response.content[0].text
            
            # JSON 파싱 시도
            try:
                result = json.loads(content)
                
                return SentimentAnalysis(
                    sentiment_score=float(result['sentiment_score']),
                    market_impact=float(result['market_impact']),
                    spy_relevance=float(result['spy_relevance']),
                    confidence=float(result['confidence']),
                    reasoning=result['reasoning'],
                    analysis_time=datetime.now(),
                    llm_model='claude-3-haiku'
                )
                
            except json.JSONDecodeError:
                logger.error(f"Claude 응답 JSON 파싱 실패: {content[:200]}...")
                return None
                
        except Exception as e:
            logger.error(f"Claude 분석 실패: {str(e)}")
            return None
    
    async def analyze_with_gpt(self, article: NewsArticle) -> Optional[SentimentAnalysis]:
        """GPT를 사용한 감정 분석"""
        if not self.openai_client:
            logger.warning("OpenAI API 키가 없습니다.")
            return None
        
        try:
            # 프롬프트 구성
            prompt = self.sentiment_prompt.format(
                title=article.title,
                description=article.description,
                date=article.published_at.strftime('%Y-%m-%d'),
                source=article.source
            )
            
            # GPT API 호출
            response = await self.openai_client.ChatCompletion.acreate(
                model=self.models['gpt']['model'],
                messages=[
                    {"role": "system", "content": "당신은 금융 뉴스 감정 분석 전문가입니다. 정확하고 객관적인 분석을 제공해주세요."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.models['gpt']['max_tokens'],
                temperature=self.models['gpt']['temperature']
            )
            
            # 응답 파싱
            content = response.choices[0].message.content
            
            # JSON 파싱 시도
            try:
                result = json.loads(content)
                
                return SentimentAnalysis(
                    sentiment_score=float(result['sentiment_score']),
                    market_impact=float(result['market_impact']),
                    spy_relevance=float(result['spy_relevance']),
                    confidence=float(result['confidence']),
                    reasoning=result['reasoning'],
                    analysis_time=datetime.now(),
                    llm_model='gpt-4o-mini'
                )
                
            except json.JSONDecodeError:
                logger.error(f"GPT 응답 JSON 파싱 실패: {content[:200]}...")
                return None
                
        except Exception as e:
            logger.error(f"GPT 분석 실패: {str(e)}")
            return None
    
    def ensemble_analysis(self, claude_result: Optional[SentimentAnalysis], 
                         gpt_result: Optional[SentimentAnalysis]) -> Optional[SentimentAnalysis]:
        """
        두 LLM 결과를 앙상블하여 최종 분석 생성
        """
        if not claude_result and not gpt_result:
            return None
        
        if claude_result and not gpt_result:
            return claude_result
        
        if gpt_result and not claude_result:
            return gpt_result
        
        # 두 결과 모두 있을 때 가중평균
        # 신뢰도 기반 가중치
        claude_weight = claude_result.confidence
        gpt_weight = gpt_result.confidence
        total_weight = claude_weight + gpt_weight
        
        if total_weight == 0:
            return claude_result  # 기본값
        
        # 가중평균 계산
        sentiment_score = (claude_result.sentiment_score * claude_weight + 
                          gpt_result.sentiment_score * gpt_weight) / total_weight
        
        market_impact = (claude_result.market_impact * claude_weight + 
                        gpt_result.market_impact * gpt_weight) / total_weight
        
        spy_relevance = (claude_result.spy_relevance * claude_weight + 
                        gpt_result.spy_relevance * gpt_weight) / total_weight
        
        confidence = max(claude_result.confidence, gpt_result.confidence)  # 높은 신뢰도 선택
        
        # 합성 reasoning
        reasoning = f"Claude: {claude_result.reasoning[:100]}... | GPT: {gpt_result.reasoning[:100]}..."
        
        return SentimentAnalysis(
            sentiment_score=sentiment_score,
            market_impact=market_impact,
            spy_relevance=spy_relevance,
            confidence=confidence,
            reasoning=reasoning,
            analysis_time=datetime.now(),
            llm_model='ensemble'
        )
    
    async def analyze_single_article(self, article: NewsArticle) -> Optional[SentimentAnalysis]:
        """단일 기사 감정 분석"""
        logger.info(f"분석 중: {article.title[:50]}...")
        
        # 병렬로 두 모델 실행 (비용 절약을 위해 선택적)
        tasks = []
        
        if self.anthropic_client:
            tasks.append(self.analyze_with_claude(article))
        
        # GPT는 비용이 더 높으므로 Claude가 없거나 중요한 뉴스일 때만
        if self.openai_client and (not self.anthropic_client or article.spy_relevance > 0.7):
            tasks.append(self.analyze_with_gpt(article))
        
        if not tasks:
            logger.error("사용 가능한 LLM API가 없습니다.")
            return None
        
        # 병렬 실행
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 분리
        claude_result = None
        gpt_result = None
        
        for i, result in enumerate(results):
            if isinstance(result, SentimentAnalysis):
                if len(tasks) == 1 or i == 0:  # Claude 결과
                    claude_result = result
                else:  # GPT 결과
                    gpt_result = result
            else:
                logger.error(f"LLM 분석 오류: {str(result)}")
        
        # 앙상블 결과 생성
        return self.ensemble_analysis(claude_result, gpt_result)
    
    async def analyze_batch(self, articles: List[NewsArticle], 
                          max_concurrent: int = 5) -> Dict[str, SentimentAnalysis]:
        """뉴스 배치 감정 분석"""
        logger.info(f"📊 {len(articles)}개 뉴스 배치 감정 분석 시작")
        
        # 동시 실행 제한
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_with_semaphore(article):
            async with semaphore:
                return await self.analyze_single_article(article)
        
        # 병렬 처리
        tasks = [analyze_with_semaphore(article) for article in articles]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 정리
        sentiment_results = {}
        successful_analyses = 0
        
        for article, result in zip(articles, results):
            if isinstance(result, SentimentAnalysis):
                # URL을 키로 사용
                sentiment_results[article.url] = result
                successful_analyses += 1
            else:
                logger.error(f"배치 분석 실패 ({article.title[:30]}...): {str(result)}")
        
        logger.info(f"✅ 배치 분석 완료: {successful_analyses}/{len(articles)} 성공")
        return sentiment_results
    
    def calculate_daily_sentiment_score(self, sentiment_results: Dict[str, SentimentAnalysis]) -> Dict[str, float]:
        """
        일일 종합 감정 점수 계산
        """
        if not sentiment_results:
            return {
                'overall_sentiment': 0.0,
                'market_impact': 0.0,
                'confidence': 0.0,
                'total_articles': 0
            }
        
        analyses = list(sentiment_results.values())
        
        # 가중평균 계산 (market_impact * spy_relevance로 가중치)
        weighted_sentiment = 0.0
        weighted_impact = 0.0
        total_weight = 0.0
        confidences = []
        
        for analysis in analyses:
            weight = analysis.market_impact * analysis.spy_relevance
            weighted_sentiment += analysis.sentiment_score * weight
            weighted_impact += analysis.market_impact * weight
            total_weight += weight
            confidences.append(analysis.confidence)
        
        if total_weight == 0:
            return {
                'overall_sentiment': 0.0,
                'market_impact': 0.0,
                'confidence': 0.0,
                'total_articles': len(analyses)
            }
        
        return {
            'overall_sentiment': weighted_sentiment / total_weight,
            'market_impact': weighted_impact / total_weight,
            'confidence': sum(confidences) / len(confidences),
            'total_articles': len(analyses),
            'positive_articles': len([a for a in analyses if a.sentiment_score > 0.1]),
            'negative_articles': len([a for a in analyses if a.sentiment_score < -0.1]),
            'neutral_articles': len([a for a in analyses if -0.1 <= a.sentiment_score <= 0.1])
        }
    
    def save_sentiment_analysis(self, date: datetime, sentiment_results: Dict[str, SentimentAnalysis]):
        """감정 분석 결과 저장"""
        # 일일 종합 점수 계산
        daily_score = self.calculate_daily_sentiment_score(sentiment_results)
        
        # 저장 데이터 구성
        save_data = {
            'date': date.strftime('%Y-%m-%d'),
            'analysis_time': datetime.now().isoformat(),
            'daily_summary': daily_score,
            'individual_analyses': {}
        }
        
        # 개별 분석 결과 저장
        for url, analysis in sentiment_results.items():
            save_data['individual_analyses'][url] = {
                'sentiment_score': analysis.sentiment_score,
                'market_impact': analysis.market_impact,
                'spy_relevance': analysis.spy_relevance,
                'confidence': analysis.confidence,
                'reasoning': analysis.reasoning,
                'llm_model': analysis.llm_model,
                'analysis_time': analysis.analysis_time.isoformat()
            }
        
        # 파일 저장
        filename = f"data/raw/sentiment_analysis_{date.strftime('%Y%m%d')}.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 감정 분석 결과 저장: {filename}")
        logger.info(f"📊 일일 종합 감정 점수: {daily_score['overall_sentiment']:.3f}")

async def main():
    """감정 분석 실행 예제"""
    analyzer = LLMSentimentAnalyzer()
    
    # 예제 뉴스 (실제로는 news_data_collector에서 로드)
    sample_articles = [
        NewsArticle(
            title="Fed Cuts Interest Rates by 0.5%",
            description="The Federal Reserve announced a surprise 0.5% rate cut to combat economic slowdown.",
            content="",
            url="https://example.com/1",
            published_at=datetime.now(),
            source="Reuters",
            spy_relevance=0.9,
            market_impact_potential=0.8
        )
    ]
    
    # 감정 분석 실행
    results = await analyzer.analyze_batch(sample_articles)
    
    # 결과 저장
    analyzer.save_sentiment_analysis(datetime.now(), results)

if __name__ == "__main__":
    asyncio.run(main())