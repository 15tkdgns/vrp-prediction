import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.features.llm_feature_extractor_improved import EnhancedLLMFeatureExtractor

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XAIAnalyzer:
    """
    XAI (Explainable AI) 분석을 위한 정적 분석기
    Chain-of-Thought 추론과 Attention Visualization을 포함한 종합적인 XAI 분석 수행
    """
    
    def __init__(self, data_path: str = "data/raw", output_path: str = "data/processed"):
        """
        XAI 분석기 초기화
        
        Args:
            data_path: 원본 데이터 경로
            output_path: 분석 결과 저장 경로
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Enhanced LLM Feature Extractor 초기화
        self.llm_extractor = EnhancedLLMFeatureExtractor()
        
        logger.info("🚀 XAI Analyzer initialized")
    
    def load_news_data(self, filename: str = "news_sentiment_data.csv") -> pd.DataFrame:
        """
        뉴스 데이터 로드
        
        Args:
            filename: 뉴스 데이터 파일명
            
        Returns:
            로드된 뉴스 데이터프레임
        """
        file_path = self.data_path / filename
        
        if not file_path.exists():
            logger.error(f"❌ News data file not found: {file_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"📰 Loaded {len(df)} news articles from {filename}")
            
            # 기본 데이터 정제
            if 'title' in df.columns:
                df = df.dropna(subset=['title'])  # 제목이 없는 행 제거
                df = df[df['title'].str.len() > 10]  # 너무 짧은 제목 제거
                logger.info(f"📝 After cleaning: {len(df)} articles")
            
            return df
        
        except Exception as e:
            logger.error(f"❌ Error loading news data: {e}")
            return pd.DataFrame()
    
    def analyze_sample_data(self, df: pd.DataFrame, sample_size: int = 10) -> pd.DataFrame:
        """
        샘플 데이터에 대한 XAI 분석 수행 (테스트용)
        
        Args:
            df: 뉴스 데이터프레임
            sample_size: 분석할 샘플 개수
            
        Returns:
            XAI 분석 결과 데이터프레임
        """
        if df.empty:
            logger.warning("⚠️ Empty dataframe provided for analysis")
            return pd.DataFrame()
        
        # 샘플 선택 (다양성을 위해 고르게 분포된 샘플)
        if len(df) > sample_size:
            indices = np.linspace(0, len(df) - 1, sample_size, dtype=int)
            sample_df = df.iloc[indices].copy()
        else:
            sample_df = df.copy()
        
        logger.info(f"🧠 Analyzing {len(sample_df)} sample articles with Enhanced LLM...")
        
        # Enhanced LLM 특성 추출
        xai_results = self.llm_extractor.extract_enhanced_features(sample_df)
        
        return xai_results
    
    def create_xai_summary(self, xai_df: pd.DataFrame) -> Dict[str, Any]:
        """
        XAI 분석 결과 요약 생성
        
        Args:
            xai_df: XAI 분석 결과 데이터프레임
            
        Returns:
            요약 통계 딕셔너리
        """
        if xai_df.empty:
            return {"error": "No XAI data available"}
        
        try:
            summary = {
                "metadata": {
                    "analysis_date": datetime.now().isoformat(),
                    "total_articles_analyzed": len(xai_df),
                    "successful_analyses": len(xai_df[xai_df['processing_status'] == 'success']),
                    "failed_analyses": len(xai_df[xai_df['processing_status'] != 'success']),
                    "model_used": xai_df['model_used'].iloc[0] if not xai_df.empty else "unknown"
                },
                
                "sentiment_analysis": {
                    "sentiment_score_stats": {
                        "mean": float(xai_df['llm_sentiment_score'].mean()),
                        "std": float(xai_df['llm_sentiment_score'].std()),
                        "min": float(xai_df['llm_sentiment_score'].min()),
                        "max": float(xai_df['llm_sentiment_score'].max()),
                        "median": float(xai_df['llm_sentiment_score'].median())
                    },
                    "market_sentiment_distribution": xai_df['market_sentiment'].value_counts().to_dict(),
                    "event_category_distribution": xai_df['event_category'].value_counts().to_dict()
                },
                
                "uncertainty_analysis": {
                    "uncertainty_score_stats": {
                        "mean": float(xai_df['uncertainty_score'].mean()),
                        "std": float(xai_df['uncertainty_score'].std()),
                        "min": float(xai_df['uncertainty_score'].min()),
                        "max": float(xai_df['uncertainty_score'].max()),
                        "median": float(xai_df['uncertainty_score'].median())
                    }
                },
                
                "attention_analysis": {
                    "avg_tokens_per_article": float(xai_df['attention_tokens'].apply(len).mean()),
                    "total_unique_tokens": len(set([token for tokens in xai_df['attention_tokens'] for token in tokens])),
                    "articles_with_attention": len(xai_df[xai_df['attention_tokens'].apply(len) > 0])
                }
            }
            
            return summary
        
        except Exception as e:
            logger.error(f"❌ Error creating XAI summary: {e}")
            return {"error": str(e)}
    
    def create_dashboard_data(self, xai_df: pd.DataFrame) -> Dict[str, Any]:
        """
        대시보드용 XAI 데이터 생성
        
        Args:
            xai_df: XAI 분석 결과 데이터프레임
            
        Returns:
            대시보드용 데이터 구조
        """
        if xai_df.empty:
            return {"error": "No XAI data available"}
        
        try:
            # 성공적으로 처리된 데이터만 필터링
            successful_df = xai_df[xai_df['processing_status'] == 'success'].copy()
            
            dashboard_data = {
                "xai_samples": []
            }
            
            for idx, row in successful_df.iterrows():
                # Chain-of-Thought 추론 단계 파싱
                reasoning_steps = self._parse_reasoning_steps(row['reasoning_chain'])
                
                # 어텐션 데이터 구조화
                attention_data = {
                    "tokens": row['attention_tokens'] if isinstance(row['attention_tokens'], list) else [],
                    "weights": row['attention_weights'] if isinstance(row['attention_weights'], list) else []
                }
                
                sample_data = {
                    "id": f"sample_{idx}",
                    "title": row['title'],
                    "date": row['date'],
                    "sentiment_score": float(row['llm_sentiment_score']),
                    "uncertainty_score": float(row['uncertainty_score']),
                    "market_sentiment": row['market_sentiment'],
                    "event_category": row['event_category'],
                    "reasoning_steps": reasoning_steps,
                    "attention": attention_data,
                    "model_used": row['model_used']
                }
                
                dashboard_data["xai_samples"].append(sample_data)
            
            return dashboard_data
        
        except Exception as e:
            logger.error(f"❌ Error creating dashboard data: {e}")
            return {"error": str(e)}
    
    def _parse_reasoning_steps(self, reasoning_chain: str) -> List[Dict[str, str]]:
        """
        추론 과정을 단계별로 파싱
        
        Args:
            reasoning_chain: 원본 추론 체인 텍스트
            
        Returns:
            단계별 추론 과정 리스트
        """
        if not reasoning_chain or reasoning_chain.strip() == "":
            return []
        
        try:
            steps = []
            current_step = None
            current_content = []
            
            for line in reasoning_chain.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # STEP으로 시작하는 라인 찾기
                if line.startswith('STEP'):
                    # 이전 단계 저장
                    if current_step and current_content:
                        steps.append({
                            "step": current_step,
                            "content": '\n'.join(current_content)
                        })
                    
                    # 새 단계 시작
                    current_step = line
                    current_content = []
                
                elif current_step:
                    current_content.append(line)
            
            # 마지막 단계 저장
            if current_step and current_content:
                steps.append({
                    "step": current_step,
                    "content": '\n'.join(current_content)
                })
            
            return steps
        
        except Exception as e:
            logger.warning(f"⚠️ Error parsing reasoning steps: {e}")
            return [{"step": "Raw Content", "content": reasoning_chain}]
    
    def save_xai_analysis(self, xai_df: pd.DataFrame, summary: Dict, dashboard_data: Dict, 
                         filename_prefix: str = "xai_analysis") -> Dict[str, str]:
        """
        XAI 분석 결과를 파일로 저장
        
        Args:
            xai_df: XAI 분석 결과 데이터프레임
            summary: 분석 요약
            dashboard_data: 대시보드용 데이터
            filename_prefix: 파일명 접두사
            
        Returns:
            저장된 파일 경로들
        """
        saved_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 1. 전체 XAI 분석 결과 CSV
            csv_path = self.output_path / f"{filename_prefix}_{timestamp}.csv"
            xai_df.to_csv(csv_path, index=False)
            saved_files['csv'] = str(csv_path)
            logger.info(f"💾 Saved XAI CSV data to: {csv_path}")
            
            # 2. 분석 요약 JSON
            summary_path = self.output_path / f"{filename_prefix}_summary_{timestamp}.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            saved_files['summary'] = str(summary_path)
            logger.info(f"📊 Saved XAI summary to: {summary_path}")
            
            # 3. 대시보드용 JSON (정적 파일명 - 대시보드에서 참조하기 위해)
            dashboard_path = self.output_path / "xai_dashboard_data.json"
            with open(dashboard_path, 'w', encoding='utf-8') as f:
                json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
            saved_files['dashboard'] = str(dashboard_path)
            logger.info(f"📱 Saved dashboard data to: {dashboard_path}")
            
            return saved_files
        
        except Exception as e:
            logger.error(f"❌ Error saving XAI analysis: {e}")
            return {"error": str(e)}
    
    def run_complete_analysis(self, sample_size: int = 10, 
                            news_file: str = "news_sentiment_data.csv") -> Dict[str, Any]:
        """
        완전한 XAI 분석 실행
        
        Args:
            sample_size: 분석할 샘플 개수
            news_file: 뉴스 데이터 파일명
            
        Returns:
            분석 결과 정보
        """
        logger.info("🚀 Starting complete XAI analysis...")
        
        try:
            # 1. 뉴스 데이터 로드
            news_df = self.load_news_data(news_file)
            if news_df.empty:
                return {"error": "Failed to load news data"}
            
            # 2. XAI 분석 수행
            xai_df = self.analyze_sample_data(news_df, sample_size)
            if xai_df.empty:
                return {"error": "XAI analysis failed"}
            
            # 3. 요약 통계 생성
            summary = self.create_xai_summary(xai_df)
            
            # 4. 대시보드 데이터 생성
            dashboard_data = self.create_dashboard_data(xai_df)
            
            # 5. 결과 저장
            saved_files = self.save_xai_analysis(xai_df, summary, dashboard_data)
            
            logger.info("✅ Complete XAI analysis finished successfully!")
            
            return {
                "status": "success",
                "articles_analyzed": len(xai_df),
                "successful_analyses": len(xai_df[xai_df['processing_status'] == 'success']),
                "saved_files": saved_files,
                "summary": summary
            }
        
        except Exception as e:
            logger.error(f"❌ Complete XAI analysis failed: {e}")
            return {"error": str(e)}


def main():
    """메인 실행 함수"""
    logger.info("🌟 Starting XAI Analysis System")
    
    # XAI 분석기 생성
    analyzer = XAIAnalyzer()
    
    # 완전한 분석 실행 (샘플 크기 조절 가능)
    result = analyzer.run_complete_analysis(sample_size=5)  # 테스트용으로 5개만 분석
    
    # 결과 출력
    if "error" in result:
        logger.error(f"❌ Analysis failed: {result['error']}")
    else:
        logger.info("✅ XAI Analysis completed successfully!")
        logger.info(f"📈 Articles analyzed: {result['articles_analyzed']}")
        logger.info(f"✨ Successful analyses: {result['successful_analyses']}")
        logger.info(f"💾 Files saved: {list(result['saved_files'].keys())}")
        
        # 간단한 통계 출력
        if 'summary' in result:
            summary = result['summary']
            logger.info(f"📊 Average sentiment score: {summary['sentiment_analysis']['sentiment_score_stats']['mean']:.3f}")
            logger.info(f"🎯 Market sentiment distribution: {summary['sentiment_analysis']['market_sentiment_distribution']}")


if __name__ == "__main__":
    main()