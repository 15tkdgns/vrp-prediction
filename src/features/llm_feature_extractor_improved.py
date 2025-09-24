import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import json
import logging
from typing import Dict, List, Tuple, Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedLLMFeatureExtractor:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        """
        Enhanced LLM Feature Extractor with Chain-of-Thought and Attention Visualization
        
        Args:
            model_name: HuggingFace model name (default: flan-t5-base for better reasoning)
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            self._initialize_model()
            logger.info(f"✅ Enhanced LLM Feature Extractor initialized with {model_name}")
        except Exception as e:
            logger.error(f"⚠️ Failed to initialize model: {e}")
            self.model = None
            self.tokenizer = None

    def _initialize_model(self):
        """모델과 토크나이저 초기화"""
        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_name,
            output_attentions=True,  # Attention 가중치 추출을 위해 필요
            return_dict=True
        )
        self.model.to(self.device)
        self.model.eval()

    def _create_cot_prompt(self, title: str) -> str:
        """
        Chain-of-Thought 프롬프트 생성
        LLM이 단계별로 사고하도록 유도하는 구조화된 프롬프트
        """
        prompt = f"""
Analyze this financial news headline step by step:

Headline: "{title}"

Please follow these reasoning steps:

STEP 1 - KEY TERMS IDENTIFICATION:
Identify the most important financial terms, company names, or market events in this headline.

STEP 2 - SENTIMENT ANALYSIS:
Analyze the sentiment step by step:
- What positive indicators do you see?
- What negative indicators do you see?
- What neutral or uncertain elements are present?

STEP 3 - MARKET IMPACT ASSESSMENT:
Based on your analysis:
- How might this affect the S&P 500?
- What is the uncertainty level?
- What category of event is this?

STEP 4 - FINAL JUDGMENT:
Provide your final assessment in this exact format:
llm_sentiment_score: [number between -1.0 and 1.0]
uncertainty_score: [number between 0.0 and 1.0] 
market_sentiment: [Bullish/Bearish/Neutral]
event_category: [M&A/Product Launch/Regulation/Financials/Other]

Reasoning:
"""
        return prompt

    def _extract_attention_weights(self, inputs: Dict, outputs) -> Dict:
        """
        어텐션 가중치 추출 및 처리
        
        Args:
            inputs: 토크나이저 입력
            outputs: 모델 출력 (어텐션 포함)
            
        Returns:
            처리된 어텐션 정보
        """
        try:
            # 입력 토큰들
            input_tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # 마지막 레이어의 어텐션 가중치 (첫 번째 헤드)
            attention_weights = outputs.encoder_attentions[-1][0, 0].detach().cpu().numpy()
            
            # 토큰별 평균 어텐션 점수 계산
            token_attention_scores = np.mean(attention_weights, axis=0)
            
            # 특수 토큰 제거 및 정규화
            filtered_tokens = []
            filtered_scores = []
            
            for i, (token, score) in enumerate(zip(input_tokens, token_attention_scores)):
                if token not in ['<pad>', '</s>', '<unk>'] and not token.startswith('▁'):
                    # ▁는 SentencePiece의 워드 시작 마커
                    clean_token = token.replace('▁', '')
                    if clean_token.strip():
                        filtered_tokens.append(clean_token)
                        filtered_scores.append(float(score))
            
            # 어텐션 점수 정규화 (0-1 범위)
            if len(filtered_scores) > 0:
                min_score = min(filtered_scores)
                max_score = max(filtered_scores)
                if max_score > min_score:
                    filtered_scores = [(s - min_score) / (max_score - min_score) for s in filtered_scores]
                
            return {
                'tokens': filtered_tokens,
                'attention_weights': filtered_scores,
                'raw_attention_shape': attention_weights.shape
            }
            
        except Exception as e:
            logger.warning(f"Attention extraction failed: {e}")
            return {
                'tokens': [],
                'attention_weights': [],
                'raw_attention_shape': None
            }

    def _parse_cot_response(self, response: str) -> Tuple[Dict, str]:
        """
        Chain-of-Thought 응답 파싱
        
        Args:
            response: LLM의 응답
            
        Returns:
            (구조화된 특성, 추론 과정)
        """
        features = {
            'llm_sentiment_score': 0.0,
            'uncertainty_score': 0.0,
            'market_sentiment': 'Neutral',
            'event_category': 'Other'
        }
        
        reasoning_chain = ""
        
        try:
            lines = response.split('\n')
            current_step = ""
            
            for line in lines:
                line = line.strip()
                
                # CoT 단계 식별
                if line.startswith('STEP'):
                    current_step = line
                    reasoning_chain += f"\n{line}\n"
                elif current_step and line:
                    reasoning_chain += f"{line}\n"
                
                # 최종 판단 파싱
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    
                    if 'sentiment_score' in key:
                        try:
                            features['llm_sentiment_score'] = float(value)
                        except ValueError:
                            pass
                    elif 'uncertainty_score' in key:
                        try:
                            features['uncertainty_score'] = float(value)
                        except ValueError:
                            pass
                    elif 'market_sentiment' in key:
                        if value in ['Bullish', 'Bearish', 'Neutral']:
                            features['market_sentiment'] = value
                    elif 'event_category' in key:
                        if value in ['M&A', 'Product Launch', 'Regulation', 'Financials', 'Other']:
                            features['event_category'] = value
            
        except Exception as e:
            logger.warning(f"Response parsing failed: {e}")
            reasoning_chain = f"Parsing error: {str(e)}"
        
        return features, reasoning_chain

    def extract_enhanced_features(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced 특성 추출 (CoT + Attention)
        
        Args:
            news_df: 뉴스 데이터프레임
            
        Returns:
            향상된 특성이 포함된 데이터프레임
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Model not initialized. Using default values.")
            return self._create_default_features(news_df)
        
        enhanced_features = []
        
        for index, row in tqdm(
            news_df.iterrows(), 
            total=news_df.shape[0], 
            desc="Extracting Enhanced LLM Features"
        ):
            current_date = row.get("date")
            current_title = row.get("title", "Unknown Title")
            
            try:
                # CoT 프롬프트 생성
                prompt = self._create_cot_prompt(current_title)
                
                # 토크나이징
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                # 모델 실행 (어텐션 포함)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=False,
                        output_attentions=True,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                
                # 응답 디코딩
                response = self.tokenizer.decode(
                    outputs.sequences[0], 
                    skip_special_tokens=True
                )
                
                # CoT 응답 파싱
                features, reasoning_chain = self._parse_cot_response(response)
                
                # 어텐션 가중치 추출
                attention_info = self._extract_attention_weights(inputs, outputs)
                
                # 결과 저장
                enhanced_features.append({
                    "date": current_date,
                    "title": current_title,
                    "llm_sentiment_score": features['llm_sentiment_score'],
                    "uncertainty_score": features['uncertainty_score'],
                    "market_sentiment": features['market_sentiment'],
                    "event_category": features['event_category'],
                    "reasoning_chain": reasoning_chain.strip(),
                    "attention_tokens": attention_info['tokens'],
                    "attention_weights": attention_info['attention_weights'],
                    "model_used": self.model_name,
                    "processing_status": "success"
                })
                
            except Exception as e:
                logger.error(f"Error processing row {index} (Title: {current_title}): {e}")
                enhanced_features.append({
                    "date": current_date,
                    "title": current_title,
                    "llm_sentiment_score": 0.0,
                    "uncertainty_score": 0.0,
                    "market_sentiment": "Neutral",
                    "event_category": "Other",
                    "reasoning_chain": f"Processing failed: {str(e)}",
                    "attention_tokens": [],
                    "attention_weights": [],
                    "model_used": self.model_name,
                    "processing_status": "error"
                })
        
        return pd.DataFrame(enhanced_features)

    def _create_default_features(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """모델 초기화 실패 시 기본값 생성"""
        default_features = []
        for index, row in news_df.iterrows():
            default_features.append({
                "date": row.get("date"),
                "title": row.get("title", "Unknown Title"),
                "llm_sentiment_score": 0.0,
                "uncertainty_score": 0.0,
                "market_sentiment": "Neutral",
                "event_category": "Other",
                "reasoning_chain": "Model initialization failed - using default values",
                "attention_tokens": [],
                "attention_weights": [],
                "model_used": "none",
                "processing_status": "model_error"
            })
        return pd.DataFrame(default_features)


def extract_enhanced_llm_features(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    편의 함수: Enhanced LLM 특성 추출
    
    Args:
        news_df: 뉴스 데이터프레임
        
    Returns:
        향상된 특성이 포함된 데이터프레임
    """
    extractor = EnhancedLLMFeatureExtractor()
    return extractor.extract_enhanced_features(news_df)


if __name__ == "__main__":
    # 테스트 실행
    try:
        # 샘플 뉴스 데이터로 테스트
        sample_data = pd.DataFrame({
            'date': ['2025-09-13', '2025-09-13'],
            'title': [
                'Apple Inc. reports record quarterly earnings beating analysts expectations',
                'Federal Reserve raises interest rates by 0.25% amid inflation concerns'
            ]
        })
        
        logger.info("Testing Enhanced LLM Feature Extractor...")
        extractor = EnhancedLLMFeatureExtractor()
        
        if extractor.model is not None:
            results = extractor.extract_enhanced_features(sample_data)
            
            # 결과 출력
            for i, row in results.iterrows():
                print(f"\n=== Test Result {i+1} ===")
                print(f"Title: {row['title']}")
                print(f"Sentiment Score: {row['llm_sentiment_score']}")
                print(f"Market Sentiment: {row['market_sentiment']}")
                print(f"Event Category: {row['event_category']}")
                print(f"Reasoning Chain Preview: {row['reasoning_chain'][:200]}...")
                print(f"Attention Tokens: {row['attention_tokens'][:5]}")
                print(f"Status: {row['processing_status']}")
                
            logger.info("✅ Enhanced LLM Feature Extractor test completed successfully!")
        else:
            logger.error("❌ Model initialization failed - cannot run test")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")