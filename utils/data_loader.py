"""
데이터 로딩 유틸리티 함수들
"""
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import json

# 경로 상수
DIAGRAM_DIR = "diagrams"
RESULTS_DIR = "data/results"


@st.cache_data
def load_json_results(filename):
    """분석 결과 JSON 로드 (캐싱)"""
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


@st.cache_data(ttl=3600)  # 1시간 캐싱
def load_spy_data():
    """SPY 데이터 및 VRP 데이터 로드 (캐싱)"""
    try:
        csv_path = "data/raw/spy_data_2020_2025.csv"
        if os.path.exists(csv_path):
            spy = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            # VIX 데이터 로드 (yfinance 사용)
            try:
                import yfinance as yf
                vix = yf.download('^VIX', start='2020-01-01', end='2025-12-31', progress=False)
                if isinstance(vix.columns, pd.MultiIndex):
                    vix.columns = vix.columns.get_level_values(0)
                spy['VIX'] = vix['Close'].reindex(spy.index).ffill()
            except Exception as vix_error:
                # VIX 다운로드 실패 시 기본값 사용
                st.warning(f"VIX 데이터 다운로드 실패: {vix_error}. 기본값 사용 중.")
                spy['VIX'] = 20.0  # 평균 VIX 값
            
            # 변동성 계산
            spy['returns'] = spy['Close'].pct_change()
            spy['RV_22d'] = spy['returns'].rolling(22).std() * np.sqrt(252) * 100
            spy['VRP'] = spy['VIX'] - spy['RV_22d']
            spy = spy.dropna()
            return spy
    except Exception as e:
        st.error(f"SPY 데이터 로드 실패: {e}")
    return None


@st.cache_resource  # 리소스 캐싱
def load_image(filename):
    """PNG 이미지 로드 (캐싱)"""
    path = os.path.join(DIAGRAM_DIR, filename)
    if os.path.exists(path):
        return Image.open(path)
    return None


# 분석 결과 로드 (모듈 import 시 자동 로드)
def load_all_results():
    """모든 JSON 결과 파일 로드"""
    return {
        'TRANSACTION_COSTS': load_json_results("transaction_costs.json"),
        'VIX_BETA': load_json_results("vix_beta_expansion.json"),
        'SUBPERIOD': load_json_results("subperiod_analysis.json"),
        'STRUCTURAL_BREAKS': load_json_results("structural_breaks.json")
    }
