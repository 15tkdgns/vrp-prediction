#!/usr/bin/env python3
"""
AI Stock Prediction System - Setup Configuration
모듈 경로 문제 해결을 위한 패키지 설정
"""

from setuptools import setup, find_packages
import os

# 현재 디렉토리의 requirements.txt 읽기
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'config', 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# README 읽기
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name="ai-stock-prediction",
    version="1.0.0",
    description="AI-powered S&P500 stock event detection and prediction system",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    
    # 패키지 정보
    packages=find_packages(),
    include_package_data=True,
    
    # Python 버전
    python_requires=">=3.8",
    
    # 의존성
    install_requires=read_requirements(),
    
    # 선택적 의존성
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=22.0.0',
            'ruff>=0.1.0',
            'pre-commit>=3.0.0',
        ],
        'dashboard': [
            'flask>=2.0.0',
            'flask-cors>=4.0.0',
        ],
        'gpu': [
            'tensorflow-gpu>=2.12.0',
            'torch>=1.13.0',
        ]
    },
    
    # 스크립트 엔트리 포인트
    entry_points={
        'console_scripts': [
            'ai-stock-train=src.models.model_training:main',
            'ai-stock-orchestrator=src.utils.system_orchestrator:main',
            'ai-stock-dashboard=dashboard.server:main',
            'ai-stock-test=src.testing.run_realtime_test:main',
        ],
    },
    
    # 메타데이터
    author="AI Stock Prediction Team",
    author_email="team@example.com",
    url="https://github.com/example/ai-stock-prediction",
    
    # 분류
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
    ],
    
    # 키워드
    keywords="ai, machine-learning, stock-prediction, finance, sp500, trading",
    
    # 라이센스
    license="MIT",
)