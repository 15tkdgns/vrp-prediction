"""
색상 팔레트 및 설정 상수
"""

# 기본 색상 팔레트
COLORS = {
    'primary': '#3498db',      # 파란색
    'success': '#2ecc71',      # 초록색  
    'danger': '#e74c3c',       # 빨간색
    'warning': '#f39c12',      # 주황색
    'secondary': '#9b59b6',    # 보라색
    'gold': '#f1c40f',         # 금색
    'gray': '#95a5a6',         # 회색
}

# 모델별 색상
MODEL_COLORS = {
    'neural': '#e74c3c',   # Neural 모델 - 빨강
    'tree': '#2ecc71',     # Tree 모델 - 초록
    'linear': '#3498db',   # Linear 모델 - 파랑
}

# 카테고리별 색상 (특성 중요도)
CATEGORY_COLORS = {
    '변동성': '#3498db',   # 파랑
    'VIX': '#e74c3c',      # 빨강
    'VRP': '#2ecc71',      # 초록
    '시장': '#9b59b6',     # 보라
}

# 자산별 색상
ASSET_COLORS = {
    'GLD': '#f1c40f',      # 금색
    'TLT': '#3498db',      # 파랑
    'EEM': '#e74c3c',      # 빨강
    'QQQ': '#9b59b6',      # 보라
    'SPY': '#2ecc71',      # 초록
}

# Plotly 색상 스킴
COLOR_SCHEMES = {
    'diverging': 'RdYlGn',     # 빨강-노랑-초록 (양/음 구분)
    'sequential': 'Viridis',   # 연속형
    'correlation': 'RdBu_r',   # 상관관계 (빨강-파랑)
    'heatmap': 'Reds',         # 히트맵
}

# 차트 기본 설정
CHART_DEFAULTS = {
    'height': 400,
    'plot_bgcolor': 'white',
    'paper_bgcolor': 'white',
    'gridcolor': 'lightgray',
}

# 데이터 경로
DATA_PATHS = {
    'spy': 'data/raw/spy_data_2020_2025.csv',
    'diagrams': 'diagrams',
    'results': 'data/results',
}
