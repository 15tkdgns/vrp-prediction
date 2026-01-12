"""
Streamlit Tabs Configuration
각 탭을 별도 모듈로 분리하여 관리
"""

# 탭 구성 정보
TAB_CONFIG = {
    'overview': {
        'name': '연구 개요',
        'icon': '📊',
        'module': 'tabs.tab_overview'
    },
    'methodology': {
        'name': '방법론',
        'icon': '🔬',
        'module': 'tabs.tab_methodology'
    },
    'results': {
        'name': '결과',
        'icon': '📈',
        'module': 'tabs.tab_results'
    },
    'validation': {
        'name': '검증',
        'icon': '✓',
        'module': 'tabs.tab_validation'
    },
    'references': {
        'name': '참고문헌',
        'icon': '📚',
        'module': 'tabs.tab_references'
    }
}

def get_tab_names():
    """탭 이름 리스트 반환"""
    return [f"{config['icon']} {config['name']}" for config in TAB_CONFIG.values()]
