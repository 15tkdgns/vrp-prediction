"""
CSS 스타일 정의
"""

def get_css_styles():
    """전체 CSS 스타일 반환"""
    return """
<style>
    .slide-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        padding: 1.5rem;
        background: #f8f9fa;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.6rem;
        font-weight: 600;
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .slide-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 3px 15px rgba(0,0,0,0.08);
    }
    .key-point {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
        line-height: 1.7;
    }
    .result-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
        line-height: 1.7;
    }
    .hypothesis-card {
        background: #ebf5fb;
        border-left: 4px solid #3498db;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .warning-card {
        background: #fef9e7;
        border-left: 4px solid #f39c12;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .explanation {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
        line-height: 1.7;
    }
    .script-box {
        background: #fff3cd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #ffc107;
        font-style: italic;
    }
    /* Sidebar 크기 조정 */
    section[data-testid="stSidebar"] {
        width: 250px !important;
    }
    section[data-testid="stSidebar"] > div {
        width: 250px !important;
    }
</style>
"""
