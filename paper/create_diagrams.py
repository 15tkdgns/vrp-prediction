#!/usr/bin/env python3
"""
학술 논문 스타일 다이어그램
========================

흑백/회색 위주의 간결한 학술 논문 플로우차트
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
import os

OUTPUT_DIR = "paper/diagrams"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 학술 논문 스타일 색상 (흑백 + 최소 색상)
COLORS = {
    'black': '#000000',
    'dark': '#333333',
    'gray': '#666666',
    'light_gray': '#999999',
    'very_light': '#e5e5e5',
    'white': '#ffffff',
    'accent': '#4a4a4a',  # 강조용 (진한 회색)
}

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10


def box(ax, x, y, width, height, text, fill='white', edge='black', 
        fontsize=10, bold=False, subtext=None):
    """학술 논문 스타일 박스"""
    rect = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.01,rounding_size=0.01",
        facecolor=fill, edgecolor=edge, linewidth=1.5
    )
    ax.add_patch(rect)
    
    weight = 'bold' if bold else 'normal'
    if subtext:
        ax.text(x, y + height*0.15, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, color=COLORS['black'])
        ax.text(x, y - height*0.18, subtext, ha='center', va='center',
                fontsize=fontsize-2, color=COLORS['gray'])
    else:
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, color=COLORS['black'])


def arrow(ax, x1, y1, x2, y2, style='-|>'):
    """학술 논문 스타일 화살표"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=COLORS['black'], lw=1.2))


def save_fig(fig, filename, dpi=200):
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {filepath}")


# ============================================================================
# 1. VRP Concept
# ============================================================================
def create_vrp_concept():
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # 수식 형태로 표현
    box(ax, 0.18, 0.5, 0.22, 0.45, 'VIX', fontsize=14, bold=True, 
        subtext='(Implied Volatility)')
    
    ax.text(0.35, 0.5, '-', ha='center', va='center', fontsize=24, fontweight='bold')
    
    box(ax, 0.52, 0.5, 0.22, 0.45, 'RV', fontsize=14, bold=True,
        subtext='(Realized Volatility)')
    
    ax.text(0.69, 0.5, '=', ha='center', va='center', fontsize=24, fontweight='bold')
    
    box(ax, 0.86, 0.5, 0.22, 0.45, 'VRP', fontsize=14, bold=True,
        subtext='(Risk Premium)', fill=COLORS['very_light'])
    
    ax.text(0.5, 0.92, 'Figure 1: Volatility Risk Premium Definition', 
            ha='center', fontsize=11, style='italic')
    
    save_fig(fig, "01_vrp_concept.png")


# ============================================================================
# 2. Research Gap
# ============================================================================
def create_research_gap():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # 왼쪽: Previous Research
    ax.text(0.22, 0.92, 'Previous Research', ha='center', fontsize=11, fontweight='bold')
    
    prev = [('S&P 500 focused', 0.72), ('Traditional models', 0.52), ('Limited scope', 0.32)]
    for text, y in prev:
        box(ax, 0.22, y, 0.32, 0.14, text, fontsize=9)
    
    # 오른쪽: This Study
    ax.text(0.78, 0.92, 'This Study', ha='center', fontsize=11, fontweight='bold')
    
    this = [('Multi-asset analysis', 0.72), ('Machine learning', 0.52), ('VIX-Beta theory', 0.32)]
    for text, y in this:
        box(ax, 0.78, y, 0.32, 0.14, text, fontsize=9, fill=COLORS['very_light'])
    
    # 화살표
    for y in [0.72, 0.52, 0.32]:
        arrow(ax, 0.40, y, 0.60, y)
    
    ax.text(0.5, 0.05, 'Figure 2: Research Contribution', 
            ha='center', fontsize=11, style='italic')
    
    save_fig(fig, "02_research_gap.png")


# ============================================================================
# 3. Hypothesis
# ============================================================================
def create_hypothesis():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    hypotheses = [
        {'x': 0.17, 'id': 'H1', 'title': 'Model Comparison', 
         'content': 'Nonlinear > Linear'},
        {'x': 0.50, 'id': 'H2', 'title': 'VIX-Beta Theory',
         'content': 'Low corr = High R-squared'},
        {'x': 0.83, 'id': 'H3', 'title': 'Economic Value',
         'content': 'Prediction-based > Buy&Hold'},
    ]
    
    for h in hypotheses:
        # 메인 박스
        box(ax, h['x'], 0.55, 0.28, 0.55, '', fill=COLORS['very_light'])
        
        # ID
        ax.text(h['x'], 0.75, h['id'], ha='center', va='center',
                fontsize=14, fontweight='bold')
        # Title
        ax.text(h['x'], 0.55, h['title'], ha='center', va='center',
                fontsize=10, fontweight='bold')
        # Content
        ax.text(h['x'], 0.38, h['content'], ha='center', va='center',
                fontsize=9, color=COLORS['gray'])
    
    ax.text(0.5, 0.95, 'Figure 3: Research Hypotheses', 
            ha='center', fontsize=11, style='italic')
    
    save_fig(fig, "03_hypothesis.png")


# ============================================================================
# 4. Pipeline (학술 논문 플로우차트)
# ============================================================================
def create_pipeline():
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    steps = [
        {'x': 0.10, 'text': 'Data\nCollection'},
        {'x': 0.28, 'text': 'Preprocessing'},
        {'x': 0.46, 'text': 'Feature\nExtraction'},
        {'x': 0.64, 'text': 'Model\nTraining'},
        {'x': 0.82, 'text': 'VRP\nPrediction'},
    ]
    
    for i, step in enumerate(steps):
        box(ax, step['x'], 0.5, 0.14, 0.40, step['text'], fontsize=9, bold=True)
        
        if i < len(steps) - 1:
            arrow(ax, step['x'] + 0.08, 0.5, steps[i+1]['x'] - 0.08, 0.5)
    
    ax.text(0.5, 0.92, 'Figure 4: Prediction Pipeline', 
            ha='center', fontsize=11, style='italic')
    
    save_fig(fig, "04_pipeline.png")


# ============================================================================
# 5. Features
# ============================================================================
def create_features():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    categories = [
        {'x': 0.15, 'label': 'Volatility', 'features': ['RV_1d', 'RV_5d', 'RV_22d']},
        {'x': 0.40, 'label': 'VIX', 'features': ['Vol_lag1', 'Vol_lag5', 'Vol_change']},
        {'x': 0.65, 'label': 'VRP', 'features': ['VRP_lag1', 'VRP_lag5', 'VRP_ma5']},
        {'x': 0.90, 'label': 'Others', 'features': ['regime', 'ret_5d', 'ret_22d']},
    ]
    
    for cat in categories:
        # 카테고리 헤더
        box(ax, cat['x'], 0.85, 0.18, 0.10, cat['label'], fontsize=10, bold=True,
            fill=COLORS['very_light'])
        
        # 특성들
        for i, feat in enumerate(cat['features']):
            y = 0.62 - i * 0.16
            box(ax, cat['x'], y, 0.16, 0.10, feat, fontsize=9)
        
        # 화살표
        ax.annotate('', xy=(0.52, 0.12), xytext=(cat['x'], 0.22),
                    arrowprops=dict(arrowstyle='-|>', color=COLORS['gray'], lw=1))
    
    # 모델 입력
    box(ax, 0.52, 0.08, 0.30, 0.08, 'Model Input (12 features)', fontsize=9, bold=True,
        fill=COLORS['very_light'])
    
    ax.text(0.5, 0.97, 'Figure 5: Feature Categories', 
            ha='center', fontsize=11, style='italic')
    
    save_fig(fig, "05_features.png")


# ============================================================================
# 6. MLP Architecture
# ============================================================================
def create_mlp():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    layers = [
        {'x': 0.12, 'n': 5, 'label': 'Input\n(12)'},
        {'x': 0.38, 'n': 4, 'label': 'Hidden 1\n(64)'},
        {'x': 0.64, 'n': 3, 'label': 'Hidden 2\n(32)'},
        {'x': 0.88, 'n': 1, 'label': 'Output\n(1)'},
    ]
    
    # 연결선 (먼저)
    for l_idx in range(len(layers) - 1):
        l1, l2 = layers[l_idx], layers[l_idx + 1]
        for i in range(l1['n']):
            for j in range(l2['n']):
                y1 = 0.18 + (i + 1) * (0.55 / (l1['n'] + 1))
                y2 = 0.18 + (j + 1) * (0.55 / (l2['n'] + 1))
                ax.plot([l1['x'], l2['x']], [y1, y2], 
                       color=COLORS['very_light'], linewidth=0.5, zorder=0)
    
    # 뉴런
    for layer in layers:
        spacing = 0.55 / (layer['n'] + 1)
        for i in range(layer['n']):
            y = 0.18 + (i + 1) * spacing
            circle = Circle((layer['x'], y), 0.025, facecolor='white',
                           edgecolor=COLORS['black'], linewidth=1.2, zorder=2)
            ax.add_patch(circle)
        
        ax.text(layer['x'], 0.05, layer['label'], ha='center', va='center',
               fontsize=9, fontweight='bold')
    
    ax.text(0.5, 0.92, 'Figure 6: MLP Architecture', 
            ha='center', fontsize=11, style='italic')
    
    save_fig(fig, "06_mlp.png")


# ============================================================================
# 7. Data Split
# ============================================================================
def create_data_split():
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Training
    rect1 = FancyBboxPatch((0.05, 0.35), 0.65, 0.30,
                           boxstyle="square", facecolor='white',
                           edgecolor=COLORS['black'], linewidth=1.5)
    ax.add_patch(rect1)
    ax.text(0.375, 0.5, 'Training Set (78%)', ha='center', va='center',
            fontsize=11, fontweight='bold')
    
    # Gap (해칭으로 표현)
    rect2 = FancyBboxPatch((0.70, 0.35), 0.05, 0.30,
                           boxstyle="square", facecolor=COLORS['very_light'],
                           edgecolor=COLORS['black'], linewidth=1.5, hatch='///')
    ax.add_patch(rect2)
    ax.text(0.725, 0.5, 'Gap', ha='center', va='center', fontsize=8, rotation=90)
    
    # Test
    rect3 = FancyBboxPatch((0.75, 0.35), 0.20, 0.30,
                           boxstyle="square", facecolor=COLORS['very_light'],
                           edgecolor=COLORS['black'], linewidth=1.5)
    ax.add_patch(rect3)
    ax.text(0.85, 0.5, 'Test Set (20%)', ha='center', va='center',
            fontsize=10, fontweight='bold')
    
    # Gap 설명
    ax.annotate('22-day gap\n(leakage prevention)', xy=(0.725, 0.33),
                xytext=(0.725, 0.12), ha='center', fontsize=9,
                arrowprops=dict(arrowstyle='->', color=COLORS['gray']))
    
    ax.text(0.5, 0.85, 'Figure 7: Train-Test Split with Gap', 
            ha='center', fontsize=11, style='italic')
    
    save_fig(fig, "07_data_split.png")


# ============================================================================
# 8. VIX-Beta Theory
# ============================================================================
def create_vix_beta():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # VIX 중앙
    box(ax, 0.5, 0.85, 0.40, 0.10, 'VIX (S&P 500 Options)', fontsize=10, bold=True,
        fill=COLORS['very_light'])
    
    # SPY
    box(ax, 0.25, 0.55, 0.30, 0.22, 'SPY', fontsize=12, bold=True,
        subtext='Corr(VIX, RV) = 0.83')
    box(ax, 0.25, 0.22, 0.25, 0.10, 'R-squared = 0.02', fontsize=9)
    ax.text(0.25, 0.08, '(Low predictability)', ha='center', fontsize=8, 
            color=COLORS['gray'])
    
    # GLD
    box(ax, 0.75, 0.55, 0.30, 0.22, 'GLD', fontsize=12, bold=True,
        subtext='Corr(VIX, RV) = 0.51', fill=COLORS['very_light'])
    box(ax, 0.75, 0.22, 0.25, 0.10, 'R-squared = 0.37', fontsize=9,
        fill=COLORS['very_light'])
    ax.text(0.75, 0.08, '(High predictability)', ha='center', fontsize=8,
            color=COLORS['gray'])
    
    # 화살표
    arrow(ax, 0.35, 0.78, 0.25, 0.68)
    arrow(ax, 0.65, 0.78, 0.75, 0.68)
    arrow(ax, 0.25, 0.42, 0.25, 0.30)
    arrow(ax, 0.75, 0.42, 0.75, 0.30)
    
    ax.text(0.5, 0.97, 'Figure 8: VIX-Beta Theory', 
            ha='center', fontsize=11, style='italic')
    
    save_fig(fig, "08_vix_beta.png")


# ============================================================================
# 9. Conclusion
# ============================================================================
def create_conclusion():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    hypotheses = [
        {'x': 0.17, 'id': 'H1', 'result': 'Accepted', 'detail': 'R-sq: 0.44 > 0.37'},
        {'x': 0.50, 'id': 'H2', 'result': 'Accepted', 'detail': 'r = -0.87'},
        {'x': 0.83, 'id': 'H3', 'result': 'Accepted', 'detail': 'Sharpe: 22.76'},
    ]
    
    for h in hypotheses:
        # 가설 박스
        box(ax, h['x'], 0.72, 0.22, 0.14, h['id'], fontsize=12, bold=True)
        
        # 화살표
        arrow(ax, h['x'], 0.63, h['x'], 0.50)
        
        # 결과 박스
        box(ax, h['x'], 0.38, 0.22, 0.20, h['result'], fontsize=11, bold=True,
            fill=COLORS['very_light'], subtext=h['detail'])
    
    ax.text(0.5, 0.92, 'Figure 9: Hypothesis Test Results', 
            ha='center', fontsize=11, style='italic')
    
    save_fig(fig, "09_conclusion.png")


# ============================================================================
# 10. Research Flow
# ============================================================================
def create_research_flow():
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    steps = [
        {'x': 0.10, 'text': 'Research\nQuestion', 'detail': 'RQ1-3'},
        {'x': 0.30, 'text': 'Hypothesis', 'detail': 'H1-3'},
        {'x': 0.50, 'text': 'Data', 'detail': '10yr, 4 assets'},
        {'x': 0.70, 'text': 'Experiment', 'detail': '24 models'},
        {'x': 0.90, 'text': 'Conclusion', 'detail': 'All accepted'},
    ]
    
    for i, step in enumerate(steps):
        # 상단 박스
        box(ax, step['x'], 0.65, 0.14, 0.20, step['text'], fontsize=9, bold=True)
        
        # 하단 세부
        box(ax, step['x'], 0.30, 0.12, 0.12, step['detail'], fontsize=8,
            fill=COLORS['very_light'])
        
        # 수직 연결선
        ax.plot([step['x'], step['x']], [0.53, 0.38], 
               color=COLORS['gray'], linewidth=1, linestyle='--')
        
        # 수평 화살표
        if i < len(steps) - 1:
            arrow(ax, step['x'] + 0.08, 0.65, steps[i+1]['x'] - 0.08, 0.65)
    
    ax.text(0.5, 0.92, 'Figure 10: Research Flow', 
            ha='center', fontsize=11, style='italic')
    
    save_fig(fig, "10_research_flow.png")


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print("Generating academic-style diagrams...")
    print("=" * 50)
    
    create_vrp_concept()
    create_research_gap()
    create_hypothesis()
    create_pipeline()
    create_features()
    create_mlp()
    create_data_split()
    create_vix_beta()
    create_conclusion()
    create_research_flow()
    
    print("=" * 50)
    print("All diagrams generated!")
    print(f"Location: {OUTPUT_DIR}/")
