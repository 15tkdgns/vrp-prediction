#!/usr/bin/env python3
"""
Draw.io XML 다이어그램 생성기
============================

Draw.io에서 열 수 있는 XML 파일 직접 생성
"""

import os
import base64
import zlib
from urllib.parse import quote

OUTPUT_DIR = "paper/diagrams"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_drawio_xml(cells_xml, filename, page_width=800, page_height=400):
    """Draw.io XML 파일 생성"""
    xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net" modified="2024-12-14T00:00:00.000Z" agent="Python Script" version="21.0.0">
  <diagram name="Page-1" id="page1">
    <mxGraphModel dx="1000" dy="600" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{page_width}" pageHeight="{page_height}" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
{cells_xml}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(xml)
    print(f"Saved: {filepath}")
    return filepath


def box(id, x, y, width, height, text, fill_color, text_color="white", font_size=14, style_extra=""):
    """박스 셀 생성"""
    return f'''        <mxCell id="{id}" value="{text}" style="rounded=1;whiteSpace=wrap;html=1;fillColor={fill_color};strokeColor=none;fontColor={text_color};fontSize={font_size};fontStyle=1;{style_extra}" vertex="1" parent="1">
          <mxGeometry x="{x}" y="{y}" width="{width}" height="{height}" as="geometry" />
        </mxCell>'''


def arrow(id, source_id, target_id, style=""):
    """화살표 셀 생성"""
    default_style = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=2;strokeColor=#333333;"
    return f'''        <mxCell id="{id}" style="{default_style}{style}" edge="1" parent="1" source="{source_id}" target="{target_id}">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>'''


def text_label(id, x, y, width, height, text, font_size=12, font_color="#000000", style_extra=""):
    """텍스트 라벨 생성"""
    return f'''        <mxCell id="{id}" value="{text}" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize={font_size};fontColor={font_color};{style_extra}" vertex="1" parent="1">
          <mxGeometry x="{x}" y="{y}" width="{width}" height="{height}" as="geometry" />
        </mxCell>'''


def circle(id, x, y, size, text, fill_color, text_color="white", font_size=12):
    """원형 셀 생성"""
    return f'''        <mxCell id="{id}" value="{text}" style="ellipse;whiteSpace=wrap;html=1;fillColor={fill_color};strokeColor=none;fontColor={text_color};fontSize={font_size};fontStyle=1;" vertex="1" parent="1">
          <mxGeometry x="{x}" y="{y}" width="{size}" height="{size}" as="geometry" />
        </mxCell>'''


# ============================================================================
# 1. VRP Concept Diagram
# ============================================================================
def create_vrp_concept():
    cells = []
    
    # Title
    cells.append(text_label("title", 250, 10, 300, 30, "VRP = VIX - RV", font_size=20, style_extra="fontStyle=1;"))
    
    # VIX Box
    cells.append(box("vix", 50, 80, 180, 100, "VIX&lt;br&gt;&lt;font style=&quot;font-size:11px&quot;&gt;Implied Volatility&lt;/font&gt;", "#e74c3c"))
    
    # Minus sign
    cells.append(text_label("minus", 240, 100, 40, 60, "-", font_size=36, style_extra="fontStyle=1;"))
    
    # RV Box
    cells.append(box("rv", 300, 80, 180, 100, "RV&lt;br&gt;&lt;font style=&quot;font-size:11px&quot;&gt;Realized Volatility&lt;/font&gt;", "#3498db"))
    
    # Equals sign
    cells.append(text_label("equals", 490, 100, 40, 60, "=", font_size=36, style_extra="fontStyle=1;"))
    
    # VRP Box
    cells.append(box("vrp", 550, 80, 180, 100, "VRP&lt;br&gt;&lt;font style=&quot;font-size:11px&quot;&gt;Risk Premium&lt;/font&gt;", "#2ecc71"))
    
    create_drawio_xml('\n'.join(cells), "01_vrp_concept.drawio")


# ============================================================================
# 2. Research Gap Diagram
# ============================================================================
def create_research_gap():
    cells = []
    
    # Title
    cells.append(text_label("title", 200, 10, 400, 30, "Research Gap & Contribution", font_size=18, style_extra="fontStyle=1;"))
    
    # Headers
    cells.append(text_label("prev_header", 50, 50, 200, 30, "Previous Research", font_size=14, font_color="#e74c3c", style_extra="fontStyle=1;"))
    cells.append(text_label("this_header", 550, 50, 200, 30, "This Study", font_size=14, font_color="#2ecc71", style_extra="fontStyle=1;"))
    
    # Previous research boxes
    cells.append(box("prev1", 50, 90, 200, 50, "S&amp;P 500 Only", "#e74c3c", font_size=12))
    cells.append(box("prev2", 50, 160, 200, 50, "Traditional Models", "#e74c3c", font_size=12))
    cells.append(box("prev3", 50, 230, 200, 50, "No Cross-Asset", "#e74c3c", font_size=12))
    
    # This study boxes
    cells.append(box("this1", 550, 90, 200, 50, "Multi-Asset", "#2ecc71", font_size=12))
    cells.append(box("this2", 550, 160, 200, 50, "Machine Learning", "#2ecc71", font_size=12))
    cells.append(box("this3", 550, 230, 200, 50, "VIX-Beta Theory", "#2ecc71", font_size=12))
    
    # Arrows
    cells.append(arrow("arr1", "prev1", "this1"))
    cells.append(arrow("arr2", "prev2", "this2"))
    cells.append(arrow("arr3", "prev3", "this3"))
    
    create_drawio_xml('\n'.join(cells), "02_research_gap.drawio", page_height=320)


# ============================================================================
# 3. Hypothesis Diagram
# ============================================================================
def create_hypothesis():
    cells = []
    
    # Title
    cells.append(text_label("title", 200, 10, 400, 30, "Three Research Hypotheses", font_size=18, style_extra="fontStyle=1;"))
    
    # H1
    cells.append(box("h1", 50, 60, 200, 120, "H1&lt;br&gt;&lt;font style=&quot;font-size:11px&quot;&gt;Model Comparison&lt;/font&gt;", "#3498db", font_size=20))
    cells.append(text_label("h1_desc", 50, 190, 200, 30, "MLP &gt; Linear", font_size=11))
    
    # H2
    cells.append(box("h2", 300, 60, 200, 120, "H2&lt;br&gt;&lt;font style=&quot;font-size:11px&quot;&gt;VIX-Beta&lt;/font&gt;", "#e74c3c", font_size=20))
    cells.append(text_label("h2_desc", 300, 190, 200, 30, "Corr ↓ = R² ↑", font_size=11))
    
    # H3
    cells.append(box("h3", 550, 60, 200, 120, "H3&lt;br&gt;&lt;font style=&quot;font-size:11px&quot;&gt;Trading&lt;/font&gt;", "#2ecc71", font_size=20))
    cells.append(text_label("h3_desc", 550, 190, 200, 30, "Sharpe &gt; B&amp;H", font_size=11))
    
    create_drawio_xml('\n'.join(cells), "03_hypothesis.drawio", page_height=250)


# ============================================================================
# 4. Prediction Pipeline
# ============================================================================
def create_pipeline():
    cells = []
    
    # Title
    cells.append(text_label("title", 250, 10, 400, 30, "Prediction Pipeline", font_size=18, style_extra="fontStyle=1;"))
    
    # Steps
    steps = [
        {"id": "s1", "x": 20, "text": "Data&lt;br&gt;Collection", "color": "#3498db"},
        {"id": "s2", "x": 170, "text": "Preprocess", "color": "#3498db"},
        {"id": "s3", "x": 320, "text": "Feature&lt;br&gt;Extraction", "color": "#2ecc71"},
        {"id": "s4", "x": 470, "text": "Model&lt;br&gt;Training", "color": "#e74c3c"},
        {"id": "s5", "x": 620, "text": "VRP&lt;br&gt;Prediction", "color": "#9b59b6"},
    ]
    
    for step in steps:
        cells.append(box(step["id"], step["x"], 60, 130, 80, step["text"], step["color"], font_size=11))
    
    # Arrows
    for i in range(len(steps) - 1):
        cells.append(arrow(f"arr{i}", steps[i]["id"], steps[i+1]["id"]))
    
    create_drawio_xml('\n'.join(cells), "04_pipeline.drawio", page_height=180)


# ============================================================================
# 5. Feature Structure
# ============================================================================
def create_features():
    cells = []
    
    # Title
    cells.append(text_label("title", 200, 10, 400, 30, "12 Features Classification", font_size=18, style_extra="fontStyle=1;"))
    
    # Categories
    categories = [
        {"id": "cat1", "x": 30, "label": "Volatility", "color": "#e74c3c", "features": ["RV_1d", "RV_5d", "RV_22d"]},
        {"id": "cat2", "x": 220, "label": "VIX", "color": "#3498db", "features": ["Vol_lag1", "Vol_lag5", "Vol_change"]},
        {"id": "cat3", "x": 410, "label": "VRP", "color": "#2ecc71", "features": ["VRP_lag1", "VRP_lag5", "VRP_ma5"]},
        {"id": "cat4", "x": 600, "label": "Others", "color": "#9b59b6", "features": ["regime", "ret_5d", "ret_22d"]},
    ]
    
    for cat in categories:
        cells.append(box(cat["id"], cat["x"], 50, 150, 40, cat["label"], cat["color"], font_size=12))
        
        for i, feat in enumerate(cat["features"]):
            feat_id = f"{cat['id']}_f{i}"
            cells.append(box(feat_id, cat["x"], 100 + i * 45, 150, 35, feat, cat["color"], font_size=10, style_extra="fillOpacity=40;fontColor=#000000;"))
    
    # Model input box
    cells.append(box("model", 280, 260, 220, 40, "Model Input (12)", "#2c3e50", font_size=12))
    
    create_drawio_xml('\n'.join(cells), "05_features.drawio", page_height=340)


# ============================================================================
# 6. MLP Structure
# ============================================================================
def create_mlp():
    cells = []
    
    # Title
    cells.append(text_label("title", 200, 10, 400, 30, "MLP Architecture", font_size=18, style_extra="fontStyle=1;"))
    
    # Layers
    layers = [
        {"x": 50, "neurons": 6, "color": "#3498db", "label": "Input (12)"},
        {"x": 230, "neurons": 5, "color": "#e74c3c", "label": "Hidden1 (64)"},
        {"x": 410, "neurons": 4, "color": "#e74c3c", "label": "Hidden2 (32)"},
        {"x": 590, "neurons": 1, "color": "#2ecc71", "label": "Output (1)"},
    ]
    
    for layer_idx, layer in enumerate(layers):
        # Layer label
        cells.append(text_label(f"label{layer_idx}", layer["x"] - 10, 280, 120, 30, layer["label"], font_size=10))
        
        # Neurons
        spacing = 220 / (layer["neurons"] + 1)
        for i in range(layer["neurons"]):
            y = 50 + (i + 1) * spacing
            cells.append(circle(f"n{layer_idx}_{i}", layer["x"], y, 40, "", layer["color"]))
    
    create_drawio_xml('\n'.join(cells), "06_mlp.drawio", page_height=340)


# ============================================================================
# 7. Data Split
# ============================================================================
def create_data_split():
    cells = []
    
    # Title
    cells.append(text_label("title", 200, 10, 400, 30, "Data Leakage Prevention: 22-day Gap", font_size=18, style_extra="fontStyle=1;"))
    
    # Training data
    cells.append(box("train", 20, 70, 540, 80, "Training Data (78%)", "#3498db", font_size=14))
    
    # Gap
    cells.append(box("gap", 570, 70, 50, 80, "Gap", "#f39c12", font_size=11))
    
    # Test data
    cells.append(box("test", 630, 70, 140, 80, "Test (20%)", "#2ecc71", font_size=12))
    
    # Gap label
    cells.append(text_label("gap_label", 545, 160, 100, 30, "22-day Gap", font_size=11, font_color="#f39c12", style_extra="fontStyle=1;"))
    
    create_drawio_xml('\n'.join(cells), "07_data_split.drawio", page_height=220)


# ============================================================================
# 8. VIX-Beta Theory
# ============================================================================
def create_vix_beta():
    cells = []
    
    # Title
    cells.append(text_label("title", 100, 10, 600, 30, "VIX-Beta Theory: Lower Correlation = Higher Predictability", font_size=16, style_extra="fontStyle=1;"))
    
    # VIX center
    cells.append(box("vix", 250, 50, 300, 50, "VIX (S&amp;P 500 Options)", "#9b59b6", font_size=12))
    
    # SPY
    cells.append(box("spy", 50, 140, 220, 80, "SPY&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;Corr = 0.83&lt;/font&gt;", "#e74c3c", font_size=14))
    cells.append(box("spy_r2", 50, 250, 220, 50, "R² = 0.02", "#e74c3c", font_size=12))
    
    # GLD
    cells.append(box("gld", 530, 140, 220, 80, "GLD&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;Corr = 0.51&lt;/font&gt;", "#2ecc71", font_size=14))
    cells.append(box("gld_r2", 530, 250, 220, 50, "R² = 0.37", "#2ecc71", font_size=12))
    
    # Arrows
    cells.append(arrow("arr1", "vix", "spy"))
    cells.append(arrow("arr2", "vix", "gld"))
    cells.append(arrow("arr3", "spy", "spy_r2"))
    cells.append(arrow("arr4", "gld", "gld_r2"))
    
    create_drawio_xml('\n'.join(cells), "08_vix_beta.drawio", page_height=340)


# ============================================================================
# 9. Conclusion Diagram
# ============================================================================
def create_conclusion():
    cells = []
    
    # Title
    cells.append(text_label("title", 150, 10, 500, 30, "Hypothesis Testing Results: All 3 Accepted", font_size=18, style_extra="fontStyle=1;"))
    
    # Hypothesis boxes
    cells.append(box("h1", 50, 60, 200, 60, "H1: Model", "#3498db", font_size=12))
    cells.append(box("h2", 300, 60, 200, 60, "H2: VIX-Beta", "#e74c3c", font_size=12))
    cells.append(box("h3", 550, 60, 200, 60, "H3: Trading", "#2ecc71", font_size=12))
    
    # Result boxes
    cells.append(box("r1", 50, 180, 200, 70, "ACCEPTED&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;R²: 0.44 &gt; 0.37&lt;/font&gt;", "#27ae60", font_size=12))
    cells.append(box("r2", 300, 180, 200, 70, "ACCEPTED&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;r = -0.87&lt;/font&gt;", "#27ae60", font_size=12))
    cells.append(box("r3", 550, 180, 200, 70, "ACCEPTED&lt;br&gt;&lt;font style=&quot;font-size:10px&quot;&gt;Sharpe 22.76&lt;/font&gt;", "#27ae60", font_size=12))
    
    # Arrows
    cells.append(arrow("arr1", "h1", "r1"))
    cells.append(arrow("arr2", "h2", "r2"))
    cells.append(arrow("arr3", "h3", "r3"))
    
    create_drawio_xml('\n'.join(cells), "09_conclusion.drawio", page_height=290)


# ============================================================================
# 10. Research Flow
# ============================================================================
def create_research_flow():
    cells = []
    
    # Title
    cells.append(text_label("title", 250, 10, 400, 30, "Research Flow", font_size=18, style_extra="fontStyle=1;"))
    
    # Steps
    steps = [
        {"id": "s1", "x": 20, "text": "Research&lt;br&gt;Question", "color": "#3498db"},
        {"id": "s2", "x": 170, "text": "Hypothesis", "color": "#3498db"},
        {"id": "s3", "x": 320, "text": "Data&lt;br&gt;Collection", "color": "#2ecc71"},
        {"id": "s4", "x": 470, "text": "Experiment", "color": "#e74c3c"},
        {"id": "s5", "x": 620, "text": "Conclusion", "color": "#9b59b6"},
    ]
    
    for step in steps:
        cells.append(box(step["id"], step["x"], 60, 130, 70, step["text"], step["color"], font_size=11))
    
    # Arrows
    for i in range(len(steps) - 1):
        cells.append(arrow(f"arr{i}", steps[i]["id"], steps[i+1]["id"]))
    
    # Details
    details = [
        {"id": "d1", "x": 20, "text": "RQ1: Model&lt;br&gt;RQ2: Asset&lt;br&gt;RQ3: Strategy"},
        {"id": "d2", "x": 170, "text": "H1, H2, H3"},
        {"id": "d3", "x": 320, "text": "10yr Daily&lt;br&gt;4 Assets"},
        {"id": "d4", "x": 470, "text": "24 Models&lt;br&gt;22d Gap"},
        {"id": "d5", "x": 620, "text": "All 3&lt;br&gt;Accepted"},
    ]
    
    for detail in details:
        cells.append(box(detail["id"], detail["x"], 170, 130, 70, detail["text"], "#ecf0f1", font_size=9, text_color="#333333"))
    
    create_drawio_xml('\n'.join(cells), "10_research_flow.drawio", page_height=280)


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print("Generating Draw.io XML files...")
    print("-" * 40)
    
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
    
    print("-" * 40)
    print("All Draw.io files generated!")
    print(f"Location: {OUTPUT_DIR}/")
    print("\nOpen these .drawio files in:")
    print("- https://app.diagrams.net/")
    print("- Draw.io Desktop app")
