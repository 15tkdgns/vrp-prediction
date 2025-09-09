#!/usr/bin/env python3
"""
Update dashboard HTML with daily prediction data
Replace existing data with new daily predictions
"""

import json
import re

def load_daily_predictions():
    """Load the generated daily predictions"""
    with open('/root/workspace/dashboard/daily_predictions.json', 'r') as f:
        return json.load(f)

def update_dashboard_html():
    """Update index.html with daily prediction data"""
    
    # Load daily predictions
    print("📊 Loading daily predictions...")
    daily_data = load_daily_predictions()
    
    # Load current HTML
    print("📖 Reading current dashboard HTML...")
    with open('/root/workspace/dashboard/index.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Prepare new data strings
    actual_data_str = json.dumps(daily_data['actual_data'], indent=12)[1:-1]  # Remove outer brackets
    predictions_str = json.dumps(daily_data['model_predictions'], indent=8)
    
    # Update data points count
    data_count = len(daily_data['actual_data'])
    
    # Update actualData in JavaScript
    actualdata_pattern = r'const actualData = \[.*?\];'
    new_actualdata = f'const actualData = [\n{actual_data_str}\n        ];'
    html_content = re.sub(actualdata_pattern, new_actualdata, html_content, flags=re.DOTALL)
    
    # Update modelPredictions in JavaScript  
    predictions_pattern = r'const modelPredictions = \{.*?\};'
    new_predictions = f'const modelPredictions = {predictions_str};'
    html_content = re.sub(predictions_pattern, new_predictions, html_content, flags=re.DOTALL)
    
    # Update data points count in HTML
    html_content = re.sub(r'<h4>\d+</h4>\s*<p>2025 H1 SPY Data</p>', 
                         f'<h4>{data_count}</h4>\n                    <p>Daily SPY Data (2025 H1)</p>', 
                         html_content)
    
    # Update title/description
    html_content = re.sub(r'반월별 예측|격주별 예측|월별 예측', '일별 예측', html_content)
    html_content = re.sub(r'Bi-weekly|Semi-monthly', 'Daily', html_content)
    html_content = re.sub(r'완전 정적 웹 뷰 - 서버 불필요', '📅 일별 SPY 가격 예측 - 완전 정적 HTML', html_content)
    
    # Update chart title
    html_content = re.sub(r'SPY Price Prediction - Best Model \(Gradient Boosting 0\.91% MAPE\)',
                         'SPY Daily Price Prediction - Best Model (Gradient Boosting 0.91% MAPE)', 
                         html_content)
    
    # Save updated HTML
    print("💾 Saving updated dashboard...")
    with open('/root/workspace/dashboard/index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return {
        'data_points': data_count,
        'period': daily_data['data_info']['period'],
        'sampling': daily_data['data_info']['sampling']
    }

if __name__ == "__main__":
    print("🚀 Updating dashboard to daily predictions...")
    
    result = update_dashboard_html()
    
    print("✅ Dashboard updated successfully!")
    print(f"📊 Data points: {result['data_points']}")
    print(f"📅 Period: {result['period']}")  
    print(f"🎯 Sampling: {result['sampling']}")
    print("🏆 Best model: Gradient Boosting (0.91% MAPE)")
    print("📂 Updated file: /root/workspace/dashboard/index.html")