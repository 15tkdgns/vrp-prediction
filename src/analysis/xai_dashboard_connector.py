#!/usr/bin/env python3
"""
XAI Dashboard Connector
검증된 XAI 분석 결과를 대시보드와 연동하는 모듈
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class XAIDashboardConnector:
    """XAI 분석 결과를 대시보드 형태로 변환하는 클래스"""

    def __init__(self, xai_results_file: str = None):
        """
        XAI Dashboard Connector 초기화

        Args:
            xai_results_file: XAI 분석 결과 JSON 파일 경로
        """
        self.xai_results_file = xai_results_file
        self.xai_data = None
        self.dashboard_data = {}

        if xai_results_file:
            self.load_xai_results(xai_results_file)

    def load_xai_results(self, filepath: str):
        """XAI 분석 결과 로드"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.xai_data = json.load(f)
            print(f"✅ XAI 분석 결과 로드: {filepath}")
        except Exception as e:
            print(f"❌ XAI 결과 로드 실패: {e}")

    def prepare_feature_importance_chart(self, top_n: int = 10) -> Dict:
        """특성 중요도 차트 데이터 준비"""
        if not self.xai_data or 'shap_analysis' not in self.xai_data:
            return {}

        shap_analysis = self.xai_data['shap_analysis']
        feature_importance = shap_analysis.get('feature_importance', [])

        # Top N 특성 선택
        top_features = feature_importance[:top_n]

        chart_data = {
            'type': 'horizontal_bar',
            'title': f'Top {top_n} Feature Importance (SHAP)',
            'subtitle': 'Based on absolute SHAP values',
            'data': {
                'labels': [f['feature'] for f in top_features],
                'values': [f['shap_importance'] for f in top_features],
                'colors': self._generate_colors(len(top_features)),
                'tooltips': [self._generate_tooltip(f) for f in top_features]
            },
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'scales': {
                    'x': {'beginAtZero': True, 'title': {'display': True, 'text': 'SHAP Importance'}},
                    'y': {'title': {'display': True, 'text': 'Features'}}
                }
            }
        }

        return chart_data

    def prepare_feature_groups_analysis(self) -> Dict:
        """특성 그룹별 분석 데이터 준비"""
        if not self.xai_data or 'shap_analysis' not in self.xai_data:
            return {}

        shap_analysis = self.xai_data['shap_analysis']
        group_analysis = shap_analysis.get('group_analysis', {})

        if not group_analysis:
            return {}

        groups_data = []
        for group_name, group_info in group_analysis.items():
            groups_data.append({
                'group': group_name.title(),
                'importance': group_info['importance'],
                'feature_count': group_info['feature_count'],
                'features': group_info['features'][:3],  # Top 3 features per group
                'percentage': (group_info['importance'] / sum(g['importance'] for g in group_analysis.values())) * 100
            })

        # 중요도로 정렬
        groups_data.sort(key=lambda x: x['importance'], reverse=True)

        return {
            'type': 'pie_and_table',
            'title': 'Feature Groups Analysis',
            'subtitle': 'Importance distribution across feature categories',
            'pie_data': {
                'labels': [g['group'] for g in groups_data],
                'values': [g['percentage'] for g in groups_data],
                'colors': self._generate_colors(len(groups_data))
            },
            'table_data': groups_data
        }

    def prepare_model_performance_summary(self) -> Dict:
        """모델 성능 요약 데이터 준비"""
        if not self.xai_data:
            return {}

        model_perf = self.xai_data.get('model_performance', {})
        insights = self.xai_data.get('professional_insights', {})

        executive_summary = insights.get('executive_summary', {})
        performance_insights = insights.get('model_performance_insights', {})

        summary_data = {
            'model_metrics': {
                'test_r2': model_perf.get('test_r2', 0),
                'train_r2': model_perf.get('train_r2', 0),
                'test_rmse': model_perf.get('test_rmse', 0),
                'best_alpha': model_perf.get('best_alpha', 1.0),
                'n_features': model_perf.get('n_features', 0),
                'test_samples': model_perf.get('test_samples', 0)
            },
            'business_assessment': {
                'quality': executive_summary.get('model_quality', 'Unknown'),
                'predictive_power': executive_summary.get('predictive_power', 'Unknown'),
                'readiness': executive_summary.get('business_readiness', 'Unknown'),
                'r2_interpretation': performance_insights.get('r2_interpretation', 'Unknown')
            },
            'key_metrics_cards': [
                {
                    'title': 'Test R²',
                    'value': f"{model_perf.get('test_r2', 0):.4f}",
                    'subtitle': 'Prediction Accuracy',
                    'color': self._get_performance_color(model_perf.get('test_r2', 0)),
                    'icon': '📈'
                },
                {
                    'title': 'Model Quality',
                    'value': executive_summary.get('model_quality', 'Unknown'),
                    'subtitle': 'Business Assessment',
                    'color': 'success' if 'Excellent' in str(executive_summary.get('model_quality', '')) else 'primary',
                    'icon': '🏆'
                },
                {
                    'title': 'Features',
                    'value': str(model_perf.get('n_features', 0)),
                    'subtitle': 'Variables Analyzed',
                    'color': 'info',
                    'icon': '🎯'
                },
                {
                    'title': 'Readiness',
                    'value': executive_summary.get('business_readiness', 'Unknown'),
                    'subtitle': 'Deployment Status',
                    'color': 'success' if 'Ready' in str(executive_summary.get('business_readiness', '')) else 'warning',
                    'icon': '🚀'
                }
            ]
        }

        return summary_data

    def prepare_business_insights(self) -> Dict:
        """비즈니스 인사이트 데이터 준비"""
        if not self.xai_data:
            return {}

        insights = self.xai_data.get('professional_insights', {})

        # 주요 인사이트 섹션들
        sections = {
            'executive_summary': {
                'title': 'Executive Summary',
                'icon': '👔',
                'data': insights.get('executive_summary', {})
            },
            'feature_insights': {
                'title': 'Feature Insights',
                'icon': '🔍',
                'data': insights.get('feature_insights', [])[:5]  # Top 5
            },
            'risk_management': {
                'title': 'Risk Management',
                'icon': '⚠️',
                'data': insights.get('risk_management_insights', {})
            },
            'trading_strategies': {
                'title': 'Trading Strategies',
                'icon': '📊',
                'data': insights.get('trading_strategy_insights', {})
            }
        }

        # 핵심 발견사항 (Key Findings)
        key_findings = []
        if 'shap_analysis' in self.xai_data:
            top_feature = self.xai_data['shap_analysis']['feature_importance'][0]
            key_findings.append({
                'title': 'Most Important Feature',
                'description': f"{top_feature['feature']} dominates predictions with {top_feature['shap_importance']:.4f} importance",
                'impact': 'High',
                'category': 'Feature Analysis'
            })

        model_r2 = self.xai_data.get('model_performance', {}).get('test_r2', 0)
        if model_r2 > 0.3:
            key_findings.append({
                'title': 'Excellent Predictive Power',
                'description': f"Model explains {model_r2*100:.1f}% of volatility variation - outstanding for financial markets",
                'impact': 'Very High',
                'category': 'Model Performance'
            })

        return {
            'sections': sections,
            'key_findings': key_findings,
            'summary_stats': {
                'total_features_analyzed': len(insights.get('feature_insights', [])),
                'business_ready': 'Production Ready' in str(insights.get('executive_summary', {}).get('business_readiness', '')),
                'analysis_date': self.xai_data.get('metadata', {}).get('analysis_timestamp', datetime.now().isoformat())
            }
        }

    def prepare_temporal_analysis(self) -> Dict:
        """시간적 분석 데이터 준비"""
        if not self.xai_data or 'shap_analysis' not in self.xai_data:
            return {}

        temporal_analysis = self.xai_data['shap_analysis'].get('temporal_analysis', {})

        if not temporal_analysis:
            return {}

        # 최근 vs 과거 특성 중요도 비교
        recent_features = temporal_analysis.get('recent_period_top_features', [])
        past_features = temporal_analysis.get('past_period_top_features', [])

        comparison_data = {
            'type': 'comparison_chart',
            'title': 'Feature Importance: Recent vs Past',
            'subtitle': 'Evolution of feature importance over time',
            'data': {
                'recent': {
                    'labels': [f['feature'] for f in recent_features[:5]],
                    'values': [f['importance'] for f in recent_features[:5]]
                },
                'past': {
                    'labels': [f['feature'] for f in past_features[:5]],
                    'values': [f['importance'] for f in past_features[:5]]
                }
            }
        }

        return comparison_data

    def prepare_interaction_network(self) -> Dict:
        """특성 상호작용 네트워크 데이터 준비"""
        if not self.xai_data or 'shap_analysis' not in self.xai_data:
            return {}

        interaction_analysis = self.xai_data['shap_analysis'].get('interaction_analysis', [])

        if not interaction_analysis:
            return {}

        # 네트워크 노드 및 엣지 준비
        nodes = set()
        edges = []

        for interaction in interaction_analysis[:8]:  # Top 8 interactions
            feature1 = interaction['feature_1']
            feature2 = interaction['feature_2']
            strength = interaction['interaction_strength']

            nodes.add(feature1)
            nodes.add(feature2)

            edges.append({
                'source': feature1,
                'target': feature2,
                'weight': strength,
                'correlation': interaction['correlation']
            })

        network_data = {
            'type': 'network_graph',
            'title': 'Feature Interactions Network',
            'subtitle': 'Correlations between top features in SHAP space',
            'nodes': [{'id': node, 'label': node} for node in nodes],
            'edges': edges,
            'options': {
                'physics': True,
                'interaction': {'hover': True},
                'layout': {'improvedLayout': True}
            }
        }

        return network_data

    def create_dashboard_json(self) -> Dict:
        """대시보드용 종합 JSON 데이터 생성"""
        dashboard_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'source': 'Verified XAI Analysis',
                'model_type': 'Ridge Regression',
                'target': '5-day Future Volatility'
            },
            'feature_importance': self.prepare_feature_importance_chart(),
            'feature_groups': self.prepare_feature_groups_analysis(),
            'model_performance': self.prepare_model_performance_summary(),
            'business_insights': self.prepare_business_insights(),
            'temporal_analysis': self.prepare_temporal_analysis(),
            'interaction_network': self.prepare_interaction_network()
        }

        return dashboard_data

    def save_dashboard_data(self, output_file: str = None) -> str:
        """대시보드 데이터를 JSON 파일로 저장"""
        dashboard_data = self.create_dashboard_json()

        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"/root/workspace/data/processed/xai_dashboard_data_{timestamp}.json"

        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dashboard_data, f, indent=2, ensure_ascii=False)

        print(f"✅ 대시보드 데이터 저장: {output_path}")
        return str(output_path)

    def _generate_colors(self, n_colors: int) -> List[str]:
        """차트용 색상 팔레트 생성"""
        colors = [
            '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0',
            '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF',
            '#4BC0C0', '#FF6384', '#36A2EB', '#FFCE56'
        ]
        return colors[:n_colors] if n_colors <= len(colors) else colors * (n_colors // len(colors) + 1)

    def _generate_tooltip(self, feature_info: Dict) -> str:
        """특성별 툴팁 텍스트 생성"""
        feature_name = feature_info['feature']
        importance = feature_info['shap_importance']

        return f"{feature_name}<br>Importance: {importance:.4f}<br>Click for details"

    def _get_performance_color(self, r2_score: float) -> str:
        """R² 점수에 따른 색상 반환"""
        if r2_score > 0.3:
            return 'success'
        elif r2_score > 0.2:
            return 'warning'
        elif r2_score > 0.1:
            return 'info'
        else:
            return 'danger'

    def generate_summary_report(self) -> str:
        """요약 보고서 생성"""
        if not self.xai_data:
            return "XAI 분석 데이터가 없습니다."

        model_perf = self.xai_data.get('model_performance', {})
        shap_analysis = self.xai_data.get('shap_analysis', {})

        report = f"""
🏆 XAI 분석 요약 보고서
========================

📊 모델 성능:
- Test R²: {model_perf.get('test_r2', 0):.4f}
- Train R²: {model_perf.get('train_r2', 0):.4f}
- 최적 Alpha: {model_perf.get('best_alpha', 1.0)}
- 특성 수: {model_perf.get('n_features', 0)}

🎯 주요 발견:
"""

        if 'feature_importance' in shap_analysis:
            top_3 = shap_analysis['feature_importance'][:3]
            for i, feature in enumerate(top_3, 1):
                report += f"- {i}위: {feature['feature']} ({feature['shap_importance']:.4f})\n"

        insights = self.xai_data.get('professional_insights', {})
        executive_summary = insights.get('executive_summary', {})

        report += f"""
💼 비즈니스 평가:
- 모델 품질: {executive_summary.get('model_quality', 'Unknown')}
- 배포 준비도: {executive_summary.get('business_readiness', 'Unknown')}
- 예측력: {executive_summary.get('predictive_power', 'Unknown')}
"""

        return report


def main():
    """메인 실행 함수"""
    print("🔗 XAI Dashboard Connector 시작...")

    # 최신 XAI 분석 파일 찾기
    xai_dir = Path("/root/workspace/data/xai_analysis")
    if not xai_dir.exists():
        print("❌ XAI 분석 디렉토리 없음")
        return

    xai_files = list(xai_dir.glob("verified_xai_analysis_*.json"))
    if not xai_files:
        print("❌ XAI 분석 파일 없음")
        return

    # 가장 최신 파일 선택
    latest_file = max(xai_files, key=lambda x: x.stat().st_mtime)
    print(f"📄 최신 XAI 파일 선택: {latest_file}")

    # Connector 초기화 및 데이터 변환
    connector = XAIDashboardConnector(str(latest_file))

    # 대시보드 데이터 생성
    dashboard_data = connector.create_dashboard_json()

    # 저장
    dashboard_file = connector.save_dashboard_data()

    # 요약 보고서 출력
    print("\n" + "="*50)
    print(connector.generate_summary_report())
    print("="*50)

    print(f"\n✅ 대시보드 연동 데이터 준비 완료!")
    print(f"📁 대시보드 데이터 파일: {dashboard_file}")

    return connector


if __name__ == '__main__':
    connector = main()