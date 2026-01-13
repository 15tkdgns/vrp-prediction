"""
Dashboard tabs module
"""
from .tab_overview import render_overview_tab
from .tab_methodology import render_methodology_tab
from .tab_results import render_results_tab
from .tab_validation import render_validation_tab
from .tab_references import render_references_tab
from .tab_literature import render_prior_work_tab

__all__ = [
    'render_overview_tab',
    'render_methodology_tab',
    'render_results_tab',
    'render_validation_tab',
    'render_references_tab',
    'render_prior_work_tab'
]
