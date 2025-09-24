"""
Refactored Stock Prediction System

A comprehensive, modular stock prediction system with clean architecture.

Key Features:
- Modular design with separation of concerns
- Configuration management
- Structured logging
- Comprehensive error handling  
- Advanced feature engineering
- Multiple ensemble methods
- Model factory pattern
- Type hints and documentation

Architecture:
- core/: Configuration, logging, exceptions
- data/: Data loading and preprocessing
- features/: Feature engineering and selection
- models/: Model creation and ensemble methods
- training/: Complete training pipeline
"""

__version__ = "2.0.0"
__author__ = "Stock Prediction AI System"
__description__ = "Modular stock prediction system with advanced ML techniques"

# Core imports for easy access (avoid circular imports)
__all__ = ['__version__', '__author__', '__description__']