"""
Feature selection module with multiple selection strategies.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, RFECV, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV

from ..core.config import CONFIG
from ..core.logger import logger
from ..core.exceptions import FeatureEngineeringError


class FeatureSelector:
    """Advanced feature selection with multiple strategies."""
    
    def __init__(self):
        self.selected_features: List[str] = []
        self.feature_scores: Dict[str, float] = {}
        self.selection_method: Optional[str] = None
    
    def select_features(self, 
                       X: pd.DataFrame, 
                       y: pd.Series,
                       method: str = 'combined',
                       k: int = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select best features using specified method.
        
        Args:
            X: Feature matrix
            y: Target vector
            method: Selection method ('univariate', 'rfe', 'lasso', 'rf', 'combined')
            k: Number of features to select
            
        Returns:
            Selected features DataFrame and feature names list
        """
        k = k or CONFIG.data.feature_selection_k
        
        logger.section(f"Feature Selection - {method.upper()}", "🎯")
        logger.info(f"Selecting {k} features from {len(X.columns)} total features")
        
        try:
            if method == 'univariate':
                return self._univariate_selection(X, y, k)
            elif method == 'rfe':
                return self._recursive_feature_elimination(X, y, k)
            elif method == 'lasso':
                return self._lasso_selection(X, y, k)
            elif method == 'rf':
                return self._random_forest_selection(X, y, k)
            elif method == 'combined':
                return self._combined_selection(X, y, k)
            else:
                raise FeatureEngineeringError(f"Unknown selection method: {method}")
                
        except Exception as e:
            raise FeatureEngineeringError(f"Feature selection failed: {str(e)}")
    
    def _univariate_selection(self, 
                             X: pd.DataFrame, 
                             y: pd.Series, 
                             k: int) -> Tuple[pd.DataFrame, List[str]]:
        """Select features using univariate statistical tests."""
        logger.info("Running univariate feature selection", "📊")
        
        # Use F-statistics for feature selection
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected features
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        # Store feature scores
        scores = selector.scores_
        self.feature_scores = dict(zip(X.columns, scores))
        
        self.selected_features = selected_features
        self.selection_method = 'univariate'
        
        logger.success(f"Selected {len(selected_features)} features using univariate selection")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features
    
    def _recursive_feature_elimination(self, 
                                     X: pd.DataFrame, 
                                     y: pd.Series, 
                                     k: int) -> Tuple[pd.DataFrame, List[str]]:
        """Select features using Recursive Feature Elimination."""
        logger.info("Running recursive feature elimination", "🔄")
        
        # Use RandomForest as the estimator
        estimator = RandomForestClassifier(
            n_estimators=100, 
            random_state=CONFIG.training.random_state,
            n_jobs=-1
        )
        
        selector = RFE(estimator=estimator, n_features_to_select=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected features
        selected_mask = selector.support_
        selected_features = X.columns[selected_mask].tolist()
        
        # Store feature rankings (lower is better)
        rankings = selector.ranking_
        self.feature_scores = dict(zip(X.columns, 1.0 / rankings))  # Convert to scores
        
        self.selected_features = selected_features
        self.selection_method = 'rfe'
        
        logger.success(f"Selected {len(selected_features)} features using RFE")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features
    
    def _lasso_selection(self, 
                        X: pd.DataFrame, 
                        y: pd.Series, 
                        k: int) -> Tuple[pd.DataFrame, List[str]]:
        """Select features using Lasso regularization."""
        logger.info("Running Lasso feature selection", "🎲")
        
        # Use LassoCV for automatic alpha selection
        lasso = LassoCV(cv=5, random_state=CONFIG.training.random_state, max_iter=1000)
        selector = SelectFromModel(lasso, max_features=k)
        
        X_selected = selector.fit_transform(X, y)
        
        # Get selected features
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        # If we didn't get enough features, fall back to top k by coefficient magnitude
        if len(selected_features) < k:
            logger.warning(f"Lasso selected only {len(selected_features)} features, using top {k} by coefficient magnitude")
            
            lasso.fit(X, y)
            coef_abs = np.abs(lasso.coef_)
            top_indices = np.argsort(coef_abs)[-k:]
            selected_features = X.columns[top_indices].tolist()
            X_selected = X[selected_features].values
        
        # Store feature scores (coefficient magnitudes)
        lasso.fit(X, y)
        self.feature_scores = dict(zip(X.columns, np.abs(lasso.coef_)))
        
        self.selected_features = selected_features
        self.selection_method = 'lasso'
        
        logger.success(f"Selected {len(selected_features)} features using Lasso")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features
    
    def _random_forest_selection(self, 
                                X: pd.DataFrame, 
                                y: pd.Series, 
                                k: int) -> Tuple[pd.DataFrame, List[str]]:
        """Select features using Random Forest feature importance."""
        logger.info("Running Random Forest feature selection", "🌲")
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=200,
            random_state=CONFIG.training.random_state,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Select top k features
        top_indices = np.argsort(importances)[-k:]
        selected_features = X.columns[top_indices].tolist()
        
        # Store feature scores
        self.feature_scores = dict(zip(X.columns, importances))
        
        self.selected_features = selected_features
        self.selection_method = 'random_forest'
        
        logger.success(f"Selected {len(selected_features)} features using Random Forest")
        
        return X[selected_features], selected_features
    
    def _combined_selection(self, 
                           X: pd.DataFrame, 
                           y: pd.Series, 
                           k: int) -> Tuple[pd.DataFrame, List[str]]:
        """Select features using combined approach from multiple methods."""
        logger.info("Running combined feature selection", "🔀")
        
        # Run multiple selection methods
        methods = ['univariate', 'rfe', 'rf']
        feature_votes = {}
        
        for method in methods:
            if method == 'univariate':
                _, features = self._univariate_selection(X, y, k)
            elif method == 'rfe':
                _, features = self._recursive_feature_elimination(X, y, k)
            elif method == 'rf':
                _, features = self._random_forest_selection(X, y, k)
            
            # Count votes for each feature
            for feature in features:
                feature_votes[feature] = feature_votes.get(feature, 0) + 1
        
        # Sort features by vote count and scores
        feature_rankings = []
        for feature, votes in feature_votes.items():
            avg_score = np.mean([
                self.feature_scores.get(feature, 0) 
                for scores in [self.feature_scores] if feature in scores
            ])
            feature_rankings.append((feature, votes, avg_score))
        
        # Sort by votes (descending) then by score (descending)
        feature_rankings.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        # Select top k features
        selected_features = [item[0] for item in feature_rankings[:k]]
        
        self.selected_features = selected_features
        self.selection_method = 'combined'
        
        logger.success(f"Selected {len(selected_features)} features using combined selection")
        
        # Log top features with vote counts
        logger.info("Top selected features:")
        for i, (feature, votes, score) in enumerate(feature_rankings[:10]):
            logger.info(f"  {i+1:2d}. {feature:30s} (votes: {votes}, score: {score:.4f})")
        
        return X[selected_features], selected_features
    
    def get_feature_importance_report(self) -> pd.DataFrame:
        """Generate feature importance report."""
        if not self.feature_scores:
            logger.warning("No feature scores available")
            return pd.DataFrame()
        
        # Create report DataFrame
        report = pd.DataFrame({
            'feature': list(self.feature_scores.keys()),
            'score': list(self.feature_scores.values()),
            'selected': [f in self.selected_features for f in self.feature_scores.keys()]
        })
        
        # Sort by score
        report = report.sort_values('score', ascending=False).reset_index(drop=True)
        report['rank'] = range(1, len(report) + 1)
        
        return report[['rank', 'feature', 'score', 'selected']]
    
    def save_selection_results(self, filepath: str) -> None:
        """Save feature selection results."""
        results = {
            'method': self.selection_method,
            'selected_features': self.selected_features,
            'feature_scores': self.feature_scores,
            'selection_timestamp': pd.Timestamp.now().isoformat()
        }
        
        pd.Series(results).to_json(filepath)
        logger.success(f"Feature selection results saved to {filepath}")