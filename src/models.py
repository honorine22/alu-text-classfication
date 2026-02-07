"""
Traditional ML Model Implementations for E-commerce Text Classification

This module implements traditional machine learning classifiers:
1. Logistic Regression
2. Support Vector Machine (SVM)
3. Random Forest
4. Multinomial Naive Bayes
5. Gradient Boosting

References:
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). Elements of Statistical Learning.
- Joachims, T. (1998). Text Categorization with Support Vector Machines.
"""

import pickle
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class BaseClassifier(ABC):
    """Abstract base class for classifiers."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_fitted = False
        self.best_params = None
        self.cv_scores = None
    
    @abstractmethod
    def get_model(self) -> Any:
        """Return the sklearn model instance."""
        pass
    
    @abstractmethod
    def get_param_grid(self) -> Dict[str, List]:
        """Return hyperparameter grid for tuning."""
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseClassifier':
        """
        Fit the model on training data.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Self
        """
        self.model = self.get_model()
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Get prediction probabilities if available.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability matrix or None
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None
    
    def tune_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = 'f1_weighted',
        n_jobs: int = -1
    ) -> 'BaseClassifier':
        """
        Tune hyperparameters using grid search with cross-validation.
        
        Args:
            X: Feature matrix
            y: Target labels
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            
        Returns:
            Self with best parameters
        """
        param_grid = self.get_param_grid()
        
        grid_search = GridSearchCV(
            self.get_model(),
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.cv_scores = grid_search.cv_results_
        self.is_fitted = True
        
        print(f"\nBest parameters for {self.name}:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return self
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = 'f1_weighted'
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Target labels
            cv: Number of folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with CV statistics
        """
        model = self.model if self.is_fitted else self.get_model()
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        return {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist()
        }
    
    def save(self, filepath: str):
        """Save model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'name': self.name,
                'model': self.model,
                'best_params': self.best_params
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseClassifier':
        """Load model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls()
        instance.model = data['model']
        instance.best_params = data.get('best_params')
        instance.is_fitted = True
        return instance


class LogisticRegressionClassifier(BaseClassifier):
    """
    Logistic Regression Classifier.
    
    Logistic Regression is effective for text classification due to its
    ability to handle high-dimensional sparse data and provide probabilistic
    outputs. The regularization parameter C controls model complexity.
    
    Reference:
        Hastie, T., et al. (2009). The Elements of Statistical Learning.
    """
    
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = 'lbfgs',
        multi_class: str = 'multinomial',
        class_weight: str = 'balanced'
    ):
        super().__init__(name='Logistic Regression')
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.multi_class = multi_class
        self.class_weight = class_weight
    
    def get_model(self) -> LogisticRegression:
        return LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver=self.solver,
            multi_class=self.multi_class,
            class_weight=self.class_weight,
            random_state=42,
            n_jobs=-1
        )
    
    def get_param_grid(self) -> Dict[str, List]:
        return {
            'C': [0.01, 0.1, 1.0, 10.0],
            'solver': ['lbfgs', 'saga'],
            'max_iter': [500, 1000]
        }


class SVMClassifier(BaseClassifier):
    """
    Support Vector Machine Classifier.
    
    SVMs are highly effective for text classification, particularly with
    high-dimensional TF-IDF features. Linear SVM is preferred for efficiency
    with large feature sets.
    
    Reference:
        Joachims, T. (1998). Text Categorization with Support Vector Machines:
        Learning with Many Relevant Features. ECML.
    """
    
    def __init__(
        self,
        C: float = 1.0,
        kernel: str = 'linear',
        class_weight: str = 'balanced'
    ):
        super().__init__(name='SVM')
        self.C = C
        self.kernel = kernel
        self.class_weight = class_weight
    
    def get_model(self) -> LinearSVC:
        return LinearSVC(
            C=self.C,
            class_weight=self.class_weight,
            random_state=42,
            max_iter=5000,
            dual=True
        )
    
    def get_param_grid(self) -> Dict[str, List]:
        return {
            'C': [0.1, 1.0, 10.0],
            'loss': ['hinge', 'squared_hinge']
        }


class RandomForestClassifierWrapper(BaseClassifier):
    """
    Random Forest Classifier.
    
    Random Forest combines multiple decision trees to improve classification
    accuracy and reduce overfitting. It handles both sparse and dense features
    and provides feature importance scores.
    
    Reference:
        Breiman, L. (2001). Random Forests. Machine Learning.
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        class_weight: str = 'balanced'
    ):
        super().__init__(name='Random Forest')
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
    
    def get_model(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            class_weight=self.class_weight,
            random_state=42,
            n_jobs=-1
        )
    
    def get_param_grid(self) -> Dict[str, List]:
        return {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
    
    def get_feature_importance(self, feature_names: List[str] = None) -> List[Tuple[str, float]]:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        importances = self.model.feature_importances_
        if feature_names:
            return sorted(zip(feature_names, importances), 
                         key=lambda x: x[1], reverse=True)
        return sorted(enumerate(importances), key=lambda x: x[1], reverse=True)


class NaiveBayesClassifier(BaseClassifier):
    """
    Multinomial Naive Bayes Classifier.
    
    Naive Bayes is a fast, probabilistic classifier that works well with
    TF-IDF features. It assumes feature independence and is particularly
    effective for text classification.
    
    Note: For dense embeddings (Word2Vec, GloVe), use GaussianNB instead.
    
    Reference:
        McCallum, A., & Nigam, K. (1998). A Comparison of Event Models for
        Naive Bayes Text Classification.
    """
    
    def __init__(self, alpha: float = 1.0, use_gaussian: bool = False):
        super().__init__(name='Naive Bayes')
        self.alpha = alpha
        self.use_gaussian = use_gaussian
    
    def get_model(self):
        if self.use_gaussian:
            return GaussianNB()
        return MultinomialNB(alpha=self.alpha)
    
    def get_param_grid(self) -> Dict[str, List]:
        if self.use_gaussian:
            return {}
        return {
            'alpha': [0.01, 0.1, 1.0, 10.0]
        }


class GradientBoostingClassifierWrapper(BaseClassifier):
    """
    Gradient Boosting Classifier.
    
    Gradient Boosting builds an ensemble of weak learners sequentially,
    with each learner correcting errors of the previous ones. It typically
    achieves high accuracy but requires more tuning.
    
    Reference:
        Friedman, J. H. (2001). Greedy Function Approximation: A Gradient
        Boosting Machine.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3
    ):
        super().__init__(name='Gradient Boosting')
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
    
    def get_model(self) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42
        )
    
    def get_param_grid(self) -> Dict[str, List]:
        return {
            'n_estimators': [50, 100],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }


class ModelTrainer:
    """
    Model training orchestrator for systematic experiments.
    
    Handles training multiple models with different embeddings,
    hyperparameter tuning, and result collection.
    """
    
    AVAILABLE_MODELS = {
        'logistic_regression': LogisticRegressionClassifier,
        'svm': SVMClassifier,
        'random_forest': RandomForestClassifierWrapper,
        'naive_bayes': NaiveBayesClassifier,
        'gradient_boosting': GradientBoostingClassifierWrapper
    }
    
    def __init__(self, models: List[str] = None):
        """
        Initialize trainer with specified models.
        
        Args:
            models: List of model names to use (default: all)
        """
        if models is None:
            models = list(self.AVAILABLE_MODELS.keys())
        
        self.model_names = models
        self.trained_models = {}
    
    def get_model_instance(
        self,
        model_name: str,
        embedding_type: str = 'tfidf',
        **kwargs
    ) -> BaseClassifier:
        """
        Get a model instance configured for the embedding type.
        
        Args:
            model_name: Name of the model
            embedding_type: Type of embedding being used
            **kwargs: Additional model parameters
            
        Returns:
            Configured classifier instance
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Special handling for Naive Bayes with dense embeddings
        if model_name == 'naive_bayes' and embedding_type in ['word2vec', 'glove']:
            kwargs['use_gaussian'] = True
        
        return self.AVAILABLE_MODELS[model_name](**kwargs)
    
    def train_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        embedding_name: str = 'unknown',
        tune_hyperparameters: bool = False,
        **kwargs
    ) -> BaseClassifier:
        """
        Train a single model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            embedding_name: Name of the embedding for logging
            tune_hyperparameters: Whether to tune hyperparameters
            **kwargs: Additional model parameters
            
        Returns:
            Trained classifier
        """
        # Determine embedding type from name
        embedding_type = 'tfidf'
        if 'word2vec' in embedding_name.lower():
            embedding_type = 'word2vec'
        elif 'glove' in embedding_name.lower():
            embedding_type = 'glove'
        
        model = self.get_model_instance(model_name, embedding_type, **kwargs)
        
        print(f"\nTraining {model.name} with {embedding_name}...")
        
        # Convert sparse to dense for models that require it
        X = X_train
        if sparse.issparse(X_train):
            if model_name in ['gradient_boosting']:
                X = X_train.toarray()
            elif model_name == 'naive_bayes' and embedding_type in ['word2vec', 'glove']:
                X = X_train
        
        if tune_hyperparameters:
            model.tune_hyperparameters(X, y_train)
        else:
            model.fit(X, y_train)
        
        # Store trained model
        key = f"{model_name}_{embedding_name}"
        self.trained_models[key] = model
        
        return model
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        embedding_name: str = 'unknown',
        tune_hyperparameters: bool = False
    ) -> Dict[str, BaseClassifier]:
        """
        Train all configured models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            embedding_name: Name of the embedding
            tune_hyperparameters: Whether to tune hyperparameters
            
        Returns:
            Dictionary of trained models
        """
        results = {}
        for model_name in self.model_names:
            try:
                model = self.train_model(
                    model_name, X_train, y_train,
                    embedding_name, tune_hyperparameters
                )
                results[model_name] = model
            except Exception as e:
                print(f"Error training {model_name}: {e}")
        
        return results
    
    def get_trained_model(self, model_name: str, embedding_name: str) -> Optional[BaseClassifier]:
        """Get a specific trained model."""
        key = f"{model_name}_{embedding_name}"
        return self.trained_models.get(key)
