"""
Evaluation Module for E-commerce Text Classification

This module provides comprehensive evaluation metrics and visualization
tools for comparing model performance across different embeddings.

Metrics included:
- Accuracy
- Precision, Recall, F1-Score (macro, weighted, per-class)
- Confusion Matrix
- Classification Report

References:
- Powers, D. M. (2011). Evaluation: From Precision, Recall and F-Measure
  to ROC, Informedness, Markedness & Correlation.
"""

import os
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve
)
from scipy import sparse


class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison.
    
    Provides methods for:
    - Computing classification metrics
    - Generating confusion matrices
    - Comparing models across embeddings
    - Creating visualization plots
    - Exporting results to tables
    """
    
    def __init__(
        self,
        class_names: List[str] = None,
        results_dir: str = 'results'
    ):
        """
        Initialize evaluator.
        
        Args:
            class_names: List of class labels
            results_dir: Directory for saving results
        """
        self.class_names = class_names
        self.results_dir = results_dir
        self.all_results = []
        
        # Create results directories
        os.makedirs(f"{results_dir}/figures", exist_ok=True)
        os.makedirs(f"{results_dir}/tables", exist_ok=True)
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
            class_name = self.class_names[i] if self.class_names and i < len(self.class_names) else f'class_{i}'
            metrics[f'precision_{class_name}'] = p
            metrics[f'recall_{class_name}'] = r
            metrics[f'f1_{class_name}'] = f
        
        return metrics
    
    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        embedding_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained classifier
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            embedding_name: Name of the embedding
            
        Returns:
            Dictionary with evaluation results
        """
        # Handle sparse matrices for certain models
        X = X_test
        if sparse.issparse(X_test):
            if hasattr(model, 'name') and 'Gradient' in model.name:
                X = X_test.toarray()
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Get probabilities if available
        y_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X)
            except:
                pass
        
        # Compute metrics
        metrics = self.compute_metrics(y_test, y_pred, y_proba)
        
        # Get confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Create result entry
        result = {
            'model': model_name,
            'embedding': embedding_name,
            'metrics': metrics,
            'confusion_matrix': cm,
            'y_true': y_test,
            'y_pred': y_pred
        }
        
        self.all_results.append(result)
        
        return result
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """
        Generate sklearn classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Formatted classification report
        """
        return classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            zero_division=0
        )
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        model_name: str,
        embedding_name: str,
        save: bool = True,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot confusion matrix heatmap.
        
        Args:
            cm: Confusion matrix array
            model_name: Name of the model
            embedding_name: Name of the embedding
            save: Whether to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names or range(cm.shape[1]),
            yticklabels=self.class_names or range(cm.shape[0]),
            ax=ax
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(f'Confusion Matrix: {model_name} + {embedding_name}', fontsize=14)
        
        plt.tight_layout()
        
        if save:
            filepath = f"{self.results_dir}/figures/cm_{model_name}_{embedding_name}.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved confusion matrix to {filepath}")
        
        return fig
    
    def create_comparison_table(
        self,
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Create comparison table across all experiments.
        
        Args:
            metrics: List of metrics to include (default: main metrics)
            
        Returns:
            DataFrame with comparison results
        """
        if metrics is None:
            metrics = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
        
        rows = []
        for result in self.all_results:
            row = {
                'Model': result['model'],
                'Embedding': result['embedding']
            }
            for metric in metrics:
                row[metric.replace('_', ' ').title()] = result['metrics'].get(metric, 0)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def create_pivot_table(
        self,
        metric: str = 'f1_weighted'
    ) -> pd.DataFrame:
        """
        Create pivot table: Models vs Embeddings.
        
        Args:
            metric: Metric to show in cells
            
        Returns:
            Pivot table DataFrame
        """
        df = self.create_comparison_table([metric])
        metric_col = metric.replace('_', ' ').title()
        
        pivot = df.pivot(
            index='Model',
            columns='Embedding',
            values=metric_col
        )
        
        return pivot
    
    def plot_model_comparison(
        self,
        metric: str = 'f1_weighted',
        save: bool = True,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot bar chart comparing models across embeddings.
        
        Args:
            metric: Metric to compare
            save: Whether to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        df = self.create_comparison_table([metric])
        metric_col = metric.replace('_', ' ').title()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        models = df['Model'].unique()
        embeddings = df['Embedding'].unique()
        x = np.arange(len(models))
        width = 0.25
        
        for i, emb in enumerate(embeddings):
            emb_data = df[df['Embedding'] == emb]
            values = [emb_data[emb_data['Model'] == m][metric_col].values[0] 
                     if len(emb_data[emb_data['Model'] == m]) > 0 else 0 
                     for m in models]
            ax.bar(x + i * width, values, width, label=emb)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(metric_col, fontsize=12)
        ax.set_title(f'Model Comparison: {metric_col}', fontsize=14)
        ax.set_xticks(x + width)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(title='Embedding')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = f"{self.results_dir}/figures/comparison_{metric}.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved comparison plot to {filepath}")
        
        return fig
    
    def plot_embedding_comparison(
        self,
        metric: str = 'f1_weighted',
        save: bool = True,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot comparison focusing on embeddings.
        
        Args:
            metric: Metric to compare
            save: Whether to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        df = self.create_comparison_table([metric])
        metric_col = metric.replace('_', ' ').title()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Group by embedding and compute mean
        emb_means = df.groupby('Embedding')[metric_col].mean().sort_values(ascending=False)
        
        colors = sns.color_palette('viridis', len(emb_means))
        bars = ax.bar(emb_means.index, emb_means.values, color=colors)
        
        ax.set_xlabel('Embedding Type', fontsize=12)
        ax.set_ylabel(f'Mean {metric_col}', fontsize=12)
        ax.set_title(f'Embedding Comparison: Average {metric_col} Across Models', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            filepath = f"{self.results_dir}/figures/embedding_comparison_{metric}.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved embedding comparison to {filepath}")
        
        return fig
    
    def save_results_to_csv(self, filename: str = None):
        """
        Save all results to CSV.
        
        Args:
            filename: Output filename (default: auto-generated)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_results_{timestamp}.csv"
        
        df = self.create_comparison_table()
        filepath = f"{self.results_dir}/tables/{filename}"
        df.to_csv(filepath, index=False)
        print(f"Saved results to {filepath}")
        
        return filepath
    
    def save_pivot_tables(self):
        """Save pivot tables for key metrics."""
        metrics = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
        
        for metric in metrics:
            pivot = self.create_pivot_table(metric)
            filepath = f"{self.results_dir}/tables/pivot_{metric}.csv"
            pivot.to_csv(filepath)
            print(f"Saved pivot table to {filepath}")
    
    def generate_report(self) -> str:
        """
        Generate text summary report.
        
        Returns:
            Formatted report string
        """
        df = self.create_comparison_table()
        
        report = []
        report.append("=" * 60)
        report.append("EXPERIMENT RESULTS SUMMARY")
        report.append("=" * 60)
        report.append("")
        
        # Best overall
        best_idx = df['F1 Weighted'].idxmax()
        best = df.loc[best_idx]
        report.append("BEST OVERALL CONFIGURATION:")
        report.append(f"  Model: {best['Model']}")
        report.append(f"  Embedding: {best['Embedding']}")
        report.append(f"  F1 Weighted: {best['F1 Weighted']:.4f}")
        report.append(f"  Accuracy: {best['Accuracy']:.4f}")
        report.append("")
        
        # Best per embedding
        report.append("BEST MODEL PER EMBEDDING:")
        for emb in df['Embedding'].unique():
            emb_df = df[df['Embedding'] == emb]
            best_idx = emb_df['F1 Weighted'].idxmax()
            best = emb_df.loc[best_idx]
            report.append(f"  {emb}: {best['Model']} (F1: {best['F1 Weighted']:.4f})")
        report.append("")
        
        # Best per model
        report.append("BEST EMBEDDING PER MODEL:")
        for model in df['Model'].unique():
            model_df = df[df['Model'] == model]
            best_idx = model_df['F1 Weighted'].idxmax()
            best = model_df.loc[best_idx]
            report.append(f"  {model}: {best['Embedding']} (F1: {best['F1 Weighted']:.4f})")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def clear_results(self):
        """Clear all stored results."""
        self.all_results = []


def print_experiment_table(results: List[Dict], metrics: List[str] = None):
    """
    Print formatted experiment results table.
    
    Args:
        results: List of result dictionaries
        metrics: Metrics to display
    """
    if metrics is None:
        metrics = ['accuracy', 'f1_weighted']
    
    try:
        from tabulate import tabulate
        
        headers = ['Model', 'Embedding'] + [m.replace('_', ' ').title() for m in metrics]
        rows = []
        
        for r in results:
            row = [r['model'], r['embedding']]
            row.extend([f"{r['metrics'].get(m, 0):.4f}" for m in metrics])
            rows.append(row)
        
        print(tabulate(rows, headers=headers, tablefmt='grid'))
    except ImportError:
        # Fallback without tabulate
        for r in results:
            print(f"{r['model']} + {r['embedding']}:")
            for m in metrics:
                print(f"  {m}: {r['metrics'].get(m, 0):.4f}")
