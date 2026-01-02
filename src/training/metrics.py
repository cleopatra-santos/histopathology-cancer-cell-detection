"""
Métricas de Avaliação

Implementação de métricas para classificação binária.
"""

import logging
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculador de métricas para classificação binária
    
    Calcula: Accuracy, Precision, Recall, F1, AUC-ROC
    """
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset acumuladores"""
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
    
    def update(self, logits, labels):
        """
        Atualizar com batch
        
        Args:
            logits: Tensor (B, 1) - logits do modelo
            labels: Tensor (B,) - labels verdadeiros
        """
        # Converter para probabilidades
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        
        # Converter logits para predições
        preds = (probs > self.threshold).astype(int)
        
        # Converter labels
        labels_np = labels.detach().cpu().numpy()
        
        # Acumular
        self.all_preds.extend(preds.flatten())
        self.all_labels.extend(labels_np.flatten())
        self.all_probs.extend(probs.flatten())
    
    def compute(self):
        """
        Calcular todas as métricas
        
        Returns:
            dict: Métricas calculadas
        """
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        probs = np.array(self.all_probs)
        
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, zero_division=0),
            'recall': recall_score(labels, preds, zero_division=0),
            'f1': f1_score(labels, preds, zero_division=0),
            'auc_roc': roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
        }
        
        return metrics
    
    def get_confusion_matrix(self):
        """
        Obter matriz de confusão
        
        Returns:
            np.array: Confusion matrix (2, 2)
        """
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        return confusion_matrix(labels, preds)
    
    def get_classification_report(self):
        """
        Obter relatório de classificação
        
        Returns:
            str: Classification report
        """
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        return classification_report(
            labels, 
            preds,
            target_names=['Normal', 'Tumor'],
            zero_division=0
        )


def calculate_batch_metrics(logits, labels, threshold=0.5):
    """
    Calcular métricas para um único batch
    
    Args:
        logits: Tensor (B, 1)
        labels: Tensor (B,)
        threshold: float, threshold para classificação
        
    Returns:
        dict: Métricas do batch
    """
    # Probabilidades
    probs = torch.sigmoid(logits)
    
    # Predições
    preds = (probs > threshold).float()
    
    # Accuracy do batch
    correct = (preds.squeeze() == labels).float()
    accuracy = correct.mean().item()
    
    return {
        'batch_accuracy': accuracy,
        'batch_size': len(labels)
    }


def format_metrics(metrics, prefix=''):
    """
    Formatar métricas para logging
    
    Args:
        metrics: Dict com métricas
        prefix: String para adicionar antes (ex: 'val_')
        
    Returns:
        str: String formatada
    """
    formatted = []
    
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{prefix}{key}: {value:.4f}")
        else:
            formatted.append(f"{prefix}{key}: {value}")
    
    return " | ".join(formatted)


if __name__ == "__main__":
    # Teste das métricas
    print("🧪 Testando MetricsCalculator...\n")
    
    # Dados dummy
    logits = torch.randn(100, 1)
    labels = torch.randint(0, 2, (100,)).float()
    
    # Criar calculator
    calc = MetricsCalculator(threshold=0.5)
    
    # Update em batches
    for i in range(0, 100, 10):
        batch_logits = logits[i:i+10]
        batch_labels = labels[i:i+10]
        calc.update(batch_logits, batch_labels)
    
    # Compute
    metrics = calc.compute()
    
    print("📊 Métricas:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1:        {metrics['f1']:.4f}")
    print(f"   AUC-ROC:   {metrics['auc_roc']:.4f}")
    
    print("\n📊 Confusion Matrix:")
    cm = calc.get_confusion_matrix()
    print(cm)
    
    print("\n📊 Classification Report:")
    print(calc.get_classification_report())
    
    print("\n✅ MetricsCalculator funciona!")
