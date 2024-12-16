
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def calculate_metrics(outputs, targets):
    """Calculate precision, recall, and F1 score."""
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, average='weighted'
    )
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def confusion_matrix(outputs, targets, num_classes=10):
    """Generate confusion matrix."""
    preds = torch.argmax(outputs, dim=1)
    conf_matrix = torch.zeros(num_classes, num_classes)
    for t, p in zip(targets.view(-1), preds.view(-1)):
        conf_matrix[t.long(), p.long()] += 1
    return conf_matrix
