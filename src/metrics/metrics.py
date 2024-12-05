import numpy as np
from typing import Dict, List, Tuple
import torch
from torch import nn

class MetricsCalculator:
    @staticmethod
    def calculate_model_complexity(model: nn.Module) -> float:
        """Calculate overall model complexity"""
        total_params = 0
        total_neurons = 0
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                total_params += param.numel()
                total_neurons += param.size(0)
                
        return float(total_params) / float(max(1, total_neurons))
    
    @staticmethod
    def calculate_layer_importance(layer: nn.Linear) -> float:
        """Calculate importance score for a layer based on weight magnitudes"""
        weights = layer.weight.data
        importance = torch.mean(torch.abs(weights)).item()
        return importance
    
    @staticmethod
    def calculate_gradient_metrics(model: nn.Module) -> Dict[str, float]:
        """Calculate gradient-based metrics"""
        metrics = {
            'mean_gradient': 0.0,
            'gradient_variance': 0.0
        }
        
        gradients = []
        for param in model.parameters():
            if param.grad is not None:
                gradients.extend(param.grad.cpu().numpy().flatten())
                
        if gradients:
            gradients = np.array(gradients)
            metrics['mean_gradient'] = float(np.mean(np.abs(gradients)))
            metrics['gradient_variance'] = float(np.var(gradients))
            
        return metrics
    
    @staticmethod
    def calculate_prediction_stats(outputs: torch.Tensor, 
                                 targets: torch.Tensor) -> Dict[str, float]:
        """Calculate prediction statistics"""
        with torch.no_grad():
            error = outputs - targets
            metrics = {
                'mse': float(torch.mean(error ** 2).item()),
                'mae': float(torch.mean(torch.abs(error)).item()),
                'output_variance': float(torch.var(outputs).item())
            }
        return metrics
    
    @staticmethod
    def calculate_layer_metrics(layer: nn.Linear) -> Dict[str, float]:
        """Calculate various metrics for a single layer"""
        weights = layer.weight.data
        metrics = {
            'weight_mean': float(torch.mean(weights).item()),
            'weight_std': float(torch.std(weights).item()),
            'weight_l1': float(torch.norm(weights, p=1).item()),
            'weight_l2': float(torch.norm(weights, p=2).item()),
            'neuron_count': weights.size(0)
        }
        return metrics
