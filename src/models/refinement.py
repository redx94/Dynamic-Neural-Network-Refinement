import torch
import numpy as np
from typing import Dict, List, Tuple
from .neural_network import DynamicNeuralNetwork

class NetworkRefinement:
    def __init__(self, model: DynamicNeuralNetwork, 
                 complexity_threshold: float = 0.7,
                 sparsity_threshold: float = 0.3):
        self.model = model
        self.complexity_threshold = complexity_threshold
        self.sparsity_threshold = sparsity_threshold
        self.metrics_history: List[Dict[str, float]] = []
        
    def calculate_layer_complexity(self, layer_idx: int) -> float:
        """Calculate complexity metric for a layer based on weight distributions"""
        layer = self.model.layers[layer_idx]
        weights = layer.linear.weight.data
        
        # Calculate weight statistics
        weight_std = torch.std(weights).item()
        weight_mean = torch.mean(torch.abs(weights)).item()
        
        # Higher values indicate more complex weight distributions
        return min(1.0, (weight_std + weight_mean) / 2)
    
    def calculate_layer_sparsity(self, layer_idx: int) -> float:
        """Calculate sparsity metric for a layer"""
        layer = self.model.layers[layer_idx]
        weights = layer.linear.weight.data
        
        # Calculate percentage of near-zero weights
        zero_threshold = 1e-3
        num_zeros = torch.sum(torch.abs(weights) < zero_threshold).item()
        total_weights = weights.numel()
        
        return num_zeros / total_weights
    
    def analyze_network(self) -> Dict[str, List[float]]:
        """Analyze the entire network and return complexity metrics"""
        metrics = {
            'complexity': [],
            'sparsity': []
        }
        
        for i in range(len(self.model.layers)):
            complexity = self.calculate_layer_complexity(i)
            sparsity = self.calculate_layer_sparsity(i)
            
            metrics['complexity'].append(complexity)
            metrics['sparsity'].append(sparsity)
        
        self.metrics_history.append({
            'avg_complexity': np.mean(metrics['complexity']),
            'avg_sparsity': np.mean(metrics['sparsity'])
        })
        
        return metrics
    
    def refine_architecture(self) -> bool:
        """Refine network architecture based on complexity analysis"""
        metrics = self.analyze_network()
        made_changes = False
        
        # Analyze each layer
        for i in range(len(self.model.layers)):
            complexity = metrics['complexity'][i]
            sparsity = metrics['sparsity'][i]
            
            # Add neurons if complexity is high
            if complexity > self.complexity_threshold:
                current_width = self.model.layers[i].linear.out_features
                new_width = int(current_width * 1.5)  # Increase width by 50%
                self.model.adjust_layer_width(i, new_width)
                made_changes = True
                
            # Remove neurons if sparsity is high
            elif sparsity > self.sparsity_threshold:
                current_width = self.model.layers[i].linear.out_features
                new_width = max(10, int(current_width * 0.7))  # Reduce width by 30%
                self.model.adjust_layer_width(i, new_width)
                made_changes = True
        
        # Add layer if overall complexity is high
        avg_complexity = np.mean(metrics['complexity'])
        if avg_complexity > self.complexity_threshold and len(self.model.layers) < 5:
            last_layer_size = self.model.layers[-1].linear.out_features
            new_layer_size = max(10, int(last_layer_size * 0.7))
            self.model.add_layer(new_layer_size)
            made_changes = True
            
        return made_changes
    
    def get_refinement_history(self) -> List[Dict[str, float]]:
        """Return the history of refinement metrics"""
        return self.metrics_history
