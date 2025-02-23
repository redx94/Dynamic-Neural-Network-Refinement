import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np

class ArchitectureController:
    def __init__(self, 
                 meta_controller: MetaController,
                 min_layers: int = 2,
                 max_layers: int = 10):
        self.meta_controller = meta_controller
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.architecture_history = []
        
    def adapt_architecture(self, 
                         current_model: nn.Module,
                         performance_metrics: Dict[str, float]) -> nn.Module:
        network_state = self._gather_network_state(current_model, performance_metrics)
        action = self.meta_controller.get_architecture_action(network_state)
        
        # Apply architectural changes
        modified_model = self._apply_architecture_action(current_model, action)
        self.architecture_history.append({
            'action': action,
            'metrics': performance_metrics,
            'architecture': self._get_architecture_summary(modified_model)
        })
        
        return modified_model
    
    def _apply_architecture_action(self, 
                                 model: nn.Module, 
                                 action: Dict[str, float]) -> nn.Module:
        # Modify layer widths
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                width_modifier = max(0.5, min(2.0, 1.0 + action['layer_width_modifier']))
                new_width = int(module.out_features * width_modifier)
                module = self._resize_layer(module, new_width)
        
        # Add skip connections based on probability
        if action['skip_connection_prob'] > 0.5:
            model = self._add_skip_connection(model)
            
        return model
    
    def _resize_layer(self, 
                     layer: nn.Linear, 
                     new_width: int) -> nn.Linear:
        new_layer = nn.Linear(layer.in_features, new_width)
        with torch.no_grad():
            if new_width > layer.out_features:
                new_layer.weight[:layer.out_features] = layer.weight
            else:
                new_layer.weight = nn.Parameter(layer.weight[:new_width])
        return new_layer
