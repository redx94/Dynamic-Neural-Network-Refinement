import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class LayerMapping:
    source_layer: str
    target_layer: str
    adaptation_method: str
    confidence: float

class TransferOptimizer:
    def __init__(self,
                 similarity_threshold: float = 0.7,
                 adaptation_lr: float = 0.01):
        self.similarity_threshold = similarity_threshold
        self.adaptation_lr = adaptation_lr
        self.layer_mappings = {}
        
    def optimize_transfer(self,
                         source_model: nn.Module,
                         target_model: nn.Module,
                         source_data: torch.Tensor,
                         target_data: torch.Tensor) -> nn.Module:
        # Find optimal layer mappings
        self.layer_mappings = self._compute_layer_mappings(
            source_model, target_model, source_data, target_data
        )
        
        # Apply knowledge transfer
        adapted_model = self._apply_transfer(source_model, target_model)
        
        # Fine-tune adaptation layers
        adapted_model = self._fine_tune_adaptations(adapted_model, target_data)
        
        return adapted_model
    
    def _compute_layer_mappings(self,
                              source_model: nn.Module,
                              target_model: nn.Module,
                              source_data: torch.Tensor,
                              target_data: torch.Tensor) -> Dict[str, LayerMapping]:
        mappings = {}
        source_features = self._extract_layer_features(source_model, source_data)
        target_features = self._extract_layer_features(target_model, target_data)
        
        for s_name, s_feat in source_features.items():
            best_mapping = self._find_best_target_layer(s_feat, target_features)
            if best_mapping.confidence > self.similarity_threshold:
                mappings[s_name] = best_mapping
                
        return mappings
