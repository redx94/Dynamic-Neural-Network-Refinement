import torch
import torch.nn as nn
from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class EnsembleMember:
    model: nn.Module
    weight: float
    specialty: Optional[Dict[str, float]] = None

class DynamicEnsemble(nn.Module):
    def __init__(self, 
                 max_members: int = 5,
                 diversity_threshold: float = 0.3):
        super().__init__()
        self.max_members = max_members
        self.diversity_threshold = diversity_threshold
        self.members: List[EnsembleMember] = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.members:
            raise ValueError("Ensemble has no members")
            
        outputs = []
        weights = []
        
        for member in self.members:
            out = member.model(x)
            outputs.append(out)
            weights.append(member.weight)
            
        outputs = torch.stack(outputs)
        weights = torch.tensor(weights, device=x.device)
        weighted_sum = torch.sum(outputs * weights.view(-1, 1, 1), dim=0)
        
        return weighted_sum
        
    def add_member(self, 
                  model: nn.Module,
                  specialty: Optional[Dict[str, float]] = None) -> bool:
        if len(self.members) >= self.max_members:
            self._remove_weakest_member()
            
        if self._is_diverse_enough(model):
            weight = 1.0 / (len(self.members) + 1)
            self._rebalance_weights(weight)
            
            self.members.append(EnsembleMember(
                model=model,
                weight=weight,
                specialty=specialty
            ))
            return True
            
        return False
        
    def _is_diverse_enough(self, new_model: nn.Module) -> bool:
        if not self.members:
            return True
            
        diversity_scores = []
        for member in self.members:
            score = self._compute_diversity(member.model, new_model)
            diversity_scores.append(score)
            
        return min(diversity_scores) > self.diversity_threshold
