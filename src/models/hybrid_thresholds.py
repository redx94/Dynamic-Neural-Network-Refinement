import torch
import torch.nn as nn
import math
from typing import Dict, Optional

class HybridThresholds:
    """
    Manages dynamic thresholds for complexity-based routing with cosine annealing.
    """
    def __init__(self, 
                 initial_thresholds: Dict[str, float],
                 annealing_start_epoch: int,
                 total_epochs: int,
                 min_threshold: float = 0.1):
        """
        Initialize hybrid thresholds manager.
        
        Args:
            initial_thresholds: Dictionary of initial threshold values for each metric
            annealing_start_epoch: Epoch to start threshold annealing
            total_epochs: Total number of training epochs
            min_threshold: Minimum threshold value after annealing
        """
        self.initial_thresholds = initial_thresholds
        self.annealing_start_epoch = annealing_start_epoch
        self.total_epochs = total_epochs
        self.min_threshold = min_threshold
        self.current_thresholds = initial_thresholds.copy()
        
    def compute_annealing_factor(self, current_epoch: int) -> float:
        """
        Compute cosine annealing factor based on current epoch.
        """
        if current_epoch < self.annealing_start_epoch:
            return 1.0
            
        progress = (current_epoch - self.annealing_start_epoch) / (self.total_epochs - self.annealing_start_epoch)
        return self.min_threshold + (1 - self.min_threshold) * (1 + math.cos(math.pi * progress)) / 2
        
    def update_thresholds(self, current_epoch: int) -> None:
        """
        Update thresholds based on current epoch using cosine annealing.
        """
        factor = self.compute_annealing_factor(current_epoch)
        self.current_thresholds = {
            k: v * factor for k, v in self.initial_thresholds.items()
        }
        
    def __call__(self, 
                 variance: torch.Tensor,
                 entropy: torch.Tensor,
                 sparsity: torch.Tensor,
                 current_epoch: int) -> Dict[str, torch.Tensor]:
        """
        Apply thresholds to complexity metrics.
        
        Args:
            variance: Batch variance metrics
            entropy: Batch entropy metrics
            sparsity: Batch sparsity metrics
            current_epoch: Current training epoch
            
        Returns:
            Dictionary of boolean tensors indicating which thresholds were exceeded
        """
        self.update_thresholds(current_epoch)
        
        return {
            'variance': variance > self.current_thresholds['variance'],
            'entropy': entropy > self.current_thresholds['entropy'],
            'sparsity': sparsity > self.current_thresholds['sparsity']
        }
        
    def get_current_thresholds(self) -> Dict[str, float]:
        """
        Get current threshold values.
        """
        return self.current_thresholds.copy()
