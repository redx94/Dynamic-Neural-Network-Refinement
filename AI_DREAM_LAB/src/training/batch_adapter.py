import torch
from typing import Optional, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class BatchStats:
    gradient_norm: float
    memory_usage: float
    processing_time: float
    convergence_rate: float

class AdaptiveBatchSizer:
    def __init__(self,
                 initial_batch_size: int,
                 min_batch_size: int = 4,
                 max_batch_size: int = 512,
                 adaptation_frequency: int = 10):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.adaptation_frequency = adaptation_frequency
        self.stats_history = []
        
    def adapt_batch_size(self, stats: BatchStats) -> int:
        self.stats_history.append(stats)
        
        if len(self.stats_history) < self.adaptation_frequency:
            return self.current_batch_size
            
        # Analyze recent performance
        recent_stats = self.stats_history[-self.adaptation_frequency:]
        grad_trend = self._analyze_gradient_trend(recent_stats)
        memory_pressure = self._analyze_memory_pressure(recent_stats)
        
        # Adjust batch size based on analysis
        new_batch_size = self._compute_optimal_batch_size(
            grad_trend, memory_pressure
        )
        
        self.current_batch_size = int(new_batch_size)
        return self.current_batch_size
    
    def _compute_optimal_batch_size(self,
                                  grad_trend: float,
                                  memory_pressure: float) -> int:
        # Use gradient trend and memory pressure to determine optimal batch size
        adjustment = 1.0 + (grad_trend * 0.2 - memory_pressure * 0.3)
        new_size = int(self.current_batch_size * adjustment)
        
        # Ensure we stay within bounds
        return max(self.min_batch_size,
                  min(self.max_batch_size, new_size))
