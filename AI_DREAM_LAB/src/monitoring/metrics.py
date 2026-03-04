from prometheus_client import Gauge, Histogram
import torch
from typing import Dict, Any

class MetricsCollector:
    def __init__(self):
        self.layer_complexity = Gauge('layer_complexity', 
                                    'Complexity score per layer',
                                    ['layer_id'])
        self.network_size = Gauge('network_size', 
                                'Total number of parameters')
        self.adaptation_time = Histogram('adaptation_time',
                                       'Time taken for network adaptation')
        
    def update_metrics(self, model: torch.nn.Module, metrics: Dict[str, Any]):
        for name, layer in model.named_modules():
            if isinstance(layer, DynamicLayer):
                self.layer_complexity.labels(layer_id=name).set(
                    layer.complexity_score)
        
        total_params = sum(p.numel() for p in model.parameters())
        self.network_size.set(total_params)
