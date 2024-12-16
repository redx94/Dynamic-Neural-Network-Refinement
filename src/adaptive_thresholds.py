
import torch
import torch.nn as nn
import math

class AdaptiveThresholds(nn.Module):
    def __init__(self, initial_values=None):
        super().__init__()
        if initial_values is None:
            initial_values = {'variance': 0.5, 'entropy': 0.7, 'sparsity': 0.3}
        self.thresholds = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(v)) 
            for k, v in initial_values.items()
        })
        self.sensitivity = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, complexities):
        adapted_thresholds = {}
        for metric, threshold in self.thresholds.items():
            if metric in complexities:
                sensitivity_factor = torch.sigmoid(self.sensitivity * complexities[metric])
                adapted_thresholds[metric] = threshold * sensitivity_factor
            else:
                adapted_thresholds[metric] = threshold
        return adapted_thresholds
