
import torch
import torch.nn as nn
from collections import defaultdict

class ComplexityAnalyzer:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.history = defaultdict(list)
        
    def calculate_complexities(self, x):
        complexities = {}
        
        # Variance complexity
        complexities['variance'] = torch.var(x, dim=1).mean()
        
        # Entropy complexity
        softmax = torch.nn.functional.softmax(x, dim=1)
        entropy = -(softmax * torch.log(softmax + 1e-10)).sum(dim=1)
        complexities['entropy'] = entropy.mean()
        
        # Sparsity complexity
        sparsity = (torch.abs(x) < 0.01).float().mean()
        complexities['sparsity'] = sparsity
        
        return complexities

    def update_history(self, complexities):
        for metric, value in complexities.items():
            self.history[metric].append(value)
            if len(self.history[metric]) > self.window_size:
                self.history[metric].pop(0)

    def detect_drift(self, threshold=0.1):
        drifts = {}
        for metric, values in self.history.items():
            if len(values) >= self.window_size:
                first_half = torch.tensor(values[:self.window_size//2]).mean()
                second_half = torch.tensor(values[self.window_size//2:]).mean()
                drifts[metric] = abs(first_half - second_half) > threshold
        return drifts
