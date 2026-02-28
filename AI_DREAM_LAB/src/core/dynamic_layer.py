import torch
import torch.nn as nn
import numpy as np

class DynamicLayer(nn.Module):
    def __init__(self, input_size, output_size, complexity_threshold=0.8):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.complexity_threshold = complexity_threshold
        
        self.layer = nn.Linear(input_size, output_size)
        self.complexity_score = None
        
    def forward(self, x):
        output = self.layer(x)
        self.analyze_complexity(output)
        return output
    
    def analyze_complexity(self, output):
        # Calculate complexity using entropy and variance
        entropy = self._calculate_entropy(output)
        variance = torch.var(output, dim=1).mean()
        self.complexity_score = 0.7 * entropy + 0.3 * variance
        
        if self.complexity_score > self.complexity_threshold:
            self._refine_layer()
    
    def _calculate_entropy(self, tensor):
        probs = torch.softmax(tensor, dim=1)
        return -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()
    
    def _refine_layer(self):
        # Adaptive layer modification based on complexity
        if self.complexity_score > self.complexity_threshold:
            new_output_size = int(self.output_size * 1.5)
            new_layer = nn.Linear(self.input_size, new_output_size)
            with torch.no_grad():
                new_layer.weight[:self.output_size] = self.layer.weight
            self.layer = new_layer
            self.output_size = new_output_size
