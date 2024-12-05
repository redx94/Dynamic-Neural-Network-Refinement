import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np

class DynamicLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(0.2)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return self.dropout(x)

class DynamicNeuralNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int] = [64, 32]):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Create initial architecture
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(DynamicLayer(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, output_size)
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)
    
    def add_layer(self, size: int) -> None:
        """Dynamically add a new layer to the network"""
        prev_size = self.layers[-1].linear.out_features
        new_layer = DynamicLayer(prev_size, size)
        
        # Adjust output layer
        old_output = self.output_layer
        self.output_layer = nn.Linear(size, self.output_size)
        
        # Initialize new output layer weights using old weights
        with torch.no_grad():
            self.output_layer.weight.data.fill_(0)
            self.output_layer.bias.data.copy_(old_output.bias.data)
        
        self.layers.append(new_layer)
    
    def remove_layer(self) -> None:
        """Remove the last hidden layer"""
        if len(self.layers) > 1:
            prev_size = self.layers[-2].linear.out_features
            self.output_layer = nn.Linear(prev_size, self.output_size)
            self.layers = self.layers[:-1]
    
    def adjust_layer_width(self, layer_idx: int, new_width: int) -> None:
        """Adjust the width of a specific layer"""
        if layer_idx >= len(self.layers):
            return
            
        old_layer = self.layers[layer_idx]
        in_features = old_layer.linear.in_features
        
        # Create new layer with adjusted width
        new_layer = DynamicLayer(in_features, new_width)
        
        # Transfer learned weights where possible
        min_width = min(old_layer.linear.out_features, new_width)
        with torch.no_grad():
            new_layer.linear.weight.data[:min_width, :] = \
                old_layer.linear.weight.data[:min_width, :]
            new_layer.linear.bias.data[:min_width] = \
                old_layer.linear.bias.data[:min_width]
        
        self.layers[layer_idx] = new_layer
        
        # Adjust next layer if necessary
        if layer_idx < len(self.layers) - 1:
            next_layer = self.layers[layer_idx + 1]
            adjusted_next = DynamicLayer(new_width, next_layer.linear.out_features)
            min_width = min(next_layer.linear.in_features, new_width)
            with torch.no_grad():
                adjusted_next.linear.weight.data[:, :min_width] = \
                    next_layer.linear.weight.data[:, :min_width]
            self.layers[layer_idx + 1] = adjusted_next
