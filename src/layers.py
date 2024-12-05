# src/layers.py

import torch.nn as nn

class BaseLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BaseLayer, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.layer(x))
