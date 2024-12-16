
import torch.nn as nn

class DynamicNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(DynamicNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dims[0])
        self.layer2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.layer3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = nn.ReLU()(self.layer1(x))
        x = nn.ReLU()(self.layer2(x))
        x = self.layer3(x)
        return x
import torch
import torch.nn as nn

class DynamicNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
