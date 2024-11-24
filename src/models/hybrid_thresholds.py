
import torch
import torch.nn as nn

class HybridThresholdsModel(nn.Module):
    def __init__(self, input_size, thresholds):
        super(HybridThresholdsModel, self).__init__()
        self.input_size = input_size
        self.thresholds = thresholds
        self.hidden_layer = nn.Linear(input_size, 128)
        self.output_layer = nn.Linear(128, 3)

    def forward(self, x, complexities):
        x = self.hidden_layer(x)
        x = torch.relu(x)
        x = self.output_layer(x)
        return x
    