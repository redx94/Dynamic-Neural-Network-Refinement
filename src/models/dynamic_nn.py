
import torch.nn as nn

class DynamicNeuralNetwork(nn.Module):
    def __init__(self):
        super(DynamicNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(784, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 10)

    def forward(self, x):
        x = nn.ReLU()(self.layer1(x))
        x = nn.ReLU()(self.layer2(x))
        x = self.layer3(x)
        return x
