import torch.nn as nn


class DynamicNN(nn.Module):
    """
    Dynamic Neural Network model that adjusts its architecture during training.
    """

    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Initializes the DynamicNN model.

        Args:
            input_size (int): Size of the input layer.
            hidden_sizes (list): List of hidden layer sizes.
            output_size (int): Size of the output layer.
        """
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
        """
        Forward pass of the dynamic neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.network(x)