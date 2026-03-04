import torch.nn as nn


class DynamicNN(nn.Module):
    """
    Dynamic Neural Network model that adjusts its architecture during training.
    """

    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', dropout_rate=0.0):
        """
        Initializes the DynamicNN model.

        Args:
            input_size (int): Size of the input layer.
            hidden_sizes (list): List of hidden layer sizes.
            output_size (int): Size of the output layer.
            activation (str): Type of activation function to use (relu, leaky_relu, elu).
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.dropout_rate = dropout_rate

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            elif activation == 'elu':
                layers.append(nn.ELU())
            else:
                raise ValueError("Invalid activation function: {}".format(activation))
            layers.append(nn.BatchNorm1d(hidden_size))
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def add_layer(self, hidden_size, activation='relu', dropout_rate=0.0):
        """
        Adds a new hidden layer to the network.

        Args:
            hidden_size (int): Size of the hidden layer.
            activation (str): Type of activation function to use (relu, leaky_relu, elu).
            dropout_rate (float): Dropout rate.
        """
        new_layer = nn.Linear(self.network[-2].out_features, hidden_size)
        self.network = nn.Sequential(
            *list(self.network.children())[:-1],
            new_layer,
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate),
            self.network[-1]
        )

    def remove_layer(self):
        """
        Removes a hidden layer from the network.
        """
        if len(self.network) > 2:
            self.network = nn.Sequential(*list(self.network.children())[:-5], self.network[-1])

    def forward(self, x):
        """
        Forward pass of the dynamic neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = x.view(-1, 784)  # Flatten the input tensor
        return self.network(x)
