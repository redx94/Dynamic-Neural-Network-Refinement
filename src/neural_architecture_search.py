
import torch
import torch.nn as nn
from ray import tune
import numpy as np

class SearchableNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList()
        self.build_network(config)
        
    def build_network(self, config):
        input_dim = config['input_dim']
        for i in range(config['num_layers']):
            self.layers.append(nn.Linear(input_dim, config[f'layer_{i}_units']))
            self.layers.append(self.get_activation(config['activation']))
            self.layers.append(nn.Dropout(config[f'layer_{i}_dropout']))
            input_dim = config[f'layer_{i}_units']
        self.layers.append(nn.Linear(input_dim, config['output_dim']))
        
    def get_activation(self, name):
        return {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'tanh': nn.Tanh()
        }[name]
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class NeuralArchitectureSearch:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.best_model = None
        self.best_score = float('-inf')
        
    def search(self, train_loader, val_loader, num_trials=10):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_layers': tune.randint(2, 6),
            'activation': tune.choice(['relu', 'leaky_relu', 'elu', 'tanh']),
            'learning_rate': tune.loguniform(1e-4, 1e-2)
        }
        
        def train_model(config):
            model = SearchableNetwork(config)
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(5):  # Quick training for search
                for batch in train_loader:
                    inputs, targets = batch
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
            score = self.evaluate(model, val_loader)
            tune.report(score=score)
            
        analysis = tune.run(
            train_model,
            config=config,
            num_samples=num_trials
        )
        
        return analysis.best_config, analysis.best_trial.last_result['score']
