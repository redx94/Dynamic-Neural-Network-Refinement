
import torch
import torch.nn as nn
import copy
from ray import tune
import numpy as np

class SearchableNetwork(nn.Module):
    def __init__(self, config):
        super(SearchableNetwork, self).__init__()
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
    def __init__(self, input_dim, output_dim, metric='accuracy'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.metric = metric
        self.best_config = None
        self.best_score = float('-inf')
        
    def get_search_space(self):
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_layers': tune.randint(2, 6),
            'layer_0_units': tune.randint(64, 512),
            'layer_1_units': tune.randint(64, 256),
            'layer_2_units': tune.randint(32, 128),
            'layer_3_units': tune.randint(32, 128),
            'layer_4_units': tune.randint(16, 64),
            'layer_0_dropout': tune.uniform(0.1, 0.5),
            'layer_1_dropout': tune.uniform(0.1, 0.5),
            'layer_2_dropout': tune.uniform(0.1, 0.5),
            'layer_3_dropout': tune.uniform(0.1, 0.5),
            'layer_4_dropout': tune.uniform(0.1, 0.5),
            'activation': tune.choice(['relu', 'leaky_relu', 'elu', 'tanh']),
            'learning_rate': tune.loguniform(1e-4, 1e-2)
        }
        
    def train_model(self, config, train_data, val_data, epochs=10):
        model = SearchableNetwork(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        best_val_score = float('-inf')
        for epoch in range(epochs):
            model.train()
            for batch in train_data:
                inputs, targets = batch
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
            val_score = self.evaluate(model, val_data)
            if val_score > best_val_score:
                best_val_score = val_score
                
        return best_val_score
        
    def evaluate(self, model, val_data):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_data:
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return correct / total
        
    def search(self, train_data, val_data, num_samples=10, epochs=10):
        def objective(config):
            score = self.train_model(config, train_data, val_data, epochs)
            tune.report(score=score)
            
        analysis = tune.run(
            objective,
            config=self.get_search_space(),
            num_samples=num_samples,
            resources_per_trial={'cpu': 1, 'gpu': 0}
        )
        
        self.best_config = analysis.get_best_config('score')
        self.best_score = analysis.get_best_trial('score').last_result['score']
        return self.best_config, self.best_score
