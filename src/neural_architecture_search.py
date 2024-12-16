
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import torch
import torch.nn as nn

class SearchSpace:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
    def sample_architecture(self):
        num_layers = tune.randint(2, 5)
        layer_sizes = [tune.randint(64, 512) for _ in range(num_layers)]
        dropout_rates = [tune.uniform(0.1, 0.5) for _ in range(num_layers)]
        activation = tune.choice(['relu', 'leaky_relu', 'elu'])
        return {
            'num_layers': num_layers,
            'layer_sizes': layer_sizes,
            'dropout_rates': dropout_rates,
            'activation': activation
        }

def build_model(config):
    layers = []
    activation_fn = getattr(nn, config['activation'].upper())()
    
    for i in range(config['num_layers']):
        if i == 0:
            layers.append(nn.Linear(config['input_size'], config['layer_sizes'][i]))
        else:
            layers.append(nn.Linear(config['layer_sizes'][i-1], config['layer_sizes'][i]))
        layers.append(activation_fn)
        layers.append(nn.Dropout(config['dropout_rates'][i]))
    
    layers.append(nn.Linear(config['layer_sizes'][-1], config['output_size']))
    return nn.Sequential(*layers)

def objective(config):
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(config['num_epochs']):
        train_loss = train_epoch(model, optimizer, config['train_loader'])
        val_loss = evaluate(model, config['val_loader'])
        
        tune.report(
            loss=val_loss,
            training_loss=train_loss,
            epoch=epoch
        )

def run_architecture_search(train_loader, val_loader, input_size, output_size,
                          num_samples=10, num_epochs=10, gpus_per_trial=0):
    config = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'input_size': input_size,
        'output_size': output_size,
        'num_epochs': num_epochs,
        **SearchSpace(input_size, output_size).sample_architecture()
    }
    
    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2
    )
    
    analysis = tune.run(
        objective,
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        resources_per_trial={'cpu': 1, 'gpu': gpus_per_trial}
    )
    
    best_trial = analysis.get_best_trial('loss', 'min', 'last')
    best_config = best_trial.config
    return best_config
