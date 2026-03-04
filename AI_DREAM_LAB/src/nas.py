import torch
import torch.nn as nn
import copy
import random
from src.layers import BaseLayer
from src.model import DynamicNeuralNetwork

class NAS:
    """
    A simple Neural Architecture Search (NAS) class that mutates model architectures
    and evaluates them.
    """

    def __init__(self, base_model, search_space, device='cpu'):
        """
        Initializes the NAS class.

        Args:
            base_model (nn.Module): The base model to mutate.
            search_space (dict): The search space for NAS.
            device (str): Device to run the models on.
        """
        self.base_model = base_model
        self.search_space = search_space
        self.device = device
        self.best_model = None
        self.best_score = float('-inf')

    def mutate(self, model):
        """
        Applies a random mutation to the model architecture.

        Args:
            model (nn.Module): The model to mutate.

        Returns:
            nn.Module: The mutated model.
        """
        mutated_model = copy.deepcopy(model)
        mutation_type = random.choice(list(self.search_space.keys()))

        if mutation_type == 'add_layer':
            layer_type = random.choice(['BaseLayer', 'Linear'])
            input_dim = random.choice([128, 256, 512])
            output_dim = random.choice([128, 256, 512])
            
            if layer_type == 'BaseLayer':
                new_layer = BaseLayer(input_dim, output_dim)
            else:
                new_layer = nn.Linear(input_dim, output_dim)
            
            # Insert the new layer at a random position
            insert_position = random.randint(0, len(mutated_model.layers))
            mutated_model.layers.insert(insert_position, new_layer)

        elif mutation_type == 'remove_layer' and len(mutated_model.layers) > 1:
            # Remove a layer at a random position
            remove_position = random.randint(0, len(mutated_model.layers) - 1)
            del mutated_model.layers[remove_position]

        elif mutation_type == 'change_layer_dim' and len(mutated_model.layers) > 0:
            # Change the dimensions of a layer at a random position
            layer_index = random.randint(0, len(mutated_model.layers) - 1)
            layer = mutated_model.layers[layer_index]
            
            if isinstance(layer, nn.Linear) or isinstance(layer, BaseLayer):
                new_input_dim = random.choice([128, 256, 512])
                new_output_dim = random.choice([128, 256, 512])
                
                if isinstance(layer, nn.Linear):
                    mutated_model.layers[layer_index] = nn.Linear(new_input_dim, new_output_dim)
                else:
                    mutated_model.layers[layer_index] = BaseLayer(new_input_dim, new_output_dim)

        return mutated_model

    def evaluate(self, model, dataloader):
        """
        Evaluates the model on the provided dataset.

        Args:
            model (nn.Module): Model to evaluate.
            dataloader (DataLoader): Dataset for evaluation.

        Returns:
            float: Evaluation score (accuracy).
        """
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in dataloader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    def run(self, dataloader, generations=5, population_size=5):
        """
        Runs the NAS process.

        Args:
            dataloader (DataLoader): Dataset for evaluation.
            generations (int): Number of generations to run.
            population_size (int): Number of models per generation.

        Returns:
            nn.Module: The best-performing model found during NAS.
        """
        population = [copy.deepcopy(self.base_model) for _ in range(population_size)]

        for gen in range(generations):
            print(f"Generation {gen + 1}")

            for i, model in enumerate(population):
                mutated_model = self.mutate(model)
                
                # Extract network configuration from the mutated model
                network_config = []
                for layer in mutated_model.layers:
                    if isinstance(layer, BaseLayer):
                        network_config.append({"type": "BaseLayer", "input_dim": layer.input_dim, "output_dim": layer.output_dim})
                    elif isinstance(layer, nn.Linear):
                        network_config.append({"type": "Linear", "input_dim": layer.in_features, "output_dim": layer.out_features})
                
                # Create a new DynamicNeuralNetwork instance with the mutated network configuration
                mutated_model = DynamicNeuralNetwork(self.base_model.hybrid_thresholds, network_config=network_config).to(self.device)
                
                score = self.evaluate(mutated_model, dataloader)
                print(f"Model {i + 1} Accuracy: {score:.4f}")

                if score > self.best_score:
                    self.best_score = score
                    self.best_model = mutated_model

        print(f"Best Accuracy: {self.best_score:.4f}")
        return self.best_model
