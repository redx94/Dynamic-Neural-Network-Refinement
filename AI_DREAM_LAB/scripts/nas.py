import torch
import torch.nn as nn
import copy
import random


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
        mutation_type = random.choice(
            ['add_layer', 'remove_layer', 'increase_units', 'decrease_units']
        )

        if mutation_type == 'add_layer' and self.search_space.get('add_layer', False):
            new_layer = nn.Linear(256, 128)
            if hasattr(mutated_model, 'layer3') and isinstance(mutated_model.layer3, nn.Sequential):
                mutated_model.layer3 = nn.Sequential(mutated_model.layer3, new_layer)
            else:
                mutated_model.layer3 = nn.Sequential(new_layer)

        elif mutation_type == 'remove_layer' and self.search_space.get('remove_layer', False):
            if hasattr(mutated_model, 'layer3'):
                del mutated_model.layer3

        elif mutation_type == 'increase_units' and self.search_space.get('increase_units', False):
            if hasattr(mutated_model.layer1, 'layer'):
                mutated_model.layer1.layer = nn.Linear(mutated_model.layer1.layer.in_features, 512)

        elif mutation_type == 'decrease_units' and self.search_space.get('decrease_units', False):
            if hasattr(mutated_model.layer1, 'layer'):
                mutated_model.layer1.layer = nn.Linear(mutated_model.layer1.layer.in_features, 128)

        return mutated_model

    def evaluate(self, model, dataloader):
        """
        Evaluates the model on the provided dataloader.

        Args:
            model (nn.Module): The model to evaluate.
            dataloader (DataLoader): The dataloader for evaluation.

        Returns:
            float: The evaluation score (e.g., accuracy).
        """
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in dataloader:
                data, labels = data.to(self.device), labels.to(self.device)

                complexities = {
                    'variance': torch.tensor([0.6] * data.size(0)).to(self.device),
                    'entropy': torch.tensor([0.6] * data.size(0)).to(self.device),
                    'sparsity': torch.tensor([0.4] * data.size(0)).to(self.device)
                }

                outputs = model(data, complexities)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return accuracy

    def run(self, dataloader, generations=5, population_size=5):
        """
        Runs the NAS process.

        Args:
            dataloader (DataLoader): The dataloader for evaluation.
            generations (int): Number of generations.
            population_size (int): Number of models per generation.

        Returns:
            nn.Module: The best-performing model found during NAS.
        """
        population = [copy.deepcopy(self.base_model) for _ in range(population_size)]

        for gen in range(generations):
            print(f"Generation {gen + 1}")

            for i in range(population_size):
                mutated_model = self.mutate(population[i])
                mutated_model.to(self.device)
                score = self.evaluate(mutated_model, dataloader)

                print(f"Model {i + 1} Accuracy: {score}")

                if score > self.best_score:
                    self.best_score = score
                    self.best_model = mutated_model

        print(f"Best Accuracy: {self.best_score}")
        return self.best_model