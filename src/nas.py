import torch
import torch.nn as nn
import copy


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
        mutation_type = torch.randint(0, 3, (1,)).item()

        if mutation_type == 0 and 'add_layer' in self.search_space:
            # Example: Add a new linear layer after layer 2
            new_layer = nn.Linear(256, 128)
            mutated_model.layer3 = nn.Sequential(new_layer, nn.ReLU())

        elif mutation_type == 1 and hasattr(mutated_model, 'layer3'):
            # Example: Remove a layer
            del mutated_model.layer3

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
                mutated_model = self.mutate(model).to(self.device)
                score = self.evaluate(mutated_model, dataloader)
                print(f"Model {i + 1} Accuracy: {score:.4f}")

                if score > self.best_score:
                    self.best_score = score
                    self.best_model = mutated_model

        print(f"Best Accuracy: {self.best_score:.4f}")
        return self.best_model