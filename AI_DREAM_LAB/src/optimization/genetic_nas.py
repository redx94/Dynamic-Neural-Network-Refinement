import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import random

@dataclass
class ArchitectureGene:
    layers: List[Dict[str, any]]
    connections: List[Tuple[int, int]]
    hyperparameters: Dict[str, float]
    fitness: Optional[float] = None

class GeneticNAS:
    def __init__(self,
                 population_size: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 generations: int = 100):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.population: List[ArchitectureGene] = []
        self.history: List[Dict] = []
        
    def evolve(self, 
               fitness_fn: callable,
               initial_population: Optional[List[ArchitectureGene]] = None) -> ArchitectureGene:
        # Initialize population
        self.population = initial_population or self._generate_initial_population()
        
        for generation in range(self.generations):
            # Evaluate fitness
            self._evaluate_population(fitness_fn)
            
            # Selection
            parents = self._select_parents()
            
            # Create new population
            new_population = []
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    # Crossover
                    parent1, parent2 = random.sample(parents, 2)
                    child = self._crossover(parent1, parent2)
                else:
                    # Mutation
                    parent = random.choice(parents)
                    child = self._mutate(parent)
                    
                new_population.append(child)
                
            self.population = new_population
            self._update_history(generation)
            
        return max(self.population, key=lambda x: x.fitness)
    
    def _crossover(self, 
                  parent1: ArchitectureGene,
                  parent2: ArchitectureGene) -> ArchitectureGene:
        # Layer-wise crossover
        child_layers = []
        for l1, l2 in zip(parent1.layers, parent2.layers):
            if random.random() < 0.5:
                child_layers.append(l1.copy())
            else:
                child_layers.append(l2.copy())
                
        # Connection crossover
        connection_mask = np.random.rand(len(parent1.connections)) < 0.5
        child_connections = [
            c1 if m else c2
            for c1, c2, m in zip(parent1.connections, parent2.connections, connection_mask)
        ]
        
        # Hyperparameter crossover
        child_hyperparams = {}
        for key in parent1.hyperparameters:
            alpha = random.random()
            child_hyperparams[key] = (
                alpha * parent1.hyperparameters[key] +
                (1 - alpha) * parent2.hyperparameters[key]
            )
            
        return ArchitectureGene(
            layers=child_layers,
            connections=child_connections,
            hyperparameters=child_hyperparams
        )
