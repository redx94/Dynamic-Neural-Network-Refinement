import torch
from typing import List, Dict
import numpy as np
from dataclasses import dataclass

@dataclass
class ArchitectureGene:
    layer_sizes: List[int]
    activation_functions: List[str]
    skip_connections: List[Tuple[int, int]]
    mutation_rate: float = 0.1

class EvolutionaryArchitectureSearch:
    def __init__(self, 
                 population_size: int = 50,
                 tournament_size: int = 5,
                 elite_size: int = 2):
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.population: List[ArchitectureGene] = []
        
    def evolve_architecture(self, 
                          fitness_scores: Dict[int, float]) -> ArchitectureGene:
        # Select parents using tournament selection
        parents = self._tournament_selection(fitness_scores)
        
        # Create next generation through crossover and mutation
        new_population = []
        
        # Preserve elite architectures
        elite = self._select_elite(fitness_scores)
        new_population.extend(elite)
        
        while len(new_population) < self.population_size:
            parent1, parent2 = np.random.choice(parents, 2, replace=False)
            child = self._crossover(parent1, parent2)
            child = self._mutate(child)
            new_population.append(child)
        
        self.population = new_population
        return max(self.population, key=lambda x: fitness_scores.get(hash(str(x)), 0))
    
    def _tournament_selection(self, fitness_scores: Dict[int, float]) -> List[ArchitectureGene]:
        selected = []
        for _ in range(self.population_size):
            tournament = np.random.choice(self.population, self.tournament_size)
            winner = max(tournament, key=lambda x: fitness_scores.get(hash(str(x)), 0))
            selected.append(winner)
        return selected
    
    def _crossover(self, parent1: ArchitectureGene, parent2: ArchitectureGene) -> ArchitectureGene:
        # Perform intelligent crossover of architecture genes
        crossover_point = np.random.randint(1, len(parent1.layer_sizes))
        
        child_layers = parent1.layer_sizes[:crossover_point] + parent2.layer_sizes[crossover_point:]
        child_activations = parent1.activation_functions[:crossover_point] + parent2.activation_functions[crossover_point:]
        
        # Merge skip connections intelligently
        child_skip = self._merge_skip_connections(parent1.skip_connections, parent2.skip_connections)
        
        return ArchitectureGene(child_layers, child_activations, child_skip)
