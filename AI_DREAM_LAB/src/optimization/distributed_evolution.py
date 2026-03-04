import ray
from typing import List, Dict, Optional
import torch.distributed as dist
from dataclasses import dataclass
import numpy as np

@dataclass
class EvolutionWorkerState:
    worker_id: str
    population: List[dict]
    best_fitness: float
    generation: int

@ray.remote
class DistributedEvolutionWorker:
    def __init__(self, 
                 worker_id: str,
                 population_size: int,
                 mutation_rate: float):
        self.worker_id = worker_id
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.state = EvolutionWorkerState(
            worker_id=worker_id,
            population=[],
            best_fitness=float('-inf'),
            generation=0
        )
        
    def evolve_population(self, 
                         shared_population: Optional[List[dict]] = None) -> Dict:
        # Initialize or update local population
        if shared_population:
            self._merge_populations(shared_population)
            
        # Perform local evolution
        for _ in range(self.local_steps):
            self._evolve_step()
            
        return {
            'worker_id': self.worker_id,
            'best_individual': self._get_best_individual(),
            'population_sample': self._sample_population()
        }
