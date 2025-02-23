import torch
from typing import Optional, Dict, List
import gc
from dataclasses import dataclass

@dataclass
class MemoryStats:
    allocated: int
    cached: int
    reserved: int
    active_bytes: int

class MemoryManager:
    def __init__(self, target_utilization: float = 0.8):
        self.target_utilization = target_utilization
        self.peak_memory = 0
        self.checkpoints = {}
        
    def monitor_memory(self) -> MemoryStats:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            stats = MemoryStats(
                allocated=torch.cuda.memory_allocated(),
                cached=torch.cuda.memory_reserved(),
                reserved=torch.cuda.max_memory_reserved(),
                active_bytes=torch.cuda.memory_allocated()
            )
        else:
            stats = MemoryStats(0, 0, 0, 0)
            
        self.peak_memory = max(self.peak_memory, stats.allocated)
        return stats
        
    def checkpoint_model(self, model: torch.nn.Module, tag: str):
        self.checkpoints[tag] = {
            'state_dict': model.state_dict(),
            'memory_stats': self.monitor_memory()
        }
        
    def optimize_memory(self, model: torch.nn.Module) -> bool:
        stats = self.monitor_memory()
        if stats.allocated / stats.reserved > self.target_utilization:
            gc.collect()
            torch.cuda.empty_cache()
            return True
        return False
