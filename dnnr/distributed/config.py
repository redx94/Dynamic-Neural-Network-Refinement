from typing import Dict
import torch.distributed as dist
from pathlib import Path
import yaml

class DistributedConfig:
    def __init__(self, config_path: Path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.world_size = self.config['distributed']['nodes'] * \
                         self.config['distributed']['gpus_per_node']

    def setup_distributed(self) -> None:
        """Initialize distributed training environment"""
        dist.init_process_group(
            backend=self.config['distributed']['backend'],
            init_method='env://'
        )

    def get_node_config(self, node_rank: int) -> Dict:
        """Get configuration for specific node"""
        return {
            'gpus': self.config['distributed']['gpus_per_node'],
            'rank': node_rank,
            'world_size': self.world_size,
            'backend': self.config['distributed']['backend']
        }

    def cleanup(self):
        """Cleanup distributed training resources"""
        dist.destroy_process_group()
