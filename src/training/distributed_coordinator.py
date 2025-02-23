import torch
import torch.distributed as dist
from typing import Dict, Optional, List
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import logging

class DistributedCoordinator:
    def __init__(self, 
                 world_size: int,
                 backend: str = 'nccl',
                 master_addr: str = 'localhost',
                 master_port: str = '12355'):
        self.world_size = world_size
        self.backend = backend
        self.master_addr = master_addr
        self.master_port = master_port
        self.logger = logging.getLogger(__name__)
        
    def setup_process_group(self, rank: int):
        os.environ['MASTER_ADDR'] = self.master_addr
        os.environ['MASTER_PORT'] = self.master_port
        
        dist.init_process_group(
            backend=self.backend,
            rank=rank,
            world_size=self.world_size
        )
        
    def distribute_model(self, model: torch.nn.Module, device_id: int) -> DistributedDataParallel:
        model = model.to(device_id)
        return DistributedDataParallel(
            model,
            device_ids=[device_id],
            output_device=device_id,
            find_unused_parameters=True
        )
    
    def synchronize_meta_controllers(self, meta_controllers: List[torch.nn.Module]):
        for meta_controller in meta_controllers:
            for param in meta_controller.parameters():
                dist.broadcast(param.data, src=0)
                
    def aggregate_metrics(self, local_metrics: Dict[str, float]) -> Dict[str, float]:
        aggregated = {}
        for key, value in local_metrics.items():
            tensor = torch.tensor(value).cuda()
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            aggregated[key] = tensor.item()
        return aggregated
