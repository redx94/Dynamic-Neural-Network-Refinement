import torch
from datetime import datetime
from typing import Dict, Optional, List
import hashlib
import json
from dataclasses import dataclass
import git

@dataclass
class ModelVersion:
    version_id: str
    architecture_hash: str
    performance_metrics: Dict[str, float]
    creation_time: datetime
    git_commit: str
    parent_version: Optional[str]

class ModelRegistry:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.versions: Dict[str, ModelVersion] = {}
        self.current_version: Optional[str] = None
        
    def register_model(self, 
                      model: torch.nn.Module,
                      metrics: Dict[str, float]) -> str:
        # Generate version ID and architecture hash
        arch_hash = self._hash_architecture(model)
        version_id = self._generate_version_id()
        
        # Store model version info
        version = ModelVersion(
            version_id=version_id,
            architecture_hash=arch_hash,
            performance_metrics=metrics,
            creation_time=datetime.now(),
            git_commit=self._get_git_commit(),
            parent_version=self.current_version
        )
        
        self.versions[version_id] = version
        self._save_model(model, version_id)
        self.current_version = version_id
        
        return version_id
        
    def _hash_architecture(self, model: torch.nn.Module) -> str:
        arch_str = str(model)
        return hashlib.sha256(arch_str.encode()).hexdigest()
        
    def _get_git_commit(self) -> str:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
