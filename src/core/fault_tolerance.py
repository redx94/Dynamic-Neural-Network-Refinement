import torch
import os
from typing import Dict, Optional, Callable
from datetime import datetime
import json
import threading
import queue

class FaultToleranceManager:
    def __init__(self, 
                 checkpoint_dir: str,
                 backup_frequency: int = 100,
                 max_checkpoints: int = 5):
        self.checkpoint_dir = checkpoint_dir
        self.backup_frequency = backup_frequency
        self.max_checkpoints = max_checkpoints
        self.checkpoint_queue = queue.Queue()
        self.recovery_states = {}
        
        # Start background checkpoint writer
        self._start_checkpoint_writer()
        
    def save_checkpoint(self, 
                       state: Dict,
                       is_emergency: bool = False) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_id = f"checkpoint_{timestamp}"
        
        if is_emergency:
            self._write_checkpoint_sync(checkpoint_id, state)
        else:
            self.checkpoint_queue.put((checkpoint_id, state))
            
        self._cleanup_old_checkpoints()
        return checkpoint_id
    
    def recover_latest(self) -> Optional[Dict]:
        checkpoints = self._get_available_checkpoints()
        if not checkpoints:
            return None
            
        latest = max(checkpoints)
        return self._load_checkpoint(latest)
    
    def _start_checkpoint_writer(self):
        def writer_thread():
            while True:
                try:
                    checkpoint_id, state = self.checkpoint_queue.get()
                    self._write_checkpoint_sync(checkpoint_id, state)
                except Exception as e:
                    print(f"Checkpoint writing error: {e}")
                    
        thread = threading.Thread(target=writer_thread, daemon=True)
        thread.start()
