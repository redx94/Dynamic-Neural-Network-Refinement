import logging
import ray
from typing import Dict, Any, Optional
from datetime import datetime
import json
from elasticsearch import Elasticsearch
from opensearch_py import OpenSearch

class DistributedLogger:
    def __init__(self, 
                 cluster_name: str,
                 elasticsearch_url: Optional[str] = None,
                 opensearch_url: Optional[str] = None):
        self.cluster_name = cluster_name
        self._setup_storage(elasticsearch_url, opensearch_url)
        self.buffer_size = 1000
        self.log_buffer = []
        
    @ray.remote
    def log_event(self, 
                  event_type: str,
                  data: Dict[str, Any],
                  metadata: Optional[Dict[str, Any]] = None):
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'cluster': self.cluster_name,
            'type': event_type,
            'data': data,
            'metadata': metadata or {}
        }
        
        self.log_buffer.append(event)
        if len(self.log_buffer) >= self.buffer_size:
            self._flush_logs()
            
    def _flush_logs(self):
        if self.elasticsearch:
            self._store_elasticsearch(self.log_buffer)
        if self.opensearch:
            self._store_opensearch(self.log_buffer)
        self.log_buffer = []
