from typing import Dict, Type, Any
from .strategies import AdaptationStrategy
import importlib

class AdapterRegistry:
    def __init__(self):
        self._strategies: Dict[str, Type[AdaptationStrategy]] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, strategy_class: Type[AdaptationStrategy], config: Dict[str, Any] = None):
        """Register a new adaptation strategy"""
        self._strategies[name] = strategy_class
        self._configs[name] = config or {}

    def get_strategy(self, name: str) -> AdaptationStrategy:
        """Get an instance of a registered strategy"""
        if name not in self._strategies:
            raise KeyError(f"Strategy {name} not found in registry")
        return self._strategies[name](self._configs[name])

    def load_plugin(self, module_path: str):
        """Dynamically load a plugin strategy"""
        module = importlib.import_module(module_path)
        if hasattr(module, 'register_strategies'):
            module.register_strategies(self)

registry = AdapterRegistry()
