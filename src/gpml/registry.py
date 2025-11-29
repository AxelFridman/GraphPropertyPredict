from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict

@dataclass
class Registry:
    name: str
    items: Dict[str, Any]

    def register(self, key: str) -> Callable[[Any], Any]:
        def deco(obj: Any) -> Any:
            if key in self.items:
                raise KeyError(f"[{self.name}] '{key}' already registered")
            self.items[key] = obj
            return obj
        return deco

    def get(self, key: str) -> Any:
        if key not in self.items:
            raise KeyError(f"[{self.name}] Unknown key '{key}'. Available: {sorted(self.items.keys())}")
        return self.items[key]

TASKS = Registry("tasks", {})
MODELS = Registry("models", {})
TRANSFORMS = Registry("transforms", {})
GRAPH_FAMILIES = Registry("graph_families", {})
