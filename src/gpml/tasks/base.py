from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Protocol, Any
import networkx as nx

TaskMode = Literal["binary", "regression", "multiclass"]

@dataclass
class TaskSpec:
    name: str
    mode: TaskMode
    params: dict[str, Any] | None = None

class BaseGraphTask(Protocol):
    name: str
    mode: TaskMode
    num_classes: int | None

    def label(self, g: nx.Graph) -> Any:
        ...
