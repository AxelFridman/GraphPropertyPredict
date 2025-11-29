"""
Copy this file, implement `label`, register it, then set task.name in config.

Example:

import networkx as nx
from gpml.registry import TASKS

@TASKS.register("my_property")
class MyPropertyTask:
    name = "my_property"
    mode = "binary"  # or "regression" or "multiclass"
    num_classes = None

    def __init__(self, **params): ...
    def label(self, g: nx.Graph):
        return 1
"""
