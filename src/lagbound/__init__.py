"""Lagbound: certified bounds for modularity maximization.

>>> import lagbound
>>> res = lagbound.bound("graph.txt")              # lower and upper bound
>>> lb = lagbound.lower_bound(G)                   # best partition found
>>> ub = lagbound.upper_bound(G, verify=True)      # certified and independently verified upper bound
"""
from .api import BoundResult, LowerBound, UpperBound, bound, lower_bound, upper_bound
from .graph import Graph, as_graph
from .modularity import modularity
from .pipeline import PipelineError

__version__ = "0.1.0"

__all__ = ["BoundResult", "Graph", "LowerBound", "PipelineError", "UpperBound", "as_graph", "bound", "lower_bound",
           "modularity", "upper_bound", "__version__"]
