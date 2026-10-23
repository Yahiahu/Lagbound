"""Lagbound: certified bounds for modularity maximization.

>>> import lagbound
>>> res = lagbound.bound("graph.txt")              # lower and upper bound
>>> lb = lagbound.lower_bound(G)                   # best partition found
>>> ub = lagbound.upper_bound(G, verify=True)      # certified and independently verified upper bound
>>> s = lagbound.score(G, partition, upper=0.42)   # how much any partition could improve on this one
"""
from .api import BoundResult, LowerBound, UpperBound, bound, lower_bound, upper_bound
from .graph import Graph, as_graph
from .modularity import modularity
from .pipeline import PipelineError
from .score import CERTIFIED, Score, score

__version__ = "0.1.0"

__all__ = ["BoundResult", "CERTIFIED", "Graph", "LowerBound", "PipelineError", "Score", "UpperBound", "as_graph",
           "bound", "lower_bound", "modularity", "score", "upper_bound", "__version__"]
