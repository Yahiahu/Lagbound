"""Atris: large-scale modularity maximization with approximation guarantees (package ``lagbound``).

A feasible partition and a verified upper bound establish proximity to optimal modularity.
For exact partition score L > 0 and accepted bound U, the partition is within 100*(U-L)/U percent of optimal.
The legacy ``gap`` field uses (U-L)/L instead. See the README for conservative percentage reporting.

>>> import lagbound
>>> res = lagbound.bound("graph.txt", verify=True) # partition and independently verified upper bound
>>> lb = lagbound.lower_bound(G)                   # best partition found
>>> ub = lagbound.upper_bound(G, verify=True)      # certified and independently verified upper bound
>>> s = lagbound.score(G, partition, upper=0.42)   # how much any partition could improve on this one
"""
from .api import BoundResult, LowerBound, UpperBound, bound, lower_bound, upper_bound
from .graph import Graph, as_graph
from .modularity import modularity
from .pipeline import PipelineError
from .score import CERTIFIED, Score, score
from .triangle import triangle_bound
from .triangle_certificate import verify_triangle_certificate

__version__ = "0.1.2"

__all__ = ["BoundResult", "CERTIFIED", "Graph", "LowerBound", "PipelineError", "Score", "UpperBound", "as_graph",
           "bound", "lower_bound", "modularity", "score", "upper_bound", "triangle_bound",
           "verify_triangle_certificate", "__version__"]
