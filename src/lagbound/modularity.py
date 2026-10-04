"""Modularity of a partition, in floating point or exactly."""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from .graph import Graph, as_graph


def as_membership(graph: Graph, partition) -> np.ndarray:
    """Accept a membership array over ``0..n-1`` or a dict keyed by the caller's node identifiers."""
    if isinstance(partition, dict):
        return np.array([partition[name] for name in graph.labels])
    membership = np.asarray(partition)
    if len(membership) != graph.n:
        raise ValueError(f"partition has {len(membership)} entries, graph has {graph.n} non-isolated nodes")
    return membership


def modularity(graph, partition, exact: bool = False):
    """Modularity of ``partition`` on ``graph`` (resolution 1).

    ``partition`` is a membership array over the graph's non-isolated nodes or a dict keyed by node identifier.
    With ``exact=True`` the value is returned as a :class:`fractions.Fraction`.
    """
    graph = as_graph(graph)
    lab = np.unique(as_membership(graph, partition), return_inverse=True)[1]
    m = graph.m
    inside = int((lab[graph.u] == lab[graph.v]).sum())
    volc = np.bincount(lab, weights=graph.degrees).astype(np.int64)
    q = Fraction(inside, m) - Fraction(sum(int(x) ** 2 for x in volc), 4 * m * m)
    return q if exact else float(q)
