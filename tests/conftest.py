import itertools

import numpy as np
import pytest


def set_partitions(n):
    """All partitions of range(n) as membership arrays (restricted growth strings)."""
    def rec(prefix, k):
        if len(prefix) == n:
            yield np.array(prefix)
            return
        for c in range(k + 1):
            yield from rec(prefix + [c], max(k, c + 1))
    yield from rec([0], 1)


def brute_force_optimum(graph):
    from lagbound import modularity
    return max(modularity(graph, p, exact=True) for p in set_partitions(graph.n))


def random_graph(n, p, seed):
    rng = np.random.default_rng(seed)
    edges = [(i, j) for i, j in itertools.combinations(range(n), 2) if rng.random() < p]
    # attach a leaf and a twin pair so that both reductions fire
    edges += [(0, n), (1, n + 1), (2, n + 1), (1, 2)]
    return np.array(edges)


@pytest.fixture
def small_graphs():
    return [random_graph(7, 0.45, seed) for seed in range(6)]
