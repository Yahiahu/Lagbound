import networkx as nx
import numpy as np
import scipy.sparse as sp

from lagbound import as_graph, modularity


def test_inputs_agree(tmp_path):
    G = nx.karate_club_graph()
    a = as_graph(G)
    b = as_graph(nx.to_scipy_sparse_array(G))
    path = tmp_path / "karate.txt"
    path.write_text("# karate\n" + "\n".join(f"{u} {v}" for u, v in G.edges()))
    c = as_graph(str(path))
    d = as_graph(np.array(G.edges()))
    for other in (b, c, d):
        assert (other.n, other.m) == (a.n, a.m)
        assert sorted(zip(other.u, other.v)) == sorted(zip(a.u, a.v))


def test_cleaning():
    g = as_graph(np.array([[0, 1], [1, 0], [1, 1], [1, 2], [5, 6], [5, 6]]))
    assert g.m == 3 and g.n == 5


def test_isolated_nodes_reported():
    G = nx.Graph()
    G.add_edges_from([("a", "b"), ("b", "c")])
    G.add_node("lonely")
    g = as_graph(G)
    part = g.partition_dict(np.zeros(g.n, int))
    assert part["lonely"] != part["a"] and set(part) == {"a", "b", "c", "lonely"}


def test_modularity_matches_networkx():
    G = nx.karate_club_graph()
    rng = np.random.default_rng(0)
    for _ in range(5):
        labels = rng.integers(0, 4, G.number_of_nodes())
        communities = [{i for i in G if labels[i] == c} for c in range(4)]
        communities = [c for c in communities if c]
        expected = nx.algorithms.community.modularity(G, communities, weight=None)
        assert abs(modularity(G, labels) - expected) < 1e-12


def test_dropped_information_warns():
    import pytest
    G = nx.Graph()
    G.add_edge(0, 1, weight=3.0)
    G.add_edge(1, 2)
    with pytest.warns(UserWarning, match="weights ignored"):
        as_graph(G)
    with pytest.warns(UserWarning, match="self-loops"):
        as_graph(np.array([[0, 1], [1, 1], [1, 2]]))
    with pytest.warns(UserWarning, match="other than 0/1"):
        as_graph(sp.csr_matrix(np.array([[0, 2], [2, 0]])))


def test_gap_needs_positive_lower_bound():
    from lagbound.api import BoundResult, LowerBound, UpperBound
    lo = LowerBound(0.0, 0, np.zeros(3, int), {}, 0.0, None)
    up = UpperBound(0.01, "certified", 0.01, None, {}, 0.0, "")
    res = BoundResult(lo, up)
    assert res.gap is None and abs(res.absolute_gap - 0.01) < 1e-15
