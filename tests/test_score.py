import json
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from lagbound import CERTIFIED, modularity, score
from lagbound.cli import main

KARATE_OPT = 0.4197896120973044  # proven optimum of Zachary's karate club


def karate():
    return nx.Graph(nx.karate_club_graph().edges())                # unweighted, as modularity is computed here


def greedy_partition(G):
    labels = np.zeros(G.number_of_nodes(), int)
    for c, nodes in enumerate(nx.algorithms.community.greedy_modularity_communities(G)):
        labels[list(nodes)] = c
    return labels


def test_bracket_contains_the_true_improvement():
    G = karate()
    labels = greedy_partition(G)
    s = score(G, labels, upper=0.42, lower=KARATE_OPT)
    q = modularity(G, labels, exact=True)
    assert s.exact == q and abs(s.modularity - float(q)) < 1e-15
    lo, hi = s.improvement
    true = (Fraction(KARATE_OPT) - q) / q
    assert lo <= true <= hi and 0 < lo
    assert s.absolute_gap >= 0.42 - float(q)


def test_without_lower_bound_the_lower_limit_is_zero():
    G = karate()
    s = score(G, greedy_partition(G), upper=0.42)
    assert s.improvement[0] == 0.0 and s.improvement[1] > 0


def test_nonpositive_modularity_has_no_relative_range():
    G = karate()
    s = score(G, np.arange(G.number_of_nodes()), upper=0.42)      # singletons: Q < 0
    assert s.modularity < 0 and s.improvement == (None, None)


def test_bound_below_modularity_is_refused():
    G = karate()
    with pytest.raises(ValueError, match="below"):
        score(G, greedy_partition(G), upper=0.1)


def test_certified_network_checks_the_graph():
    with pytest.raises(ValueError, match="does not apply"):
        score(karate(), np.zeros(34, int), network="coPapersDBLP")
    assert set(CERTIFIED) >= {"Hollywood-2011", "ogbn-products"}
    for ref in CERTIFIED.values():
        assert ref["lower"] < ref["upper"] < 1


def test_partition_must_cover_every_node():
    G = karate()
    with pytest.raises(ValueError, match="no cluster"):
        score(G, {0: 0, 1: 0}, upper=0.42)


def test_cli_score(tmp_path, capsys):
    G = karate()
    graph = tmp_path / "karate.txt"
    graph.write_text("\n".join(f"{u} {v}" for u, v in G.edges()))
    part = tmp_path / "part.txt"
    part.write_text("\n".join(f"{i} {c}" for i, c in enumerate(greedy_partition(G))))
    assert main(["score", str(graph), str(part), "--upper", "0.42", "--lower", str(KARATE_OPT), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["improvement_at_least"] > 0 and out["improvement_at_most"] >= out["improvement_at_least"]
    assert main(["score", str(graph), str(part), "--network", "coPapersDBLP"]) == 2
