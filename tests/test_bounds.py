import json

import networkx as nx
import pytest

from conftest import brute_force_optimum
from lagbound import as_graph, bound, lower_bound, upper_bound
from lagbound.cli import main

KARATE_OPT = 0.4197896120973044  # proven optimum of Zachary's karate club


@pytest.mark.parametrize("seed", [0, 1])
def test_bounds_bracket_the_optimum(seed, small_graphs):
    g = as_graph(small_graphs[seed])
    q_star = float(brute_force_optimum(g))
    res = bound(g, time_limit=20, seeds=3, refine_time=2, verify=True)
    assert res.lower.value <= q_star + 1e-12
    assert res.upper.status == "verified"
    assert res.upper.value >= q_star - 1e-12
    assert res.gap >= 0


def test_lower_only():
    lb = lower_bound(nx.karate_club_graph(), seeds=5, refine_time=2)
    assert abs(lb.value - KARATE_OPT) < 1e-9
    assert len(lb.partition) == 34 and lb.workdir is None


def test_upper_only_then_verify(tmp_path):
    ub = upper_bound(nx.karate_club_graph(), time_limit=30, workdir=str(tmp_path))
    assert ub.status == "certified"
    assert KARATE_OPT <= ub.estimate <= ub.value < 0.44
    check = ub.verify()
    assert check["passed"] and check["UB"] >= KARATE_OPT


def test_estimate_only():
    ub = upper_bound(nx.karate_club_graph(), time_limit=15, certify=False)
    assert ub.status == "estimate" and ub.certificate is None
    assert ub.value >= KARATE_OPT


def test_cli_lower_only(tmp_path, capsys):
    path = tmp_path / "karate.txt"
    path.write_text("\n".join(f"{u} {v}" for u, v in nx.karate_club_graph().edges()))
    part = tmp_path / "part.txt"
    assert main([str(path), "--lower-only", "--seeds", "3", "--refine-time", "1", "--json", "-q",
                 "--partition-out", str(part)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert abs(out["lower"]["value"] - KARATE_OPT) < 1e-9 and "upper" not in out
    assert len(part.read_text().splitlines()) == 34


def test_hollywood_settings():
    # levels and certificate settings of the Hollywood runs: starting partition as a level, triangle filter, bisection
    ub = upper_bound(nx.karate_club_graph(), time_limit=20, levels=(1, 16), vc_level=16, fill_filter=True,
                     margin=1.5, bisect=1, bisect_lo=0.7, verify=True, verify_max_gb=4)
    assert ub.status == "verified" and ub.value >= KARATE_OPT


def test_failed_verification_is_not_reported_as_passed(tmp_path):
    ub = upper_bound(nx.karate_club_graph(), time_limit=10, workdir=str(tmp_path))
    assert ub.verify()["passed"]                       # leaves a passing verify.json behind
    ub.certificate = str(tmp_path / "missing.npz")
    assert not ub.verify()["passed"]


@pytest.mark.parametrize("graph, q_star", [(nx.complete_graph(12), 0.0),
                                           (nx.disjoint_union(nx.complete_graph(5), nx.complete_graph(6)), 0.48)])
def test_solved_by_reductions(graph, q_star):
    # twin merges leave no edge between groups: the optimum is exact, with no solver run and no certificate
    ub = upper_bound(graph, verify=True)
    assert ub.status == "verified" and ub.certificate is None and ub.details["solved_by_reduction"]
    assert q_star <= ub.value <= q_star + 1e-15


def test_small_reduced_instance_converges():
    # Les Miserables reduces to 46 nodes; as a single (Jacobi) batch its rows locked antipodal and the bound diverged
    # (U = 25 on the MaxSAT benchmark). Optimum 72259/129032 (Ansotegui et al., ICCS 2025).
    ub = upper_bound(nx.les_miserables_graph(), time_limit=60, verify=True)
    assert ub.status == "verified" and 72259 / 129032 <= ub.value < 0.58
