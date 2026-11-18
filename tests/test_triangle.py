import json
import math
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from conftest import brute_force_optimum
from lagbound import as_graph, bound, triangle_bound, verify_triangle_certificate
from lagbound.api import UpperBound
from lagbound.triangle import _separate, pair_lp_memory_gb
from lagbound.triangle_certificate import upper_float


@pytest.mark.parametrize("seed", range(8))
def test_exact_replay_brackets_enumerated_optimum(seed, tmp_path):
    pytest.importorskip("highspy")
    g = as_graph(nx.gnp_random_graph(6, .55, seed=seed))
    optimum = brute_force_optimum(g)
    result = triangle_bound(g, time_limit=10, workdir=tmp_path)
    replay = verify_triangle_certificate(result.certificate, g)
    assert result.status == "verified" and replay["passed"]
    assert Fraction(replay["upper_exact"]) >= optimum
    assert Fraction(result.value) >= Fraction(replay["upper_exact"])
    assert result.verify() == replay


def test_complete_graph_exact_equality(tmp_path):
    pytest.importorskip("highspy")
    g = nx.complete_graph(7)
    result = triangle_bound(g, partition=[0] * 7, workdir=tmp_path)
    assert result.value == 0 and result.verify()["optimal"]
    proof = json.loads((tmp_path / "triangle_certificate.json").read_text())
    assert proof["partition"]["modularity_exact"] == "0"


@pytest.mark.parametrize("q", [Fraction(23763, 31250), Fraction(1, 10), Fraction(307, 5618), Fraction(0)])
def test_float_is_outward(q):
    value = upper_float(q)
    assert Fraction(value) >= q
    assert Fraction(math.nextafter(value, -math.inf)) < q


@pytest.mark.parametrize("options, reason", [({"time_limit": 0}, "time budget"),
                                           ({"memory_gb": 0}, "memory estimate"),
                                           ({"target_percent": 100, "partition": [0, 0, 1, 1]}, "target met")])
def test_skips_without_solver_or_dense_allocation(options, reason, tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "highspy", None)
    result = triangle_bound(nx.path_graph(4), workdir=tmp_path, **options)
    assert result.status == "verified" and result.value == 1
    assert result.details["stop"] == reason


@pytest.mark.parametrize("options", [{"time_limit": -1}, {"time_limit": float("nan")},
                                    {"memory_gb": -1}, {"target_percent": 101}])
def test_invalid_budgets(options, tmp_path):
    with pytest.raises(ValueError):
        triangle_bound(nx.path_graph(4), workdir=tmp_path, **options)


@pytest.mark.parametrize("corrupt", ["upper", "graph", "triangle", "negative_dual", "partition"])
def test_replay_rejects_tampering(corrupt, tmp_path):
    pytest.importorskip("highspy")
    result = triangle_bound(nx.cycle_graph(5), partition=[0, 0, 1, 1, 1], workdir=tmp_path)
    p = tmp_path / "triangle_certificate.json"
    proof = json.loads(p.read_text())
    if corrupt == "upper":
        proof["upper_exact"] = "-1"
    elif corrupt == "graph":
        proof["graph"]["edges"].pop()
    elif corrupt == "triangle":
        proof["triangles"].append([0, 0, 1, "1"])
    elif corrupt == "negative_dual":
        proof["triangles"].append([0, 1, 2, "-1"])
    else:
        proof["partition"]["modularity_exact"] = "1"
    p.write_text(json.dumps(proof))
    assert not result.verify()["passed"]


def test_graph_identity(tmp_path):
    r = triangle_bound(nx.path_graph(5), time_limit=0, workdir=tmp_path)
    assert not verify_triangle_certificate(r.certificate, nx.star_graph(4))["passed"]


def test_bounded_separation_and_incomplete_scan():
    n = 40
    iu, ju = np.triu_indices(n, 1)
    x = np.random.default_rng(5).uniform(0, 1, len(iu))
    cuts, complete = _separate(n, iu, ju, x, set(), 7, float("inf"))
    assert complete and len(cuts) == 7
    X = np.zeros((n, n)); X[iu, ju] = x; X += X.T
    assert all(X[i, j] + X[j, k] - X[i, k] > 1 + 1e-7 for _, i, j, k in cuts)
    _, complete = _separate(n, iu, ju, x, set(), 7, 0)
    assert not complete


def test_large_network_memory_gate():
    assert pair_lp_memory_gb(281903) > 8


@pytest.mark.parametrize("old, status, winner", [(1.0, "verified", "triangle_lp"),
                                               (7 / 32, "verified", None),
                                               (.01, "rejected", "triangle_lp")])
def test_integration_keeps_best_accepted_bound(old, status, winner, tmp_path, monkeypatch):
    pytest.importorskip("highspy")
    import lagbound.api as api
    base = UpperBound(old, status, old, None, {}, 0, str(tmp_path))
    monkeypatch.setattr(api, "_sdp_upper_bound", lambda *a, **k: base)
    result = api.upper_bound(nx.path_graph(5), tighten=True, workdir=tmp_path)
    assert result.details.get("method") == winner
    assert result.status == "verified"
    assert result.value <= (old if status == "verified" else 1)
    assert result.details["tightening"]["run"]


def test_target_uses_incumbent_not_untrusted_lower(tmp_path, monkeypatch):
    pytest.importorskip("highspy")
    import lagbound.api as api
    base = UpperBound(.3, "verified", .3, None, {}, 0, str(tmp_path))
    monkeypatch.setattr(api, "_sdp_upper_bound", lambda *a, **k: base)
    result = api.upper_bound(nx.path_graph(4), partition=[0, 0, 1, 1], lower=1,
                             tighten=True, target_percent=0, workdir=tmp_path)
    assert result.details["tightening"]["run"]


def test_incumbent_without_heuristic_and_exact_percentage(tmp_path, monkeypatch):
    import lagbound.api as api
    monkeypatch.setattr(api, "upper_bound", lambda *a, **k: UpperBound(.25, "verified", .25, None, {}, 0, str(tmp_path)))
    result = bound(nx.path_graph(4), lower=False, partition=[0, 0, 1, 1], workdir=tmp_path)
    assert result.lower.exact == Fraction(1, 6)
    assert result.within_percent == Fraction(100, 3)
    assert json.loads((tmp_path / "incumbent.json").read_text())["modularity_exact"] == "1/6"


def test_zero_or_rejected_has_no_relative_guarantee(tmp_path, monkeypatch):
    import lagbound.api as api
    monkeypatch.setattr(api, "upper_bound", lambda *a, **k: UpperBound(1, "verified", 1, None, {}, 0, str(tmp_path)))
    result = bound(nx.path_graph(4), lower=False, partition=[0] * 4, workdir=tmp_path)
    assert result.within_percent is None
    result = bound(nx.path_graph(4), lower=False, partition=[0, 0, 1, 1], workdir=tmp_path)
    result.upper.status = "rejected"
    assert result.within_percent is None


def test_failed_tightening_keeps_verified_sdp(tmp_path, monkeypatch):
    pytest.importorskip("highspy")
    import lagbound.api as api
    import lagbound.triangle as triangle
    base = UpperBound(.3, "verified", .3, None, {}, 0, str(tmp_path))
    monkeypatch.setattr(api, "_sdp_upper_bound", lambda *a, **k: base)
    def fail(*args, **kwargs):
        raise MemoryError("test allocation failure")
    monkeypatch.setattr(triangle, "triangle_bound", fail)
    result = api.upper_bound(nx.path_graph(4), tighten=True, workdir=tmp_path)
    assert result is base and result.status == "verified"
    assert result.details["tightening"]["stop"] == "failed"


def test_target_skips_lp_and_cannot_be_claimed_without_tightening(tmp_path, monkeypatch):
    pytest.importorskip("highspy")
    import lagbound.api as api
    base = UpperBound(.17, "verified", .17, None, {}, 0, str(tmp_path))
    monkeypatch.setattr(api, "_sdp_upper_bound", lambda *a, **k: base)
    result = api.upper_bound(nx.path_graph(4), partition=[0, 0, 1, 1], tighten=True,
                             target_percent=5, workdir=tmp_path)
    assert result is base and result.details["tightening"] == {"run": False, "stop": "target met"}
    with pytest.raises(ValueError, match="requires tighten"):
        api.upper_bound(nx.path_graph(4), target_percent=5)


def test_rejected_sdp_does_not_skip_tightening_for_target(tmp_path, monkeypatch):
    pytest.importorskip("highspy")
    import lagbound.api as api
    base = UpperBound(.01, "rejected", .01, None, {}, 0, str(tmp_path))
    monkeypatch.setattr(api, "_sdp_upper_bound", lambda *a, **k: base)
    result = api.upper_bound(nx.path_graph(4), partition=[0, 0, 1, 1], tighten=True,
                             target_percent=100, workdir=tmp_path)
    assert result.status == "verified" and result.details["tightening"]["run"]


def test_lp_uses_cumulative_time_limit(tmp_path, monkeypatch):
    highspy = pytest.importorskip("highspy")
    Original = highspy.Highs
    instances = []
    class TrackingHighs(Original):
        def __init__(self):
            super().__init__()
            self.calls = 0
            instances.append(self)
        def getRunTime(self):
            return 12345.0
        def setOptionValue(self, name, value):
            if name == "time_limit":
                assert 12345 < value <= 12355
                self.calls += 1
            return super().setOptionValue(name, value)
    monkeypatch.setattr(highspy, "Highs", TrackingHighs)
    triangle_bound(nx.cycle_graph(5), time_limit=10, workdir=tmp_path)
    assert instances[0].calls >= 2


def test_cli_json_survives_native_solver_output(tmp_path):
    pytest.importorskip("highspy")
    import os
    import subprocess
    import sys
    # Exercise the CLI with a real LP while substituting only the costly SDP stage.
    graph = tmp_path / "graph.txt"
    graph.write_text("0 1\n1 2\n2 3\n")
    part = tmp_path / "part.txt"
    part.write_text("0 0\n1 0\n2 1\n3 1\n")
    code = """
import sys
import lagbound.api as api
from lagbound.cli import main
api._sdp_upper_bound = lambda *a, **k: api.UpperBound(1., 'verified', 1., None, {}, 0., sys.argv[3])
raise SystemExit(main([sys.argv[1], '--partition', sys.argv[2], '--upper-only', '--tighten', '--json',
                       '--workdir', sys.argv[3], '-q']))
"""
    proc = subprocess.run([sys.executable, "-c", code, str(graph), str(part), str(tmp_path)],
                          capture_output=True, text=True, env=os.environ, check=True)
    record = json.loads(proc.stdout)
    assert record["upper"]["status"] == "verified"
    assert record["within_percent"] == "0.00" and record["within_percent_exact"] == "0"


def test_score_rounding_is_outward():
    from lagbound.score import _round
    for q in (Fraction(23763, 31250), Fraction(1, 10), Fraction(307, 5618)):
        assert Fraction(_round(q, 9, up=True)) >= q
        assert Fraction(_round(q, 9, up=False)) <= q
