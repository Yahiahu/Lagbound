import json

import numpy as np
import pytest

from lagbound._core import verify_cert as vc


def write_instances(tmp_path, edges, groups):
    """Original graph and its contraction by ``groups`` (one group id per original node), in the verifier's format."""
    edges = np.sort(np.asarray(edges, dtype=np.int64), axis=1)
    no = int(edges.max()) + 1
    grp = np.unique(np.asarray(groups), return_inverse=True)[1].astype(np.int64)
    n = int(grp.max()) + 1
    deg = np.bincount(edges.ravel(), minlength=no)
    orig, red = tmp_path / "orig", tmp_path / "red"
    orig.mkdir(); red.mkdir()
    json.dump({"n": no, "m": len(edges)}, open(orig / "graph.json", "w"))
    np.save(orig / "keys.npy", edges[:, 0] * no + edges[:, 1])
    gu, gv = grp[edges[:, 0]], grp[edges[:, 1]]
    inside = gu == gv
    kk, w = np.unique(np.minimum(gu, gv)[~inside] * n + np.maximum(gu, gv)[~inside], return_counts=True)
    json.dump({"n": n, "m": len(kk)}, open(red / "graph.json", "w"))
    np.save(red / "keys.npy", kk)
    np.save(red / "weights.npy", w.astype(np.float64))
    np.save(red / "vol.npy", np.bincount(grp, weights=deg, minlength=n).astype(np.int64))
    np.save(red / "group_of.npy", grp)
    json.dump({"L_orig": 2 * len(edges), "leaf_edges": int(inside.sum()), "inside_edges": int(inside.sum())},
              open(red / "meta.json", "w"))
    return str(orig), str(red)


def check(tmp_path, edges, groups):
    orig, red = write_instances(tmp_path, edges, groups)
    return vc.check_reduction(orig, red, vc.load_instance(red), None)


# K4 on 0..3 with leaf 7 at node 0 and a triangle 4-5-6 hanging off node 3: 1, 2 are twins (degree 3, 9 < 2m = 22),
# 5, 6 are twins (degree 2), and 7 is a leaf of 0.
GRAPH = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (4, 6), (0, 7)]


def test_eligible_contractions_pass(tmp_path):
    res = check(tmp_path, GRAPH, [0, 1, 1, 2, 3, 4, 4, 0])
    assert res["bookkeeping"] == "ok"
    assert res["contractions"] == dict(eligible=True, leaf_unions=1, twin_unions=2, ineligible_inside_edges=0)


def test_merging_adjacent_non_twins_is_rejected(tmp_path):
    with pytest.raises(vc.Fail, match="eligible"):
        check(tmp_path, GRAPH, [0, 1, 2, 3, 3, 4, 5, 6])          # 3 and 4 are adjacent but not twins


def test_merging_non_adjacent_nodes_is_rejected(tmp_path):
    square = [(0, 1), (1, 2), (2, 3), (3, 0)]                      # 0 and 2 share all neighbours but are not adjacent
    with pytest.raises(vc.Fail, match="eligible"):
        check(tmp_path, square, [0, 1, 0, 2])


def test_twins_with_large_degree_are_rejected(tmp_path):
    # 0 and 1 are adjacent twins on 2, 3, 4: degree 4, and 4^2 = 16 >= 2m = 14, so merging them is not justified
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)]
    with pytest.raises(vc.Fail, match="eligible"):
        check(tmp_path, edges, [0, 0, 1, 2, 3])


def test_partial_contraction_passes(tmp_path):
    # merging only some eligible pairs is fine: a group that is a subset of an eligible component keeps the optimum
    res = check(tmp_path, GRAPH, [0, 1, 1, 2, 3, 4, 5, 6])
    assert res["contractions"]["twin_unions"] == 1 and res["contractions"]["leaf_unions"] == 0


# ---------------------------------------------------------------------------------------------------- Nov 14 2026 audit
def test_all_in_one_contraction_of_a_path_is_rejected(tmp_path):
    # P4 merged into one node: the bookkeeping is consistent, but 1-2 is neither a leaf nor a twin pair, and the
    # reduced optimum (0) is below the original one (1/6).  The public commit c92d74d accepted this.
    with pytest.raises(vc.Fail, match="eligible"):
        check(tmp_path, [(0, 1), (1, 2), (2, 3)], [0, 0, 0, 0])


def test_contracting_non_adjacent_endpoints_of_two_edges_is_rejected(tmp_path):
    # two disjoint edges, groups {0, 2} and {1, 3} (Oct 4 2026 audit, finding 1)
    with pytest.raises(vc.Fail, match="eligible"):
        check(tmp_path, [(0, 1), (2, 3)], [0, 1, 0, 1])


def test_accept_price_dominates_its_evaluation_error():
    from fractions import Fraction
    charged, ev = vc.accept_price(0.15, 2.5e8, 1e-300)
    assert Fraction(charged) >= Fraction(0.15)/(1 - vc.gam_frac(2.5e8)) and ev["margin"] > 9e-4
    with pytest.raises(vc.Fail, match="evaluation error"):
        vc.accept_price(0.15, 1e13, 0.0)              # gamma(1e13) > 1e-3: the 1.001 factor no longer covers it
    with pytest.raises(vc.Fail, match="evaluation error"):
        vc.accept_price(1e-310, 10, 1e-300)           # underflow loss larger than the allowance


def test_certifier_final_sum_is_exact():
    from fractions import Fraction
    from lagbound._core.certify_lifted_vc import certified_total
    # nested correctly rounded sums + one upward rounding fall below the exact value here (Codex review, Nov 14 2026)
    g = np.array([-1e16, 1.0])
    ub, tot = certified_total(1e16, np.zeros(0), np.zeros(0), g, np.zeros(1), 0.0, 0.0, 2)
    assert tot == Fraction(1) and Fraction(ub) >= Fraction(1, 2)


def test_neighbour_rows_are_refused(tmp_path):
    import os
    import networkx as nx
    from lagbound import upper_bound
    from lagbound.pipeline import TAG, Workspace
    ub = upper_bound(nx.karate_club_graph(), time_limit=20, workdir=str(tmp_path / "w"))
    inst = os.path.join(Workspace(None, ub.workdir).data, TAG + "__rx")
    good = vc.verify(inst, ub.certificate, "ldl")
    assert good["passed"] and good["psd"]["ldl"]["info"]["price_eval"]["margin"] > 0
    assert good["assembly"]["price_eval"]["margin"] > 0
    z = dict(np.load(ub.certificate))
    mu = np.zeros(int(z["n"])); mu[0] = 0.25
    z["mu"] = mu
    bad = str(tmp_path / "with_mu.npz"); np.savez(bad, **z)
    res = vc.verify(inst, bad, "ldl")
    assert not res["passed"] and "mu / om" in res["fail"]


def test_reduction_verifier_cache_is_importable_from_package(tmp_path):
    """A subprocess-generated cache must also work for a later package import."""
    import os
    import subprocess
    import sys

    script = """
import sys
from pathlib import Path
import networkx as nx
from lagbound import upper_bound

root = Path(sys.argv[1])
result = upper_bound(nx.complete_graph(6), verify=True, workdir=str(root))
assert result.status == "verified"
from lagbound._core import verify_cert as verifier
original, reduced = str(root / "data/g"), str(root / "data/g__rx")
check = verifier.check_reduction(original, reduced, verifier.load_instance(reduced), None)
assert check["contractions"]["eligible"]
"""
    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    subprocess.run([sys.executable, "-c", script, str(tmp_path / "run")],
                   env=env, check=True, capture_output=True, text=True, timeout=120)
