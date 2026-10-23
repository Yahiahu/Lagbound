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
