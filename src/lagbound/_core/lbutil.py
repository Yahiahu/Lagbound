"""Shared helpers for lower-bound (partition) improvement: graph loading, exact modularity, saving."""
import json, os, time
from fractions import Fraction
import numpy as np
import igraph as ig

# lagbound: graphs are read from DATA_DIR/<tag> (tol5 format) and results written to LAGBOUND_OUT
# (the original research script used a hard-coded registry of graphs and paths here).
OUT = os.environ.get("LAGBOUND_OUT", os.getcwd())


class _Registry(dict):
    def __missing__(self, tag):
        return dict(dir=os.path.join(os.environ.get("DATA_DIR", "."), tag), lb=float("nan"), ub=float("nan"), prev=None, V="")


GRAPHS = _Registry()


def load(tag):
    d = GRAPHS[tag]["dir"]
    g = json.load(open(d + "/graph.json"))
    n, m = int(g["n"]), int(g["m"])
    keys = np.load(d + "/keys.npy").astype(np.int64)
    u = (keys // n).astype(np.int64); v = (keys % n).astype(np.int64)
    assert len(u) == m, (len(u), m)
    assert (u != v).all()
    deg = np.bincount(np.r_[u, v], minlength=n).astype(np.int64)
    return n, m, u, v, deg


def to_igraph(n, u, v):
    return ig.Graph(n=n, edges=np.column_stack([u, v]).tolist(), directed=False)


def relabel(lab):
    _, inv = np.unique(np.asarray(lab), return_inverse=True)
    return inv.astype(np.int64)


def Q_exact(lab, n, m, u, v, deg):
    """Exact Q = sum_C [e_C/m - (vol_C/2m)^2] as Fraction."""
    lab = relabel(lab)
    same = int((lab[u] == lab[v]).sum())
    vol = np.zeros(lab.max() + 1, dtype=np.int64)
    np.add.at(vol, lab, deg)
    s2 = sum(int(x) * int(x) for x in vol.tolist())
    num = 4 * m * same - s2
    # Q = same/m - s2/(4m^2) = (4m*same - s2)/(4m^2)
    return Fraction(num, 4 * m * m)


def Q_float(lab, m, u, v, deg):
    lab = np.asarray(lab)
    same = (lab[u] == lab[v]).sum()
    vol = np.bincount(lab, weights=deg.astype(float))
    return same / m - (vol ** 2).sum() / (4.0 * m * m)


def save_best(tag, lab, method, t_elapsed, n, m, u, v, deg, extra=None):
    """Save labels if better than existing saved best; returns Q float."""
    lab = relabel(lab)
    Qf = Q_exact(lab, n, m, u, v, deg)
    q = float(Qf)
    jp = f"{OUT}/best_labels_{tag}.json"
    old = None
    if os.path.exists(jp):
        old = json.load(open(jp))
    if old is None or q > old["Q_float"] + 1e-15:
        np.save(f"{OUT}/best_labels_{tag}.npy", lab)
        info = dict(tag=tag, Q_exact=f"{Qf.numerator}/{Qf.denominator}", Q_float=q, ncomm=int(lab.max() + 1),
                    method=method, elapsed_s=round(t_elapsed, 1), old_LB=GRAPHS[tag]["lb"], UB=GRAPHS[tag]["ub"],
                    gap_old=(GRAPHS[tag]["ub"] - GRAPHS[tag]["lb"]) / GRAPHS[tag]["lb"],
                    gap_new=(GRAPHS[tag]["ub"] - q) / q, time=time.strftime("%Y-%m-%d %H:%M:%S"))
        if extra: info.update(extra)
        json.dump(info, open(jp, "w"), indent=1)
        print(f"[save] {tag} Q={q:.10f} ({method})", flush=True)
    return q
