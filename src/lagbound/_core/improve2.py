"""Lower-bound improvement v2 (igraph native C Leiden, ~20-40x faster than leidenalg here).

Pipeline:
  1. pool <- previous best labels (refined), N Leiden-from-scratch runs (random seeds), optional SDP roundings
     (k-means on unit-normalised SDP vectors), each refined by Leiden started from its membership.
  2. iterated ensemble (CGGCi-style): meet of top-K pool partitions -> contract graph (self-loops keep
     internal weight so strengths = volumes) -> many Leiden runs on the core graph -> pull back -> refine on
     the full graph -> add to pool. Plus "dissolve-and-reoptimise" perturbations of the incumbent
     (random subset of communities split to singletons, then Leiden from that membership).
  3. exact Fraction modularity of every saved incumbent (lbutil.save_best).
usage: python improve2.py TAG BUDGET_S [--nscratch N] [--kmeans k1,k2] [--topk K] [--ens_seeds S]
"""
import time, argparse, random, resource
import numpy as np
from lbutil import *

ap = argparse.ArgumentParser()
ap.add_argument("tag"); ap.add_argument("budget", type=float)
ap.add_argument("--nscratch", type=int, default=30)
ap.add_argument("--kmeans", default="")
ap.add_argument("--topk", type=int, default=8)
ap.add_argument("--ens_seeds", type=int, default=12)
ap.add_argument("--seed0", type=int, default=0)
ap.add_argument("--start", default=None, help="extra start labels .npy (comma-separated)")
ap.add_argument("--dissolve", type=float, default=0.05)
A = ap.parse_args()
T0 = time.time(); tag = A.tag
rs = random.Random(A.seed0); ig.set_random_number_generator(rs)


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


def log(*a):
    print(f"[{time.time()-T0:7.1f}s rss={rss():.2f}G]", *a, flush=True)


def left():
    return A.budget - (time.time() - T0)


n, m, u, v, deg = load(tag)
G = to_igraph(n, u, v)
log(tag, "n", n, "m", m)


def leiden(graph, init=None, weights=None, node_weights=None):
    # NB: igraph's C Leiden mishandles self-loops (quality inconsistent), so the core graph is built
    # loop-free and the core volumes are passed explicitly as node_weights (normalised by their sum = 2m).
    c = graph.community_leiden(objective_function="modularity", weights=weights, initial_membership=init,
                               n_iterations=-1, node_weights=node_weights)
    return np.array(c.membership, dtype=np.int64)


def refine(memb):
    lab = relabel(memb)
    best = Q_float(lab, m, u, v, deg)
    for _ in range(20):
        lab2 = leiden(G, lab.tolist())
        q2 = Q_float(lab2, m, u, v, deg)
        if q2 <= best + 1e-13:
            if q2 > best: lab, best = lab2, q2
            break
        lab, best = relabel(lab2), q2
    return lab, best


pool = []


def add(lab, q, name):
    for p in pool:  # skip exact duplicates (same Q and same partition)
        if abs(p[0] - q) < 1e-14:
            return
    pool.append((q, lab, name)); pool.sort(key=lambda x: -x[0]); del pool[max(3 * A.topk, 16):]
    save_best(tag, lab, name, time.time() - T0, n, m, u, v, deg)


starts = ([GRAPHS[tag]["prev"]] if GRAPHS[tag]["prev"] else []) + ([s for s in A.start.split(",")] if A.start else [])
for f in starts:
    lab0 = np.load(f)
    q0 = Q_float(lab0, m, u, v, deg)
    lab, q = refine(lab0)
    log(f"start {os.path.basename(f)} Q={q0:.10f} refined={q:.10f}")
    add(lab, q, f"start({os.path.basename(f)})+leiden")

if A.kmeans:
    from scipy.cluster.vq import kmeans2
    Z = np.load(GRAPHS[tag]["V"]); V = np.asarray(Z["V"], dtype=np.float64); del Z
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(A.seed0 + 7)
    for k in [int(x) for x in A.kmeans.split(",")]:
        for rep in range(2):
            _, lk = kmeans2(V, k, minit="++", seed=int(rng.integers(1 << 30)), iter=20)
            lab, q = refine(lk)
            log(f"sdp kmeans k={k} raw={Q_float(lk, m, u, v, deg):.6f} refined={q:.8f}")
            add(lab, q, f"sdp-kmeans{k}+leiden")
    del V

qs = []
for s in range(A.nscratch):
    if left() < 0.6 * A.budget: break
    lab, q = refine(leiden(G))
    qs.append(q)
    add(lab, q, "leiden-scratch")
if qs:
    log(f"scratch runs {len(qs)}: max={max(qs):.8f} mean={np.mean(qs):.8f} pool best={pool[0][0]:.10f}")

rnd = 0
stale = 0
while left() > 0:
    rnd += 1
    K = min(A.topk, len(pool))
    key = np.zeros(n, dtype=np.int64)
    for p in pool[:K]:
        key = relabel(key * (int(p[1].max()) + 1) + p[1])
    ncore = int(key.max()) + 1
    H = G.copy(); H.es["weight"] = 1
    H.contract_vertices(key.tolist(), combine_attrs=None)
    H.simplify(multiple=True, loops=True, combine_edges="sum")
    nw = np.zeros(ncore, dtype=np.int64); np.add.at(nw, key, deg); nw = nw.astype(float).tolist()
    before = pool[0][0]
    snap = [p[1] for p in pool[:K]]
    _, proj = np.unique(key, return_index=True)
    bq = -1
    for s in range(A.ens_seeds):
        if left() < 0: break
        if s < min(3, K):  # start from a pool partition projected onto the cores (monotone step)
            cl = leiden(H, init=relabel(snap[s][proj]).tolist(), weights="weight", node_weights=nw)
        else:
            cl = leiden(H, weights="weight", node_weights=nw)
        lab, q = refine(cl[key])
        if s < 3: log(f"  core-seed {s}: Q(start)={Q_float(snap[min(s, K-1)], m, u, v, deg):.10f} -> {q:.10f}")
        bq = max(bq, q)
        add(lab, q, f"ensemble-r{rnd}(K={K},cores={ncore})")
    del H
    log(f"ensemble r{rnd}: K={K} cores={ncore} best_this={bq:.10f} pool_best={pool[0][0]:.10f}")
    # dissolve-and-reoptimise perturbations of top partitions
    rng2 = np.random.default_rng(1000 + rnd)
    for s in range(A.ens_seeds):
        if left() < 0: break
        lab = pool[min(s % 3, len(pool) - 1)][1].copy()
        cs = np.unique(lab)
        kill = rng2.choice(cs, size=max(1, int(A.dissolve * len(cs))), replace=False)
        mask = np.isin(lab, kill)
        lab[mask] = lab.max() + 1 + np.arange(mask.sum())
        lab, q = refine(lab)
        add(lab, q, f"dissolve-r{rnd}")
    # fresh diversity
    for s in range(max(2, A.ens_seeds // 3)):
        if left() < 0: break
        lab, q = refine(leiden(G)); add(lab, q, "leiden-scratch")
    log(f"after r{rnd}: pool_best={pool[0][0]:.10f} top={[round(float(p[0]), 7) for p in pool[:6]]}")
    stale = stale + 1 if pool[0][0] <= before + 1e-13 else 0

log("DONE best", pool[0][0], pool[0][2])
