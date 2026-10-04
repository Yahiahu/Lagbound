"""Exact (optimum-preserving) reduction for modularity: merge vertex groups that share a community in EVERY optimal
partition, so the union of rules is again satisfied by every optimal partition.
 Rule L (leaves): a degree-1 vertex is with its neighbour (move gain >= 1/(2m^2) > 0).
 Rule T (true twins): adjacent u,v with N[u]=N[v] and d^2 < 2m are together (the two cross-moves gain (2m-d^2)/m^2 in sum).
Reduced instance: group graph with integer edge weights w_PQ = #edges between groups, volumes vol_P = sum of degrees,
original L = 2m, constant C0 = 2 * #edges inside groups:  2m Q(X) = C0 + <B', X'>,  B' = W - vol vol^T/L.
Writes data/<tag>__rx/{graph.json, keys.npy, weights.npy, vol.npy, meta.json, group_of.npy} + restricted label files
(the label of a group = label of its first member, after enforcing the rules on the partition), with an exact Q check.
usage: reduce_exact.py <tag> [label files ...]"""
import sys, os, json, numpy as np
from fractions import Fraction
tag = sys.argv[1]; D = f"data/{tag}/"; Dz = f"data/{tag}__rx/"; os.makedirs(Dz, exist_ok=True)
n = json.load(open(D+"graph.json"))["n"]; keys = np.load(D+"keys.npy"); u = keys//n; v = keys % n; m = len(keys); L = 2*m
d = np.bincount(np.r_[u, v], minlength=n)
import scipy.sparse as sp
A = sp.csr_matrix((np.ones(2*m, np.int8), (np.r_[u, v], np.r_[v, u])), shape=(n, n)); A.sort_indices()
par = np.arange(n)
def find(x):
    r = x
    while par[r] != r: r = par[r]
    while par[x] != r: par[x], x = r, par[x]
    return r
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: par[max(ra, rb)] = min(ra, rb)
# rule L
leafedge = (d[u] == 1) | (d[v] == 1)
for a, b in zip(u[leafedge], v[leafedge]): union(a, b)
nL = int(leafedge.sum())
# rule T: candidate pairs by closed-neighbourhood hash, then exact check
rng = np.random.default_rng(11); r = rng.integers(1, 2**62, size=n, dtype=np.int64).astype(np.uint64)
h = r.copy(); np.add.at(h, u, r[v]); np.add.at(h, v, r[u])
cand = (h[u] == h[v]) & (d[u] == d[v]) & (d[u].astype(float)**2 < 2*m) & (d[u] > 1)
nT = 0
for a, b in zip(u[cand], v[cand]):
    Na = set(A.indices[A.indptr[a]:A.indptr[a+1]].tolist()) | {a}; Nb = set(A.indices[A.indptr[b]:A.indptr[b+1]].tolist()) | {b}
    if Na == Nb: union(a, b); nT += 1
root = np.array([find(x) for x in range(n)]); reps, grp = np.unique(root, return_inverse=True); N = len(reps)
vol = np.bincount(grp, weights=d, minlength=N).astype(np.int64)
gu, gv = grp[u], grp[v]; inside = gu == gv; C0e = int(inside.sum())
a_ = np.minimum(gu[~inside], gv[~inside]); b_ = np.maximum(gu[~inside], gv[~inside])
kk, w = np.unique(a_*N + b_, return_counts=True)
assert vol.sum() == L and w.sum() + C0e == m
json.dump({"n": int(N), "m": int(len(kk))}, open(Dz+"graph.json", "w")); np.save(Dz+"keys.npy", kk.astype(np.int64))
np.save(Dz+"weights.npy", w.astype(np.float64)); np.save(Dz+"vol.npy", vol); np.save(Dz+"group_of.npy", grp)
json.dump({"L_orig": int(L), "leaf_edges": int(C0e), "inside_edges": int(C0e), "n_orig": int(n), "m_orig": int(m),
           "leaf_unions": nL, "twin_unions": nT}, open(Dz+"meta.json", "w"))
print(f"{tag}: n {n} -> {N} (leaf edges {nL}, twin pairs {nT}), m {m} -> {len(kk)} weighted (max w {w.max()}), "
      f"inside edges {C0e}, C0/L {2*C0e/L:.6f}")
for f in sys.argv[2:]:
    lab = np.load(f).astype(np.int64)
    # enforce rules on the partition: every group takes the label of its representative's majority? use first member
    first = np.full(N, -1); order = np.argsort(grp, kind="stable"); gs = grp[order]; firsts = order[np.r_[True, gs[1:] != gs[:-1]]]
    lab2 = lab[firsts][grp]; P = lab2.max()+1
    volc = np.bincount(lab2, weights=d, minlength=P).astype(np.int64); ein = int((lab2[u] == lab2[v]).sum())
    Qo = Fraction(2*ein, L) - Fraction(int((volc.astype(object)**2).sum()), L*L)
    lr = lab[firsts]; volr = np.bincount(lr, weights=vol, minlength=P).astype(np.int64)
    ein_r = int(w[lr[kk//N] == lr[kk % N]].sum())
    Qr = Fraction(2*C0e + 2*ein_r, L) - Fraction(int((volr.astype(object)**2).sum()), L*L)
    assert Qo == Qr, (float(Qo), float(Qr))
    Qorig = Fraction(2*int((lab[u] == lab[v]).sum()), L) - Fraction(int((np.bincount(lab, weights=d).astype(np.int64).astype(object)**2).sum()), L*L)
    np.save(Dz+os.path.basename(f), lr); print(f"  {os.path.basename(f)}: original Q {float(Qorig):.6f}; with rules enforced {float(Qo):.6f} == reduced {float(Qr):.6f} OK")
