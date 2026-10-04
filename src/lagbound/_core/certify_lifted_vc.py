#!/usr/bin/env python3
"""[brk/verify COPY of ../certvc/certify_lifted_vc.py: identical numerics + `--dump_cert FILE.npz` which saves the exact
certificate (multipliers used, g, shift diagonal, lifted tree, cluster->node maps) for the independent verify_cert.py.]
Sparse lifted certificate WITH vertex-cluster (VC) multipliers (Oct 1 2026; extends ../../../certify_lifted.py).

Certifies, for the Lagrangian of an alm_vcs.py / est_dual_vcs.py checkpoint at FIXED multipliers y,
    max over partitions X of   C0 + sum(lam) - sum(mu) + <C(y), X>   <=  UB * L,
    C(y) = A_w - d d^T/L + T(lam) + Nbr(mu) + Conn(om) + Pc(psi) + sum_levels S_pair(Sig) + sum_VC S_vc(sig)
(the operator of est_dual_vcs.py; psi pair cuts are included here, est_dual_vcs ignores them -- dropping a
multiplier is always valid, so --use_psi 0 reproduces est exactly).  Weak duality then gives Q* <= UB.

LIFTED SPACE ("a-units", every lifted coordinate has norm <= 1 on any feasible X):
  vertex vectors v_i (unit), and auxiliary a_p = s_p / vol_p for internal nodes p of binary partial-sum trees,
  s_p = sum of children s (leaf s = d_i v_i).  One tree per distinct clustering (pair levels and VC levels share a
  tree when the partitions are identical; a coarser clustering is built on top of the cluster nodes of a finer one
  when it is refined by it).  Cluster R is represented by (node_R, coef_R) with s_R = coef_R x_node: an aux node
  (coef = vol_R) or, for a singleton {j}, the vertex itself (coef = d_j).  The root of the coarsest clustering
  carries the penalty: d d^T/L term = ||s_root||^2/L = L ||a_root||^2.
  Links: s_p - sum_c s_c = 0, i.e. e_p = vol_p e_{a_p} - sum_c coef_c e_c (integer coefficients), stiffness
  mu_p = c/vol_p (same as the original).  For every partition (indeed every Gram X of unit vectors) the lifted Gram
  Xt is PSD, Xt_kk <= 1 for all k (= 1 on vertices) and e_p^T Xt e_p = 0.
  Terms: pair level  Sig_QR h_QR = Sig_QR (coef_Q coef_R <x_Q,x_R> - [Q=R] sum_{i in Q} d_i^2);
         VC entry    sig_iR g_iR = sig_iR d_i (coef_R <v_i, x_R> - [R = R(i)] d_i)   (2 lifted coordinates: v_i, node_R);
         vertex-vertex terms (A, triangles, mu, om, psi) as in C; the -[..] d_i^2 corrections go on the vertex diagonal.
CERTIFICATE:  Mt = Diag(g (+) 0) - Ct + sum_p mu_p e_p e_p^T + Diag(shift * w)   (w = s_v on vertices, sa on aux).
  If Mt_exact = R + E with R PSD then <C,X> <= sum(g) + sum(shift*w) + sum_kl |E_kl|   (|Xt_kl| <= 1).
RIGOR (no forming of R^T R, cost O(nnz(factor))):
  * assembly: Ms = fl(Mt) entrywise; |Ms - Mt_exact|_kl <= gamma(h_kl+4) * sum|terms_kl|  (h = number of summands,
    each term a product of <= 3 floats);  price_asm = sum over entries.
  * cholmod backend (supernodal LL^T, natural order after METIS ND):  Higham Thm 10.3: L L^T = Ms + dA,
    |dA| <= gamma(k+4) |L||L|^T  (k = max row/col count of L)  ->  price_round = gamma * || |L|^T 1 ||^2.
  * superlu backend (no pivoting, asserted): Higham Thm 9.3: L U = Ms + dA, |dA| <= gamma(k+4) |L||U|; with D = diag U > 0
    (asserted), U^T D^{-1} U is PSD exactly and LU - U^T D^{-1} U = F U, F = L - U^T D^{-1} (computed with a 4u
    margin) -> price = gamma (|L|^T 1).(|U| 1) + (|F|^T 1).(|U| 1).
  * all multipliers asserted >= 0; constants summed with math.fsum (correctly rounded) then rounded up; vector
    price sums of nonnegative terms inflated by (1 + 4 nnz u).
  UB = (C0 + sum lam - sum mu + sum g + sum shift*w + price_asm + price_round) / L, rounded up.
Usage (from tol5/):  DATA_DIR=... python brk/certvc/certify_lifted_vc.py TAG CKPT.npz [--initV est_saveW.npz] [--mu 1e4]
  --symbolic_only 1 : build the lifted pattern, ND order, report fill / flops / memory and exit (no numerics).
  --vc_drop t       : drop VC multipliers < t (always valid; re-polish g with --polish_rgd).
"""
import os, sys, time, json, math, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(HERE, "numba_cache"))
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
from numba import njit

U_ = 2.0**-53
def gam(k): k = np.asarray(k, dtype=np.float64); return k*U_/(1.0 - k*U_)
def up(x): return float(np.nextafter(float(x), np.inf))
def fsum_up(a): return up(math.fsum(np.asarray(a, dtype=np.float64).ravel().tolist())) if np.size(a) else 0.0
T0 = time.time()
def rss():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform == "darwin" else 2**10)
def log(*s): print(f"[{time.time()-T0:8.1f}s {rss():7.0f}MB]", *s, flush=True)


# ----------------------------------------------------------------------------- problem loading (mirrors est_dual_vcs)
def load_problem(tag, ckpt, D, use_psi=1, initV=""):
    if not D.endswith("/"): D += "/"
    g_ = json.load(open(D+tag+"/graph.json")); n = g_["n"]
    keys = np.load(D+tag+"/keys.npy"); u = keys//n; v = keys % n; m = len(keys); L = 2*m
    assert len(np.unique(keys)) == m and (u < v).all()
    d = np.bincount(np.r_[u, v], minlength=n).astype(float)
    W_ = np.load(D+tag+"/weights.npy").astype(np.float64) if os.path.exists(D+tag+"/weights.npy") else np.ones(m)
    C0 = 0.0
    if os.path.exists(D+tag+"/vol.npy"):
        meta = json.load(open(D+tag+"/meta.json")); d = np.load(D+tag+"/vol.npy").astype(np.float64)
        L = int(meta["L_orig"]); C0 = 2.0*meta["leaf_edges"]; assert int(d.sum()) == L
    assert (d == np.round(d)).all() and (d >= 0).all() and d.max() < 2**26
    z = np.load(ckpt)
    levels = []
    for li in range(int(z["nlev"])):
        assert int(z[f"pair{li}"]) == 1, "cluster-pair levels only"
        assert not (f"nested{li}" in z.files and int(z[f"nested{li}"])), "nested levels not supported"
        lab = np.unique(z[f"lab{li}"], return_inverse=True)[1].astype(np.int64); P = int(lab.max())+1
        sig = z[f"sig{li}"].astype(np.float64); assert sig.shape == (P, P) and (sig >= 0).all() and (sig == sig.T).all()
        levels.append(dict(lab=lab, P=P, sig=sig))
    TI = z["TI"].astype(np.int64).reshape(-1, 3); lam = z["lam"].astype(np.float64)
    mu = z["mu"].astype(np.float64) if "mu" in z.files and z["mu"].shape == (n,) else np.zeros(n)
    CI = z["CI"].astype(np.int64).reshape(-1, 3) if "CI" in z.files else np.zeros((0, 3), np.int64)
    om = z["om"].astype(np.float64) if "om" in z.files else np.zeros(0)
    PI = z["PI"].astype(np.int64).reshape(-1, 2) if ("PI" in z.files and use_psi) else np.zeros((0, 2), np.int64)
    psi = z["psi"].astype(np.float64) if ("psi" in z.files and use_psi) else np.zeros(0)
    vcs = []
    for mm in range(int(z["nvc"]) if "nvc" in z.files else 0):          # dense VC level -> sparse form
        lab = np.unique(z[f"vclab{mm}"], return_inverse=True)[1].astype(np.int64); P = int(lab.max())+1
        S = sp.csr_matrix(z[f"vcsig{mm}"].astype(np.float64)); assert S.shape == (n, P)
        vcs.append(dict(lab=lab, P=P, ptr=S.indptr.astype(np.int64), col=S.indices.astype(np.int64), sig=S.data.astype(np.float64)))
    for mm in range(int(z["nvcs"]) if "nvcs" in z.files else 0):
        lab = np.unique(z[f"vcslab{mm}"], return_inverse=True)[1].astype(np.int64); P = int(lab.max())+1
        vcs.append(dict(lab=lab, P=P, ptr=z[f"vcsptr{mm}"].astype(np.int64), col=z[f"vcscol{mm}"].astype(np.int64),
                        sig=z[f"vcssig{mm}"].astype(np.float64)))
    V = z["V"].astype(np.float64); z.close()
    if initV: V = np.load(initV)["V"].astype(np.float64)
    return finish_problem(dict(n=n, L=L, d=d, C0=C0, u=u, v=v, w=W_, levels=levels, TI=TI, lam=lam, mu=mu, CI=CI, om=om,
                               PI=PI, psi=psi, vcs=vcs, V=V, tag=tag, ckpt=ckpt))


def finish_problem(P):
    """validate, merge duplicate triangles (as est), drop zero multipliers."""
    n = P["n"]
    for x in ("lam", "mu", "om", "psi"): assert (P[x] >= 0).all() and np.isfinite(P[x]).all(), x
    for vc in P["vcs"]:
        assert (vc["sig"] >= 0).all() and vc["ptr"][-1] == len(vc["col"]) and len(vc["ptr"]) == n+1
        assert len(vc["col"]) == 0 or ((vc["col"] >= 0).all() and (vc["col"] < vc["P"]).all())
    TI, lam = P["TI"], P["lam"]
    if len(TI):
        TI = np.c_[np.minimum(TI[:, 0], TI[:, 2]), TI[:, 1], np.maximum(TI[:, 0], TI[:, 2])]
        kk = (TI[:, 0]*n+TI[:, 1])*n+TI[:, 2]; _, first, inv = np.unique(kk, return_index=True, return_inverse=True)
        if len(first) < len(TI): lam = np.bincount(inv, weights=lam); TI = TI[first]
        keep = lam > 0; TI = TI[keep]; lam = lam[keep]
    P["TI"], P["lam"] = TI, lam
    if len(P["om"]): k_ = P["om"] > 0; P["CI"], P["om"] = P["CI"][k_], P["om"][k_]
    if len(P["psi"]): k_ = P["psi"] > 0; P["PI"], P["psi"] = P["PI"][k_], P["psi"][k_]
    V = P["V"]; V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-300); P["V"] = np.ascontiguousarray(V)
    return P


def vertex_terms(P):
    """COO (rows, cols, vals) of the zero-diagonal vertex-vertex part of C: A_w + T + Nbr + Conn + Pc (both halves)."""
    u, v, w, TI, lam, mu, CI, om, PI, psi = (P[k] for k in ("u", "v", "w", "TI", "lam", "mu", "CI", "om", "PI", "psi"))
    R = [u, v]; Cc = [v, u]; X = [w, w]
    if mu.any(): t = 0.5*(mu[u]+mu[v])*w; R += [u, v]; Cc += [v, u]; X += [t, t]
    if len(lam):
        h = 0.5*lam; i, j, k = TI.T
        R += [i, j, j, k, i, k]; Cc += [j, i, k, j, k, i]; X += [-h, -h, -h, -h, h, h]
    if len(om):
        h = 0.5*om; i, j, s_ = CI.T
        R += [i, s_, i, j]; Cc += [s_, i, j, i]; X += [h, h, -h, -h]
    if len(psi):
        h = 0.5*psi; i, j = PI.T
        R += [i, j]; Cc += [j, i]; X += [h, h]
    rr = np.concatenate(R).astype(np.int64); cc = np.concatenate(Cc).astype(np.int64)
    assert (rr != cc).all(), "vertex-vertex terms must be off-diagonal"
    return rr, cc, np.concatenate(X).astype(np.float64)


# ----------------------------------------------------------------------------- operator C(y) (for g and polishing)
class Op:
    def __init__(self, P):
        n = P["n"]; self.P = P; d = P["d"]
        rr, cc, xx = vertex_terms(P); self.Cvv = sp.csr_matrix((xx, (rr, cc)), shape=(n, n))
        self.lv = []
        for Lv in P["levels"]:
            W = sp.csr_matrix((d, (Lv["lab"], np.arange(n))), shape=(Lv["P"], n))
            self.lv.append((W, Lv["sig"], np.diag(Lv["sig"])[Lv["lab"]]*d**2, Lv["lab"]))
        self.vc = []
        for vc in P["vcs"]:
            S = sp.csr_matrix((vc["sig"], vc["col"], vc["ptr"]), shape=(n, vc["P"]))
            W = sp.csr_matrix((d, (vc["lab"], np.arange(n))), shape=(vc["P"], n))
            own = np.asarray(S[np.arange(n), vc["lab"]]).ravel()
            self.vc.append((S, W, own*d**2, vc["lab"]))
    def __call__(self, X):
        P = self.P; d = P["d"]; L = P["L"]
        out = self.Cvv@X - np.outer(d, d@X)/L
        for W, sig, dc, lab in self.lv: out += d[:, None]*(sig@(W@X))[lab] - dc[:, None]*X
        for S, W, dc, lab in self.vc: out += 0.5*d[:, None]*(S@(W@X) + (S.T@(d[:, None]*X))[lab]) - dc[:, None]*X
        return out
    def dense(self):
        return self(np.eye(self.P["n"]))


def polish_rgd(op, V, iters, tlim):
    """preconditioned Riemannian ascent of <C, V V^T> on the oblique manifold (est_dual_vcs polish_rgd, condensed)."""
    L = op.P["L"]; const = op.P["C0"] + op.P["lam"].sum() - op.P["mu"].sum()
    def value(W): CW = op(W); g = np.einsum("ij,ij->i", CW, W); return (const+g.sum())/L, g, CW
    tp = time.time(); f, g, CW = value(V); Cref = f; Q = 1.0; alpha = 2.0
    nr = np.maximum(np.linalg.norm(CW, axis=1), 1e-12); Rr = CW - g[:, None]*V; R = Rr/nr[:, None]
    for it in range(iters):
        if time.time()-tp > tlim: break
        gr = float(np.einsum("ij,ij->", Rr, R))/L
        for bt in range(12):
            Vn = V + alpha*R; Vn /= np.linalg.norm(Vn, axis=1, keepdims=True); fn, gn, CWn = value(Vn)
            if fn >= Cref + 1e-4*alpha*gr: break
            alpha *= 0.5
        S_ = Vn - V; nrn = np.maximum(np.linalg.norm(CWn, axis=1), 1e-12); Rrn = CWn - gn[:, None]*Vn; Rn = Rrn/nrn[:, None]
        Y_ = Rn - R; sy = float(np.einsum("ij,ij->", S_, Y_*nrn[:, None])); ss = float(np.einsum("ij,ij->", S_, S_*nrn[:, None]))
        yy = float(np.einsum("ij,ij->", Y_, Y_*nrn[:, None]))
        alpha = ((ss/(-sy)) if it % 2 else (-sy/yy)) if sy < 0 else min(2*alpha, 50.0); alpha = float(np.clip(alpha, 0.05, 50.0))
        V, f, g, CW, R, Rr = Vn, fn, gn, CWn, Rn, Rrn; Qn = 0.85*Q+1; Cref = (0.85*Q*Cref+f)/Qn; Q = Qn
    return V, f


# ----------------------------------------------------------------------------- symbolic analysis
@njit(cache=True)
def etree_colcounts(N, ip, ix, cap):
    """ip/ix: CSR of the strictly-lower pattern (row k lists cols i < k). Returns parent, column counts of L (incl.
    diagonal), work, ok (row-subtree traversal, O(nnz(L)))."""
    parent = -np.ones(N, np.int64); anc = -np.ones(N, np.int64)
    for k in range(N):
        for p in range(ip[k], ip[k+1]):
            i = ix[p]
            while i != -1 and i < k:
                nx = anc[i]; anc[i] = k
                if nx == -1: parent[i] = k
                i = nx
    cnt = np.ones(N, np.int64); mark = -np.ones(N, np.int64); work = 0
    for k in range(N):
        mark[k] = k
        for p in range(ip[k], ip[k+1]):
            i = ix[p]
            while mark[i] != k:
                cnt[i] += 1; mark[i] = k; i = parent[i]; work += 1
        if work > cap: return parent, cnt, work, False
    return parent, cnt, work, True


@njit(cache=True)
def in_pattern(N, ip, ix, parent, qp, qa, out):
    """out[e] = 1 iff L[b, qa[e]] != 0 for queries grouped by row b (qp CSR over b, qa[e] < b): row-subtree marking."""
    mark = -np.ones(N, np.int64)
    for k in range(N):
        mark[k] = k
        for p in range(ip[k], ip[k+1]):
            i = ix[p]
            while mark[i] != k:
                mark[i] = k; i = parent[i]
        for e in range(qp[k], qp[k+1]):
            out[e] = 1 if mark[qa[e]] == k else 0


def nd_order(indptr, indices):
    import pymetis
    N = len(indptr)-1
    if N <= 2: return np.arange(N)
    return np.asarray(pymetis.nested_dissection(pymetis.CSRAdjacency(indptr, indices))[0], dtype=np.int64)


# ----------------------------------------------------------------------------- lifted construction
def build_lifted(P, gd, opts, want_H=False):
    """returns dict with COO term arrays (rows, cols, vals) of Mt WITHOUT the shift, N, weights w, H (optional)."""
    n, d, L = P["n"], P["d"], P["L"]; c_mu = opts["mu"]
    # vertex locality key: ND position on the vertex graph
    Apat = sp.csr_matrix((np.ones(2*len(P["u"]), np.int8), (np.r_[P["u"], P["v"]], np.r_[P["v"], P["u"]])), shape=(n, n))
    perm_v = nd_order(Apat.indptr.astype(np.int64), Apat.indices.astype(np.int64)); key = np.empty(n, np.int64); key[perm_v] = np.arange(n); del Apat
    # distinct clusterings, finest first
    cls = []
    def find_or_add(lab):
        for ci, c in enumerate(cls):
            if c["P"] == lab.max()+1 and np.array_equal(c["lab"], lab): return ci
        cls.append(dict(lab=lab, P=int(lab.max())+1)); return len(cls)-1
    lv_cl = [find_or_add(Lv["lab"]) for Lv in P["levels"]]
    vc_cl = [find_or_add(vc["lab"]) for vc in P["vcs"]]
    order = sorted(range(len(cls)), key=lambda c: -cls[c]["P"])
    nxt = [n]; links = []          # (p, volp, [(child, coef)...])
    Hrows = []                      # for want_H: members of every aux node
    def build_tree(items):          # items: list of (node, coef, key, members-or-None); returns root item
        while len(items) > 1:
            new = []
            for t in range(0, len(items)-1, 2):
                a_, b_ = items[t], items[t+1]; vp = a_[1]+b_[1]
                if a_[1] == 0: new.append(b_); continue
                if b_[1] == 0: new.append(a_); continue
                p = nxt[0]; nxt[0] += 1; links.append((p, vp, [(a_[0], a_[1]), (b_[0], b_[1])]))
                mem = (a_[3] + b_[3]) if want_H else None
                if want_H: Hrows.append((p, mem))
                new.append((p, vp, min(a_[2], b_[2]), mem))
            if len(items) % 2: new.append(items[-1])
            items = new
        return items[0]
    built = {}
    for ci in order:
        c = cls[ci]; lab = c["lab"]; Pc = c["P"]
        fine = None
        for cj in built:              # finer clustering refining this one
            f = cls[cj]
            if f["P"] > Pc and len(np.unique(f["lab"]*Pc + lab)) == f["P"]: fine = cj if fine is None or cls[cj]["P"] < cls[fine]["P"] else fine
        groups = [[] for _ in range(Pc)]
        if fine is None:
            for i in np.argsort(key, kind="stable"): groups[lab[i]].append((int(i), float(d[i]), int(key[i]), [int(i)] if want_H else None))
        else:
            fn = built[fine]; flab = cls[fine]["lab"]; fmap = np.zeros(cls[fine]["P"], np.int64); fmap[flab] = lab
            items = sorted(range(cls[fine]["P"]), key=lambda R: fn[R][2])
            for R in items: groups[fmap[R]].append(fn[R])
        nodes = []
        for Q in range(Pc):
            it = build_tree(groups[Q]) if groups[Q] else (-1, 0.0, 0, [])
            if it[1] == 0: it = (-1, 0.0, it[2], it[3])
            nodes.append(it)
        built[ci] = nodes
    # root over the coarsest clustering (or all vertices)
    if cls:
        top = built[order[-1]]; r = build_tree([x for x in top if x[0] >= 0])
    else:
        r = build_tree([(int(i), float(d[i]), int(key[i]), [int(i)] if want_H else None) for i in np.argsort(key)])
    root, rootvol = r[0], r[1]; assert rootvol == L, (rootvol, L)
    N = nxt[0]
    # ---- COO terms
    TR, TC, TV = [], [], []
    def put(rr, cc, vv): TR.append(np.asarray(rr, np.int64)); TC.append(np.asarray(cc, np.int64)); TV.append(np.asarray(vv, np.float64))
    # links (vectorised): e = [(p, volp), (c, -coef_c)...], mu = c_mu/volp
    if links:
        lp = np.array([x[0] for x in links]); lv = np.array([x[1] for x in links])
        ch = np.array([[x[2][0][0], x[2][1][0]] for x in links]); cf = np.array([[x[2][0][1], x[2][1][1]] for x in links])
        mu_ = c_mu/lv
        E_idx = np.c_[lp, ch]; E_w = np.c_[lv, -cf]
        for a_ in range(3):
            for b_ in range(3): put(E_idx[:, a_], E_idx[:, b_], mu_*E_w[:, a_]*E_w[:, b_])
    put([root], [root], [rootvol*(rootvol/L)])
    for li, Lv in enumerate(P["levels"]):
        nodes = built[lv_cl[li]]; nd_ = np.array([x[0] for x in nodes]); cf_ = np.array([x[1] for x in nodes]); isaux = nd_ >= n
        Qi, Ri = np.nonzero(Lv["sig"]); ok = (nd_[Qi] >= 0) & (nd_[Ri] >= 0) & ~((Qi == Ri) & ~isaux[Qi])
        Qi, Ri = Qi[ok], Ri[ok]
        put(nd_[Qi], nd_[Ri], -Lv["sig"][Qi, Ri]*cf_[Qi]*cf_[Ri])
        sd = np.diag(Lv["sig"]); put(np.arange(n), np.arange(n), np.where(isaux[Lv["lab"]], sd[Lv["lab"]], 0.0)*d**2)
    nvc_used = 0; nvc_drop = 0
    for mi, vc in enumerate(P["vcs"]):
        nodes = built[vc_cl[mi]]; nd_ = np.array([x[0] for x in nodes]); cf_ = np.array([x[1] for x in nodes]); isaux = nd_ >= n
        rows = np.repeat(np.arange(n), np.diff(vc["ptr"])); col = vc["col"]; s_ = vc["sig"]
        keep = (s_ > 0) & (s_ >= opts.get("vc_drop", 0.0)) & (nd_[col] >= 0) & (d[rows] > 0)
        nvc_drop += int(((s_ > 0) & ~keep).sum())
        own = col == vc["lab"][rows]
        keep &= ~(own & ~isaux[col])                   # own singleton: g_iR = 0 identically
        rows, col, s_, own = rows[keep], col[keep], s_[keep], own[keep]; nvc_used += len(s_)
        t = -0.5*s_*d[rows]*cf_[col]
        put(rows, nd_[col], t); put(nd_[col], rows, t)
        put(rows[own], rows[own], s_[own]*d[rows[own]]**2)     # one own entry per vertex and level
    rr, cc, xx = vertex_terms(P); put(rr, cc, -xx); del rr, cc, xx
    put(np.arange(n), np.arange(n), gd)
    lv_nodes = [(np.array([x[0] for x in built[lv_cl[li]]], np.int64), np.array([x[1] for x in built[lv_cl[li]]], np.float64)) for li in range(len(P["levels"]))]
    vc_nodes = [(np.array([x[0] for x in built[vc_cl[mi]]], np.int64), np.array([x[1] for x in built[vc_cl[mi]]], np.float64)) for mi in range(len(P["vcs"]))]
    out = dict(lv_nodes=lv_nodes, vc_nodes=vc_nodes, rootvol=rootvol, links=(lp, lv, ch, cf) if links else None, R=TR, C=TC, V=TV, N=N, n=n, root=root, naux=N-n, nlinks=len(links), nvc_used=nvc_used, nvc_drop=nvc_drop,
               nclust=[cls[c]["P"] for c in order])
    if want_H:
        H = sp.lil_matrix((N, n)); H[np.arange(n), np.arange(n)] = 1.0
        for p, mem in Hrows:
            vp = d[mem].sum()
            for i in mem: H[p, i] = d[i]/vp
        out["H"] = H.tocsr()
    return out


@njit(cache=True)
def _lift(Xl, lp, lv, ch, cf):
    for t in range(len(lp)):
        p = lp[t]
        for q in range(Xl.shape[1]): Xl[p, q] = (cf[t, 0]*Xl[ch[t, 0], q] + cf[t, 1]*Xl[ch[t, 1], q])/lv[t]


def lift_V(lf, V):
    """lifted BM point: vertex rows V, aux rows a_p = s_p/vol_p (links in creation order: children first)."""
    Xl = np.zeros((lf["N"], V.shape[1])); Xl[:V.shape[0]] = V
    if lf["links"] is not None:
        lp, lv, ch, cf = lf["links"]; _lift(Xl, lp.astype(np.int64), lv.astype(np.float64), ch.astype(np.int64), cf.astype(np.float64))
    return Xl


def assemble(lf, shift_diag):
    """Ms = fl(sum of terms) (CSR, symmetric), plus the assembly price sum_kl gamma(h+4) sum|terms|."""
    N = lf["N"]
    R = np.concatenate(lf["R"] + [np.arange(N)]); C = np.concatenate(lf["C"] + [np.arange(N)])
    V = np.concatenate(lf["V"] + [shift_diag])
    if N < 2**31-1: R = R.astype(np.int32); C = C.astype(np.int32)
    M = sp.csr_matrix((V, (R, C)), shape=(N, N)); M.sum_duplicates(); M = (0.5*(M + M.T)).tocsr(); M.sort_indices()   # exact symmetry
    Ma = sp.csr_matrix((np.abs(V), (R, C)), shape=(N, N)); Ma.sum_duplicates(); Ma.sort_indices()
    Mc = sp.csr_matrix((np.ones(len(V), np.float32), (R, C)), shape=(N, N)); Mc.sum_duplicates(); Mc.sort_indices()
    del R, C, V
    assert np.array_equal(Mc.indptr, Ma.indptr) and np.array_equal(Mc.indices, Ma.indices)
    hmax = float(Mc.data.max())
    # entry = fl((a+b)/2), a, b sums of h computed terms (each <= 3 roundings): error <= gamma(h+5) sum|terms| (x1.0001 slack)
    price = float(np.dot(gam(Mc.data.astype(np.float64)+5.0), Ma.data))*(1+4*len(Ma.data)*U_)*1.0001
    return M, up(price), hmax


# ----------------------------------------------------------------------------- factorization with rigorous residual price
def factor_price(Ms, perm, backend):
    """returns (ok, price_round, info).  Ms is factored as Ms[perm][:,perm] in natural order."""
    N = Ms.shape[0]; Mp = Ms[perm][:, perm].tocsc(); Mp.sort_indices(); info = {}
    t1 = time.time()
    if backend == "cholcert":            # memory-lean C helper (cholcert.c): supernodal factor never copied
        import ctypes
        lib = ctypes.CDLL(os.path.join(HERE, "libcholcert.so"))
        Lw = sp.tril(Mp).tocsc(); del Mp; Lw.sort_indices()
        Ap = np.ascontiguousarray(Lw.indptr, np.int64); Ai = np.ascontiguousarray(Lw.indices, np.int64)
        Ax = np.ascontiguousarray(Lw.data, np.float64); del Lw
        out = np.zeros(10); P_ = ctypes.POINTER(ctypes.c_long); D_ = ctypes.POINTER(ctypes.c_double)
        rc = lib.cholcert(ctypes.c_long(N), Ap.ctypes.data_as(P_), Ai.ctypes.data_as(P_), Ax.ctypes.data_as(D_), out.ctypes.data_as(D_))
        info.update(factor_s=time.time()-t1, rc=int(rc), nnzL=float(out[1]), k=int(out[2]), min_diag=float(out[4]),
                    max_diag=float(out[5]), flops=float(out[6]), status=int(out[7]), minor=int(out[8]))
        if rc != 0 or out[0] != 1: return False, None, info
        price = min(float(gam(out[2]+4))*float(out[3]), float(out[9]))*(1+4*(out[1]+N)*U_)*1.0001
        info.update(p_global=float(gam(out[2]+4))*float(out[3]), p_rowweighted=float(out[9]))
        return True, up(price), info
    if backend == "cholmod":
        from sksparse.cholmod import cholesky, CholmodNotPositiveDefiniteError
        try:
            f = cholesky(Mp, ordering_method="natural", mode="supernodal")
        except CholmodNotPositiveDefiniteError as e:
            return False, None, dict(err=str(e))
        assert (f.P() == np.arange(N)).all(), "cholmod permuted"
        Lf = f.L().tocsc(); del f
        info["factor_s"] = time.time()-t1; info["nnzL"] = int(Lf.nnz)
        dg = Lf.diagonal()
        if not (dg > 0).all() or not np.isfinite(Lf.data).all(): return False, None, dict(err="nonpositive/nonfinite diag")
        k = max(int(np.diff(Lf.indptr).max()), int(np.bincount(Lf.indices, minlength=N).max()))
        absL = abs(Lf); s1 = np.asarray(absL.sum(axis=0)).ravel()          # |L|^T 1
        rc = np.bincount(Lf.indices, minlength=N).astype(np.float64)           # row counts of L
        wr = np.sqrt((rc + 4.0)*U_/(1.0 - (k + 4.0)*U_))                       # gamma(min(a,b)+4) <= w_a w_b
        s2 = np.asarray((sp.diags(wr)@absL).sum(axis=0)).ravel()
        price = min(float(gam(k+4))*float(np.dot(s1, s1)), float(np.dot(s2, s2)))*(1+4*(Lf.nnz+N)*U_)*1.0001
        info.update(k=k, min_diag=float(dg.min()))
        return True, up(price), info
    # superlu
    lu = sla.splu(Mp, permc_spec="NATURAL", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
    if not ((lu.perm_r == np.arange(N)).all() and (lu.perm_c == np.arange(N)).all()): return False, None, dict(err="pivoting occurred")
    Lf = lu.L.tocsc(); Uf = lu.U.tocsr(); del lu
    info["factor_s"] = time.time()-t1; info["nnzL"] = int(Lf.nnz); info["nnzU"] = int(Uf.nnz)
    Dg = Uf.diagonal()
    if not (Dg > 0).all() or not (np.isfinite(Lf.data).all() and np.isfinite(Uf.data).all()): return False, None, dict(err="nonpositive pivot")
    k = max(int(np.diff(Lf.indptr).max()), int(np.diff(Uf.indptr).max()),
            int(np.bincount(Lf.indices, minlength=N).max()), int(np.bincount(Uf.indices, minlength=N).max()))
    aL = abs(Lf); aU = abs(Uf)
    lsum = np.asarray(aL.sum(axis=0)).ravel(); usum = np.asarray(aU.sum(axis=1)).ravel()     # |L|^T 1, |U| 1
    rcL = np.diff(Lf.tocsr().indptr).astype(np.float64); ccU = np.diff(Uf.tocsc().indptr).astype(np.float64)
    wL = np.sqrt((rcL + 4.0)*U_/(1.0 - (k + 4.0)*U_)); wU = np.sqrt((ccU + 4.0)*U_/(1.0 - (k + 4.0)*U_))
    p1 = min(float(gam(k+4))*float(np.dot(lsum, usum)),
             float(np.dot(np.asarray((sp.diags(wL)@aL).sum(axis=0)).ravel(), np.asarray((aU@sp.diags(wU)).sum(axis=1)).ravel())))
    Q = (sp.diags(1.0/Dg)@Uf).T.tocsc()                    # U^T D^{-1} (computed)
    F = (Lf - Q).tocsc(); Fb = abs(F) + 4*U_*(aL + abs(Q))  # |F_exact| <= |fl F| + 4u(|L| + |Q|)
    fsum_ = np.asarray(Fb.sum(axis=0)).ravel()
    p2 = float(np.dot(fsum_, usum))
    nn = Lf.nnz + Uf.nnz + Fb.nnz + N
    info.update(k=k, min_pivot=float(Dg.min()), p_backward=p1, p_F=p2)
    return True, up((p1+p2)*(1+4*nn*U_)*1.0001), info


def lmin_pencil(M, w, X0=None, maxiter=500, tol=1e-6, seed=0):
    """smallest eigenvalue of (M, Diag(w)) -- an ESTIMATE, only used to choose the shift (validity never depends on it)."""
    s = 1/np.sqrt(w); Mh = (sp.diags(s)@M@sp.diags(s)).tocsr(); N = M.shape[0]
    if N <= 12000:
        return float(np.linalg.eigvalsh(Mh.toarray())[0])
    import warnings
    rng = np.random.default_rng(seed)
    if X0 is not None:
        Uq, Sq, _ = np.linalg.svd(X0*np.sqrt(w)[:, None], full_matrices=False); X0 = Uq[:, Sq > 1e-6*Sq[0]]
    b = 24 if X0 is None else X0.shape[1] + 16
    X = rng.standard_normal((N, b))
    if X0 is not None: X[:, :X0.shape[1]] = X0
    dg = Mh.diagonal(); Pc = sla.LinearOperator((N, N), matmat=lambda Y: Y/np.maximum(np.abs(dg), 1e-3*np.abs(dg).mean())[:, None],
                                               matvec=lambda y: y.ravel()/np.maximum(np.abs(dg), 1e-3*np.abs(dg).mean()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev, Xe = sla.lobpcg(Mh, X, M=Pc, largest=False, maxiter=maxiter, tol=tol)
    i = int(np.argmin(ev)); x = Xe[:, i]; th = float(x@(Mh@x)/(x@x))
    return th            # Ritz value (>= true lmin): a too-small shift is caught by the factorization and increased


def tri_fill_filter(P, opts):
    """FALLBACK (valid: dropping multipliers): keep only triangles whose 3 vertex pairs lie in the symbolic pattern of
    the factor of the triangle-free lifted matrix (same ordering) -> the triangles add no fill.  g must be re-polished."""
    n = P["n"]; Pn = dict(P); Pn["TI"] = P["TI"][:0]; Pn["lam"] = P["lam"][:0]
    lf = build_lifted(Pn, np.ones(n), opts); N = lf["N"]; M0, _, _ = assemble(lf, np.zeros(N)); del lf
    Gp = M0.copy(); Gp.setdiag(0); Gp.eliminate_zeros(); Gp.data[:] = 1; del M0
    deg = np.diff(Gp.indptr); thr = float(opts.get("dense_deg") or 1000); dense = np.flatnonzero(deg > thr); sp_ = np.flatnonzero(deg <= thr)
    Gs = Gp[sp_][:, sp_].tocsr(); ps = nd_order(Gs.indptr.astype(np.int64), Gs.indices.astype(np.int64)); del Gs
    perm = np.r_[sp_[ps], dense[np.argsort(deg[dense], kind="stable")]]; pos = np.empty(N, np.int64); pos[perm] = np.arange(N)
    Tl = sp.tril(Gp[perm][:, perm], -1).tocsr(); del Gp
    parent, cnt, work, ok = etree_colcounts(N, Tl.indptr.astype(np.int64), Tl.indices.astype(np.int64), int(1e13))
    TI = P["TI"]; prs = np.r_[TI[:, [0, 1]], TI[:, [1, 2]], TI[:, [0, 2]]]
    a_ = pos[prs[:, 0]]; b_ = pos[prs[:, 1]]; lo = np.minimum(a_, b_); hi = np.maximum(a_, b_)
    o = np.argsort(hi, kind="stable"); qp = np.r_[0, np.cumsum(np.bincount(hi, minlength=N))].astype(np.int64)
    res = np.zeros(len(o), np.int64); in_pattern(N, Tl.indptr.astype(np.int64), Tl.indices.astype(np.int64), parent, qp, lo[o].astype(np.int64), res)
    inp = np.empty(len(o), bool); inp[o] = res.astype(bool); nt = len(TI)
    keep = inp[:nt] & inp[nt:2*nt] & inp[2*nt:]
    log(f"tri_fill_filter: triangle-free fill {cnt.sum():.3e}; keep {keep.sum()}/{nt} triangles, lam mass {P['lam'][keep].sum()/P['lam'].sum():.4f}")
    Q = dict(P); Q["TI"] = TI[keep]; Q["lam"] = P["lam"][keep]; return Q


# ----------------------------------------------------------------------------- main certify routine
def certify(P, opts):
    n, d, L = P["n"], P["d"], P["L"]; t0 = time.time()
    if opts.get("tri_fill_filter") and len(P["lam"]):
        P = tri_fill_filter(P, opts); opts = dict(opts); opts["order_notri"] = 1     # same ordering -> no triangle fill
    op = Op(P); V = P["V"]
    if opts.get("polish_rgd", 0) > 0:
        V, f = polish_rgd(op, V, opts["polish_rgd"], opts.get("polish_time", 1e9)); log(f"rgd polished dv {f:.6f}")
    gd = np.einsum("ij,ij->i", op(V), V); del op
    const = math.fsum([P["C0"]] + P["lam"].tolist() + (-P["mu"]).tolist())
    dv = (const + math.fsum(gd.tolist()))/L
    log(f"dv (L-normalised value of the BM point, lower end) {dv:.6f}")
    lf = build_lifted(P, gd, opts, want_H=opts.get("want_H", False))
    N = lf["N"]; log(f"lifted N {N} (aux {lf['naux']}, links {lf['nlinks']}, clusterings {lf['nclust']}, VC entries used {lf['nvc_used']}, dropped {lf['nvc_drop']})")
    sv = np.maximum(d, 1.0) if opts.get("sv", "vol") == "vol" else np.ones(n)
    w = np.r_[sv, np.full(N-n, float(opts.get("sa", 1.0)))]
    M0, _, _ = assemble(lf, np.zeros(N))
    res = dict(n=n, N=N, nnzM=int(M0.nnz), dv=dv, mu=opts["mu"], nvc_used=lf["nvc_used"], nvc_drop=lf["nvc_drop"])
    # ND ordering of the lifted pattern
    if opts.get("order_notri") and len(P["lam"]):   # ordering computed WITHOUT the triangle pairs (any order is valid)
        Pn = dict(P); Pn["TI"] = P["TI"][:0]; Pn["lam"] = P["lam"][:0]
        Gp, _, _ = assemble(build_lifted(Pn, gd, opts), np.zeros(N)); del Pn
    else: Gp = M0.copy()
    Gp.setdiag(0); Gp.eliminate_zeros(); Gp.data[:] = 1
    deg = np.diff(Gp.indptr); thr = float(opts.get("dense_deg") or 1000)
    dense = np.flatnonzero(deg > thr); sparse_ = np.flatnonzero(deg <= thr)
    if len(dense):     # dense rows (VC/pair cluster hubs) ordered LAST; METIS ND on the rest (classic dense-row rule)
        Gs = Gp[sparse_][:, sparse_].tocsr()
        ps = nd_order(Gs.indptr.astype(np.int64), Gs.indices.astype(np.int64)); del Gs
        perm = np.r_[sparse_[ps], dense[np.argsort(deg[dense], kind="stable")]]
    else:
        perm = nd_order(Gp.indptr.astype(np.int64), Gp.indices.astype(np.int64))
    del Gp; res["ndense"] = int(len(dense)); log(f"ordering: {len(dense)} dense nodes (deg > {thr:g}) last, ND on the rest")
    Tl = sp.tril(M0[perm][:, perm], -1).tocsr()
    _, cnt, work, ok = etree_colcounts(N, Tl.indptr.astype(np.int64), Tl.indices.astype(np.int64), int(opts.get("max_fill", 1e12))); del Tl
    fill = int(cnt.sum()); flops = float((cnt.astype(np.float64)**2).sum())
    res.update(fill=fill, flops=flops, maxcol=int(cnt.max())); log(f"symbolic nnz(L) {fill:.3e}  flops ~{flops:.3e}  max col {cnt.max()}  ok {ok}")
    if opts.get("symbolic_only"):
        if opts.get("backend") == "cholcert":       # exact CHOLMOD supernodal storage (rectangle padding included)
            import ctypes
            lib = ctypes.CDLL(os.path.join(HERE, "libcholcert.so"))
            Lw = sp.tril(M0[perm][:, perm]).tocsc(); Lw.sort_indices()
            Ap = np.ascontiguousarray(Lw.indptr, np.int64); Ai = np.ascontiguousarray(Lw.indices, np.int64); del Lw
            o = np.zeros(6); P_ = ctypes.POINTER(ctypes.c_long); D_ = ctypes.POINTER(ctypes.c_double)
            lib.cholsym(ctypes.c_long(N), Ap.ctypes.data_as(P_), Ai.ctypes.data_as(P_), o.ctypes.data_as(D_))
            res.update(sn_xsize=float(o[0]), sn_GB=float(o[0])*8/1e9, sn_ssize=float(o[1]), nsuper=float(o[2]), sn_flops=float(o[3]), sn_lnz=float(o[4]))
            log(f"CHOLMOD supernodal: L->x {o[0]:.3e} doubles = {o[0]*8/1e9:.1f} GB, nsuper {o[2]:.0f}, flops {o[3]:.3e}, lnz {o[4]:.3e}")
        return res
    assert ok, "fill exceeds cap"
    lmin = None
    if opts.get("shift") is not None: shift = float(opts["shift"])
    elif opts.get("shift_est"):                     # est_dual_vcs json (s = vol pencil): lifted lmin ~>= dense lmin
        e = json.load(open(opts["shift_est"])); shift = max(0.0, -float(e["s_vol"]["lmin_est"]))*float(opts.get("margin") or 1.05) + 1e-9
    else:
        lmin = lmin_pencil(M0, w, X0=lift_V(lf, V)); shift = max(0.0, -lmin)*float(opts.get("margin") or 1.005) + 1e-9
        log(f"lmin(pencil, lifted) {lmin:.4e}")
    del M0
    backend = opts.get("backend", "auto")
    if backend == "auto":
        try: import sksparse.cholmod; backend = "cholmod"
        except Exception: backend = "superlu"
    if opts.get("dump_only"):                      # save the certificate at this shift WITHOUT factorizing (PSD is checked by verify_cert.py)
        Ms, p_asm, hmax = assemble(lf, shift*w); del Ms; sdiag = shift*w
        res.update(UB=float("nan"), shift=shift, price_asm=p_asm/L, dump_only=1, seconds=time.time()-t0, rss_MB=rss())
        log(f"dump_only at shift {shift:.9e} (no factorization)")
        dump_cert(opts["dump_cert"], P, opts, lf, gd, shift, w, sdiag, res); return res
    best = None; hi = None; lo = float(opts.get("bisect_lo") or 0.0)*shift; steps = 0
    while True:
        Ms, p_asm, hmax = assemble(lf, shift*w)
        ok, p_round, info = factor_price(Ms, perm, backend); del Ms
        log(f"shift {shift:.6e}: factor ok {ok} {info}")
        if ok:
            best = (shift, p_asm, hmax, p_round, info); hi = shift
        else:
            lo = shift
        steps += 1
        if hi is None:
            assert steps < 40, "could not factor"
            shift = shift*1.25 + 1e-8; continue
        if steps > int(opts.get("bisect", 0)) or hi - lo <= 2e-3*hi: break
        shift = 0.5*(lo + hi)
    shift, p_asm, hmax, p_round, info = best; sdiag = shift*w
    p_shift = fsum_up(sdiag)
    tot = math.fsum([const, math.fsum(gd.tolist()), p_shift, p_asm, p_round])
    tot = up(tot)
    ub = up(tot/L)
    res.update(UB=ub, shift=shift, lmin=lmin, price_shift=p_shift/L, price_asm=p_asm/L, price_round=p_round/L, hmax=hmax,
               backend=backend, factor=info, seconds=time.time()-t0, rss_MB=rss())
    log(f"CERTIFIED UB {ub:.6f}  (dv {dv:.6f} + shift {p_shift/L:.2e} + asm {p_asm/L:.2e} + round {p_round/L:.2e})")
    if opts.get("dump_cert"): dump_cert(opts["dump_cert"], P, opts, lf, gd, shift, w, sdiag, res)
    return res


def dump_cert(fn, P, opts, lf, gd, shift, w, sdiag, res):
    """save EXACTLY what was certified (after tri filter / merging / vc_drop) for brk/verify/verify_cert.py."""
    n = P["n"]; D = dict(n=n, N=lf["N"], L=P["L"], C0=P["C0"], d=P["d"], g=gd, shift=shift, w=w, sdiag=sdiag, mu_c=float(opts["mu"]),
                        root=lf["root"], rootvol=lf["rootvol"], TI=P["TI"], lam=P["lam"], mu=P["mu"], CI=P["CI"], om=P["om"], PI=P["PI"], psi=P["psi"],
                        nlev=len(P["levels"]), nvc=len(P["vcs"]), cert_UB=res["UB"], cert_json=json.dumps(res, default=float))
    if lf["links"] is not None:
        lp, lv, ch, cf = lf["links"]; D.update(link_p=lp, link_vol=lv, link_ch=ch, link_cf=cf)
    else:
        D.update(link_p=np.zeros(0, np.int64), link_vol=np.zeros(0), link_ch=np.zeros((0, 2), np.int64), link_cf=np.zeros((0, 2)))
    for li, Lv in enumerate(P["levels"]):
        D[f"lev{li}_lab"] = Lv["lab"]; D[f"lev{li}_sig"] = Lv["sig"]; D[f"lev{li}_node"], D[f"lev{li}_coef"] = lf["lv_nodes"][li]
    t = float(opts.get("vc_drop", 0.0) or 0.0)
    for mi, vc in enumerate(P["vcs"]):
        s_ = vc["sig"].copy(); s_[s_ < t] = 0.0          # entries dropped by --vc_drop are not part of the certificate
        D[f"vc{mi}_lab"] = vc["lab"]; D[f"vc{mi}_ptr"] = vc["ptr"]; D[f"vc{mi}_col"] = vc["col"]; D[f"vc{mi}_sig"] = s_
        D[f"vc{mi}_node"], D[f"vc{mi}_coef"] = lf["vc_nodes"][mi]
    np.savez_compressed(fn, **D); log(f"certificate dumped to {fn}")


def main():
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("tag"); pa.add_argument("ckpt"); pa.add_argument("--initV", default="")
    pa.add_argument("--polish_rgd", type=int, default=0); pa.add_argument("--polish_time", type=float, default=1e9)
    pa.add_argument("--mu", type=float, default=1e4, help="link stiffness c: mu_p = c/vol_p")
    pa.add_argument("--shift", type=float, default=None); pa.add_argument("--shift_est", default="", help="est_dual_vcs json: shift = margin*|lmin_est(s=vol)|")
    pa.add_argument("--margin", type=float, default=None); pa.add_argument("--bisect", type=int, default=0, help="extra factorizations to bisect the shift down");
    pa.add_argument("--bisect_lo", type=float, default=0.0, help="bisection lower end as a fraction of the starting shift"); pa.add_argument("--sv", default="vol"); pa.add_argument("--sa", type=float, default=0.01)
    pa.add_argument("--backend", default="auto"); pa.add_argument("--max_fill", type=float, default=2e10)
    pa.add_argument("--symbolic_only", type=int, default=0); pa.add_argument("--dense_deg", type=float, default=1000); pa.add_argument("--order_notri", type=int, default=0); pa.add_argument("--tri_fill_filter", type=int, default=0); pa.add_argument("--vc_drop", type=float, default=0.0)
    pa.add_argument("--no_vc", type=int, default=0); pa.add_argument("--no_tri", type=int, default=0); pa.add_argument("--no_pair", type=int, default=0); pa.add_argument("--use_psi", type=int, default=1)
    pa.add_argument("--dump_only", type=int, default=0); pa.add_argument("--out", default=""); pa.add_argument("--dump_cert", default="", help="save the exact certificate (npz) for verify_cert.py")
    a = pa.parse_args()
    D = os.environ.get("DATA_DIR", "/Users/ashrafkhalil/Desktop/MIE 498 RT/output/decomp_audit_2026-09-28/data/")
    P = load_problem(a.tag, a.ckpt, D, use_psi=a.use_psi, initV=a.initV)
    if a.no_vc: P["vcs"] = []
    if a.no_tri: P["TI"] = P["TI"][:0]; P["lam"] = P["lam"][:0]          # dropping multipliers is always valid
    if a.no_pair: P["levels"] = []
    log(f"tag {a.tag} n {P['n']} L {P['L']} levels {[Lv['P'] for Lv in P['levels']]} VC {[(vc['P'], len(vc['sig'])) for vc in P['vcs']]} "
        f"ntri {len(P['lam'])} mu {bool(P['mu'].any())} nconn {len(P['om'])} npsi {len(P['psi'])} C0/L {P['C0']/P['L']:.6f}")
    if a.symbolic_only: P["V"] = P["V"][:, :1]
    res = certify(P, vars(a)); res.update(tag=a.tag, ckpt=a.ckpt, initV=a.initV, args=vars(a))
    print(json.dumps(res, indent=1, default=float))
    out = a.out or a.ckpt.replace(".npz", "_CERTVC.json")
    json.dump(res, open(out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
