#!/usr/bin/env python3
"""INDEPENDENT verifier of a modularity upper-bound certificate (brk/verify, Oct 2 2026).

It imports NOTHING from the certifier (certify_lifted_vc*.py, est_dual_vcs.py, alm_*.py).  Inputs:
  INSTANCE  reduced instance directory (graph.json, keys.npy, [weights.npy, vol.npy, meta.json]) -- the modularity data
  CERT      npz written by `certify_lifted_vc_dump.py ... --dump_cert CERT` (any producer may write this format):
            multipliers y = (lam on triangles TI, mu, om on CI, psi on PI, pair levels (lab, sig), vertex-cluster (VC)
            levels (lab, CSR ptr/col/sig)), the vector g, the diagonal shift sdiag (length N), link stiffness mu_c,
            the lifted tree (link_p, link_vol, link_ch, link_cf, root) and the cluster -> lifted node maps.

CLAIM CHECKED.  For every partition X (X_ij = [i, j same community]) of the reduced graph,
    L * Q(X) = C0 + <B, X>,  B = W - d d^T / L                                    (exact reduction, see --orig)
            <= C0 + sum(lam) - sum(mu) + <C(y), X>                                   (valid constraints, y >= 0)
            <= C0 + sum(lam) - sum(mu) + sum(g) + sum(sdiag) + price_asm + price_psd = L * UB.
  C(y) = B + T(lam) + Nbr(mu) + Conn(om) + Pc(psi) + sum_levels S_pair + sum_VC S_vc  with the multiplier x slack terms
    triangle (i,j,k), centre j:  lam (1 - X_ij - X_jk + X_ik) >= 0        (every partition)
    psi (i,j):  psi X_ij >= 0;   pair level:  sig_QR sum_{i in Q, j in R, i != j} d_i d_j X_ij >= 0;
    VC (i,R):   s_iR d_i sum_{j in R, j != i} d_j X_ij >= 0                (every partition)
    mu_i (sum_j w_ij X_ij - 1) >= 0, om (X_is - X_ij) >= 0                  (OPTIMAL partitions only -> reported)
LIFTED PROOF.  Lifted vectors x_k: x_i = v_i (vertices), x_p = s_p / vol_p for tree nodes, s_p = sum of children's s
  (s_i = d_i v_i).  The verifier recomputes every node's exact integer leaf vector S_p (s_p = sum_i S_pi v_i) and checks
  vol_p = sum_i S_pi (=> ||x_p|| <= 1), child coefficients, S_root = d, and S_node(R) = d * 1_R for every cluster R.
  Then for the Gram Xt of the x's: Xt PSD, |Xt_kl| <= 1, e_p^T Xt e_p = 0 (e_p = vol_p e_p - sum_c coef_c e_c), and the
  independently assembled  Mt = Diag(g (+) 0) - Ct + sum_p mu_p e_p e_p^T + Diag(sdiag)  satisfies
  <Mt - Diag(sdiag), Xt> = sum(g) - <C, X>  (Ct = lifted image of C; checked numerically below against a separate
  vertex-space implementation of C).  If Mt_exact = PSD + E then <C,X> <= sum g + sum sdiag + sum_kl |E_kl|.
PSD PROOF (a different method from the certifier's CHOLMOD + a-priori Higham bound):
  --method ldl : own numba up-looking simplicial LDL^T (no BLAS/CHOLMOD) of fl(Mt) in a METIS ND order; L D L^T with
                 D > 0 is exactly PSD; the residual fl(Mt) - L D L^T is bounded ENTRY BY ENTRY by a running error
                 bound accumulated during the factorization (|r_kj| <= gamma(h+3) a_kj + u |y_j|, a_kj = |A_kj| +
                 sum |L_ji y_i|, h = row pattern size + 1; derivation in NOTES.md).  price_psd = sum over the full matrix.
  --method eig : dense eigh, A = V Lam V^T + E; lambda_min(A) >= lam_1 (1 -/+ ||V^T V - I||_F) - ||E||_F with all
                 products bounded by gamma(N+2) |.||.| terms;  price_psd = N max(0, -lambda_lb).   (N <= ~15000)
  --method sn  : supernodal right-looking Cholesky A + dA = L L^T (own numba diagonal/triangular kernels, BLAS DGEMM for
                 all block updates, multithreaded); a-priori componentwise bound valid for ANY summation order / FMA:
                 |dA_ij| <= gamma(min(rc_i, rc_j) + 3) (|L||L|^T)_ij, rc = row counts of the factor's pattern, so
                 price_psd = sum_k (sum_i sqrt(gamma(rc_i+3)) |l_ik|)^2 + underflow term (NOTES.md "Method sn").
  price_asm = sum_kl gamma(h_kl + 5) sum |terms_kl| (assembly rounding of fl(Mt)).  All sums of the bound are exact
  rationals (fractions) of the stored floats; UB is rounded up.
Usage:
  python verify_cert.py INSTANCE_DIR CERT.npz [--method ldl|eig|sn|both|ldlsn|all] [--orig ORIG_DIR --labels LAB.npy] [--out R.json]
"""
import os, sys, time, json, math, argparse
from fractions import Fraction
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(HERE, "numba_cache"))
import numpy as np, scipy.sparse as sp
from numba import njit, prange

U = 2.0**-53
T0 = time.time()
def gam(k):
    k = np.asarray(k, dtype=np.float64); return k*U/(1.0 - k*U)
def rss():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform == "darwin" else 2**10)
def log(*s): print(f"[{time.time()-T0:8.1f}s {rss():7.0f}MB]", *s, flush=True)
class Fail(Exception): pass
def need(cond, msg):
    if not cond: raise Fail(msg)


def exact_sum(a):
    """exact rational sum of a float64 array (mantissa/exponent decomposition, python big ints)."""
    a = np.asarray(a, np.float64).ravel(); a = a[a != 0]
    if len(a) == 0: return Fraction(0)
    need(np.isfinite(a).all(), "non-finite value in a summed vector")
    m, e = np.frexp(a); mi = (m*2.0**53).astype(np.int64); e = e.astype(np.int64) - 53
    o = np.argsort(e, kind="stable"); e = e[o]; mi = mi[o]; cut = np.flatnonzero(np.diff(e)) + 1
    tot = Fraction(0)
    for es, ms in zip(np.split(e, cut), np.split(mi, cut)):
        tot += Fraction(sum(ms.tolist()))*(Fraction(2)**int(es[0]))
    return tot
def frac_up(x):
    f = float(x)
    return f if Fraction(f) >= x else float(np.nextafter(f, np.inf))


# ============================================================================ 1. instance (from the files, from scratch)
def load_instance(D):
    g = json.load(open(os.path.join(D, "graph.json"))); n = int(g["n"])
    keys = np.load(os.path.join(D, "keys.npy")).astype(np.int64); u = keys//n; v = keys % n; m = len(keys)
    need(len(np.unique(keys)) == m and (u < v).all() and (keys >= 0).all() and (v < n).all(), "keys: not a simple graph")
    wf = os.path.join(D, "weights.npy")
    w = np.load(wf).astype(np.float64) if os.path.exists(wf) else np.ones(m)
    need(len(w) == m and (w > 0).all() and (w == np.round(w)).all(), "weights must be positive integers")
    wdeg = np.bincount(np.r_[u, v], weights=np.r_[w, w], minlength=n).astype(np.int64)
    if os.path.exists(os.path.join(D, "vol.npy")):
        meta = json.load(open(os.path.join(D, "meta.json")))
        d = np.load(os.path.join(D, "vol.npy")).astype(np.int64); L = int(meta["L_orig"])
        ins = int(meta.get("inside_edges", meta["leaf_edges"])); need(ins == int(meta["leaf_edges"]), "meta inside != leaf edges")
        C0 = 2*ins
        need(len(d) == n and int(d.sum()) == L, "sum(vol) != L_orig")
        need(int(d.sum()) - 2*int(w.astype(np.int64).sum()) == C0, "sum(vol) - 2 sum(w) != C0 (contraction bookkeeping)")
        need((d >= wdeg).all(), "vol < weighted degree")
    else:
        need((w == 1).all(), "unreduced instance with weights: unsupported"); d = wdeg; L = 2*m; C0 = 0
    return dict(n=n, u=u, v=v, w=w, d=d, L=L, C0=C0, m=m)


# ============================================================================ 2. certificate file
def load_cert(fn, inst, tiny=1e-200):
    z = np.load(fn, allow_pickle=False); f = set(z.files); n = inst["n"]
    c = dict(n=int(z["n"]), N=int(z["N"]), L=int(z["L"]), C0=float(z["C0"]), d=z["d"].astype(np.float64),
             g=z["g"].astype(np.float64), sdiag=z["sdiag"].astype(np.float64), mu_c=float(z["mu_c"]), root=int(z["root"]),
             cert_UB=float(z["cert_UB"]) if "cert_UB" in f else float("nan"))
    need(c["n"] == n and c["L"] == inst["L"] and c["C0"] == inst["C0"], "cert n/L/C0 differ from the instance")
    need(np.array_equal(c["d"], inst["d"].astype(np.float64)), "cert vol vector differs from the instance")
    N = c["N"]; need(len(c["g"]) == n and len(c["sdiag"]) == N and N >= n, "g / sdiag shapes")
    need(np.isfinite(c["g"]).all() and np.isfinite(c["sdiag"]).all() and (c["sdiag"] >= 0).all(), "g/sdiag non-finite or sdiag < 0")
    def mult(name, idx, k):
        val = z[name].astype(np.float64) if name in f else np.zeros(0)
        I = z[idx].astype(np.int64).reshape(-1, k) if idx in f else np.zeros((0, k), np.int64)
        need(np.isfinite(val).all() and (val >= 0).all(), f"multiplier {name} must be finite and >= 0")
        val = np.where(val < tiny, 0.0, val)              # (dropping a multiplier is always valid; avoids subnormals)
        return val, I
    c["lam"], c["TI"] = mult("lam", "TI", 3); c["om"], c["CI"] = mult("om", "CI", 3); c["psi"], c["PI"] = mult("psi", "PI", 2)
    mu = z["mu"].astype(np.float64) if "mu" in f else np.zeros(n)
    if mu.shape != (n,): mu = np.zeros(n)
    need(np.isfinite(mu).all() and (mu >= 0).all(), "mu must be >= 0"); c["mu"] = np.where(mu < tiny, 0.0, mu)
    for nm, I, k in (("TI", c["TI"], 3), ("CI", c["CI"], 3), ("PI", c["PI"], 2)):
        need(len(I) == len(c[{"TI": "lam", "CI": "om", "PI": "psi"}[nm]]), f"{nm} length")
        if len(I):
            need((I >= 0).all() and (I < n).all(), f"{nm} index range")
            for a in range(k):
                for b in range(a+1, k): need((I[:, a] != I[:, b]).all(), f"{nm}: repeated vertex in a constraint")
    c["levels"] = []
    for li in range(int(z["nlev"])):
        lab = z[f"lev{li}_lab"].astype(np.int64); sig = z[f"lev{li}_sig"].astype(np.float64)
        P = sig.shape[0]; need(sig.shape == (P, P) and (sig == sig.T).all() and np.isfinite(sig).all() and (sig >= 0).all(), f"level {li} sig")
        need(len(lab) == n and lab.min() >= 0 and lab.max() < P, f"level {li} labels")
        sig = np.where(sig < tiny, 0.0, sig)
        c["levels"].append(dict(lab=lab, P=P, sig=sig, node=z[f"lev{li}_node"].astype(np.int64), coef=z[f"lev{li}_coef"].astype(np.float64)))
    c["vcs"] = []
    for mi in range(int(z["nvc"])):
        lab = z[f"vc{mi}_lab"].astype(np.int64); ptr = z[f"vc{mi}_ptr"].astype(np.int64); col = z[f"vc{mi}_col"].astype(np.int64)
        s = z[f"vc{mi}_sig"].astype(np.float64); node = z[f"vc{mi}_node"].astype(np.int64); P = len(node)
        need(len(lab) == n and lab.min() >= 0 and lab.max() < P, f"VC {mi} labels")
        need(len(ptr) == n+1 and ptr[0] == 0 and ptr[-1] == len(col) == len(s) and (np.diff(ptr) >= 0).all(), f"VC {mi} CSR")
        need(len(col) == 0 or (col.min() >= 0 and col.max() < P), f"VC {mi} column range")
        need(np.isfinite(s).all() and (s >= 0).all(), f"VC {mi} sig >= 0")
        rows = np.repeat(np.arange(n), np.diff(ptr)); need(len(np.unique(rows*P + col)) == len(col), f"VC {mi} duplicate entries")
        s = np.where(s < tiny, 0.0, s)
        c["vcs"].append(dict(lab=lab, P=P, rows=rows, col=col, sig=s, node=node, coef=z[f"vc{mi}_coef"].astype(np.float64)))
    c["link_p"] = z["link_p"].astype(np.int64); c["link_vol"] = z["link_vol"].astype(np.float64)
    c["link_ch"] = z["link_ch"].astype(np.int64).reshape(-1, 2); c["link_cf"] = z["link_cf"].astype(np.float64).reshape(-1, 2)
    z.close(); return c


# ============================================================================ 3. lifted tree: exact leaf vectors
def check_tree(inst, c):
    """returns S (N x n int64 CSR; row k = exact leaf vector of lifted node k, s_k = sum_i S_ki v_i) and vol (N)."""
    n, N, d = inst["n"], c["N"], inst["d"]
    lp, lv, ch, cf = c["link_p"], c["link_vol"], c["link_ch"], c["link_cf"]
    need(len(lp) == N - n and len(np.unique(lp)) == len(lp) and (len(lp) == 0 or (lp.min() >= n and lp.max() < N)),
         "every aux node must be defined by exactly one link")
    need(len(ch) == 0 or (ch.min() >= 0 and ch.max() < N), "link child range")
    need((ch != lp[:, None]).all() and (ch[:, 0] != ch[:, 1]).all(), "link with repeated node")
    vol = np.zeros(N, np.float64); vol[:n] = d; vol[lp] = lv
    need((lv > 0).all() and (lv == np.round(lv)).all() and lv.max(initial=0) < 2**52, "link volumes must be positive integers")
    need(np.array_equal(cf, vol[ch]), "child coefficient != child volume (vertex: d_i, aux: vol)")
    need(np.array_equal(lv, cf[:, 0] + cf[:, 1]), "vol_p != sum of child coefficients")
    # exact leaf vectors: S = E + G S  (G = child adjacency), iterate to the fixed point (depth steps; cycle => no fixed point)
    G = sp.csr_matrix((np.ones(2*len(lp), np.int64), (np.repeat(lp, 2), ch.ravel())), shape=(N, N))
    from scipy.sparse.csgraph import connected_components
    need(connected_components(G, directed=True, connection="strong")[0] == N, "lifted tree has a cycle")
    E = sp.csr_matrix((d.astype(np.int64), (np.arange(n), np.arange(n))), shape=(N, n))
    S = E.copy(); it = 0
    while True:
        Sn = (E + G@S).tocsr(); Sn.eliminate_zeros(); it += 1
        if Sn.nnz == S.nnz and (Sn != S).nnz == 0: break
        S = Sn; need(it < 10000, "lifted tree has a cycle (no fixed point)")
    S.sort_indices()
    need((S.data > 0).all(), "negative leaf weight")
    rs = np.asarray(S.sum(axis=1)).ravel()
    need(np.array_equal(rs[lp], lv.astype(np.int64)), "vol_p != total leaf weight of p (||x_p|| <= 1 not guaranteed)")
    root = c["root"]; need(0 <= root < N, "root index")
    Sr = S[root].toarray().ravel(); need(np.array_equal(Sr, d), "leaf vector of the root != d (penalty term invalid)")
    need(int(vol[root]) == inst["L"], "root volume != L")
    def check_map(lab, P, node, coef, what):
        need(len(node) == P and len(coef) == P, f"{what}: node map length")
        Dl = sp.csr_matrix((d.astype(np.int64), (lab, np.arange(n))), shape=(P, n)); Dl.eliminate_zeros(); Dl.sort_indices()
        volR = np.bincount(lab, weights=d, minlength=P)
        empty = node < 0
        need((volR[empty] == 0).all(), f"{what}: cluster with positive volume has no lifted node")
        ok = ~empty; need((node[ok] < N).all(), f"{what}: node range")
        need(np.array_equal(coef[ok], vol[node[ok]]) and np.array_equal(coef[ok], volR[ok]), f"{what}: cluster coefficient != cluster volume")
        diff = (S[node[ok]] - Dl[np.flatnonzero(ok)]); diff.eliminate_zeros()
        need(diff.nnz == 0, f"{what}: lifted node leaf set != cluster")
    for li, Lv in enumerate(c["levels"]): check_map(Lv["lab"], Lv["P"], Lv["node"], Lv["coef"], f"pair level {li}")
    for mi, vc in enumerate(c["vcs"]): check_map(vc["lab"], vc["P"], vc["node"], vc["coef"], f"VC level {mi}")
    return S, vol, it


# ============================================================================ 4. assembly of Mt (lower triangle)
def assemble(inst, c, vol):
    """lower-triangle COO terms of the symmetric Mt (one copy per unordered pair).  Returns CSR lower (value), price_asm."""
    n, N, d = inst["n"], c["N"], inst["d"].astype(np.float64)
    R, Cc, V = [], [], []
    def put(r, cc, val):
        r = np.asarray(r, np.int64); cc = np.asarray(cc, np.int64); val = np.broadcast_to(np.asarray(val, np.float64), r.shape)
        R.append(np.maximum(r, cc)); Cc.append(np.minimum(r, cc)); V.append(np.array(val))
    # (a) diagonal: g on vertices, shift on all nodes
    put(np.arange(n), np.arange(n), c["g"]); put(np.arange(N), np.arange(N), c["sdiag"])
    # (b) penalty d d^T / L = L ||x_root||^2  -> +L at (root, root)   (S_root = d checked)
    put([c["root"]], [c["root"]], [float(inst["L"])])
    # (c) links mu_p e_p e_p^T, e_p = vol_p e_p - cf1 e_c1 - cf2 e_c2, mu_p = fl(mu_c / vol_p) (any value is valid)
    lp, lv, ch, cf = c["link_p"], c["link_vol"], c["link_ch"], c["link_cf"]
    if len(lp):
        mup = c["mu_c"]/lv; idx = np.c_[lp, ch]; co = np.c_[lv, -cf]
        for a in range(3):
            for b in range(a, 3): put(idx[:, a], idx[:, b], mup*co[:, a]*co[:, b])
    # (d) pair levels: Ct(node_Q, node_R) = sig_QR coef_Q coef_R ; vertex diagonal correction -sig_QQ d_i^2 (Mt = -Ct)
    for Lv in c["levels"]:
        node, coef, sig = Lv["node"], Lv["coef"], Lv["sig"]
        Q, Rr = np.nonzero(np.triu(sig)); ok = (node[Q] >= 0) & (node[Rr] >= 0); Q, Rr = Q[ok], Rr[ok]
        put(node[Q], node[Rr], -sig[Q, Rr]*coef[Q]*coef[Rr])
        sq = np.diag(sig)[Lv["lab"]]; nz = (sq > 0) & (d > 0)
        put(np.flatnonzero(nz), np.flatnonzero(nz), sq[nz]*d[nz]*d[nz])
    # (e) VC entries: Ct(i, node_R) = Ct(node_R, i) = s d_i coef_R / 2 ; own cluster: vertex diagonal -s d_i^2
    for vc in c["vcs"]:
        rows, col, s, node, coef = vc["rows"], vc["col"], vc["sig"], vc["node"], vc["coef"]
        k = (s > 0) & (node[col] >= 0) & (d[rows] > 0); rows, col, s = rows[k], col[k], s[k]
        nd = node[col]; diag = nd == rows                 # R = {i}: the lifted node is the vertex itself
        t = -0.5*s*d[rows]*coef[col]
        put(rows[~diag], nd[~diag], t[~diag]); put(rows[diag], rows[diag], 2.0*t[diag])
        own = col == vc["lab"][rows]; put(rows[own], rows[own], s[own]*d[rows[own]]*d[rows[own]])
    # (f) vertex-vertex part of C (off-diagonal), Mt = -C
    u, v, w = inst["u"], inst["v"], inst["w"]
    put(u, v, -w)
    if c["mu"].any(): put(u, v, -(0.5*(c["mu"][u] + c["mu"][v]))*w)
    if len(c["lam"]):
        i, j, k_ = c["TI"].T; h = 0.5*c["lam"]; put(i, j, h); put(j, k_, h); put(i, k_, -h)
    if len(c["om"]):
        i, j, s_ = c["CI"].T; h = 0.5*c["om"]; put(i, s_, -h); put(i, j, h)
    if len(c["psi"]):
        i, j = c["PI"].T; put(i, j, -0.5*c["psi"])
    r = np.concatenate(R); cc = np.concatenate(Cc); val = np.concatenate(V); del R, Cc, V
    need(np.isfinite(val).all(), "non-finite assembly term")
    it = np.int32 if N < 2**31 - 1 else np.int64; r = r.astype(it); cc = cc.astype(it)
    M = sp.csr_matrix((val, (r, cc)), shape=(N, N)); M.sum_duplicates(); M.sort_indices()
    A = sp.csr_matrix((np.abs(val), (r, cc)), shape=(N, N)); A.sum_duplicates(); A.sort_indices()
    H = sp.csr_matrix((np.ones(len(val)), (r, cc)), shape=(N, N)); H.sum_duplicates(); H.sort_indices()
    nterms = len(val); del r, cc, val
    need(np.array_equal(M.indptr, A.indptr) and np.array_equal(M.indices, A.indices) and np.array_equal(M.indices, H.indices), "pattern mismatch")
    # each term: <= 3 roundings; recursive sum of h terms: <= h-1 more;  gamma(h+5) covers both (+ the abs-sum rounding)
    rowi = np.repeat(np.arange(N), np.diff(M.indptr)); mult = np.where(rowi == M.indices, 1.0, 2.0)
    price = float(np.sum(mult*gam(H.data + 5.0)*A.data))*1.001
    return M, price, dict(nterms=nterms, nnz_lower=int(M.nnz), hmax=float(H.data.max()))


def full_sym(Ml):
    """exact symmetric full matrix from the lower triangle (no arithmetic)."""
    Mc = Ml.tocoo(); off = Mc.row != Mc.col
    return sp.csr_matrix((np.r_[Mc.data, Mc.data[off]], (np.r_[Mc.row, Mc.col[off]], np.r_[Mc.col, Mc.row[off]])), shape=Ml.shape)


# ============================================================================ 5a. PSD proof: own LDL^T with running residual bound
@njit(cache=True)
def _etree(N, rp, ci):
    parent = -np.ones(N, np.int64); anc = -np.ones(N, np.int64)
    for k in range(N):
        for p in range(rp[k], rp[k+1]):
            i = ci[p]
            while i != -1 and i < k:
                nx = anc[i]; anc[i] = k
                if nx == -1: parent[i] = k
                i = nx
    return parent

@njit(cache=True)
def _colcounts(N, rp, ci, parent):
    cnt = np.zeros(N, np.int64); flag = -np.ones(N, np.int64)
    for k in range(N):
        flag[k] = k
        for p in range(rp[k], rp[k+1]):
            i = ci[p]
            while flag[i] != k:
                cnt[i] += 1; flag[i] = k; i = parent[i]
    return cnt                                      # strictly-lower entries per column

@njit(cache=True)
def _ldl(N, rp, ci, ax, parent, Lp, Li, Lx, D):
    """up-looking LDL^T (row k of L from the rows of A(k, 0:k)), with a running bound on |A - L D L^T| per entry.
    returns (status, k_fail, price_off (one triangle), price_diag)."""
    Y = np.zeros(N); YA = np.zeros(N); flag = -np.ones(N, np.int64); patt = np.zeros(N, np.int64); Lnz = np.zeros(N, np.int64)
    poff = 0.0; pdiag = 0.0
    for k in range(N):
        top = N; flag[k] = k
        for p in range(rp[k], rp[k+1]):
            i = ci[p]; Y[i] += ax[p]; YA[i] += abs(ax[p])
            ln = 0
            while flag[i] != k:
                patt[ln] = i; ln += 1; flag[i] = k; i = parent[i]
            while ln > 0:
                top -= 1; ln -= 1; patt[top] = patt[ln]
        dk = Y[k]; dka = YA[k]; Y[k] = 0.0; YA[k] = 0.0
        h = N - top + 1.0; gh = (h + 3.0)*U/(1.0 - (h + 3.0)*U)
        for t in range(top, N):
            i = patt[t]; yi = Y[i]; yai = YA[i]; Y[i] = 0.0; YA[i] = 0.0
            p2 = Lp[i] + Lnz[i]
            for p in range(Lp[i], p2):
                r = Li[p]; prod = Lx[p]*yi; Y[r] -= prod; YA[r] += abs(prod)
            lki = yi/D[i]
            pr = lki*yi; dk -= pr; dka += abs(pr)
            Li[p2] = k; Lx[p2] = lki; Lnz[i] += 1
            poff += gh*yai + U*abs(yi)
        D[k] = dk; pdiag += gh*dka
        if not (dk > 0.0) or not np.isfinite(dk): return 1, k, poff, pdiag
    for i in range(N):
        if Lnz[i] != Lp[i+1] - Lp[i]: return 2, i, poff, pdiag
        if Y[i] != 0.0 or YA[i] != 0.0: return 3, i, poff, pdiag       # an update outside the row pattern (must not happen)
    return 0, -1, poff, pdiag


def nd_perm(Mfull, dense_deg=1000):
    """fill-reducing order (any order is valid): METIS ND on the non-dense rows, dense rows (lifted hubs) last."""
    N = Mfull.shape[0]; G = Mfull.copy(); G.setdiag(0); G.eliminate_zeros(); G.data[:] = 1; deg = np.diff(G.indptr)
    dense = np.flatnonzero(deg > dense_deg); spr = np.flatnonzero(deg <= dense_deg)
    Gs = G[spr][:, spr].tocsr()
    try:
        import pymetis
        ps = np.asarray(pymetis.nested_dissection(pymetis.CSRAdjacency(Gs.indptr.astype(np.int64), Gs.indices.astype(np.int64)))[0], np.int64) if len(spr) > 2 else np.arange(len(spr))
        how = "metis-nd"
    except ImportError:
        from scipy.sparse.csgraph import reverse_cuthill_mckee
        ps = np.asarray(reverse_cuthill_mckee(Gs, symmetric_mode=True), np.int64); how = "rcm"
    return np.r_[spr[ps], dense[np.argsort(deg[dense], kind="stable")]], how, len(dense)


def psd_ldl(Ml, max_nnz=4e9):
    N = Ml.shape[0]; Mf = full_sym(Ml)
    perm, how, nd = nd_perm(Mf); pos = np.empty(N, np.int64); pos[perm] = np.arange(N)
    Mc = Mf.tocoo(); del Mf
    keep = pos[Mc.row] >= pos[Mc.col]
    A = sp.csr_matrix((Mc.data[keep], (pos[Mc.row[keep]], pos[Mc.col[keep]])), shape=(N, N)); del Mc, keep
    A.sum_duplicates(); A.sort_indices()                    # row k: columns <= k (exact copy of fl(Mt) entries)
    rp = A.indptr.astype(np.int64); ci = A.indices.astype(np.int64); ax = A.data.astype(np.float64); del A
    t0 = time.time(); parent = _etree(N, rp, ci); cnt = _colcounts(N, rp, ci, parent)
    nnzL = int(cnt.sum()); flops = float((cnt.astype(np.float64)**2).sum())
    log(f"  ldl: order {how} ({nd} dense last), nnz(L) {nnzL:.3e}, flops ~{flops:.3e}, symbolic {time.time()-t0:.1f}s")
    need(nnzL < max_nnz, f"factor too large for the verifier ({nnzL:.2e} > {max_nnz:.1e})")
    Lp = np.r_[0, np.cumsum(cnt)].astype(np.int64); Li = np.zeros(max(nnzL, 1), np.int64)
    Lx = np.zeros(max(nnzL, 1)); D = np.zeros(N)
    t0 = time.time(); st, kf, poff, pdiag = _ldl(N, rp, ci, ax, parent, Lp, Li, Lx, D); tf = time.time() - t0
    info = dict(order=how, ndense=nd, nnzL=nnzL, flops=flops, factor_s=tf)
    if st != 0:
        info.update(status=int(st), fail_at=int(kf), fail_frac=float(kf)/N); return False, None, info, None
    info.update(minD=float(D.min()), maxD=float(D.max()), poff=poff, pdiag=pdiag)
    price = (2.0*poff + pdiag)*1.001
    return True, price, info, (perm, Lp, Li, Lx, D)


# ============================================================================ 5b. PSD proof: dense eigendecomposition
def psd_eig(Ml):
    N = Ml.shape[0]; A = full_sym(Ml).toarray()
    lam, Vm = np.linalg.eigh(A); absV = np.abs(Vm)
    E = A - (Vm*lam)@Vm.T                                          # computed residual
    bE = gam(N + 2)*(np.abs(A) + (absV*np.abs(lam))@absV.T)*(1 + gam(N + 2))   # |E_exact - E_fl| entrywise
    nE = float(np.sqrt(np.sum((np.abs(E) + bE)**2)))*(1 + gam(2*N*N + 8)); del E, bE, A
    Gm = Vm.T@Vm; Gm[np.diag_indices(N)] -= 1.0
    bG = gam(N + 2)*(absV.T@absV)*(1 + gam(N + 2))
    nG = float(np.sqrt(np.sum((np.abs(Gm) + bG)**2)))*(1 + gam(2*N*N + 8)); del Gm, bG
    l1 = float(lam[0])
    lb = (l1*(1 - nG) if l1 >= 0 else l1*(1 + nG)) - nE
    lb -= abs(lb)*1e-12 + 1e-300
    price = N*max(0.0, -lb)*1.001
    return True, price, dict(lam1=l1, normE=nE, normVtV_I=nG, lambda_lb=lb), None


# ============================================================================ 5c. PSD proof: supernodal Cholesky (BLAS-3), a-priori bound
# Only DGEMM is taken from BLAS (scipy's OpenBLAS via scipy.linalg.cython_blas, called through ctypes so that
# sub-blocks of one big array can be addressed with a leading dimension).  Diagonal-block factorizations and
# triangular solves are our own numba code (true division, sqrt, no fastmath).  Soundness: NOTES.md, "method sn".
import ctypes as _ct
def _load_dgemm():
    from scipy.linalg import cython_blas
    cap = cython_blas.__pyx_capi__["dgemm"]; api = _ct.pythonapi
    api.PyCapsule_GetName.restype = _ct.c_char_p; api.PyCapsule_GetName.argtypes = [_ct.py_object]
    api.PyCapsule_GetPointer.restype = _ct.c_void_p; api.PyCapsule_GetPointer.argtypes = [_ct.py_object, _ct.c_char_p]
    return _ct.CFUNCTYPE(None, *([_ct.c_void_p]*13))(api.PyCapsule_GetPointer(cap, api.PyCapsule_GetName(cap)))
_DGEMM = None
_GC = dict(N=_ct.c_char(b"N"), T=_ct.c_char(b"T"), m1=_ct.c_double(-1.0), p1=_ct.c_double(1.0), z0=_ct.c_double(0.0))
_GSTAT = dict(calls=0, flops=0.0)

def _gemm_nt(m, n, k, pa, lda, pb, ldb, pc, ldc, alpha, beta):
    """C (m x n, col-major at address pc, ld ldc) := beta*C + alpha * A B^T,  A (m x k) at pa / lda, B (n x k) at pb / ldb."""
    global _DGEMM
    if m <= 0 or n <= 0 or k <= 0: return
    if _DGEMM is None: _DGEMM = _load_dgemm()
    iv = (_ct.c_int*6)(m, n, k, lda, ldb, ldc); a = _ct.addressof; sz = _ct.sizeof(_ct.c_int)
    _DGEMM(a(_GC["N"]), a(_GC["T"]), a(iv), a(iv) + sz, a(iv) + 2*sz, a(_GC[alpha]), pa, a(iv) + 3*sz, pb, a(iv) + 4*sz,
           a(_GC[beta]), pc, a(iv) + 5*sz)
    _GSTAT["calls"] += 1; _GSTAT["flops"] += 2.0*m*n*k

@njit(cache=True)
def _colrowcounts(N, rp, ci, parent):
    """strictly-lower nonzeros per column (cnt) and per row (rc) of the Cholesky factor of the pattern (rp, ci)."""
    cnt = np.zeros(N, np.int64); rc = np.zeros(N, np.int64); flag = -np.ones(N, np.int64)
    for k in range(N):
        flag[k] = k
        for p in range(rp[k], rp[k+1]):
            i = ci[p]
            while flag[i] != k:
                cnt[i] += 1; rc[k] += 1; flag[i] = k; i = parent[i]
    return cnt, rc

@njit(cache=True)
def _postorder(N, parent):
    head = -np.ones(N, np.int64); nxt = -np.ones(N, np.int64)
    for j in range(N - 1, -1, -1):
        p = parent[j]
        if p >= 0: nxt[j] = head[p]; head[p] = j
    post = np.empty(N, np.int64); stack = np.empty(N, np.int64); k = 0
    for r in range(N):
        if parent[r] != -1: continue
        top = 0; stack[0] = r
        while top >= 0:
            p = stack[top]; c = head[p]
            if c == -1:
                top -= 1; post[k] = p; k += 1
            else:
                head[p] = nxt[c]; top += 1; stack[top] = c
    return post

@njit(cache=True)
def _supernodes(N, parent, cnt, W, nr0, nr1, nr2, zr0, zr1, zr2):
    """fundamental supernodes -> relaxed amalgamation (CHOLMOD-style rule, adjacent child only) -> split to width <= W.
    returns column starts (ns+1).  Any partition into contiguous column ranges is valid; this only affects speed."""
    fs = [0]
    for j in range(1, N):
        if not (parent[j-1] == j and cnt[j-1] == cnt[j] + 1): fs.append(j)
    fs.append(N); f = np.array(fs, np.int64); nf = len(f) - 1
    snof = np.empty(N, np.int64)
    for s in range(nf): snof[f[s]:f[s+1]] = s
    first = f[:-1].copy(); ncol = (f[1:] - f[:-1]).astype(np.float64); zer = np.zeros(nf); keep = np.ones(nf, np.bool_)
    for s in range(nf - 1):
        lc = f[s+1] - 1; p = parent[lc]
        if p != f[s+1]: continue                                # parent supernode must start right after s
        q = s + 1; ls = float(cnt[lc]); lq = float(cnt[f[q+1] - 1])
        extra = ncol[s]*(ncol[q] + lq - ls)
        if extra < 0: extra = 0.0
        nc = ncol[s] + ncol[q]; z = zer[s] + extra; tot = nc*(nc + 1)/2 + nc*lq; zf = z/tot
        if nc <= nr0 or (nc <= nr1 and zf < zr0) or (nc <= nr2 and zf < zr1) or zf < zr2:
            keep[s] = False; first[q] = first[s]; ncol[q] = nc; zer[q] = z
    out = []
    for s in range(nf):
        if not keep[s]: continue
        a = first[s]; b = f[s+1]
        while a < b:
            out.append(a); a += W
    out.append(N)
    return np.array(out, np.int64)

@njit(cache=True)
def _sn_struct(N, ns, sfirst, snof, parent, cnt, up, ui, sptr):
    """row structure below each supernode: (A columns of s) U (children's structures), restricted to rows > last col.
    The size must equal cnt[last col] (checked).  Returns sidx (int64) or a failing supernode."""
    sidx = np.empty(sptr[ns], np.int64); mark = -np.ones(N, np.int64)
    head = -np.ones(ns, np.int64); nxt = -np.ones(ns, np.int64)
    for s in range(ns - 1, -1, -1):
        p = parent[sfirst[s+1] - 1]
        if p >= 0:
            ps = snof[p]; nxt[s] = head[ps]; head[ps] = s
    buf = np.empty(N, np.int64)
    for s in range(ns):
        last = sfirst[s+1] - 1; k = 0
        for j in range(sfirst[s], last + 1):
            for p in range(up[j], up[j+1]):
                i = ui[p]
                if i > last and mark[i] != s: mark[i] = s; buf[k] = i; k += 1
        c = head[s]
        while c != -1:
            for p in range(sptr[c], sptr[c+1]):
                i = sidx[p]
                if i > last and mark[i] != s: mark[i] = s; buf[k] = i; k += 1
            c = nxt[c]
        if k != sptr[s+1] - sptr[s]: return sidx, s
        sidx[sptr[s]:sptr[s+1]] = np.sort(buf[:k])
    return sidx, -1

@njit(cache=True)
def _findpos(sidx, a, b, g):
    """index of g in sorted sidx[a:b] (relative to a), -1 if absent."""
    lo = a; hi = b
    while lo < hi:
        mid = (lo + hi)//2
        if sidx[mid] < g: lo = mid + 1
        else: hi = mid
    if lo < b and sidx[lo] == g: return lo - a
    return -1

@njit(parallel=True, cache=True)
def _load_A(X, ns, sfirst, sptr, sidx, nrows, off, up, ui, ux):
    bad = np.zeros(ns, np.int64)
    for s in prange(ns):
        f = sfirst[s]; l = sfirst[s+1]; nc = l - f; nr = nrows[s]
        for j in range(f, l):
            base = off[s] + (j - f)*nr
            for p in range(up[j], up[j+1]):
                i = ui[p]
                if i < l: pos = i - f
                else:
                    q = _findpos(sidx, sptr[s], sptr[s+1], i)
                    if q < 0: bad[s] += 1; continue
                    pos = nc + q
                X[base + pos] = ux[p]
    return bad

@njit(cache=True)
def _diag_block(X, o, nr, c0, c1):
    """Cholesky of the diagonal sub-block [c0,c1) (columns c < c0 already applied).  0 = ok, else 1 + failing column."""
    for c in range(c0, c1):
        cc = o + c*nr; d = X[cc + c]
        for cp in range(c0, c):
            v = X[o + cp*nr + c]; d -= v*v
        if not (d > 0.0): return 1 + c
        r = np.sqrt(d); X[cc + c] = r
        for i in range(c + 1, c1):
            v = X[cc + i]
            for cp in range(c0, c): v -= X[o + cp*nr + i]*X[o + cp*nr + c]
            X[cc + i] = v/r
    return 0

def _rows_solve_f(X, o, nr, c0, c1, r0):
    """rows r0..nr of columns [c0,c1):  x_ic = (x_ic - sum_{c0<=c'<c} x_ic' l_cc') / l_cc   (row-independent)."""
    B = 128; nch = (nr - r0 + B - 1)//B
    for ch in prange(nch):
        i0 = r0 + ch*B; i1 = min(nr, i0 + B)
        for c in range(c0, c1):
            cc = o + c*nr; r = X[cc + c]
            for cp in range(c0, c):
                l = X[o + cp*nr + c]; cb = o + cp*nr
                for i in range(i0, i1): X[cc + i] -= X[cb + i]*l
            for i in range(i0, i1): X[cc + i] = X[cc + i]/r
_rows_solve_par = njit(parallel=True, cache=True)(_rows_solve_f)
_rows_solve_ser = njit(cache=True)(_rows_solve_f)

@njit(cache=True)
def _gallop(arr, q, hi, g):
    """first index >= q in sorted arr[:hi] with arr[idx] >= g (exponential then binary search)."""
    step = 1; lo = q
    while q + step < hi and arr[q + step] < g:
        lo = q + step; step *= 2
    h = min(q + step, hi)
    while lo < h:
        mid = (lo + h)//2
        if arr[mid] < g: lo = mid + 1
        else: h = mid
    return lo

def _plan_rel_f(p0, m, sidx, sptr, sfirst, ra, rb, rt, roff, relbuf, dense, nrows, direct, okr):
    for r in prange(len(ra)):
        t = rt[r]; tf = sfirst[t]; tnc = sfirst[t+1] - tf; a = ra[r]; b = rb[r]
        colsc = sidx[p0 + b - 1] - sidx[p0 + a] == b - 1 - a
        if dense[t]:
            if sidx[p0 + m - 1] - tf >= nrows[t]: okr[r] = False
            direct[r] = colsc and (sidx[p0 + m - 1] - sidx[p0 + a] == m - 1 - a)
        else:
            q0 = roff[r]; tp = sptr[t]; te = sptr[t+1]; q = tp
            for i in range(a, m):
                g = sidx[p0 + i]
                if i < b: relbuf[q0 + i - a] = g - tf
                else:
                    q = _gallop(sidx, q, te, g)
                    if q >= te or sidx[q] != g: okr[r] = False; relbuf[q0 + i - a] = 0
                    else: relbuf[q0 + i - a] = tnc + q - tp
            direct[r] = colsc and (relbuf[q0 + m - 1 - a] - relbuf[q0] == m - 1 - a)
_plan_rel_par = njit(parallel=True, cache=True)(_plan_rel_f)
_plan_rel_ser = njit(cache=True)(_plan_rel_f)

@njit(cache=True)
def _plan_runs(s, sptr, sidx, snof, dense):
    """update targets of supernode s: runs [ra, rb) of its below-rows that are columns of target supernode rt;
    roff = offsets of the relative-position buffer (non-dense targets only)."""
    p0 = sptr[s]; m = sptr[s+1] - p0
    nrun = 0; prev = -1
    for i in range(m):
        t = snof[sidx[p0 + i]]
        if t != prev: nrun += 1; prev = t
    ra = np.empty(nrun, np.int64); rb = np.empty(nrun, np.int64); rt = np.empty(nrun, np.int64)
    roff = np.zeros(nrun + 1, np.int64); runof = np.empty(m, np.int64)
    r = -1; prev = -1
    for i in range(m):
        t = snof[sidx[p0 + i]]
        if t != prev:
            r += 1; ra[r] = i; rt[r] = t; prev = t
        rb[r] = i + 1; runof[i] = r
    for r in range(nrun):
        roff[r+1] = roff[r] + (0 if dense[rt[r]] else m - ra[r])
    return ra, rb, rt, roff, runof

def _plan(s, sfirst, sptr, sidx, snof, nrows, dense):
    """runs + relative row positions in the targets (relbuf, non-dense targets); direct = target rows contiguous."""
    ra, rb, rt, roff, runof = _plan_runs(s, sptr, sidx, snof, dense)
    p0 = sptr[s]; m = sptr[s+1] - p0; nrun = len(ra)
    relbuf = np.empty(roff[nrun], np.int64); direct = np.zeros(nrun, np.bool_); okr = np.ones(nrun, np.bool_)
    (_plan_rel_par if roff[nrun] > 200000 and nrun > 1 else _plan_rel_ser)(p0, m, sidx, sptr, sfirst, ra, rb, rt, roff, relbuf, dense, nrows, direct, okr)
    return ra, rb, rt, roff, relbuf, direct, runof, bool(okr.all())

def _scatter_f(X, T, ld, c0, c1, m, sidx, p0, runof, ra, rt, roff, relbuf, dense, sfirst, nrows, off):
    """X[target] -= T  for the lower part (rows i >= j) of the update columns [c0, c1) of the source supernode."""
    for j in prange(c0, c1):
        r = runof[j]; t = rt[r]; tf = sfirst[t]; base = off[t] + (sidx[p0 + j] - tf)*nrows[t]; tb = (j - c0)*ld - c0
        if dense[t]:
            for i in range(j, m): X[base + sidx[p0 + i] - tf] -= T[tb + i]
        else:
            q = roff[r] - ra[r]
            for i in range(j, m): X[base + relbuf[q + i]] -= T[tb + i]
_scatter_par = njit(parallel=True, cache=True)(_scatter_f)
_scatter_ser = njit(cache=True)(_scatter_f)

@njit(parallel=True, cache=True)
def _sn_price(X, ns, sfirst, sptr, sidx, nrows, off, sg):
    """per supernode: sum_k (sum_i sg_i |l_ik|)^2 over its columns (lower part incl. diagonal), #non-finite entries,
    max |l|, min diagonal."""
    pr = np.zeros(ns); nbad = np.zeros(ns, np.int64); amax = np.zeros(ns); dmin = np.full(ns, np.inf)
    for s in prange(ns):
        f = sfirst[s]; nc = sfirst[s+1] - f; nr = nrows[s]; p0 = sptr[s]
        for c in range(nc):
            base = off[s] + c*nr; acc = 0.0
            dv = X[base + c]
            if dv < dmin[s]: dmin[s] = dv
            for pos in range(c, nr):
                v = X[base + pos]
                if not np.isfinite(v): nbad[s] += 1; continue
                g = f + pos if pos < nc else sidx[p0 + pos - nc]
                av = abs(v); acc += sg[g]*av
                if av > amax[s]: amax[s] = av
            pr[s] += acc*acc
    return pr, nbad, amax, dmin

@njit(parallel=True, cache=True)
def _sn_LTx(X, ns, sfirst, sptr, sidx, nrows, off, x, absval):
    """y = L^T x (or |L|^T x), L = computed Cholesky factor (lower, incl. diagonal)."""
    y = np.zeros(len(x))
    for s in prange(ns):
        f = sfirst[s]; nc = sfirst[s+1] - f; nr = nrows[s]; p0 = sptr[s]
        for c in range(nc):
            base = off[s] + c*nr; acc = 0.0
            for pos in range(c, nr):
                g = f + pos if pos < nc else sidx[p0 + pos - nc]
                v = X[base + pos]
                acc += (abs(v) if absval else v)*x[g]
            y[f + c] = acc
    return y

@njit(cache=True)
def _sn_Lx(X, ns, sfirst, sptr, sidx, nrows, off, x, absval):
    y = np.zeros(len(x))
    for s in range(ns):
        f = sfirst[s]; nc = sfirst[s+1] - f; nr = nrows[s]; p0 = sptr[s]
        for c in range(nc):
            base = off[s] + c*nr; xc = x[f + c]
            for pos in range(c, nr):
                g = f + pos if pos < nc else sidx[p0 + pos - nc]
                v = X[base + pos]
                y[g] += (abs(v) if absval else v)*xc
    return y


def psd_sn(Ml, max_GB=120.0, W=256, nb=64, budget=2**27, relax=(4, 16, 48, 0.8, 0.1, 0.05)):
    """supernodal right-looking Cholesky of fl(Mt) (permuted), multithreaded BLAS-3 GEMM updates.
    price_psd = 1.001 * [ sum_k (sum_i sqrt(gamma(rc_i+3)) |l_ik|)^2 + underflow term ]  (NOTES.md, method sn)."""
    N = Ml.shape[0]; Mf = full_sym(Ml)
    perm, how, nd = nd_perm(Mf); pos = np.empty(N, np.int64); pos[perm] = np.arange(N)
    def lower_perm(pos):
        Mc = Mf.tocoo(); keep = pos[Mc.row] >= pos[Mc.col]
        A = sp.csr_matrix((Mc.data[keep], (pos[Mc.row[keep]], pos[Mc.col[keep]])), shape=(N, N))
        A.sum_duplicates(); A.sort_indices(); return A
    t0 = time.time(); A = lower_perm(pos)
    parent = _etree(N, A.indptr.astype(np.int64), A.indices.astype(np.int64))
    post = _postorder(N, parent); perm = perm[post]; pos = np.empty(N, np.int64); pos[perm] = np.arange(N)
    del A; A = lower_perm(pos); del Mf
    need(A.nnz == (Ml.nnz), "permuted lower triangle lost / duplicated entries")
    rp = A.indptr.astype(np.int64); ci = A.indices.astype(np.int64)
    parent = _etree(N, rp, ci); cnt, rc = _colrowcounts(N, rp, ci, parent)
    U = A.T.tocsr(); U.sort_indices(); up = U.indptr.astype(np.int64); ui = U.indices.astype(np.int64); ux = U.data.astype(np.float64)
    del A, U, rp, ci
    nnzL = int(cnt.sum()) + N; flops = float(((cnt + 1).astype(np.float64)**2).sum())
    sfirst = _supernodes(N, parent, cnt, W, *relax); ns = len(sfirst) - 1
    snof = np.repeat(np.arange(ns), np.diff(sfirst)); nc_s = np.diff(sfirst)
    slen = cnt[sfirst[1:] - 1]; sptr = np.r_[0, np.cumsum(slen)].astype(np.int64)
    sidx, bad = _sn_struct(N, ns, sfirst, snof, parent, cnt, up, ui, sptr)
    need(bad < 0, f"supernodal structure inconsistent with the column counts (supernode {bad})")
    nrows = (nc_s + slen).astype(np.int64); sz = nrows*nc_s; off = np.r_[0, np.cumsum(sz)].astype(np.int64)
    dense = np.zeros(ns, np.bool_)
    for s in range(ns):     # target rows contiguous: struct = (last+1, last+2, ...)
        if slen[s] == 0 or (sidx[sptr[s+1]-1] - sidx[sptr[s]] == slen[s] - 1 and sidx[sptr[s]] == sfirst[s+1]): dense[s] = True
    store = int(off[-1]); flops_sn = float(np.sum(nc_s*(nrows - nc_s)**2.0 + nc_s**2.0*(nrows - nc_s) + nc_s**3/3.0))
    info = dict(order=how + "+postorder", ndense=nd, nnzL=nnzL, flops=flops, nsuper=ns, store=store, store_GB=store*8/2**30,
                pad=store/nnzL - 1, flops_sn=flops_sn, W=W, nb=nb, max_rc=int(rc.max()), max_cc=int(cnt.max()), symbolic_s=time.time()-t0)
    log(f"  sn: order {how}+postorder ({nd} dense last), nnz(L) {nnzL:.3e}, supernodes {ns}, storage {store:.3e} "
        f"({store*8/2**30:.1f} GB, pad {store/nnzL-1:.2%}), flops~{flops:.3e}, symbolic {time.time()-t0:.1f}s")
    need(store*8/2**30 < max_GB, f"supernodal factor too large ({store*8/2**30:.1f} GB > {max_GB} GB)")
    X = np.zeros(max(store, 1)); xa = X.ctypes.data
    badA = _load_A(X, ns, sfirst, sptr, sidx, nrows, off, up, ui, ux)
    need(badA.sum() == 0, "an entry of A lies outside the supernodal structure"); del ui, ux
    # ------------------------------------------------------------------ numeric factorization (right-looking)
    t0 = time.time(); g0 = dict(_GSTAT); tpanel = tgemm = tscat = tplan = 0.0; nscat = 0; ndir = 0; last_log = t0; scat_vol = 0.0
    log_every = float(os.environ.get("VERIFY_SN_LOG", "300"))
    budget = int(max(budget, int((nrows - nc_s).max(initial=1))))      # a chunk holds >= 1 full column
    Tbuf = np.empty(budget); ta = Tbuf.ctypes.data; status = 0
    for s in range(ns):
        f = sfirst[s]; nc = int(nc_s[s]); nr = int(nrows[s]); o = int(off[s]); m = nr - nc
        t1 = time.time()
        for c0 in range(0, nc, nb):
            c1 = min(c0 + nb, nc)
            st = _diag_block(X, o, nr, c0, c1)
            if st: status = 1; kfail = f + st - 1; break
            if nr > c1:
                (_rows_solve_par if (nr - c1)*(c1 - c0)**2 > 200000 else _rows_solve_ser)(X, o, nr, c0, c1, c1)
            if c1 < nc:
                _gemm_nt(nr - c1, nc - c1, c1 - c0, xa + 8*(o + c0*nr + c1), nr, xa + 8*(o + c0*nr + c1), nr,
                         xa + 8*(o + c1*nr + c1), nr, "m1", "p1")
        if status: break
        t2 = time.time(); tpanel += t2 - t1
        if m == 0: continue
        ra, rb, rt, roff, relbuf, direct, runof, okp = _plan(s, sfirst, sptr, sidx, snof, nrows, dense)
        need(okp, f"update target structure inconsistent (supernode {s})")
        t2b = time.time(); tplan += t2b - t2; t2 = t2b
        pb = xa + 8*(o + nc)                                   # below block: element (i, c) at o + c*nr + nc + i
        for r in range(len(ra)):
            if not direct[r]: continue
            a = int(ra[r]); b = int(rb[r]); t = int(rt[r]); tf = int(sfirst[t]); g = int(sidx[sptr[s] + a])
            relr = g - tf
            if not dense[t]: relr = int(relbuf[roff[r]])
            _gemm_nt(m - a, b - a, nc, pb + 8*a, nr, pb + 8*a, nr, xa + 8*(int(off[t]) + (g - tf)*int(nrows[t]) + relr), int(nrows[t]), "m1", "p1")
            ndir += 1
        t3 = time.time(); tgemm += t3 - t2
        r = 0; nrun = len(ra)
        while r < nrun:                                        # spans of consecutive non-direct runs
            if direct[r]: r += 1; continue
            r2 = r
            while r2 + 1 < nrun and not direct[r2 + 1]: r2 += 1
            j0 = int(ra[r]); j1 = int(rb[r2]); r = r2 + 1
            while j0 < j1:
                w = max(1, min(j1 - j0, budget//(m - j0))); jj = j0 + w
                ta1 = time.time()
                _gemm_nt(m - j0, w, nc, pb + 8*j0, nr, pb + 8*j0, nr, ta, m - j0, "p1", "z0")
                ta2 = time.time(); tgemm += ta2 - ta1
                (_scatter_par if (m - j0)*w > 100000 else _scatter_ser)(X, Tbuf, m - j0, j0, jj, m, sidx, sptr[s], runof, ra, rt, roff, relbuf, dense, sfirst, nrows, off)
                tscat += time.time() - ta2; nscat += 1; scat_vol += (m - j0)*w - w*(w - 1)/2.0; j0 = jj
        if time.time() - last_log > log_every:
            last_log = time.time(); el = last_log - t0
            log(f"    sn factor: supernode {s}/{ns} col {f}/{N}, {el:.0f}s, gemm {(_GSTAT['flops']-g0['flops'])/1e12:.2f} TF "
                f"({(_GSTAT['flops']-g0['flops'])/max(tgemm,1e-9)/1e9:.0f} GF/s in gemm {tgemm:.0f}s), panel {tpanel:.0f}s plan {tplan:.0f}s "
                f"scatter {tscat:.0f}s ({scat_vol:.2e} entries)")
    tf_ = time.time() - t0
    info.update(factor_s=tf_, t_panel=tpanel, t_gemm=tgemm, t_scatter=tscat, t_plan=tplan, scatter_entries=scat_vol, n_scatter=nscat, n_direct=ndir,
                gemm_calls=_GSTAT["calls"] - g0["calls"], gemm_flops=_GSTAT["flops"] - g0["flops"])
    log(f"  sn: factor {tf_:.1f}s (panel {tpanel:.1f}, gemm {tgemm:.1f}, scatter {tscat:.1f}; gemm {info['gemm_flops']:.3e} flop, "
        f"{info['gemm_flops']/max(tgemm,1e-9)/1e9:.0f} GF/s)")
    if status:
        info.update(status=1, fail_at=int(kfail), fail_frac=float(kfail)/N); return False, None, info, None
    # ------------------------------------------------------------------ bound
    sg = np.sqrt(gam(rc.astype(np.float64) + 3.0))
    pr, nbad, amax, dmin = _sn_price(X, ns, sfirst, sptr, sidx, nrows, off, sg)
    if nbad.sum() > 0 or not (dmin.min() > 0):
        info.update(status=4, nonfinite=int(nbad.sum()), min_diag=float(dmin.min())); return False, None, info, None
    rmax = float(amax.max()); hmax = float(rc.max())
    under = 2.0*(2.0*store)*(4.0*(2.0*hmax + 3.0) + 2.0*rmax)*2.0**-1022        # underflow / flush-to-zero (NOTES.md)
    price_main = float(exact_sum(pr))
    price = (price_main + under)*1.001
    info.update(status=0, min_diag=float(dmin.min()), max_abs_L=rmax, price_main=price_main, price_under=under)
    return True, price, info, (perm, sfirst, sptr, sidx, nrows, off, X, rc, sg, store)


def sn_freivalds(Ml, fac, seed=12345):
    """random-sign test  |A z - L (L^T z)| <= sg o (|L| (|L|^T sg)) + gamma terms (all quantities computed); a violation
    means the factor is not what the proof assumes (a bug), so verification is refused.  Returns max ratio."""
    perm, sfirst, sptr, sidx, nrows, off, X, rc, sg, store = fac; N = Ml.shape[0]; ns = len(sfirst) - 1
    pos = np.empty(N, np.int64); pos[perm] = np.arange(N)
    Mf = full_sym(Ml).tocoo(); Ap = sp.csr_matrix((Mf.data, (pos[Mf.row], pos[Mf.col])), shape=(N, N)); del Mf
    z = np.random.default_rng(seed).choice([-1.0, 1.0], N)
    y1 = Ap@z; aA = abs(Ap)@np.ones(N); kA = float(np.diff(Ap.indptr).max()); del Ap
    w = _sn_LTx(X, ns, sfirst, sptr, sidx, nrows, off, z, False); y2 = _sn_Lx(X, ns, sfirst, sptr, sidx, nrows, off, w, False)
    wa = _sn_LTx(X, ns, sfirst, sptr, sidx, nrows, off, sg, True); bnd = sg*_sn_Lx(X, ns, sfirst, sptr, sidx, nrows, off, wa, True)
    w1 = _sn_LTx(X, ns, sfirst, sptr, sidx, nrows, off, np.ones(N), True); LL1 = _sn_Lx(X, ns, sfirst, sptr, sidx, nrows, off, w1, True)
    K = float(rc.max() + np.diff(sptr).max() + np.diff(sfirst).max() + 4)
    tol = (bnd + gam(kA + 2)*aA + 3.0*gam(2*K + 4)*LL1)*1.01 + 1e-300
    ratio = np.abs(y1 - y2)/tol
    return float(ratio.max()), float(np.abs(y1 - y2).max()), float(tol.max())


# ============================================================================ 6. vertex-space C (separate implementation)
def vertex_C(inst, c):
    """returns apply(Z) = C(y) Z for the Lagrangian operator, written directly in vertex space."""
    n, d, L = inst["n"], inst["d"].astype(np.float64), inst["L"]; u, v, w = inst["u"], inst["v"], inst["w"]
    R, Cc, V = [u, v], [v, u], [w, w]
    if c["mu"].any(): t = 0.5*(c["mu"][u] + c["mu"][v])*w; R += [u, v]; Cc += [v, u]; V += [t, t]
    if len(c["lam"]):
        i, j, k = c["TI"].T; h = 0.5*c["lam"]
        R += [i, j, j, k, i, k]; Cc += [j, i, k, j, k, i]; V += [-h, -h, -h, -h, h, h]
    if len(c["om"]):
        i, j, s = c["CI"].T; h = 0.5*c["om"]; R += [i, s, i, j]; Cc += [s, i, j, i]; V += [h, h, -h, -h]
    if len(c["psi"]):
        i, j = c["PI"].T; h = 0.5*c["psi"]; R += [i, j]; Cc += [j, i]; V += [h, h]
    Cs = sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(Cc))), shape=(n, n))
    lev = []
    for Lv in c["levels"]:
        Dl = sp.csr_matrix((d, (Lv["lab"], np.arange(n))), shape=(Lv["P"], n)); lev.append((Dl, Lv["sig"], Lv["lab"], np.diag(Lv["sig"])[Lv["lab"]]))
    vcs = []
    for vc in c["vcs"]:
        S = sp.csr_matrix((vc["sig"], (vc["rows"], vc["col"])), shape=(n, vc["P"]))
        Dl = sp.csr_matrix((d, (vc["lab"], np.arange(n))), shape=(vc["P"], n)); own = np.asarray(S[np.arange(n), vc["lab"]]).ravel()
        vcs.append((S, Dl, vc["lab"], own))
    def apply(Z):
        out = Cs@Z - np.outer(d, d@Z)/L
        for Dl, sig, lab, sq in lev: out += d[:, None]*(sig@(Dl@Z))[lab] - (sq*d*d)[:, None]*Z
        for S, Dl, lab, own in vcs:
            out += 0.5*d[:, None]*(S@(Dl@Z) + (S.T@(d[:, None]*Z))[lab]) - (own*d*d)[:, None]*Z
        return out
    return apply


def identity_check(inst, c, S, vol, Ml, dense_max=3000, k=6, seed=1):
    """max relative deviation of H^T (Mt - Diag(sdiag)) H  from  Diag(g) - C  (H = lifting map)."""
    n, N = inst["n"], c["N"]
    Hs = sp.diags(1.0/np.maximum(vol, 1e-300))@S.astype(np.float64)
    Hs = Hs.tolil(); dz = np.flatnonzero(inst["d"] == 0); Hs[dz, dz] = 1.0; Hs = Hs.tocsr()
    M0 = full_sym(Ml) - sp.diags(c["sdiag"]); Ca = vertex_C(inst, c)
    if n <= dense_max:
        Z = np.eye(n); mode = "dense"
    else:
        Z = np.random.default_rng(seed).standard_normal((n, k)); mode = f"random k={k}"
    Y1 = Hs.T@(M0@(Hs@Z)); Y2 = c["g"][:, None]*Z - Ca(Z)
    scale = abs(Hs).T@(abs(M0)@(abs(Hs)@np.abs(Z))) + np.abs(c["g"])[:, None]*np.abs(Z) + 1e-300
    return float(np.max(np.abs(Y1 - Y2)/scale.max(axis=0, keepdims=True))), mode


# ============================================================================ 8. exact reduction check
def check_reduction(orig_dir, red_dir, inst, labels_file):
    """original graph -> reduced instance bookkeeping (exact), and Q_orig(partition) == (C0 + <B', X'>)/L (fractions)."""
    g = json.load(open(os.path.join(orig_dir, "graph.json"))); no = int(g["n"])
    keys = np.load(os.path.join(orig_dir, "keys.npy")).astype(np.int64); uo = keys//no; vo = keys % no; mo = len(keys)
    need(len(np.unique(keys)) == mo and (uo < vo).all(), "original graph not simple")
    grp = np.load(os.path.join(red_dir, "group_of.npy")).astype(np.int64); n = inst["n"]
    need(len(grp) == no and grp.min() == 0 and grp.max() == n - 1, "group_of shape")
    deg = np.bincount(np.r_[uo, vo], minlength=no).astype(np.int64); L = 2*mo
    res = dict(n_orig=no, m_orig=mo)
    need(L == inst["L"], "L_orig != 2 m_orig")
    need(np.array_equal(np.bincount(grp, weights=deg, minlength=n).astype(np.int64), inst["d"]), "vol != summed original degrees")
    gu, gv = grp[uo], grp[vo]; inside = gu == gv
    need(2*int(inside.sum()) == inst["C0"], "C0 != 2 * #inside edges")
    a = np.minimum(gu[~inside], gv[~inside]); b = np.maximum(gu[~inside], gv[~inside])
    kk, cnt = np.unique(a*n + b, return_counts=True)
    rk = inst["u"]*n + inst["v"]; o = np.argsort(rk)
    need(np.array_equal(kk, rk[o]) and np.array_equal(cnt.astype(np.float64), inst["w"][o]), "reduced weights != original edge counts")
    res["bookkeeping"] = "ok"
    if labels_file:
        lab = np.load(labels_file).astype(np.int64)
        if len(lab) == n: lr = lab; src = "reduced"
        else:
            need(len(lab) == no, "labels length matches neither graph"); src = "original"
            _, first = np.unique(grp, return_index=True); lr = lab[first]          # first member of every group
            res["labels_constant_on_groups"] = bool(np.array_equal(lab, lr[grp]))
        lr = np.unique(lr, return_inverse=True)[1]; lo = lr[grp]; P = int(lr.max()) + 1
        ein = int((lo[uo] == lo[vo]).sum()); volc = np.bincount(lo, weights=deg, minlength=P).astype(np.int64)
        Qo = Fraction(2*ein, L) - Fraction(sum(int(x)**2 for x in volc), L*L)
        same = lr[inst["u"]] == lr[inst["v"]]; einw = int(inst["w"][same].astype(np.int64).sum())
        volr = np.bincount(lr, weights=inst["d"], minlength=P).astype(np.int64)
        BX = Fraction(2*einw) - Fraction(sum(int(x)**2 for x in volr), L)          # <B', X'>
        Qr = (inst["C0"] + BX)/L
        need(Qo == Qr, f"reduction identity FAILED: Q_orig {float(Qo)} != reduced {float(Qr)}")
        res.update(labels=os.path.basename(labels_file), labels_on=src, Q_orig=float(Qo), Q_reduced=float(Qr), Q_exact=str(Qo) if len(str(Qo)) < 80 else None, identity="exact")
        res["Q_frac"] = Qo
    return res


# ============================================================================ main
METHODS = dict(ldl=["ldl"], eig=["eig"], sn=["sn"], both=["ldl", "eig"], all=["ldl", "eig", "sn"], ldlsn=["ldl", "sn"])

def verify(inst_dir, cert_file, method="ldl", orig=None, labels=None, id_check=True, max_nnz=4e9, sn_opts=None):
    out = dict(instance=inst_dir, cert=cert_file, method=method, passed=False)
    try:
        inst = load_instance(inst_dir); log(f"instance n {inst['n']} m {inst['m']} L {inst['L']} C0 {inst['C0']}")
        c = load_cert(cert_file, inst)
        fam = dict(tri=int((c["lam"] > 0).sum()), psi=int((c["psi"] > 0).sum()), mu=int((c["mu"] > 0).sum()), om=int((c["om"] > 0).sum()),
                   pair_levels=[Lv["P"] for Lv in c["levels"]], vc_entries=[int((vc["sig"] > 0).sum()) for vc in c["vcs"]])
        out.update(n=inst["n"], N=c["N"], families=fam, cert_UB=c["cert_UB"])
        out["validity"] = ("all constraint families are valid for EVERY partition" if fam["mu"] == 0 and fam["om"] == 0 else
                           "uses mu/om (neighbour / connectivity) constraints: valid for OPTIMAL partitions only")
        log(f"certificate N {c['N']} families {fam}")
        S, vol, depth = check_tree(inst, c); out["tree"] = dict(ok=True, fixed_point_iters=depth, nnzS=int(S.nnz))
        log(f"lifted tree OK (depth {depth - 1}, leaf-vector nnz {S.nnz})")
        Ml, p_asm, ainfo = assemble(inst, c, vol); out["assembly"] = ainfo; log(f"assembled Mt: {ainfo}, price_asm {p_asm:.3e}")
        if id_check:
            err, mode = identity_check(inst, c, S, vol, Ml); out["identity_relerr"] = err; out["identity_mode"] = mode
            log(f"lifted identity H^T Mt0 H = Diag(g) - C: rel err {err:.2e} ({mode})")
            need(err < 1e-9, f"lifted matrix does not represent Diag(g) - C (rel err {err:.2e})")
        del S
        const = Fraction(inst["C0"]) + exact_sum(c["lam"]) - exact_sum(c["mu"]) + exact_sum(c["g"]) + exact_sum(c["sdiag"])
        out["dv_plus_shift"] = float(const/inst["L"])
        for k in ("vcs", "levels", "lam", "TI", "om", "CI", "psi", "PI"): c[k] = None       # free memory before the PSD proof
        res = {}
        for meth in METHODS[method]:
            if meth == "eig" and c["N"] > 15000: res[meth] = dict(skipped=f"N {c['N']} > 15000"); continue
            if meth == "sn":
                ok, price, info, fac = psd_sn(Ml, **(sn_opts or {}))
                if ok:
                    t0 = time.time(); ratio, dmax, tmax = sn_freivalds(Ml, fac)
                    info.update(freivalds_ratio=ratio, freivalds_absdiff=dmax, freivalds_s=time.time() - t0)
                    log(f"  sn: a-posteriori random-sign check |Az - L(L^T z)| / bound = {ratio:.3e} (must be <= 1)")
                    if not (ratio <= 1.0): ok = False; info["status"] = 5
                del fac
            else:
                ok, price, info, _ = (psd_ldl(Ml, max_nnz) if meth == "ldl" else psd_eig(Ml))
            if not ok:
                res[meth] = dict(passed=False, info=info); log(f"  {meth}: PSD proof FAILED {info}"); continue
            ubf = (const + Fraction(p_asm) + Fraction(price))/inst["L"]
            res[meth] = dict(passed=True, UB=frac_up(ubf), price_psd=price/inst["L"], price_asm=p_asm/inst["L"], info=info)
            log(f"  {meth}: PSD OK, price_psd {price/inst['L']:.3e}, VERIFIED UB {frac_up(ubf):.9f}")
        out["psd"] = res
        good = [r["UB"] for r in res.values() if r.get("passed")]
        if good:
            out["passed"] = True; out["UB"] = min(good)
        if orig:
            rr = check_reduction(orig, inst_dir, inst, labels); qf = rr.pop("Q_frac", None); out["reduction"] = rr
            log(f"reduction check: {rr}")
            if qf is not None and good: out["gap_vs_labels"] = (out["UB"] - float(qf))/float(qf)
    except Fail as e:
        out["passed"] = False; out["fail"] = str(e); log("FAIL:", e)
    out["seconds"] = time.time() - T0; out["rss_MB"] = rss()
    return out


def main():
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("instance"); pa.add_argument("cert"); pa.add_argument("--method", default="ldl", choices=list(METHODS))
    pa.add_argument("--orig", default=None, help="original (unreduced) instance dir for the exact reduction check")
    pa.add_argument("--labels", default=None, help="partition (reduced or original labels) for the reduction identity / gap")
    pa.add_argument("--no_identity", action="store_true"); pa.add_argument("--max_nnz", type=float, default=4e9)
    pa.add_argument("--sn_W", type=int, default=256, help="sn: max supernode (panel) width")
    pa.add_argument("--sn_nb", type=int, default=64, help="sn: inner block width of the own diagonal/triangular kernels")
    pa.add_argument("--sn_max_GB", type=float, default=120.0)
    pa.add_argument("--out", default="")
    a = pa.parse_args()
    r = verify(a.instance, a.cert, a.method, a.orig, a.labels, not a.no_identity, a.max_nnz,
               sn_opts=dict(W=a.sn_W, nb=a.sn_nb, max_GB=a.sn_max_GB))
    s = json.dumps(r, indent=1, default=float); print(s)
    if a.out: open(a.out, "w").write(s)
    print("VERIFIED" if r["passed"] else "NOT VERIFIED", f"UB {r.get('UB', float('nan')):.9f}" if r["passed"] else r.get("fail", "PSD proof failed"),
          f"(certifier claimed {r.get('cert_UB', float('nan')):.9f})")
    sys.exit(0 if r["passed"] else 1)


if __name__ == "__main__":
    main()
