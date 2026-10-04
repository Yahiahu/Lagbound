#!/usr/bin/env python3
"""Estimate the Lagrangian dual bound d(y) at FIXED multipliers y stored in an alm_big.py checkpoint.

    d(y) = C0 + sum(lam) - sum(mu) + max_{X psd, diag X = 1} <C(y), X>,   C(y) = B + S(sig) + T(lam) + Nbr(mu) + Conn(om)

Weak duality: for ANY vector g and ANY positive weights s,
    max <C,X> <= sum(g) + max(0, -mu_s) * sum(s),   mu_s = lambda_min of the pencil (Diag(g) - C, Diag(s)),
because Diag(g) - C - mu_s Diag(s) is PSD and <Diag(s), X> = sum(s).  (s = 1: the usual n*|lambda_min|.)
So UB(g, s) = (C0 + sum lam - sum mu + sum g + max(0,-mu_s) sum s) / L is an upper bound on d(y)/L as long as mu_s
is not OVER-estimated.  g comes from a polished BM factor W (g_i = (C W W^T)_ii); dv = (C0 + sum lam - sum mu + sum g)/L
is a LOWER bound on d(y)/L (W W^T is feasible).  So d(y) in [dv, UB]: the bracket width is the estimation error.

Pieces:
  * polish: damped batched-parallel mixing (alm_big kernel, rho = 0) and/or row-preconditioned Riemannian gradient
    ascent with BB steps + nonmonotone line search (one C*W product per step), with a dv convergence test.
  * lambda_min: preconditioned block LOBPCG on the pencil (Diag(g)-C, Diag(s)), initial block = [W, random],
    Jacobi preconditioner; reported value = theta_1 - ||r_1|| (residual-hedged, Ritz values alone are optimistic).
  * --dense: exact pencil eigenvalue by dense eigh (validation, n <= ~50k).
NESTED (nested/NOTES.md): copy of tol5/est/est_dual.py that also reads nested (block-diagonal) pair levels written by
alm_nested.py (nested{li} = 1, blk{li}, flat sig{li}); C(y) gets d_i d_j sig_QR on same-parent fine-cluster pairs only.
--polish mix is not supported with nested levels (use rgd / none).
VC (est_dual_vc.py): also reads vertex-cluster levels (nvc, vclab{m}, vcsig{m}) written by alm_vc.py: C += S_vc,
S_vc[i,j] = d_i d_j (sig_{i,R(j)} + sig_{j,R(i)})/2 (i != j); valid for any sig >= 0 (asserted). mix polish not supported.
VCS (vcsparse/est_dual_vcs.py): also reads SPARSE VC levels (nvcs, vcslab{m}, vcsptr{m}, vcscol{m}, vcssig{m}) written
by vcsparse/alm_vcs.py; S_vc applied matrix-free (vcsops.vcs_apply: 1/2 d (Sig W X + (Sig^T d X)[lab]) - own d^2 X).
This is an ESTIMATE (not a certificate): the eigensolver can in principle miss an eigenvalue it never saw.
"""
import os, sys, time, json, math, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(HERE, "numba_cache"))
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla, scipy.linalg as sl
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "nested")); sys.path.insert(0, HERE)
from nestops import NestedLevel, nest_apply
import vcops as VCO
import vcsops as VCS
from numba import njit, prange

# ----------------------------------------------------------------------------- kernels (copied from alm_big.py)
@njit(cache=True)
def incidence(n, TI):
    nt = TI.shape[0]; cnt = np.zeros(n+1, np.int64)
    for t in range(nt):
        for r in range(3): cnt[TI[t, r]+1] += 1
    for i in range(n): cnt[i+1] += cnt[i]
    pos = cnt[:-1].copy(); codes = np.empty(3*nt, np.int64)
    for t in range(nt):
        for r in range(3):
            i = TI[t, r]; codes[pos[i]] = 3*t+r; pos[i] += 1
    return cnt, codes

@njit(parallel=True, cache=True)
def tri_apply(TI, w, ptr, codes, V, out):
    n, k = V.shape
    for i in prange(n):
        for e in range(ptr[i], ptr[i+1]):
            code = codes[e]; t = code//3; r = code % 3; h = 0.5*w[t]
            if h == 0.0: continue
            a, b, c = TI[t, 0], TI[t, 1], TI[t, 2]
            if r == 0:
                for q in range(k): out[i, q] += h*(V[c, q]-V[b, q])
            elif r == 1:
                for q in range(k): out[i, q] -= h*(V[a, q]+V[c, q])
            else:
                for q in range(k): out[i, q] += h*(V[a, q]-V[b, q])

@njit(parallel=True, cache=True)
def conn_apply(CI, w, ptr, codes, V, out):
    n, k = V.shape
    for r_ in prange(n):
        for e in range(ptr[r_], ptr[r_+1]):
            code = codes[e]; t = code//3; r = code % 3; h = 0.5*w[t]
            if h == 0.0: continue
            i, j, s_ = CI[t, 0], CI[t, 1], CI[t, 2]
            if r == 0:
                for q in range(k): out[r_, q] += h*(V[s_, q] - V[j, q])
            elif r == 1:
                for q in range(k): out[r_, q] -= h*V[i, q]
            else:
                for q in range(k): out[r_, q] += h*V[i, q]

@njit(parallel=True, cache=True)
def spmm(indptr, indices, data, X, out):
    n, k = X.shape
    for i in prange(out.shape[0]):
        for q in range(k): out[i, q] = 0.0
        for e in range(indptr[i], indptr[i+1]):
            j = indices[e]; w = data[e]
            for q in range(k): out[i, q] += w*X[j, q]

@njit(parallel=True, cache=True)
def batch_step(rows, V, indptr, indices, adata, d, z, L, labs, Poff, Mall, pdiag, TI, lam, ptr, codes, damp, pn, CI, om, cptr, ccodes):
    """alm_big batch_step at rho = 0 (plain damped mixing at fixed multipliers)."""
    k = V.shape[1]; nl = labs.shape[0]
    for idx in prange(rows.shape[0]):
        i = rows[idx]; di = d[i]
        g = np.empty(k); q = np.empty(k)
        for c in range(k): g[c] = -(di/L)*(z[c] - di*V[i, c])
        for e in range(indptr[i], indptr[i+1]):
            j = indices[e]; w = adata[e]*(1.0 + 0.5*(pn[i] + pn[j]))
            for c in range(k): g[c] += w*V[j, c]
        for l in range(nl):
            Q = Poff[l] + labs[l, i]
            for c in range(k): g[c] += di*(Mall[Q, c] - pdiag[Q]*di*V[i, c])
        for e in range(ptr[i], ptr[i+1]):
            code = codes[e]; t = code//3; r = code % 3; a_, b_, c_ = TI[t, 0], TI[t, 1], TI[t, 2]
            pt = lam[t]
            if pt > 0.0:
                for c in range(k):
                    if r == 0: g[c] -= 0.5*pt*(V[b_, c] - V[c_, c])
                    elif r == 1: g[c] -= 0.5*pt*(V[a_, c] + V[c_, c])
                    else: g[c] -= 0.5*pt*(V[b_, c] - V[a_, c])
        for e in range(cptr[i], cptr[i+1]):
            code = ccodes[e]; t = code//3; r = code % 3; a_, b_, c_ = CI[t, 0], CI[t, 1], CI[t, 2]
            pc = om[t]
            if pc > 0.0:
                for c in range(k):
                    if r == 0: g[c] += 0.5*pc*(V[c_, c] - V[b_, c])
                    elif r == 1: g[c] -= 0.5*pc*V[a_, c]
                    else: g[c] += 0.5*pc*V[a_, c]
        gn = 0.0
        for c in range(k): gn += g[c]*g[c]
        w = damp*np.sqrt(gn); nv = 0.0
        for c in range(k):
            q[c] = g[c] + w*V[i, c]; nv += q[c]*q[c]
        if nv <= 0.0: continue
        nv = 1.0/np.sqrt(nv)
        for c in range(k): V[i, c] = q[c]*nv

@njit(parallel=True, cache=True)
def rowdot(X, Y):
    n, k = X.shape; out = np.empty(n)
    for i in prange(n):
        s = 0.0
        for q in range(k): s += X[i, q]*Y[i, q]
        out[i] = s
    return out

# ----------------------------------------------------------------------------- args / data
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--tag", required=True); p.add_argument("--ckpt", required=True)
p.add_argument("--polish", default="mix", help="mix | rgd | mix+rgd | none")
p.add_argument("--max_sweeps", type=int, default=200); p.add_argument("--polish_time", type=float, default=600)
p.add_argument("--chk", type=int, default=5, help="evaluate dv every chk mixing sweeps")
p.add_argument("--ptol", type=float, default=2e-5, help="stop polishing when dv (L-normalised) gains < ptol per chk sweeps")
p.add_argument("--damp", type=float, default=0.5); p.add_argument("--nbatch", type=int, default=64)
p.add_argument("--rgd_iters", type=int, default=400)
p.add_argument("--scale", default="vol,ones", help="comma list of pencil weights s: vol | ones | sqrtvol")
p.add_argument("--eig_extra", type=int, default=16, help="LOBPCG block = k + eig_extra")
p.add_argument("--eig_maxiter", type=int, default=400); p.add_argument("--eig_tol", type=float, default=1e-9)
p.add_argument("--eig_chunk", type=int, default=20, help="LOBPCG iterations between convergence checks")
p.add_argument("--eig_atol", type=float, default=1e-5, help="stop when residual and theta change, in UB units (x sum(s)/L), are below this")
p.add_argument("--eig_time", type=float, default=1200)
p.add_argument("--dense", type=int, default=0, help="1: also compute the exact pencil eigenvalue densely")
p.add_argument("--seed", type=int, default=0); p.add_argument("--out", default="")
p.add_argument("--saveW", default=""); p.add_argument("--initV", default="", help="npz with V (e.g. a previous --saveW) replacing the checkpoint V")
p.add_argument("--no_w_init", type=int, default=0, help="1: LOBPCG initial block purely random (independence check)"); p.add_argument("--lb", type=float, default=float("nan"))
p.add_argument("--escape", type=int, default=2, help="rank-escape rounds: append isolated bottom eigenvectors of the pencil to W, re-polish, re-estimate; best round kept (every round is a valid estimate)")
p.add_argument("--escape_min", type=float, default=0.002, help="only escape when the shift (UB units) exceeds this")
p.add_argument("--escape_p", type=int, default=4, help="max eigenvectors appended per round")
p.add_argument("--escape_ratio", type=float, default=0.3, help="append eigenvectors with theta_j <= ratio*theta_1 (theta < 0)")
a = p.parse_args()
rng = np.random.default_rng(a.seed)

D = os.environ.get("DATA_DIR", "/Users/ashrafkhalil/Desktop/MIE 498 RT/output/decomp_audit_2026-09-28/data/")
if not D.endswith("/"): D += "/"
t0 = time.time()
import resource
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform == "darwin" else 2**10)
def log(*s): print(f"[{time.time()-t0:8.1f}s {rss():6.0f}MB]", *s, flush=True)

g_ = json.load(open(D+a.tag+"/graph.json")); n = g_["n"]
keys = np.load(D+a.tag+"/keys.npy"); u = keys//n; v = keys % n; m = len(keys); L = 2*m; del keys
d = np.bincount(np.r_[u, v], minlength=n).astype(float)
W_ = np.load(D+a.tag+"/weights.npy") if os.path.exists(D+a.tag+"/weights.npy") else np.ones(m)
A = sp.csr_matrix((np.r_[W_, W_], (np.r_[u, v], np.r_[v, u])), shape=(n, n)); del u, v
C0 = 0.0
if os.path.exists(D+a.tag+"/vol.npy"):
    meta_ = json.load(open(D+a.tag+"/meta.json")); d = np.load(D+a.tag+"/vol.npy").astype(float)
    L = int(meta_["L_orig"]); C0 = 2.0*meta_["leaf_edges"]; assert int(d.sum()) == L
A.sort_indices(); Ap = A.indptr.astype(np.int64); Ai = A.indices.astype(np.int64); Ad = A.data.astype(np.float64)

z0 = np.load(a.ckpt)
nl = int(z0["nlev"]); levels = []
for li in range(nl):
    if f"nested{li}" in z0.files and int(z0[f"nested{li}"]):
        lab_ = z0[f"lab{li}"].astype(np.int64); NL = NestedLevel(lab_, z0[f"blk{li}"].astype(np.int64)[lab_], d)
        assert np.array_equal(NL.lab, lab_), "nested level labels must be in canonical (parent-sorted) order"
        sig = z0[f"sig{li}"].astype(np.float64); assert sig.shape == (NL.S,) and (sig >= 0).all()
        NL.sig = sig
        levels.append(dict(lab=NL.lab, P=NL.P, sig=sig, W=NL.W, sdiag=sig[NL.diag_idx].copy(), nested=NL)); continue
    lab = np.unique(z0[f"lab{li}"], return_inverse=True)[1]; P = lab.max()+1
    sig = z0[f"sig{li}"].astype(np.float64); assert sig.shape == (P, P) and (sig >= 0).all()
    Wl = sp.csr_matrix((d, (lab, np.arange(n))), shape=(P, n))
    levels.append(dict(lab=lab, P=P, sig=sig, W=Wl, sdiag=np.diag(sig).copy()))
TI = z0["TI"].astype(np.int64); lam = z0["lam"].astype(np.float64)
mu = z0["mu"].astype(np.float64) if "mu" in z0.files and z0["mu"].shape == (n,) else np.zeros(n)
CI = z0["CI"].astype(np.int64) if "CI" in z0.files else np.zeros((0, 3), np.int64)
om = z0["om"].astype(np.float64) if "om" in z0.files else np.zeros(0)
vcl = []
for mm in range(int(z0["nvc"]) if "nvc" in z0.files else 0):
    Lv_ = VCO.VCLevel(z0[f"vclab{mm}"], d); Lv_.sig = z0[f"vcsig{mm}"].astype(np.float64)
    assert Lv_.sig.shape == (n, Lv_.P) and (Lv_.sig >= 0).all(); vcl.append(Lv_)
vcsl = []
for mm in range(int(z0["nvcs"]) if "nvcs" in z0.files else 0):
    Lv_ = VCS.VCSLevel(z0[f"vcslab{mm}"], d); Lv_.set_ws(z0[f"vcsptr{mm}"], z0[f"vcscol{mm}"], z0[f"vcssig{mm}"].astype(np.float64))
    assert (Lv_.sig >= 0).all() and Lv_.ptr[-1] == len(Lv_.col) and (Lv_.col >= 0).all() and (Lv_.col < Lv_.P).all(); vcsl.append(Lv_)
V = z0["V"].astype(np.float64); z0.close()
if a.initV: V = np.load(a.initV)["V"].astype(np.float64)
assert (lam >= 0).all() and (mu >= 0).all() and (om >= 0).all()
if len(TI):
    TI = np.c_[np.minimum(TI[:, 0], TI[:, 2]), TI[:, 1], np.maximum(TI[:, 0], TI[:, 2])]
    kk = (TI[:, 0]*n+TI[:, 1])*n+TI[:, 2]; uq, first, inv = np.unique(kk, return_index=True, return_inverse=True)
    if len(first) < len(TI): lam = np.bincount(inv, weights=lam); TI = TI[first]; log("merged duplicate triangles ->", len(TI))
    keep = lam > 0; TI = np.ascontiguousarray(TI[keep]); lam = lam[keep]
tptr, tcodes = incidence(n, TI)
if len(om):
    keep = om > 0; CI = np.ascontiguousarray(CI[keep]); om = om[keep]
cptr, ccodes = incidence(n, CI)
use_mu = bool(mu.any())
const = C0 + lam.sum() - mu.sum()
V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-300); V = np.ascontiguousarray(V); k = V.shape[1]
labs = np.ascontiguousarray(np.array([Lv["lab"] for Lv in levels], np.int64).reshape(nl, n))
Poff = np.r_[0, np.cumsum([Lv["P"] for Lv in levels])].astype(np.int64)
log(f"tag {a.tag} ckpt {a.ckpt} n {n} m {m} L {L} k {k} vc levels {[Lv_.P for Lv_ in vcl]} vcs levels {[(Lv_.P, Lv_.nnz) for Lv_ in vcsl]} levels {[Lv['P'] for Lv in levels]} ntri {len(lam)} nconn {len(om)} mu {use_mu} C0/L {C0/L:.6f} const/L {const/L:.6f}")

# ----------------------------------------------------------------------------- operator C(y)
def CV(X):
    X = np.ascontiguousarray(X); out = np.empty_like(X)
    spmm(Ap, Ai, Ad, X, out)
    out -= np.outer(d, d@X)/L
    for Lv in levels:
        Y = Lv["W"]@X
        SY = nest_apply(Lv["nested"], Lv["sig"], Y) if "nested" in Lv else Lv["sig"]@Y
        out += d[:, None]*SY[Lv["lab"]] - (Lv["sdiag"][Lv["lab"]]*d**2)[:, None]*X
    for Lv_ in vcl: out += VCO.vc_apply(Lv_, Lv_.sig, X, d)
    for Lv_ in vcsl: out += VCS.vcs_apply(Lv_, Lv_.sig, X, d)
    if len(lam): tri_apply(TI, lam, tptr, tcodes, X, out)
    if use_mu:
        AX = np.empty_like(X); spmm(Ap, Ai, Ad, X, AX); out += 0.5*mu[:, None]*AX
        spmm(Ap, Ai, Ad, np.ascontiguousarray(mu[:, None]*X), AX); out += 0.5*AX
    if len(om): conn_apply(CI, om, cptr, ccodes, X, out)
    return out
diagC = -d**2/L        # S has zero diagonal, T/Conn/A have zero diagonal

def value(W):
    CW = CV(W); g = rowdot(CW, W); return (const + g.sum())/L, g, CW

# ----------------------------------------------------------------------------- polishing
def mix_sweeps(V, nsw, damp):
    z = d@V; pn = mu if use_mu else np.zeros(n)
    for _ in range(nsw):
        Ys = [Lv["W"]@V for Lv in levels]; Ms = [Lv["sig"]@Y for Lv, Y in zip(levels, Ys)]
        pdiag = np.concatenate([Lv["sdiag"] for Lv in levels]) if nl else np.zeros(0)
        order = rng.permutation(n).astype(np.int64)
        for rows in np.array_split(order, a.nbatch):
            Mall = np.ascontiguousarray(np.concatenate(Ms)) if nl else np.zeros((0, k))
            Vold = V[rows]
            batch_step(rows, V, Ap, Ai, Ad, d, z, float(L), labs, Poff, Mall, pdiag, TI, lam, tptr, tcodes, float(damp), pn, CI, om, cptr, ccodes)
            dd = d[rows][:, None]*(V[rows] - Vold); z += dd.sum(0)
            for Lv, Y, M in zip(levels, Ys, Ms):
                dY = np.zeros_like(Y); np.add.at(dY, Lv["lab"][rows], dd); T = np.flatnonzero(np.abs(dY).sum(1) > 0)
                Y[T] += dY[T]; M += Lv["sig"][:, T]@dY[T]
    return V

def polish_mix(V, hist):
    tp = time.time(); dv_prev, _, _ = value(V); hist.append(("mix", 0, time.time()-tp, dv_prev)); log(f"polish start dv {dv_prev:.6f}")
    s = 0
    while s < a.max_sweeps and time.time()-tp < a.polish_time:
        V = mix_sweeps(V, a.chk, a.damp); s += a.chk
        dv, _, _ = value(V); hist.append(("mix", s, time.time()-tp, dv))
        log(f"mix sweeps {s} dv {dv:.6f} (+{dv-dv_prev:.2e})")
        if dv - dv_prev < a.ptol: break
        dv_prev = dv
    return V

def polish_rgd(V, hist):
    """Maximise f(W) = <C, W W^T> on the oblique manifold: W+ = rownormalize(W + alpha * D^{-1} R), R = Riemannian
    gradient (rows of CW minus their radial part), D = diag(|CW rows|) (alpha = 2 is damp-0.5 Jacobi mixing).
    alpha from BB (D-metric), Zhang-Hager nonmonotone acceptance."""
    tp = time.time(); f, g, CW = value(V); Cref = f; Q = 1.0; alpha = 2.0
    nr = np.maximum(np.linalg.norm(CW, axis=1), 1e-12)
    R = (CW - g[:, None]*V)/nr[:, None]             # preconditioned ascent direction D^{-1} R
    Rr = CW - g[:, None]*V
    hist.append(("rgd", 0, time.time()-tp, f)); log(f"rgd start dv {f:.6f}")
    fbest = f; it = 0; last_chk = f
    while it < a.rgd_iters and time.time()-tp < a.polish_time:
        it += 1; gr = float(np.einsum("ij,ij->", Rr, R))/L         # directional derivative (L-normalised, /2)
        for bt in range(12):
            Vn = V + alpha*R; Vn /= np.linalg.norm(Vn, axis=1, keepdims=True)
            fn, gn, CWn = value(Vn)
            if fn >= Cref + 1e-4*alpha*gr: break
            alpha *= 0.5
        S_ = Vn - V; nrn = np.maximum(np.linalg.norm(CWn, axis=1), 1e-12)
        Rrn = CWn - gn[:, None]*Vn; Rn = Rrn/nrn[:, None]
        Y_ = Rn - R                                    # change of the preconditioned direction
        sy = float(np.einsum("ij,ij->", S_, Y_*nrn[:, None])); ss = float(np.einsum("ij,ij->", S_, S_*nrn[:, None]))
        yy = float(np.einsum("ij,ij->", Y_, Y_*nrn[:, None]))
        if sy < 0: alpha = (ss/(-sy)) if it % 2 else (-sy/yy)
        else: alpha = min(2*alpha, 50.0)
        alpha = float(np.clip(alpha, 0.05, 50.0))
        V, f, g, CW, R, Rr = Vn, fn, gn, CWn, Rn, Rrn
        Qn = 0.85*Q + 1; Cref = (0.85*Q*Cref + f)/Qn; Q = Qn
        fbest = max(fbest, f)
        if it % 10 == 0:
            hist.append(("rgd", it, time.time()-tp, f)); log(f"rgd it {it} dv {f:.6f} alpha {alpha:.3g} bt {bt}")
            if f - last_chk < a.ptol*2: break
            last_chk = f
    return V


hist = []
if (vcl or vcsl or any("nested" in Lv for Lv in levels)) and "mix" in a.polish.split("+"): raise SystemExit("--polish mix not supported with nested levels")
def do_polish(V):
    for meth in a.polish.split("+"):
        if meth == "mix": V = polish_mix(V, hist)
        elif meth == "rgd": V = polish_rgd(V, hist)
    return V
tpol = time.time()
dv0 = value(V)[0]
V = do_polish(V)
tpol = time.time() - tpol
dv, g, CW = value(V)
log(f"POLISHED dv {dv:.6f} (start {dv0:.6f}) in {tpol:.0f}s")
res = dict(tag=a.tag, ckpt=a.ckpt, n=n, L=L, k=k, dv0=dv0, dv=dv, polish_time=tpol, polish_hist=hist, lb=a.lb)

# ----------------------------------------------------------------------------- lambda_min of the pencil (Diag(g)-C, Diag(s))
def eig_round(V, dv, g):
    diagM = g - diagC
    def Mop(X):
        X = np.asarray(X, dtype=np.float64).reshape(n, -1); return g[:, None]*X - CV(X)
    recs, vecs = {}, {}
    for sname in a.scale.split(","):
        s = {"ones": np.ones(n), "vol": np.maximum(d, 1.0), "sqrtvol": np.sqrt(np.maximum(d, 1.0))*math.sqrt(L/n)}[sname]
        te = time.time()
        b = V.shape[1] + a.eig_extra
        Uw, Sw, _ = np.linalg.svd(V, full_matrices=False); r_ = 0 if a.no_w_init else int((Sw > 1e-6*Sw[0]).sum())
        X0 = np.c_[Uw[:, :r_], rng.standard_normal((n, b-r_))]; del Uw
        prec_d = 1.0/(np.abs(diagM) + 1e-2*s)   # Jacobi
        Aop = sla.LinearOperator((n, n), matvec=lambda x: Mop(x)[:, 0], matmat=lambda X: Mop(X), dtype=float)
        Bop = sla.LinearOperator((n, n), matvec=lambda x: s*x.ravel(), matmat=lambda X: s[:, None]*X, dtype=float)
        Pop = sla.LinearOperator((n, n), matvec=lambda x: prec_d*x.ravel(), matmat=lambda X: prec_d[:, None]*X, dtype=float)
        import warnings
        sums = float(s.sum()); Xe = X0; it_tot = 0; th_prev = np.inf; trace = []
        while True:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ev, Xe = sla.lobpcg(Aop, Xe, B=Bop, M=Pop, largest=False, maxiter=a.eig_chunk, tol=a.eig_tol, retLambdaHistory=False)
            it_tot += a.eig_chunk
            o = np.argsort(ev); ev = ev[o]; Xe = Xe[:, o]
            MX = Mop(Xe); xs = np.einsum("ij,ij->j", Xe, s[:, None]*Xe); th = np.einsum("ij,ij->j", Xe, MX)/xs
            Rres = MX - th[None, :]*(s[:, None]*Xe); rn = np.sqrt(np.einsum("ij,ij->j", Rres, Rres/s[:, None])/xs); del MX, Rres
            trace.append((it_tot, time.time()-te, float(th[0]), float(rn[0])))
            log(f"  [s={sname}] lobpcg its {it_tot} theta1 {th[0]:.6e} r1 {rn[0]:.2e}  (shift-units: theta {max(0,-th[0])*sums/L:.6f} r {rn[0]*sums/L:.2e})")
            if rn[0]*sums/L < a.eig_atol and abs(th[0]-th_prev)*sums/L < a.eig_atol: break
            if it_tot >= a.eig_maxiter or time.time()-te > a.eig_time: break
            th_prev = th[0]
        lmin_est = float(th[0] - rn[0])
        shift = max(0.0, -lmin_est)*sums/L
        ub = dv + shift
        rec = dict(theta=th[:6].tolist(), resid=rn[:6].tolist(), lmin_est=lmin_est, shift=shift, ub=ub, sum_s=sums, eig_time=time.time()-te, eig_trace=trace,
                   ub_ritz=dv + max(0.0, -float(th[0]))*sums/L, dv=dv, k=int(V.shape[1]))
        log(f"[s={sname}] theta {th[:4]} resid {rn[:4]} -> lmin_est {lmin_est:.3e}  shift {shift:.6f}  UB {ub:.6f}  ({time.time()-te:.0f}s)")
        recs[sname] = rec; vecs[sname] = (th, Xe)
    return recs, vecs

best = np.inf; eigvecs = {}; Vbest = V; rounds = []
for rnd in range(a.escape + 1):
    recs, vecs = eig_round(V, dv, g)
    ub_r = min(r["ub"] for r in recs.values()); rounds.append(dict(round=rnd, dv=dv, ub=ub_r, k=int(V.shape[1])))
    log(f"[round {rnd}] k {V.shape[1]} dv {dv:.6f} UB {ub_r:.6f}")
    if ub_r < best:
        best = ub_r; Vbest = V; res["dv"] = dv
        for s_, r_ in recs.items(): res[f"s_{s_}"] = r_
        eigvecs = {s_: v_[1][:, :4].astype(np.float32) for s_, v_ in vecs.items()}
    if rnd == a.escape: break
    s0 = a.scale.split(",")[0]; th, Xe = vecs[s0]
    if recs[s0]["shift"] < a.escape_min: break
    sel = [j for j in range(len(th)) if th[j] < 0 and th[j] <= a.escape_ratio*th[0]][:a.escape_p]
    if not sel: break
    U = Xe[:, sel]; U = 0.5*U/np.abs(U).max(axis=0, keepdims=True)
    V = np.c_[V, U]; V /= np.linalg.norm(V, axis=1, keepdims=True)
    tp_ = time.time(); V = do_polish(V); dv, g, CW = value(V)
    log(f"ESCAPE round {rnd+1}: appended {len(sel)} eigenvectors (theta {th[sel]}), re-polished dv {dv:.6f} k {V.shape[1]} ({time.time()-tp_:.0f}s)")
res["escape_rounds"] = rounds
res["ub"] = best; res["time"] = time.time()-t0
if np.isfinite(a.lb): res["gap"] = (best - a.lb)/a.lb
log(f"RESULT dv {res['dv']:.6f}  UB {best:.6f}" + (f"  gap {(best-a.lb)/a.lb*100:.2f}%" if np.isfinite(a.lb) else "") + f"  total {time.time()-t0:.0f}s")
if a.out: json.dump(res, open(a.out, "w"), indent=1)
if a.saveW: np.savez(a.saveW, V=Vbest.astype(np.float32), **{f"eig_{k_}": v_ for k_, v_ in eigvecs.items()})
