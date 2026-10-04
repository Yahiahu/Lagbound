#!/usr/bin/env python3
"""Scalable augmented-Lagrangian (augmented mixing) dual solver for the cluster-pair-aggregated DNN +
triangle relaxation of modularity. Same relaxation and NPZ format as ../alm_multi/alm_multi.py (--mode gs),
so ../certify_multi.py (dense) and ../certify_lifted.py (sparse) read the output unchanged.

Primal:  max <B,X>, X = V V^T with unit rows; h_QR(X) = sum_{i in Q, j in R, i!=j} d_i d_j X_ij >= 0 for every
         cluster pair of every level; tau_t(X) = X_ab + X_bc - X_ac <= 1 for pooled wedge triangles (b = centre).
PHR augmented Lagrangian, per-constraint penalty rho*r_c (r_QR = |Q||R|/(vol_Q vol_R)^2, triangles r = tri_weight):
         L(V;y) = <B,X> - sum_c ([y_c - rho r_c g_c(X)]_+^2 - y_c^2) / (2 rho r_c).
Bound:   any y >= 0:  2m Q <= sum(lam) + max_{X unit-diag PSD} <C(y), X>,  C = B + S(sig) + T(lam).

Differences from alm_multi (all about scale, n ~ 1e6):
  * Batched parallel Gauss-Seidel: rows are processed in random batches; inside a batch every row does the
    exact single-row minorant step in parallel (Hogwild reads of neighbour rows), while the GLOBAL aggregates
    (z = d^T V, cluster sums Y_Q, H_QR = Y_Q.Y_R, effective pair multipliers p, M = p Y, curvature Kq) are
    refreshed between batches. Per-row cost is O(deg k + tri_i k + levels k) instead of O(sum_l P_l k).
  * Parallel two-pass triangle separation over chunks of centres.
  * --init_partition: start from a feasible partition matrix (greedy colouring of the cluster graph with k
    colours, so that clusters sharing a coordinate are non-adjacent), i.e. a primal point of value ~LB.
Estimates (mixing at fixed y + one Lanczos eigenvalue) are NOT bounds; certify the NPZ separately.

MERGED (Sep 30 2026, see merge_NOTES.md): tol5/alm_big.py (lead's MAIN) + from tol5/algo/alm_big.py: fast exact
cluster-pair bookkeeping in sweeps (no H inside a sweep, row gathers, transposed BLAS; ne_split edge aggregate kept,
pp stays in the shared ppbuf), --ckpt_every, warm-started eigsh/LOBPCG, --est_escape, --est_only (+ --ls_* line
search / EST_DIAG), --init_sig/--init_tri, --sep_polish, --dual_step/--dual_mode. MAIN defaults unchanged.

FAST (Oct 1 2026, see fast_NOTES.md): alm_big_merged.py semantics (every flag/default) + the tol5/perf optimizations:
lazy exact cluster-pair rows per batch (pair_all: M[Q], pp_QQ only for the clusters of the next batch, from the
current Y; with --ne_split the full fresh pp ROW of every such Q is written into ppbuf, so every pp[Q(i),Q(j)] the
kernel reads is current, and the frozen per-sweep edge aggregate EAs enters h; r_QR = rq_Q rq_R * pairlvl_weight),
numba cluster-sum updates (d_i dv_i produced by the kernel), batch_step with incidence-ordered triangle records,
fused augmented(), fused one-pass separation (separate_both), parallel spmm, OMP_WAIT_POLICY=PASSIVE default.
--pair_refresh != 1 or --slow_bk 1 fall back to the canonical sweeps (bit-for-bit the merged code path).

V2 (Oct 1 2026, see v2_NOTES.md): alm_big_fast.py + options ported from mech/alm_mech.py (all default off; with them
off the iteration is alm_big_fast's):
  --pw_auto c        graph-adaptive pair weight w = c / R, R = mobility ratio (row curvature of the pair penalty,
                     trace rule, all pairs active / |objective row gradient|, volume-weighted) at w = 1, measured once
                     at the start V (after --init_mix / resume).
  --pw_rule c1,c2    w = min(c1/R, c2/(rho n^2/L))  (mobility cap and multiplier-step cap; defaults 0.1,0.05).
  --phase2_at s / --phase2_at_it N / --phase2_w f / --phase2_inner S (+ --phase2_tri_every, --phase2_est_every):
                     proximal-point phase: from the first outer that starts after s seconds (wall clock since data
                     load, as alm_mech) or after N completed outers, every multiplier update gets S augmented sweeps at
                     pair weight f (absolute) -- or f x the computed weight with --pw_auto / --pw_rule.
  --kq_mode spec     row-step curvature = top eigenvalue of sum_R act_QR Y_R Y_R^T (+diag) instead of its trace.
The pair weight enters only as r_QR = (rq_Q rq_R) * w everywhere (Lv.r, lw_all for pair_all, a.pairlvl_weight for
level_aug); set_pair_weight() updates all three. Nothing caches r or Kq across outers (Kq per sweep, pp rows per batch).

NESTED (Oct 1 2026, see nested/NOTES.md): copy of alm_big_v2.py + nested (block-diagonal) fine pair levels
  --pair_nested fine.npy parent.npy  (repeatable): fine clusters = intersection (parent, fine); pairs (Q, R) only for
  Q, R with the same parent, multipliers stored per parent block (nested/nestops.py). Same r_QR, pairlvl_weight, PHR
  update, Kq trace rule and dual-bound formula as a dense level (= dense fine level with cross-parent sig fixed at 0).
  Checkpoint: lab{li} (fine), nested{li} = 1, blk{li} (parent block of each fine cluster), sig{li} flat (sum Pc^2).

VC (Oct 1 2026, see nested/NOTES.md, nested/vcops.py): vertex-cluster nonnegativity levels
  --vc_labels a.npy [b.npy ...]: g_iR = d_i sum_{j in R, j != i} d_j X_ij >= 0 for every vertex i and cluster R
  (dense n x P multipliers per level), PHR penalty rho r_iR, r_iR = vc_weight * pairlvl_weight * rq_R / d_i^2
  (rq_R = |R|/vol_R^2), multiplier update sig <- max(0, sig - rho r g), row gradient/curvature in vcops.py,
  dual matrix C += S_vc (zero diagonal), checkpoint nvc, vclab{m}, vcsig{m} (float32).

VCS (Oct 1 2026, see vcsparse/NOTES.md, vcsparse/vcsops.py): SPARSE active-set VC levels
  --vcs_labels a.npy [...]: working set of (i, R) entries (row CSR), grown by separation every --vcs_every outers
  (G = V Y^T in chunks; the --vcs_add best violated non-ws entries, scored by --vcs_score, total cap --vcs_max) and
  pruned of entries with sig = 0 and slack for --vcs_drop consecutive outers. Penalty / multipliers / dual only on ws.
  --vcs_full a.npy [...]: same code with the working set = all n x P entries (no separation) = the dense level.
  Same r_iR, vc_weight, vc_kap2 as the dense VC. Checkpoint: nvcs, vcslab{m}, vcsptr{m}, vcscol{m}, vcssig{m} (float32),
  vcsfull{m}. VCS_EXACT=1 (validation): per-batch einsum |Y_R|^2 and BLAS Z rebuild as the dense code.
"""
import os, sys, time, json, math, argparse, signal
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(HERE, "numba_cache"))
# idle OpenMP workers sleep instead of spinning (perf/: many short parallel regions per sweep on a shared machine)
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "nested")); sys.path.insert(0, HERE)
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
import numba
from numba import njit, prange
from aggops import PairLevel
import vcops as VCO
import vcsops as VCS
from nestops import NestedLevel, nest_apply, nest_symmetrize, nest_sig_csr, nest_kq as _nest_kq, nest_aug as _nest_aug

# ----------------------------------------------------------------------------- kernels
@njit(parallel=True, cache=True)
def tri_tau(TI, V):
    nt = TI.shape[0]; k = V.shape[1]; out = np.empty(nt)
    for t in prange(nt):
        a, b, c = TI[t, 0], TI[t, 1], TI[t, 2]; s = 0.0
        for q in range(k):
            s += V[a, q]*V[b, q] + V[b, q]*V[c, q] - V[a, q]*V[c, q]
        out[t] = s
    return out

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
    """out += T(w) V, T(w) = sum_t w_t/2 (-E_ab - E_bc + E_ac) (symmetrised), b = centre."""
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
def conn_val(CI, V):
    """c_t = X_is - X_ij for CI[t] = (i, j, s):  connectivity cut c_t >= 0 (s separates i from j)."""
    nt = CI.shape[0]; k = V.shape[1]; out = np.empty(nt)
    for t in prange(nt):
        i, j, s_ = CI[t, 0], CI[t, 1], CI[t, 2]; a = 0.0
        for q in range(k): a += V[i, q]*(V[s_, q] - V[j, q])
        out[t] = a
    return out

@njit(parallel=True, cache=True)
def conn_apply(CI, w, ptr, codes, V, out):
    """out += Cc V,  Cc = sum_t w_t/2 (E_is + E_si - E_ij - E_ji)."""
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
def pair_val(PI, V):
    nt = PI.shape[0]; k = V.shape[1]; out = np.empty(nt)
    for t in prange(nt):
        i, j = PI[t, 0], PI[t, 1]; a = 0.0
        for q in range(k): a += V[i, q]*V[j, q]
        out[t] = a
    return out

@njit(cache=True)
def incidence2(n, PI):
    nt = PI.shape[0]; cnt = np.zeros(n+1, np.int64)
    for t in range(nt):
        for r in range(2): cnt[PI[t, r]+1] += 1
    for i in range(n): cnt[i+1] += cnt[i]
    pos = cnt[:-1].copy(); codes = np.empty(2*nt, np.int64)
    for t in range(nt):
        for r in range(2):
            i = PI[t, r]; codes[pos[i]] = 2*t+r; pos[i] += 1
    return cnt, codes

@njit(parallel=True, cache=True)
def pair_apply(PI, w, ptr, codes, V, out):
    """out += Pc V,  Pc = sum_t w_t/2 (E_ij + E_ji)  (pair nonnegativity cuts X_ij >= 0)."""
    n, k = V.shape
    for r_ in prange(n):
        for e in range(ptr[r_], ptr[r_+1]):
            code = codes[e]; t = code//2; r = code % 2; h = 0.5*w[t]
            if h == 0.0: continue
            o = PI[t, 1] if r == 0 else PI[t, 0]
            for q in range(k): out[r_, q] += h*V[o, q]

@njit(parallel=True, cache=True)
def batch_step(rows, V, indptr, indices, adata, d, z, L, labs, Poff, Mall, pdiag, Kq, TI, lam, ptr, codes, rt, damp, pn, rn, CI, om, cptr, ccodes, rc, PI, ps_, pptr, pcodes, rp, nesplit, ppbuf, Hoff, Pcnt):
    """One parallel batch of single-row augmented-mixing steps with frozen global aggregates.
    Row i: g = sum_{j~i} v_j - (d_i/L)(z - d_i v_i) + sum_l d_i (M_Q - p_QQ d_i v_i) - 1/2 sum_t pt_t q_t,
           kappa = sum_l d_i^2 Kq_Q + rt sum_{active t} |q_t|^2,   v_i <- normalize(g + (kappa/2 + damp|g|) v_i)."""
    k = V.shape[1]; nl = labs.shape[0]
    for idx in prange(rows.shape[0]):
        i = rows[idx]; di = d[i]
        g = np.empty(k); q = np.empty(k)
        for c in range(k): g[c] = -(di/L)*(z[c] - di*V[i, c])
        ns = np.zeros(k)
        for e in range(indptr[i], indptr[i+1]):
            j = indices[e]; w = adata[e]*(1.0 + 0.5*(pn[i] + pn[j]))   # edge weight x neighbour-cut reweighting
            if nesplit:                                       # non-edge-split aggregates exclude edge pairs (i,j)
                corr = 0.0
                for l in range(nl):
                    corr += ppbuf[Hoff[l] + labs[l, i]*Pcnt[l] + labs[l, j]]
                w -= di*d[j]*corr
            for c in range(k): g[c] += w*V[j, c]; ns[c] += adata[e]*V[j, c]
        kap = 0.0
        if pn[i] > 0.0:
            for c in range(k): kap += rn*ns[c]*ns[c]
        for l in range(nl):
            Q = Poff[l] + labs[l, i]
            for c in range(k): g[c] += di*(Mall[Q, c] - pdiag[Q]*di*V[i, c])
            kap += di*di*Kq[Q]
        for e in range(ptr[i], ptr[i+1]):
            code = codes[e]; t = code//3; r = code % 3; a_, b_, c_ = TI[t, 0], TI[t, 1], TI[t, 2]
            tau = 0.0; q2 = 0.0
            for c in range(k):
                tau += V[a_, c]*V[b_, c] + V[b_, c]*V[c_, c] - V[a_, c]*V[c_, c]
                if r == 0: q[c] = V[b_, c] - V[c_, c]
                elif r == 1: q[c] = V[a_, c] + V[c_, c]
                else: q[c] = V[b_, c] - V[a_, c]
                q2 += q[c]*q[c]
            pt = lam[t] - rt*(1.0 - tau)
            if pt > 0.0:
                for c in range(k): g[c] -= 0.5*pt*q[c]
                kap += rt*q2
        for e in range(cptr[i], cptr[i+1]):            # connectivity cuts X_is - X_ij >= 0
            code = ccodes[e]; t = code//3; r = code % 3; a_, b_, c_ = CI[t, 0], CI[t, 1], CI[t, 2]
            cv = 0.0; q2 = 0.0
            for c in range(k):
                cv += V[a_, c]*(V[c_, c] - V[b_, c])
                if r == 0: q[c] = V[c_, c] - V[b_, c]
                elif r == 1: q[c] = -V[a_, c]
                else: q[c] = V[a_, c]
                q2 += q[c]*q[c]
            pc = om[t] - rc*cv
            if pc > 0.0:
                for c in range(k): g[c] += 0.5*pc*q[c]
                kap += rc*q2
        for e in range(pptr[i], pptr[i+1]):            # pair nonnegativity cuts X_ij >= 0
            code = pcodes[e]; t = code//2; r = code % 2; o = PI[t, 1] if r == 0 else PI[t, 0]
            xv = 0.0; q2 = 0.0
            for c in range(k): xv += V[i, c]*V[o, c]; q2 += V[o, c]*V[o, c]
            pc = ps_[t] - rp*xv
            if pc > 0.0:
                for c in range(k): g[c] += 0.5*pc*V[o, c]
                kap += rp*q2
        gn = 0.0
        for c in range(k): gn += g[c]*g[c]
        w = 0.5*kap + damp*np.sqrt(gn); nv = 0.0
        for c in range(k):
            q[c] = g[c] + w*V[i, c]; nv += q[c]*q[c]
        if nv <= 0.0: continue
        nv = 1.0/np.sqrt(nv)
        for c in range(k): V[i, c] = q[c]*nv

# ---- perf kernels (ported from tol5/perf/alm_big.py, adapted to the merged features)
def aligned(X, al=64):
    """Copy of X whose data starts on a cache-line boundary."""
    buf = np.empty(X.nbytes + al, np.uint8); off = (-buf.ctypes.data) % al
    Z = buf[off:off+X.nbytes].view(X.dtype).reshape(X.shape); Z[...] = X; return Z

@njit(parallel=True, cache=True)
def batch_step6(rows, V, indptr, indices, adata, d, z, L, labs, Poff, Mall, pdiag, Kq, IABC, LINC, ptr, codes, rt, damp, pn, rn, CI, om, cptr, ccodes, rc, PI, ps_, pptr, pcodes, rp, nesplit, ppbuf, Hoff, Pcnt, use_pn, chunk, DD, usevc, GX, KX):
    """batch_step (merged: neighbour cuts, --ne_split edge correction from ppbuf, connectivity cuts, pair cuts) with
    the perf/ batch_step5 layout: rows in chunks (scratch once per chunk), triangle records in incidence order
    (IABC[e] = TI[t], LINC[e] = lam[t]), role branches hoisted, neighbour-cut reweighting skipped when pn == 0
    (w = a_e exactly), and DD[idx] = d_i (v_i,new - v_i,old) returned for the cluster-sum update. Per-row arithmetic
    identical to batch_step."""
    k = V.shape[1]; nl = labs.shape[0]; nr = rows.shape[0]; nch = (nr + chunk - 1)//chunk
    for ch in prange(nch):
        g = np.empty(k); q = np.empty(k); ns = np.zeros(k)
        for idx in range(ch*chunk, min(nr, (ch+1)*chunk)):
            i = rows[idx]; di = d[i]
            for c in range(k): g[c] = -(di/L)*(z[c] - di*V[i, c])
            if use_pn:
                for c in range(k): ns[c] = 0.0
                for e in range(indptr[i], indptr[i+1]):
                    j = indices[e]; w = adata[e]*(1.0 + 0.5*(pn[i] + pn[j]))
                    if nesplit:
                        corr = 0.0
                        for l in range(nl): corr += ppbuf[Hoff[l] + labs[l, i]*Pcnt[l] + labs[l, j]]
                        w -= di*d[j]*corr
                    for c in range(k): g[c] += w*V[j, c]; ns[c] += adata[e]*V[j, c]
            elif nesplit:
                for e in range(indptr[i], indptr[i+1]):
                    j = indices[e]; w = adata[e]
                    corr = 0.0
                    for l in range(nl): corr += ppbuf[Hoff[l] + labs[l, i]*Pcnt[l] + labs[l, j]]
                    w -= di*d[j]*corr
                    for c in range(k): g[c] += w*V[j, c]
            else:
                for e in range(indptr[i], indptr[i+1]):
                    j = indices[e]; w = adata[e]
                    for c in range(k): g[c] += w*V[j, c]
            kap = 0.0
            if use_pn and pn[i] > 0.0:
                for c in range(k): kap += rn*ns[c]*ns[c]
            for l in range(nl):
                Q = Poff[l] + labs[l, i]; mq = Mall[Q]; pd = pdiag[Q]*di
                for c in range(k): g[c] += di*(mq[c] - pd*V[i, c])
                kap += di*di*Kq[Q]
            if usevc:                                       # VC levels (vcops.vc_pre, computed before the batch)
                for c in range(k): g[c] += GX[idx, c]
                kap += KX[idx]
            for e in range(ptr[i], ptr[i+1]):
                r = codes[e] % 3; a_, b_, c_ = IABC[e, 0], IABC[e, 1], IABC[e, 2]
                tau = 0.0; q2 = 0.0
                if r == 0:
                    for c in range(k):
                        va = V[a_, c]; vb = V[b_, c]; vc = V[c_, c]
                        tau += va*vb + vb*vc - va*vc; q[c] = vb - vc; q2 += q[c]*q[c]
                elif r == 1:
                    for c in range(k):
                        va = V[a_, c]; vb = V[b_, c]; vc = V[c_, c]
                        tau += va*vb + vb*vc - va*vc; q[c] = va + vc; q2 += q[c]*q[c]
                else:
                    for c in range(k):
                        va = V[a_, c]; vb = V[b_, c]; vc = V[c_, c]
                        tau += va*vb + vb*vc - va*vc; q[c] = vb - va; q2 += q[c]*q[c]
                pt = LINC[e] - rt*(1.0 - tau)
                if pt > 0.0:
                    for c in range(k): g[c] -= 0.5*pt*q[c]
                    kap += rt*q2
            for e in range(cptr[i], cptr[i+1]):            # connectivity cuts X_is - X_ij >= 0
                code = ccodes[e]; t = code//3; r = code % 3; a_, b_, c_ = CI[t, 0], CI[t, 1], CI[t, 2]
                cv = 0.0
                for c in range(k): cv += V[a_, c]*(V[c_, c] - V[b_, c])
                pc = om[t] - rc*cv
                if pc > 0.0:
                    q2 = 0.0
                    if r == 0:
                        for c in range(k): q[c] = V[c_, c] - V[b_, c]
                    elif r == 1:
                        for c in range(k): q[c] = -V[a_, c]
                    else:
                        for c in range(k): q[c] = V[a_, c]
                    for c in range(k): q2 += q[c]*q[c]
                    for c in range(k): g[c] += 0.5*pc*q[c]
                    kap += rc*q2
            for e in range(pptr[i], pptr[i+1]):            # pair nonnegativity cuts X_ij >= 0
                code = pcodes[e]; t = code//2; r = code % 2; o = PI[t, 1] if r == 0 else PI[t, 0]
                xv = 0.0; q2 = 0.0
                for c in range(k): xv += V[i, c]*V[o, c]; q2 += V[o, c]*V[o, c]
                pc = ps_[t] - rp*xv
                if pc > 0.0:
                    for c in range(k): g[c] += 0.5*pc*V[o, c]
                    kap += rp*q2
            gn = 0.0
            for c in range(k): gn += g[c]*g[c]
            w = 0.5*kap + damp*np.sqrt(gn); nv = 0.0
            for c in range(k):
                q[c] = g[c] + w*V[i, c]; nv += q[c]*q[c]
            if nv <= 0.0:
                for c in range(k): DD[idx, c] = 0.0
                continue
            nv = 1.0/np.sqrt(nv)
            for c in range(k):
                vo = V[i, c]; vn = q[c]*nv; V[i, c] = vn; DD[idx, c] = di*(vn - vo)    # = d_i (V_new - V_old)

@njit(parallel=True, cache=True)
def inc_abc(TI, codes, out):
    for e in prange(codes.shape[0]):
        t = codes[e]//3; out[e, 0] = TI[t, 0]; out[e, 1] = TI[t, 1]; out[e, 2] = TI[t, 2]

@njit(parallel=True, cache=True)
def inc_lam(codes, lam, out):
    for e in prange(codes.shape[0]): out[e] = lam[codes[e]//3]

@njit(cache=True)
def sig_csr(sig):
    """Row-compressed nonzeros of a level's multiplier matrix (sig is fixed during a sweeps() call)."""
    P = sig.shape[0]; ptr = np.zeros(P+1, np.int64)
    for Q in range(P):
        c = 0
        for R in range(P):
            if sig[Q, R] != 0.0: c += 1
        ptr[Q+1] = ptr[Q] + c
    idx = np.empty(ptr[P], np.int64); val = np.empty(ptr[P])
    for Q in range(P):
        m = ptr[Q]
        for R in range(P):
            if sig[Q, R] != 0.0: idx[m] = R; val[m] = sig[Q, R]; m += 1
    return ptr, idx, val

@njit(parallel=True, cache=True, fastmath={"reassoc", "contract"})
def pair_all(items, ilev, Y, clo, chi, pbase, sptr, sidx, sval, rq, dsq, lw, rho, M, pdiag, nesplit, eabuf, ppbuf):
    """Fresh exact pair-state rows, all levels in one parallel region (items = global cluster ids, ilev = levels;
    Y, sig CSR (global column ids), rq, dsq, M, pdiag concatenated over levels), at the CURRENT cluster sums Y:
      h_QR = Y_Q.Y_R - [Q=R] dsq_Q (- EAs_QR with --ne_split: frozen symmetrised edge aggregate, eabuf),
      pp_QR = max(0, sig_QR - rho r_QR h_QR),  r_QR = (rq_Q rq_R) lw_l  (lw = --pairlvl_weight, as Lv.r),
      M[Q] = sum_R pp_QR Y_R,  pdiag[Q] = pp_QQ;  rho = 0: pp = sig.
    With nesplit (rho > 0) the whole row pp[Q, :] is also written to ppbuf: the batch kernel's edge correction reads
    pp[Q(i), Q(j)] only for rows Q(i) of the batch, i.e. exactly these freshly computed rows."""
    k = Y.shape[1]
    for t in prange(items.shape[0]):
        Q = items[t]; l = ilev[t]; o0 = clo[Q]; o1 = chi[Q]; acc = np.zeros(k)   # NESTED: per-cluster column range
        if rho > 0.0:
            e = sptr[Q]; e1 = sptr[Q+1]; rqQ = rq[Q]; pq = 0.0; wl = lw[l]
            base = pbase[Q]
            for R in range(o0, o1):
                h = 0.0
                for c in range(k): h += Y[Q, c]*Y[R, c]
                if R == Q: h -= dsq[Q]
                if nesplit: h -= eabuf[base + R]
                sg = 0.0
                if e < e1 and sidx[e] == R:
                    sg = sval[e]; e += 1
                p = sg - (rho*((rqQ*rq[R])*wl))*h
                if p < 0.0: p = 0.0
                if nesplit: ppbuf[base + R] = p
                if R == Q: pq = p
                if p != 0.0:
                    for c in range(k): acc[c] += p*Y[R, c]
            pdiag[Q] = pq
        else:
            pq = 0.0
            for e in range(sptr[Q], sptr[Q+1]):
                R = sidx[e]; p = sval[e]
                if R == Q: pq = p
                for c in range(k): acc[c] += p*Y[R, c]
            pdiag[Q] = pq
        for c in range(k): M[Q, c] = acc[c]

@njit(parallel=True, cache=True)
def kq_level(H, sig, r, y2, rho, out):
    """Kq = 2 rho ((pp>0) r) y2 + 2 rho diag((pp>0) r) y2, pp = max(0, sig - rho r H) (canonical _kq)."""
    P = H.shape[0]
    for Q in prange(P):
        s = 0.0
        for R in range(P):
            p = sig[Q, R] - rho*r[Q, R]*H[Q, R]
            if p > 0.0: s += r[Q, R]*y2[R]
        pq = sig[Q, Q] - rho*r[Q, Q]*H[Q, Q]
        dg = r[Q, Q]*y2[Q] if pq > 0.0 else 0.0
        out[Q] = 2*rho*(s + dg)

@njit(cache=True)
def _acc_rows(rows, dd, z, labs, Poff, Y):
    k = dd.shape[1]; nl = labs.shape[0]
    for idx in range(rows.shape[0]):
        i = rows[idx]
        for c in range(k):
            x = dd[idx, c]; z[c] += x
            for l in range(nl):
                Q = Poff[l] + labs[l, i]; Y[Q, c] += x

@njit(parallel=True, cache=True)
def _acc_par(rows, dd, z, labs, Poff, Y):
    """_acc_rows parallel over (level, column) pairs: same additions in the same (row) order -> bitwise identical."""
    k = dd.shape[1]; nl = labs.shape[0]
    for task in prange(nl*k + k):
        if task < k:
            c = task
            for idx in range(rows.shape[0]): z[c] += dd[idx, c]
        else:
            l = (task - k)//k; c = (task - k) % k; o = Poff[l]
            for idx in range(rows.shape[0]):
                Y[o + labs[l, rows[idx]], c] += dd[idx, c]

@njit(cache=True)
def touched_all(rows, labs, Poff, mark, stamp):
    """Distinct (global) clusters of the batch rows over all levels, and their levels."""
    nl = labs.shape[0]; out = np.empty(rows.shape[0]*nl, np.int64); lv = np.empty(rows.shape[0]*nl, np.int64); m = 0
    for l in range(nl):
        for idx in range(rows.shape[0]):
            Q = Poff[l] + labs[l, rows[idx]]
            if mark[Q] != stamp: mark[Q] = stamp; out[m] = Q; lv[m] = l; m += 1
    return out[:m], lv[:m]

@njit(parallel=True, cache=True)
def quadA(indptr, indices, data, V):
    """sum_ij A_ij V_i.V_j  (so <B V, V> = quadA - |d^T V|^2 / L without forming B V)."""
    n = indptr.shape[0]-1; k = V.shape[1]; tot = 0.0
    for i in prange(n):
        s = 0.0
        for e in range(indptr[i], indptr[i+1]):
            j = indices[e]; dt = 0.0
            for c in range(k): dt += V[i, c]*V[j, c]
            s += data[e]*dt
        tot += s
    return tot

@njit(parallel=True, cache=True)
def level_aug(H, sig, rq, w, rho, pp, nesplit, EAs):
    """Canonical augmented() level term: h = sym(H) (- EAs with --ne_split), rr = rho r (r = rq rq^T * w),
    pp = max(0, sig - rr h); returns (sum (pp^2 - sig^2)/(2 rr), #(h < -1e-9)); writes pp."""
    P = H.shape[0]; tot = 0.0; nv = 0
    for Q in prange(P):
        s = 0.0; c = 0
        for R in range(P):
            h = 0.5*(H[Q, R] + H[R, Q])
            if nesplit: h = h - EAs[Q, R]
            rr = rho*((rq[Q]*rq[R])*w); sg = sig[Q, R]
            p = sg - rr*h
            if p < 0.0: p = 0.0
            pp[Q, R] = p; s += (p*p - sg*sg)/(2*rr)
            if h < -1e-9: c += 1
        tot += s; nv += c
    return tot, nv

@njit(parallel=True, cache=True)
def tri_aug(TI, V, lam, rt, gt, pt):
    """gt = 1 - tau(V), pt = max(0, lam - rt gt); returns (sum (pt^2 - lam^2)/(2 rt), min gt)."""
    nt = TI.shape[0]; k = V.shape[1]; tot = 0.0; gmin = np.inf
    for t in prange(nt):
        a, b, c = TI[t, 0], TI[t, 1], TI[t, 2]; s = 0.0
        for q in range(k):
            s += V[a, q]*V[b, q] + V[b, q]*V[c, q] - V[a, q]*V[c, q]
        g = 1.0 - s; gt[t] = g; p = lam[t] - rt*g
        if p < 0.0: p = 0.0
        pt[t] = p; tot += (p*p - lam[t]*lam[t])/(2*rt); gmin = min(gmin, g)
    return tot, gmin

@njit(parallel=True, cache=True)
def cluster_sums(labs, Poff, d, V, nthr, out):
    """out[Poff[l] + lab_l(i)] = sum_i d_i V_i for every level (= Wall @ V), per-thread partial sums."""
    n, k = V.shape; nl = labs.shape[0]; sP = out.shape[0]
    part = np.zeros((nthr, sP, k)); cs = (n + nthr - 1)//nthr
    for t in prange(nthr):
        for i in range(t*cs, min(n, (t+1)*cs)):
            di = d[i]
            for l in range(nl):
                Q = Poff[l] + labs[l, i]
                for c in range(k): part[t, Q, c] += di*V[i, c]
    for Q in prange(sP):
        for c in range(k):
            s = 0.0
            for t in range(nthr): s += part[t, Q, c]
            out[Q, c] = s

@njit(parallel=True, cache=True)
def spmm(indptr, indices, data, X, out):
    n = indptr.shape[0]-1; k = X.shape[1]
    for i in prange(n):
        for c in range(k): out[i, c] = 0.0
        for e in range(indptr[i], indptr[i+1]):
            j = indices[e]; w = data[e]
            for c in range(k): out[i, c] += w*X[j, c]

@njit(parallel=True, cache=True)
def edge_vals(indptr, indices, Ew, V, out):
    """out[e] = d_i d_j X_ij for the ordered edge pairs e = (i, indices[e]) (CSR order = edge_agg's E_r/E_c order)."""
    n = indptr.shape[0]-1; k = V.shape[1]
    for i in prange(n):
        for e in range(indptr[i], indptr[i+1]):
            j = indices[e]; s = 0.0
            for c in range(k): s += V[i, c]*V[j, c]
            out[e] = s*Ew[e]

@njit(cache=True)
def _ea_acc(indptr, indices, x, lab, EA):
    n = indptr.shape[0]-1
    for i in range(n):
        Q = lab[i]
        for e in range(indptr[i], indptr[i+1]): EA[Q, lab[indices[e]]] += x[e]

@njit(cache=True)
def _centre(j, indptr, indices, V, thresh, maxh):
    k = V.shape[1]; lo, hi = indptr[j], indptr[j+1]
    m = 0; H = np.empty(hi-lo, np.int64); xh = np.empty(hi-lo)
    for e in range(lo, hi):
        i = indices[e]; s = 0.0
        for q in range(k): s += V[i, q]*V[j, q]
        if s > thresh: H[m] = i; xh[m] = s; m += 1
    H = H[:m]; xh = xh[:m]
    if m > maxh:
        o = np.argsort(-xh)[:maxh]; H = H[o]; xh = xh[o]
    return H, xh

@njit(parallel=True, cache=True)
def sep_hist(indptr, indices, V, thresh, maxh, vtol, nchunk, nb):
    n, k = V.shape; hist = np.zeros((nchunk, nb), np.int64); cs = (n + nchunk - 1)//nchunk
    for ch in prange(nchunk):
        for j in range(ch*cs, min(n, (ch+1)*cs)):
            if indptr[j+1]-indptr[j] < 2: continue
            H, xh = _centre(j, indptr, indices, V, thresh, maxh)
            for x in range(len(H)):
                for y in range(x+1, len(H)):
                    s = 0.0
                    for q in range(k): s += V[H[x], q]*V[H[y], q]
                    vm = xh[x]+xh[y]-s-1.0
                    if vm > vtol: hist[ch, min(int((vm-vtol)/(2.0-vtol)*nb), nb-1)] += 1
    return hist

@njit(parallel=True, cache=True)
def sep_collect(indptr, indices, V, thresh, maxh, vmin, nchunk, counts, offs, out, val, chs):
    n, k = V.shape; cs = (n + nchunk - 1)//nchunk
    for ich in prange(chs.shape[0]):
        ch = chs[ich]
        m = offs[ch]; end = offs[ch] + counts[ch]
        for j in range(ch*cs, min(n, (ch+1)*cs)):
            if indptr[j+1]-indptr[j] < 2: continue
            H, xh = _centre(j, indptr, indices, V, thresh, maxh)
            for x in range(len(H)):
                for y in range(x+1, len(H)):
                    s = 0.0
                    for q in range(k): s += V[H[x], q]*V[H[y], q]
                    vm = xh[x]+xh[y]-s-1.0
                    if vm >= vmin and m < end:
                        out[m, 0] = min(H[x], H[y]); out[m, 1] = j; out[m, 2] = max(H[x], H[y]); val[m] = vm; m += 1

@njit(cache=True)
def _isadj(indptr, indices, j, x):
    lo = indptr[j]; hi = indptr[j+1]
    while lo < hi:
        mid = (lo+hi)//2
        if indices[mid] < x: lo = mid+1
        else: hi = mid
    return lo < indptr[j+1] and indices[lo] == x

@njit(cache=True)
def _centre1(j, indptr, indices, V, thresh, maxh, maxk, hub2, cap2):
    """Edge arms H (top-maxh neighbours of j by X_ij > thresh) and non-edge arms K (2-hop vertices, non-adjacent
    to j, top-maxk by X_jk > thresh; 2-hop walk through the hub2 strongest arms, at most cap2 neighbours each)."""
    H, xh = _centre(j, indptr, indices, V, thresh, maxh)
    k = V.shape[1]; nh = min(len(H), hub2); tot = 0
    for a in range(nh): tot += min(cap2, indptr[H[a]+1]-indptr[H[a]])
    K = np.empty(tot, np.int64); xk = np.empty(tot); m = 0
    for a in range(nh):
        i2 = H[a]; lo = indptr[i2]; hi = min(indptr[i2+1], lo+cap2)
        for e in range(lo, hi):
            c = indices[e]
            if c == j or _isadj(indptr, indices, j, c): continue
            s = 0.0
            for q in range(k): s += V[c, q]*V[j, q]
            if s > thresh: K[m] = c; xk[m] = s; m += 1
    K = K[:m]; xk = xk[:m]
    if m > 1:
        o = np.argsort(K); K = K[o]; xk = xk[o]; keep = np.ones(m, np.bool_)
        for t in range(1, m):
            if K[t] == K[t-1]: keep[t] = False
        K = K[keep]; xk = xk[keep]
    if len(K) > maxk:
        o = np.argsort(-xk)[:maxk]; K = K[o]; xk = xk[o]
    return H, xh, K, xk

@njit(parallel=True, cache=True)
def sep1_hist(indptr, indices, V, thresh, maxh, maxk, hub2, cap2, vtol, nchunk, nb):
    n, k = V.shape; hist = np.zeros((nchunk, nb), np.int64); cs = (n + nchunk - 1)//nchunk
    for ch in prange(nchunk):
        for j in range(ch*cs, min(n, (ch+1)*cs)):
            if indptr[j+1]-indptr[j] < 1: continue
            H, xh, K, xk = _centre1(j, indptr, indices, V, thresh, maxh, maxk, hub2, cap2)
            for x in range(len(H)):
                for y in range(len(K)):
                    if H[x] == K[y]: continue
                    s = 0.0
                    for q in range(k): s += V[H[x], q]*V[K[y], q]
                    vm = xh[x]+xk[y]-s-1.0
                    if vm > vtol: hist[ch, min(int((vm-vtol)/(2.0-vtol)*nb), nb-1)] += 1
    return hist

@njit(parallel=True, cache=True)
def sep1_collect(indptr, indices, V, thresh, maxh, maxk, hub2, cap2, vmin, nchunk, counts, offs, out, val, chs):
    n, k = V.shape; cs = (n + nchunk - 1)//nchunk
    for ich in prange(chs.shape[0]):
        ch = chs[ich]
        m = offs[ch]; end = offs[ch] + counts[ch]
        for j in range(ch*cs, min(n, (ch+1)*cs)):
            if indptr[j+1]-indptr[j] < 1: continue
            H, xh, K, xk = _centre1(j, indptr, indices, V, thresh, maxh, maxk, hub2, cap2)
            for x in range(len(H)):
                for y in range(len(K)):
                    if H[x] == K[y]: continue
                    s = 0.0
                    for q in range(k): s += V[H[x], q]*V[K[y], q]
                    vm = xh[x]+xk[y]-s-1.0
                    if vm >= vmin and m < end:
                        out[m, 0] = min(H[x], K[y]); out[m, 1] = j; out[m, 2] = max(H[x], K[y]); val[m] = vm; m += 1

@njit(cache=True)
def _compact(ch, cnt, theta, B, bo, bv):
    """Keep (in generation order) only the buffered candidates strictly above the B-th largest value; raise theta."""
    c = cnt[ch]; tmp = np.sort(bv[ch, :c]); thr = tmp[c-B]; m = 0
    for x in range(c):
        if bv[ch, x] > thr:
            bv[ch, m] = bv[ch, x]; bo[ch, m, 0] = bo[ch, x, 0]; bo[ch, m, 1] = bo[ch, x, 1]; bo[ch, m, 2] = bo[ch, x, 2]; m += 1
    cnt[ch] = m; theta[ch] = thr

@njit(cache=True)
def _push(ch, vm, x0, x1, x2, cnt, theta, B, bo, bv):
    if vm > theta[ch]:
        if cnt[ch] == 2*B: _compact(ch, cnt, theta, B, bo, bv)
        if vm > theta[ch]:
            m = cnt[ch]; bo[ch, m, 0] = x0; bo[ch, m, 1] = x1; bo[ch, m, 2] = x2; bv[ch, m] = vm; cnt[ch] = m+1

@njit(parallel=True, cache=True)
def sep_pass(indptr, indices, V, thresh, maxh, vtol, nchunk, nb, B, hist, bo, bv, cnt, theta):
    """sep_hist + buffered candidates: per chunk, every violation > theta_ch is kept (theta_ch only rises, by
    compaction to the top B), in generation order. Same arithmetic as sep_hist/sep_collect."""
    n, k = V.shape; cs = (n + nchunk - 1)//nchunk
    for ch in prange(nchunk):
        cnt[ch] = 0; theta[ch] = vtol
        for j in range(ch*cs, min(n, (ch+1)*cs)):
            if indptr[j+1]-indptr[j] < 2: continue
            H, xh = _centre(j, indptr, indices, V, thresh, maxh)
            mh = len(H)
            if mh < 2: continue
            VT = np.empty((k, mh)); srow = np.empty(mh)          # transposed gather: dot products vectorised
            for y in range(mh):                                    # across y, same per-pair summation order
                for q in range(k): VT[q, y] = V[H[y], q]
            for x in range(mh):
                for y in range(x+1, mh): srow[y] = 0.0
                for q in range(k):
                    vq = VT[q, x]
                    for y in range(x+1, mh): srow[y] += vq*VT[q, y]
                for y in range(x+1, mh):
                    vm = xh[x]+xh[y]-srow[y]-1.0
                    if vm > vtol:
                        hist[ch, min(int((vm-vtol)/(2.0-vtol)*nb), nb-1)] += 1
                        _push(ch, vm, min(H[x], H[y]), j, max(H[x], H[y]), cnt, theta, B, bo, bv)

@njit(cache=True)
def _centre1f(j, indptr, indices, V, thresh, maxh, maxk, hub2, cap2, mk):
    """_centre1 with identical output: adjacency test by a per-thread marker array (stamp j+1) instead of a
    binary search, and duplicate 2-hop candidates dropped BEFORE their dot product (duplicates carry identical
    values, and the reference sorts by id and keeps one of each anyway)."""
    H, xh = _centre(j, indptr, indices, V, thresh, maxh)
    k = V.shape[1]; nh = min(len(H), hub2); tot = 0; st = j + 1
    for e in range(indptr[j], indptr[j+1]): mk[indices[e]] = st
    mk[j] = st
    for a in range(nh): tot += min(cap2, indptr[H[a]+1]-indptr[H[a]])
    C = np.empty(tot, np.int64); mc = 0
    for a in range(nh):
        i2 = H[a]; lo = indptr[i2]; hi = min(indptr[i2+1], lo+cap2)
        for e in range(lo, hi):
            c = indices[e]
            if mk[c] == st or mk[c] == -st: continue
            mk[c] = -st; C[mc] = c; mc += 1
    K = np.empty(mc, np.int64); xk = np.empty(mc); m = 0
    for x in range(mc):
        c = C[x]; s = 0.0
        for q in range(k): s += V[c, q]*V[j, q]
        if s > thresh: K[m] = c; xk[m] = s; m += 1
    K = K[:m]; xk = xk[:m]
    if m > 1:
        o = np.argsort(K); K = K[o]; xk = xk[o]
    if len(K) > maxk:
        o = np.argsort(-xk)[:maxk]; K = K[o]; xk = xk[o]
    return H, xh, K, xk

@njit(parallel=True, cache=True)
def sep1_pass(indptr, indices, V, thresh, maxh, maxk, hub2, cap2, marks, vtol, nchunk, nb, B, hist, bo, bv, cnt, theta):
    n, k = V.shape; cs = (n + nchunk - 1)//nchunk
    for ch in prange(nchunk):
        cnt[ch] = 0; theta[ch] = vtol; mrk = marks[numba.get_thread_id()]
        for j in range(ch*cs, min(n, (ch+1)*cs)):
            if indptr[j+1]-indptr[j] < 1: continue
            H, xh, K, xk = _centre1f(j, indptr, indices, V, thresh, maxh, maxk, hub2, cap2, mrk)
            mk = len(K)
            if mk == 0 or len(H) == 0: continue
            VT = np.empty((k, mk)); srow = np.empty(mk)
            for y in range(mk):
                for q in range(k): VT[q, y] = V[K[y], q]
            for x in range(len(H)):
                hx = H[x]
                for y in range(mk): srow[y] = 0.0
                for q in range(k):
                    vq = V[hx, q]
                    for y in range(mk): srow[y] += vq*VT[q, y]
                for y in range(mk):
                    if hx == K[y]: continue
                    vm = xh[x]+xk[y]-srow[y]-1.0
                    if vm > vtol:
                        hist[ch, min(int((vm-vtol)/(2.0-vtol)*nb), nb-1)] += 1
                        _push(ch, vm, min(hx, K[y]), j, max(hx, K[y]), cnt, theta, B, bo, bv)

@njit(parallel=True, cache=True)
def sep_fill(ok, vmin, counts, offs, cnt, bo, bv, out, val):
    """Reference sep_collect semantics (generation order, vm >= vmin, at most counts[ch]) from the buffers."""
    for ch in prange(ok.shape[0]):
        if not ok[ch]: continue
        m = offs[ch]; end = offs[ch] + counts[ch]
        for x in range(cnt[ch]):
            if bv[ch, x] >= vmin and m < end:
                out[m, 0] = bo[ch, x, 0]; out[m, 1] = bo[ch, x, 1]; out[m, 2] = bo[ch, x, 2]; val[m] = bv[ch, x]; m += 1

SEPSTAT = {}
def _bufs(cap, nchunk, nb):
    B = int(max(2048, 4*cap//nchunk))
    return [B, np.zeros((nchunk, nb), np.int64), np.empty((nchunk, 2*B, 3), np.int64), np.empty((nchunk, 2*B)),
            np.zeros(nchunk, np.int64), np.zeros(nchunk)]

def _finish(fam, bufs, collf, pargs, vtol, cap, nchunk, nb):
    """Reference threshold/bin selection on the full histogram; candidates from the buffers, chunks whose buffer
    threshold rose above vmin recomputed with the reference collect kernel (identical values)."""
    B, hist, bo, bv, cnt, theta = bufs
    cum = np.cumsum(hist.sum(0)[::-1])[::-1]; okb = np.flatnonzero(cum <= cap); b0 = okb[0] if len(okb) else nb
    if b0 >= nb: return np.zeros((0, 3), np.int64), np.zeros(0)
    vmin = vtol + (2.0-vtol)*b0/nb
    counts = hist[:, b0:].sum(1).astype(np.int64); offs = np.r_[0, np.cumsum(counts)[:-1]].astype(np.int64)
    out = np.zeros((counts.sum(), 3), np.int64); val = np.zeros(counts.sum())
    ok = (theta < vmin) | (theta == vtol)
    sep_fill(ok, float(vmin), counts, offs, cnt, bo, bv, out, val)
    bad = np.flatnonzero(~ok).astype(np.int64)
    if len(bad):
        with numba.parallel_chunksize(1):
            collf(*pargs, vmin, nchunk, counts, offs, out, val, bad)
    SEPSTAT[fam] = SEPSTAT.get(fam, 0) + len(bad)
    keep = val > 0; return out[keep], val[keep]

def _sep_generic(fam, passf, collf, pargs, vtol, cap, nchunk, nb, extra=()):
    bufs = _bufs(cap, nchunk, nb)
    with numba.parallel_chunksize(1):
        passf(*pargs, *extra, vtol, nchunk, nb, *bufs)
    return _finish(fam, bufs, collf, pargs, vtol, cap, nchunk, nb)

@njit(parallel=True, cache=True)
def gather_nb(indices, V, G):
    k = V.shape[1]
    for e in prange(indices.shape[0]):
        i = indices[e]
        for q in range(k): G[e, q] = V[i, q]

@njit(cache=True)
def _centreG(j, indptr, indices, G, V, thresh, maxh):
    """_centre reading neighbour rows from the edge-ordered copy G = V[indices] (streaming instead of random
    row reads); also returns the edge positions of the arms. Same values and order as _centre."""
    k = V.shape[1]; lo, hi = indptr[j], indptr[j+1]
    m = 0; H = np.empty(hi-lo, np.int64); xh = np.empty(hi-lo); Hp = np.empty(hi-lo, np.int64)
    for e in range(lo, hi):
        s = 0.0
        for q in range(k): s += G[e, q]*V[j, q]
        if s > thresh: H[m] = indices[e]; xh[m] = s; Hp[m] = e; m += 1
    H = H[:m]; xh = xh[:m]; Hp = Hp[:m]
    if m > maxh:
        o = np.argsort(-xh)[:maxh]; H = H[o]; xh = xh[o]; Hp = Hp[o]
    return H, xh, Hp

@njit(cache=True)
def _walkG(j, H, indptr, indices, G, V, thresh, maxk, hub2, cap2, mrk, hk, hs, hv):
    """Non-edge arms of _centre1 (same output): marker adjacency test, duplicates dropped before the dot product,
    candidate rows read from G."""
    jj = np.int64(j); k = V.shape[1]; nh = min(len(H), hub2); tot = 0; st = jj + 1; dj = indptr[j+1] - indptr[j]
    for a in range(nh): tot += min(cap2, indptr[H[a]+1]-indptr[H[a]])
    C = np.empty(tot, np.int64); Cp = np.empty(tot, np.int64); mc = 0
    TS = hk.shape[0]
    if 2*(dj + 1 + tot) <= TS:
        # small open-addressing set (L1/L2 resident, stamped: no clearing): 1 = adjacent to j or j, 2 = seen
        mask = TS - 1
        for e in range(indptr[j], indptr[j+1] + 1):
            c = indices[e] if e < indptr[j+1] else jj
            h = (c*2654435761) & mask
            while hs[h] == st and hk[h] != c: h = (h + 1) & mask
            hs[h] = st; hk[h] = c; hv[h] = 1
        for a in range(nh):
            i2 = H[a]; lo = indptr[i2]; hi = min(indptr[i2+1], lo+cap2)
            for e in range(lo, hi):
                c = indices[e]; h = (c*2654435761) & mask
                while hs[h] == st and hk[h] != c: h = (h + 1) & mask
                if hs[h] == st: continue                  # adjacent, j itself, or already a candidate
                hs[h] = st; hk[h] = c; hv[h] = 2; C[mc] = c; Cp[mc] = e; mc += 1
    else:
        for e in range(indptr[j], indptr[j+1]): mrk[indices[e]] = st
        mrk[j] = st
        for a in range(nh):
            i2 = H[a]; lo = indptr[i2]; hi = min(indptr[i2+1], lo+cap2)
            for e in range(lo, hi):
                c = indices[e]
                if mrk[c] == st or mrk[c] == -st: continue
                mrk[c] = -st; C[mc] = c; Cp[mc] = e; mc += 1
    K = np.empty(mc, np.int64); xk = np.empty(mc); Kp = np.empty(mc, np.int64); m = 0
    sv = np.empty(mc); x = 0
    while x + 4 <= mc:                                   # 4 independent dot products at a time (ILP);
        e0 = Cp[x]; e1 = Cp[x+1]; e2 = Cp[x+2]; e3 = Cp[x+3]  # each keeps the reference summation order
        s0 = 0.0; s1 = 0.0; s2 = 0.0; s3 = 0.0
        for q in range(k):
            vj = V[j, q]; s0 += G[e0, q]*vj; s1 += G[e1, q]*vj; s2 += G[e2, q]*vj; s3 += G[e3, q]*vj
        sv[x] = s0; sv[x+1] = s1; sv[x+2] = s2; sv[x+3] = s3; x += 4
    while x < mc:
        e0 = Cp[x]; s0 = 0.0
        for q in range(k): s0 += G[e0, q]*V[j, q]
        sv[x] = s0; x += 1
    for x in range(mc):
        if sv[x] > thresh: K[m] = C[x]; xk[m] = sv[x]; Kp[m] = Cp[x]; m += 1
    K = K[:m]; xk = xk[:m]; Kp = Kp[:m]
    if m > maxk:
        # reference: sort by id, then argsort(-xk)[:maxk]. With distinct values this is exactly the top-maxk by
        # xk in descending order, which a partition + small sort gives without the two O(m log m) sorts.
        kth = np.partition(-xk, maxk-1)[maxk-1]; sel = np.empty(maxk, np.int64); ns = 0
        for x in range(m):
            if -xk[x] < kth and ns < maxk: sel[ns] = x; ns += 1
        for x in range(m):
            if -xk[x] == kth and ns < maxk: sel[ns] = x; ns += 1
        K = K[sel]; xk = xk[sel]; Kp = Kp[sel]
        o = np.argsort(-xk); K = K[o]; xk = xk[o]; Kp = Kp[o]
    elif m > 1:
        o = np.argsort(K); K = K[o]; xk = xk[o]; Kp = Kp[o]
    return K, xk, Kp

@njit(parallel=True, cache=True)
def sep_both_pass(indptr, indices, G, V, thresh, maxh, maxk, hub2, cap2, marks, HK, HS, HV, vtol, nchunk, nb,
                  B0, hist0, bo0, bv0, cnt0, th0, B1, hist1, bo1, bv1, cnt1, th1):
    """sep_pass and sep1_pass fused (shared edge arms), all row reads from G or the gathered arm rows."""
    n, k = V.shape; cs = (n + nchunk - 1)//nchunk
    for ch in prange(nchunk):
        cnt0[ch] = 0; th0[ch] = vtol; cnt1[ch] = 0; th1[ch] = vtol; tid = numba.get_thread_id()
        mrk = marks[tid]; hk = HK[tid]; hs = HS[tid]; hv = HV[tid]
        for j in range(ch*cs, min(n, (ch+1)*cs)):
            dj = indptr[j+1]-indptr[j]
            if dj < 1: continue
            H, xh, Hp = _centreG(j, indptr, indices, G, V, thresh, maxh)
            mh = len(H)
            VH = np.empty((k, mh))
            for y in range(mh):
                for q in range(k): VH[q, y] = G[Hp[y], q]
            if dj >= 2 and mh >= 2:
                srow = np.empty(mh)
                for x in range(mh):
                    for y in range(x+1, mh): srow[y] = 0.0
                    for q in range(k):
                        vq = VH[q, x]
                        for y in range(x+1, mh): srow[y] += vq*VH[q, y]
                    for y in range(x+1, mh):
                        vm = xh[x]+xh[y]-srow[y]-1.0
                        if vm > vtol:
                            hist0[ch, min(int((vm-vtol)/(2.0-vtol)*nb), nb-1)] += 1
                            _push(ch, vm, min(H[x], H[y]), j, max(H[x], H[y]), cnt0, th0, B0, bo0, bv0)
            K, xk, Kp = _walkG(j, H, indptr, indices, G, V, thresh, maxk, hub2, cap2, mrk, hk, hs, hv)
            mk = len(K)
            if mk == 0 or mh == 0: continue
            VK = np.empty((k, mk)); srow1 = np.empty(mk)
            for y in range(mk):
                for q in range(k): VK[q, y] = G[Kp[y], q]
            for x in range(mh):
                hx = H[x]
                for y in range(mk): srow1[y] = 0.0
                for q in range(k):
                    vq = VH[q, x]
                    for y in range(mk): srow1[y] += vq*VK[q, y]
                for y in range(mk):
                    if hx == K[y]: continue
                    vm = xh[x]+xk[y]-srow1[y]-1.0
                    if vm > vtol:
                        hist1[ch, min(int((vm-vtol)/(2.0-vtol)*nb), nb-1)] += 1
                        _push(ch, vm, min(hx, K[y]), j, max(hx, K[y]), cnt1, th1, B1, bo1, bv1)

SEP_BMAX = int(os.environ.get("ALM_SEP_BMAX", "16384"))   # per-chunk candidate buffer cap (512 chunks x 2B x 32 bytes ~ 0.5 GB)
def _big(cap, nchunk): return 4*cap//nchunk > SEP_BMAX

def separate_both(indptr, indices, V, thresh, maxh, maxk, hub2, cap2, vtol, cap0, cap1, nchunk=512, nb=2000):
    """(separate(...), separate1(...)) in one fused pass; identical outputs. Huge caps (buffers > SEP_BMAX per chunk)
    fall back to the reference two-pass kernels (same output, bounded memory)."""
    if _big(cap0, nchunk) or _big(cap1, nchunk):
        return (separate_ref(indptr, indices, V, thresh, maxh, vtol, cap0, nchunk, nb),
                separate1_ref(indptr, indices, V, thresh, maxh, maxk, hub2, cap2, vtol, cap1, nchunk, nb))
    G = np.empty((len(indices), V.shape[1])); gather_nb(indices, V, G)
    nt = numba.get_num_threads(); marks = np.zeros((nt, len(indptr)-1), np.int64)
    TS = 1 << max(12, int(np.ceil(np.log2(4*(hub2*cap2 + 64)))))
    HK = np.zeros((nt, TS), np.int64); HS = np.zeros((nt, TS), np.int64); HV = np.zeros((nt, TS), np.int64)
    b0 = _bufs(cap0, nchunk, nb); b1 = _bufs(cap1, nchunk, nb)
    with numba.parallel_chunksize(1):
        sep_both_pass(indptr, indices, G, V, thresh, maxh, maxk, hub2, cap2, marks, HK, HS, HV, vtol, nchunk, nb, *b0, *b1)
    del G, marks
    r0 = _finish(0, b0, sep_collect, (indptr, indices, V, thresh, maxh), vtol, cap0, nchunk, nb)
    r1 = _finish(1, b1, sep1_collect, (indptr, indices, V, thresh, maxh, maxk, hub2, cap2), vtol, cap1, nchunk, nb)
    return r0, r1

def separate1(indptr, indices, V, thresh, maxh, maxk, hub2, cap2, vtol, cap, nchunk=512, nb=2000):
    if _big(cap, nchunk): return separate1_ref(indptr, indices, V, thresh, maxh, maxk, hub2, cap2, vtol, cap, nchunk, nb)
    marks = np.zeros((numba.get_num_threads(), len(indptr)-1), np.int64)
    return _sep_generic(1, sep1_pass, sep1_collect, (indptr, indices, V, thresh, maxh, maxk, hub2, cap2), vtol, cap, nchunk, nb, (marks,))

def separate(indptr, indices, V, thresh, maxh, vtol, cap, nchunk=512, nb=2000):
    if _big(cap, nchunk): return separate_ref(indptr, indices, V, thresh, maxh, vtol, cap, nchunk, nb)
    return _sep_generic(0, sep_pass, sep_collect, (indptr, indices, V, thresh, maxh), vtol, cap, nchunk, nb)

def separate_ref(indptr, indices, V, thresh, maxh, vtol, cap, nchunk=512, nb=2000):
    hist = sep_hist(indptr, indices, V, thresh, maxh, vtol, nchunk, nb)
    tot_b = hist.sum(0); cum = np.cumsum(tot_b[::-1])[::-1]
    ok = np.flatnonzero(cum <= cap); b0 = ok[0] if len(ok) else nb
    if b0 >= nb: return np.zeros((0, 3), np.int64), np.zeros(0)
    vmin = vtol + (2.0-vtol)*b0/nb
    counts = hist[:, b0:].sum(1).astype(np.int64); offs = np.r_[0, np.cumsum(counts)[:-1]].astype(np.int64)
    out = np.zeros((counts.sum(), 3), np.int64); val = np.zeros(counts.sum())
    sep_collect(indptr, indices, V, thresh, maxh, vmin, nchunk, counts, offs, out, val, np.arange(nchunk, dtype=np.int64))
    keep = val > 0; return out[keep], val[keep]

def separate1_ref(indptr, indices, V, thresh, maxh, maxk, hub2, cap2, vtol, cap, nchunk=512, nb=2000):
    hist = sep1_hist(indptr, indices, V, thresh, maxh, maxk, hub2, cap2, vtol, nchunk, nb)
    cum = np.cumsum(hist.sum(0)[::-1])[::-1]; ok = np.flatnonzero(cum <= cap); b0 = ok[0] if len(ok) else nb
    if b0 >= nb: return np.zeros((0, 3), np.int64), np.zeros(0)
    vmin = vtol + (2.0-vtol)*b0/nb
    counts = hist[:, b0:].sum(1).astype(np.int64); offs = np.r_[0, np.cumsum(counts)[:-1]].astype(np.int64)
    out = np.zeros((counts.sum(), 3), np.int64); val = np.zeros(counts.sum())
    sep1_collect(indptr, indices, V, thresh, maxh, maxk, hub2, cap2, vmin, nchunk, counts, offs, out, val, np.arange(nchunk, dtype=np.int64))
    keep = val > 0; return out[keep], val[keep]

@njit(parallel=True, cache=True)
def _hubpairs_hist(H, V, d, vtol, nb, wmax):
    h = len(H); k = V.shape[1]; hist = np.zeros((h, nb), np.int64)
    for a in prange(h):
        i = H[a]
        for b in range(a+1, h):
            j = H[b]; x = 0.0
            for q in range(k): x += V[i, q]*V[j, q]
            if x < -vtol:
                w = -x*d[i]*d[j]; hist[a, min(int(w/wmax*nb), nb-1)] += 1
    return hist

@njit(parallel=True, cache=True)
def _hubpairs_collect(H, V, d, vtol, wmin, counts, offs, out, val):
    h = len(H); k = V.shape[1]
    for a in prange(h):
        i = H[a]; m = offs[a]; end = offs[a] + counts[a]
        for b in range(a+1, h):
            j = H[b]; x = 0.0
            for q in range(k): x += V[i, q]*V[j, q]
            if x < -vtol:
                w = -x*d[i]*d[j]
                if w >= wmin and m < end: out[m, 0] = min(i, j); out[m, 1] = max(i, j); val[m] = w; m += 1

def separate_pairs(V, d, nhub, cap, vtol=1e-3, nb=4000):
    """Most negative d_i d_j X_ij among all pairs of the nhub highest-volume vertices (pair nonnegativity cuts)."""
    H = np.argsort(-d)[:nhub].astype(np.int64); wmax = float(d[H[0]]*d[H[1]]*2.0) + 1e-12
    hist = _hubpairs_hist(H, V, d, vtol, nb, wmax); cum = np.cumsum(hist.sum(0)[::-1])[::-1]
    ok = np.flatnonzero(cum <= cap); b0 = ok[0] if len(ok) else nb
    if b0 >= nb: return np.zeros((0, 2), np.int64)
    wmin = wmax*b0/nb; counts = hist[:, b0:].sum(1).astype(np.int64); offs = np.r_[0, np.cumsum(counts)[:-1]].astype(np.int64)
    out = np.zeros((counts.sum(), 2), np.int64); val = np.zeros(counts.sum())
    _hubpairs_collect(H, V, d, vtol, wmin, counts, offs, out, val)
    return out[val > 0]


# ----------------------------------------------------------------------------- args / data
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--tag", required=True); p.add_argument("--pair_labels", nargs="*", default=[])
p.add_argument("--resume", default=""); p.add_argument("--init_V", default="")
p.add_argument("--init_partition", default="", help="labels .npy: start V at a colouring-embedded partition matrix")
p.add_argument("--k", type=int, default=32); p.add_argument("--seed", type=int, default=0)
p.add_argument("--save", required=True); p.add_argument("--save_last", type=int, default=1)
p.add_argument("--outer", type=int, default=10**9); p.add_argument("--inner", type=int, default=3)
p.add_argument("--nbatch", type=int, default=64, help="batches per sweep (aggregates refreshed between batches)")
p.add_argument("--damp", type=float, default=0.0)
p.add_argument("--time_limit", type=float, default=3600); p.add_argument("--rho", type=float, default=0.02)
p.add_argument("--tri_weight", type=float, default=1.0)
p.add_argument("--tri", type=int, default=1); p.add_argument("--tri_add", type=int, default=500000)
p.add_argument("--tri_every", type=int, default=3); p.add_argument("--tri_max", type=int, default=5000000)
p.add_argument("--tri_thresh", type=float, default=0.3); p.add_argument("--tri_maxh", type=int, default=100)
p.add_argument("--est_every", type=int, default=100); p.add_argument("--est_mix", type=int, default=20)
p.add_argument("--eig_tol", type=float, default=1e-4); p.add_argument("--init_mix", type=int, default=0)
p.add_argument("--final_polish", type=float, default=0); p.add_argument("--final_batch", type=int, default=20)
p.add_argument("--lb", type=float, default=float("nan"))
p.add_argument("--max_clusters", type=int, default=0, help="per level: keep the largest-volume clusters, merge the rest into one (any grouping is a valid aggregate)")
p.add_argument("--pair_refresh", type=int, default=1, help="refresh the cluster-pair state (H, p, M) every R batches (cluster sums Y and z are always current)")
p.add_argument("--shift_p", type=int, default=8, help="estimate: per-vertex shift from this many lowest eigenpairs (0 = uniform n*|lmin|)")
p.add_argument("--extra_tri", default="", help="npz with TI (a,centre,c) triangles to merge into the pool at start (lam 0)")
p.add_argument("--tri1_add", type=int, default=0, help="one-edge-arm triangles added per separation round (0 = off)")
p.add_argument("--tri1_maxk", type=int, default=60); p.add_argument("--tri1_hub", type=int, default=8); p.add_argument("--tri1_cap", type=int, default=200)
p.add_argument("--eig_maxiter", type=int, default=300)
p.add_argument("--est_damp", type=float, default=0.5, help="damping of the fixed-multiplier polishing sweeps in estimates (no penalty curvature there)")
p.add_argument("--nbr_cut", type=int, default=0, help="dualize sum_{u in N(v)} X_vu >= 1 (every vertex shares its community with a neighbour)")
p.add_argument("--nbr_weight", type=float, default=1.0)
p.add_argument("--extra_conn", default="", help="npz with CI (i,j,s) connectivity cuts X_ij <= X_is (s separates i,j)")
p.add_argument("--conn_weight", type=float, default=1.0)
p.add_argument("--diag_out", default="", help="with --outer 0: polish, then dump per-vertex dual terms and partition terms (no eigen step) and exit")
p.add_argument("--diag_labels", default="")
p.add_argument("--extra_pairs", default="", help="npz with PI (i,j) pair nonnegativity cuts X_ij >= 0")
p.add_argument("--pair_weight", type=float, default=1.0)
p.add_argument("--pair_add", type=int, default=0, help="pair nonnegativity cuts added per separation round (hub-pair scan)")
p.add_argument("--pair_hubs", type=int, default=20000)
p.add_argument("--pair_max", type=int, default=5000000)
p.add_argument("--ne_split", type=int, default=0, help="aggregated nonnegativity over NON-EDGE pairs only (edges excluded from each cluster-pair sum)")
p.add_argument("--pairlvl_weight", type=float, default=1.0, help="scale of the cluster-pair penalty/step weights r_QR")
p.add_argument("--mom", type=float, default=0.0, help="max Nesterov momentum on the multiplier (prox) updates; 0 = off; adaptive restart when L rises")
# ---- merged from tol5/algo/alm_big.py (all default off)
p.add_argument("--est_escape", type=int, default=0, help="estimates: rank escape along this many negative eigenvectors of Diag(gd)-C before the next polish")
p.add_argument("--est_only", type=int, default=0, help="resume, run this many estimate rounds (polish est_mix sweeps + eig), print, exit")
p.add_argument("--sep_polish", type=int, default=0, help="separate triangles (and pair cuts) on a copy of V polished this many plain-mixing sweeps at the current multipliers (0 = on V)")
p.add_argument("--dual_step", type=float, default=0.0, help="with --sep_polish: extra projected subgradient step y <- [y - eta*rho*r*g(W)]_+ at the polished W")
p.add_argument("--dual_mode", default="full", help="full | tri_shrink (only shrink triangle multipliers with slack at W, step dual_step)")
p.add_argument("--init_sig", default="", help="npz: warm-start pair multipliers sig (multilevel / oracle test)")
p.add_argument("--init_tri", type=int, default=0, help="with --init_sig: also take TI/lam from that npz")
p.add_argument("--ls_etas", default=""); p.add_argument("--ls_mode", default="shrink,full"); p.add_argument("--ls_what", default="both")
p.add_argument("--ls_rounds", type=int, default=3)
p.add_argument("--ls_add", default=""); p.add_argument("--ls_add_n", type=int, default=200000)
p.add_argument("--ckpt_every", type=int, default=0, help="save <save>_last.npz (V, multipliers) every N outers without an estimate")
p.add_argument("--slow_bk", type=int, default=0, help="1 = MAIN's original per-batch cluster-pair bookkeeping (reference / A-B only)")
# ---- V2: ported from mech/alm_mech.py (all default off)
p.add_argument("--pw_auto", type=float, default=0.0, help=">0: graph-adaptive pair weight w = c/R(1): R = volume-weighted mean ratio (pair-penalty row curvature, trace rule, all pairs active) / |objective row gradient| at the start V (alm_mech semantics: w = pairlvl_weight * c / R(pairlvl_weight))")
p.add_argument("--pw_rule", default="", help="'c1,c2' (or 'default' = 0.1,0.05): w = min(c1/R(1), c2/(rho n^2/L)), n = vertices of the (reduced) instance, L = original 2m")
p.add_argument("--phase2_at", type=float, default=0.0, help=">0: from the first outer starting after this many seconds (since data load): proximal-point phase, inner=phase2_inner sweeps per multiplier update at pair weight phase2_w (x computed weight with pw_auto/pw_rule)")
p.add_argument("--phase2_at_it", type=int, default=0, help=">0: same switch after this many completed outers (outer phase2_at_it+1 is the first phase-2 outer)")
p.add_argument("--phase2_inner", type=int, default=300); p.add_argument("--phase2_w", type=float, default=0.2)
p.add_argument("--phase2_tri_every", type=int, default=1); p.add_argument("--phase2_est_every", type=int, default=1)
p.add_argument("--kq_mode", default="trace", help="trace: row curvature = sum_R r||Y_R||^2 (default) | spec: top eigenvalue of sum_R r Y_R Y_R^T")
p.add_argument("--pair_nested", nargs=2, action="append", default=[], metavar=("FINE", "PARENT"),
               help="NESTED: block-diagonal fine pair level (pairs of fine clusters with the same parent only); repeatable")
p.add_argument("--vc_labels", nargs="*", default=[], help="VC: vertex-cluster nonnegativity levels (labels .npy, max_clusters cap applies)")
p.add_argument("--vc_weight", type=float, default=1.0, help="VC: r_iR = vc_weight * pairlvl_weight * rq_R / d_i^2")
p.add_argument("--vc_max_clusters", type=int, default=-1, help="VC: cluster cap per VC level (-1 = --max_clusters); memory ~4 n x P float64 per level")
p.add_argument("--vc_kap2", type=float, default=1.0, help="VC: factor on the transposed-term curvature (1 = trace rule)")
p.add_argument("--vcs_labels", nargs="*", default=[], help="VCS: sparse active-set VC levels (no cluster cap)")
p.add_argument("--vcs_full", nargs="*", default=[], help="VCS: VC levels with the working set = all entries (dense level in sparse code)")
p.add_argument("--vcs_every", type=int, default=10, help="VCS: separation every N outers")
p.add_argument("--vcs_first", type=int, default=2, help="VCS: first separation after this outer")
p.add_argument("--vcs_add", type=int, default=1000000, help="VCS: max entries added per separation round (per level)")
p.add_argument("--vcs_max", type=int, default=20000000, help="VCS: max working-set size per level")
p.add_argument("--vcs_drop", type=int, default=5, help="VCS: drop entries with new sig = 0 and g > 0 for this many consecutive outers (0 = never)")
p.add_argument("--vcs_tol", type=float, default=1e-4, help="VCS: candidate if mean X over R minus i < -tol")
p.add_argument("--vcs_score", default="pen", help="VCS: pen (wv rq_R s^2 = r g^2) | mean (-xbar) | mass (-g)")
a = p.parse_args()
if a.pw_auto > 0 and a.pw_rule: raise SystemExit("--pw_auto and --pw_rule are exclusive")
if a.kq_mode not in ("trace", "spec"): raise SystemExit("bad kq_mode")

D = os.environ.get("DATA_DIR", "/Users/ashrafkhalil/Desktop/MIE 498 RT/output/decomp_audit_2026-09-28/data/")
if not D.endswith("/"): D += "/"
g_ = json.load(open(D+a.tag+"/graph.json")); n = g_["n"]
keys = np.load(D+a.tag+"/keys.npy"); u = keys//n; v = keys % n; m = len(keys); L = 2*m; del keys
d = np.bincount(np.r_[u, v], minlength=n).astype(float)
W_ = np.load(D+a.tag+"/weights.npy") if os.path.exists(D+a.tag+"/weights.npy") else np.ones(m)   # reduced instances: edge multiplicities
A = sp.csr_matrix((np.r_[W_, W_], (np.r_[u, v], np.r_[v, u])), shape=(n, n)); del u, v
C0 = 0.0      # constant part of 2m*Q (leaf reduction: every leaf edge is internal)
if os.path.exists(D+a.tag+"/vol.npy"):        # leaf-reduced instance (reduce_leaves.py): volumes, original L, constant
    meta_ = json.load(open(D+a.tag+"/meta.json")); d = np.load(D+a.tag+"/vol.npy").astype(float)
    L = int(meta_["L_orig"]); C0 = 2.0*meta_["leaf_edges"]; assert int(d.sum()) == L
deg_r = np.asarray(A.sum(1)).ravel()
nb_ok = (deg_r >= 1) & (np.abs(d - deg_r) < 0.5)        # original neighbourhood == reduced neighbourhood (no absorbed leaves)
mu = np.zeros(n)
A.sort_indices(); A_indptr = A.indptr.astype(np.int64); A_indices = A.indices.astype(np.int64); A_data = A.data.astype(np.float64)
t0 = time.time()
import resource
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform == "darwin" else 2**10)
def log(*s): print(f"[{time.time()-t0:8.0f}s {rss():6.0f}MB]", *s, flush=True)

z0 = np.load(a.resume) if a.resume else None
levels = []
if z0 is not None and not a.pair_labels:
    for li in range(int(z0["nlev"])):
        assert int(z0[f"pair{li}"]), "only cluster-pair levels are supported"
        if f"nested{li}" in z0.files and int(z0[f"nested{li}"]):        # NESTED level: fine labels + parent block per cluster
            lab_ = z0[f"lab{li}"].astype(np.int64); levels.append(NestedLevel(lab_, z0[f"blk{li}"].astype(np.int64)[lab_], d))
        else: levels.append(PairLevel(z0[f"lab{li}"], d))
else:
    for f in a.pair_labels:
        lab_ = np.unique(np.load(f), return_inverse=True)[1]
        if a.max_clusters and lab_.max()+1 > a.max_clusters:
            vol_ = np.bincount(lab_, weights=d); keep_ = np.argsort(-vol_)[:a.max_clusters-1]
            mp_ = np.full(len(vol_), a.max_clusters-1); mp_[keep_] = np.arange(len(keep_)); lab_ = mp_[lab_]
            log(f"level {f.split('/')[-1]}: capped to {a.max_clusters} clusters (merged volume share {1-vol_[keep_].sum()/vol_.sum():.4f})")
        levels.append(PairLevel(lab_, d))
    for ff, fp in a.pair_nested:                     # NESTED levels after the dense ones
        Lv_ = NestedLevel(np.load(ff), np.load(fp), d); levels.append(Lv_)
        log(f"nested level {ff.split('/')[-1]} | {fp.split('/')[-1]}: {Lv_.P} fine clusters in {Lv_.nb} parents, max block {Lv_.pc.max()}, "
            f"sum Pc^2 {Lv_.S} (dense P^2 {Lv_.P**2}), big blocks (>= {len(Lv_.big) and Lv_.pc[Lv_.big].min()}) {len(Lv_.big)}")
NEST = [bool(getattr(Lv, "nested", False)) for Lv in levels]
def _cap(lab_, nm):
    lab_ = np.unique(lab_, return_inverse=True)[1]; mc_ = a.max_clusters if a.vc_max_clusters < 0 else a.vc_max_clusters
    if mc_ and lab_.max()+1 > mc_:
        vol_ = np.bincount(lab_, weights=d); keep_ = np.argsort(-vol_)[:mc_-1]
        mp_ = np.full(len(vol_), mc_-1); mp_[keep_] = np.arange(len(keep_)); lab_ = mp_[lab_]
        log(f"vc level {nm}: capped to {mc_} clusters (merged volume share {1-vol_[keep_].sum()/vol_.sum():.4f})")
    return lab_
vcl = []
if a.vc_labels: vcl = [VCO.VCLevel(_cap(np.load(f), f.split('/')[-1]), d) for f in a.vc_labels]
elif z0 is not None and "nvc" in z0.files: vcl = [VCO.VCLevel(z0[f"vclab{m}"], d) for m in range(int(z0["nvc"]))]
if z0 is not None and "nvc" in z0.files:
    for m, Lv in enumerate(vcl):
        if m < int(z0["nvc"]) and z0[f"vcsig{m}"].shape == Lv.sig.shape: Lv.sig = z0[f"vcsig{m}"].astype(np.float64); log(f"resumed vc sig {m} sum {Lv.sig.sum():.4g}")
for Lv in vcl: log(f"VC level: P {Lv.P}, multipliers {Lv.sig.shape} ({Lv.sig.nbytes/2**20:.0f} MB each)")
if vcl and (a.mom > 0 or a.dual_step > 0 or a.ls_etas or a.ls_add or os.environ.get("EST_DIAG") or a.slow_bk or a.pair_refresh != 1 or os.environ.get("ALM_CANON_SWEEPS", "0") == "1"):
    raise SystemExit("VC levels: --mom, --dual_step, --ls_*, EST_DIAG, canonical sweeps not supported")
def wvc(): return float(a.vc_weight)*float(a.pairlvl_weight)
VCS_EXACT = os.environ.get("VCS_EXACT", "0") == "1"
vcsl = []
if a.vcs_labels or a.vcs_full:
    vcsl = [VCS.VCSLevel(np.load(f), d, full=False, name=f.split('/')[-1]) for f in a.vcs_labels] + \
           [VCS.VCSLevel(np.load(f), d, full=True, name=f.split('/')[-1]) for f in a.vcs_full]
if z0 is not None and "nvcs" in z0.files:
    if not vcsl: vcsl = [VCS.VCSLevel(z0[f"vcslab{m}"], d, full=False) for m in range(int(z0["nvcs"]))]
    for m, Lv in enumerate(vcsl):
        if m < int(z0["nvcs"]) and np.array_equal(Lv.lab, np.unique(z0[f"vcslab{m}"], return_inverse=True)[1]):
            Lv.full = bool(int(z0[f"vcsfull{m}"])); Lv.set_ws(z0[f"vcsptr{m}"], z0[f"vcscol{m}"], z0[f"vcssig{m}"].astype(np.float64))
            log(f"resumed vcs level {m}: nnz {Lv.nnz} sig sum {Lv.sig.sum():.4g}")
for Lv in vcsl: log(f"VCS level {Lv.name}: P {Lv.P} {'FULL' if Lv.full else 'active-set'} nnz {Lv.nnz} ({Lv.nbytes()/2**20:.0f} MB)")
if vcsl and (a.mom > 0 or a.dual_step > 0 or a.ls_etas or a.ls_add or os.environ.get("EST_DIAG") or a.slow_bk or a.pair_refresh != 1 or os.environ.get("ALM_CANON_SWEEPS", "0") == "1"):
    raise SystemExit("VCS levels: --mom, --dual_step, --ls_*, EST_DIAG, canonical sweeps not supported")
NTH = int(numba.get_num_threads())
if any(NEST):
    bad_ = [nm for nm, c in [("--ne_split", a.ne_split), ("--kq_mode spec", a.kq_mode == "spec"), ("--pair_refresh != 1", a.pair_refresh != 1),
                             ("--slow_bk", a.slow_bk), ("ALM_CANON_SWEEPS", os.environ.get("ALM_CANON_SWEEPS", "0") == "1"),
                             ("--dual_step", a.dual_step > 0), ("--ls_etas", bool(a.ls_etas)), ("EST_DIAG", bool(os.environ.get("EST_DIAG")))] if c]
    if bad_: raise SystemExit(f"nested levels do not support {bad_}")
for Lv in levels:
    cnt = np.bincount(Lv.lab, minlength=Lv.P).astype(float); vol = np.maximum(np.bincount(Lv.lab, weights=d, minlength=Lv.P), 1.0)
    rq = cnt/vol**2; Lv.rq = rq
    Lv.r = None if getattr(Lv, "nested", False) else np.outer(rq, rq)*a.pairlvl_weight
    log("pair level clusters", Lv.P, "(nested)" if getattr(Lv, "nested", False) else "")
TI = np.zeros((0, 3), np.int64); lam = np.zeros(0)
rng = np.random.default_rng(a.seed)
if z0 is not None:
    V = z0["V"].astype(np.float64); TI = z0["TI"].astype(np.int64); lam = z0["lam"].astype(np.float64)
    if "mu" in z0.files and z0["mu"].shape == (n,): mu = z0["mu"].astype(np.float64)
    for li, Lv in enumerate(levels):
        if f"sig{li}" in z0.files and z0[f"sig{li}"].shape == Lv.sig.shape: Lv.sig = z0[f"sig{li}"].astype(np.float64)
    if len(TI):
        TI = np.c_[np.minimum(TI[:, 0], TI[:, 2]), TI[:, 1], np.maximum(TI[:, 0], TI[:, 2])]
        kk = (TI[:, 0]*n+TI[:, 1])*n+TI[:, 2]; uq, first, inv = np.unique(kk, return_index=True, return_inverse=True)
        if len(first) < len(TI): lam = np.bincount(inv, weights=lam); TI = TI[first]; log("merged duplicate triangles ->", len(TI))
    z0.close()
elif a.init_partition:
    lab = np.unique(np.load(a.init_partition), return_inverse=True)[1]; P = lab.max()+1
    # greedy colouring of the cluster graph: clusters sharing a coordinate should be non-adjacent (merging two
    # non-adjacent clusters costs only 2 vol_Q vol_R / L^2); among forced clashes pick the smallest vol product
    Cg = sp.csr_matrix((np.ones(A.nnz), (lab[np.repeat(np.arange(n), np.diff(A.indptr))], lab[A.indices])), shape=(P, P))
    vol = np.bincount(lab, weights=d, minlength=P); col = -np.ones(P, np.int64); cvol = np.zeros(a.k)
    for Q in np.argsort(-vol):
        nb_ = Cg.indices[Cg.indptr[Q]:Cg.indptr[Q+1]]; used = np.zeros(a.k, bool); used[col[nb_][col[nb_] >= 0]] = True
        free = np.flatnonzero(~used)
        col[Q] = free[np.argmin(cvol[free])] if len(free) else int(np.argmin(cvol))
        cvol[col[Q]] += vol[Q]
    V = np.zeros((n, a.k)); V[np.arange(n), col[lab]] = 1.0
    V += 1e-3*rng.standard_normal(V.shape)
    log(f"init partition: {P} clusters -> {a.k} colours, {int((col[lab][Cg.tocoo().row] == col[lab][Cg.tocoo().col]).sum())} clashing cluster adjacencies")
elif a.init_V:
    V = np.load(a.init_V).astype(np.float64)
else:
    V = rng.standard_normal((n, a.k))
V /= np.linalg.norm(V, axis=1, keepdims=True); V = aligned(np.ascontiguousarray(V)); k = V.shape[1]
if a.extra_tri:
    E_ = np.load(a.extra_tri)["TI"].astype(np.int64); E_ = np.c_[np.minimum(E_[:, 0], E_[:, 2]), E_[:, 1], np.maximum(E_[:, 0], E_[:, 2])]
    ke = (E_[:, 0]*n+E_[:, 1])*n+E_[:, 2]; kx = (TI[:, 0]*n+TI[:, 1])*n+TI[:, 2] if len(TI) else np.zeros(0, np.int64)
    _, fi = np.unique(ke, return_index=True); E_ = E_[fi]; ke = ke[fi]; E_ = E_[~np.isin(ke, kx)]
    TI = np.vstack([TI, E_]) if len(TI) else E_; lam = np.r_[lam, np.zeros(len(E_))]; log(f"merged {len(E_)} extra triangles -> {len(TI)}")
if a.init_sig:          # warm-start the cluster-pair multipliers (and optionally the triangle pool) from another run's npz
    zs_ = np.load(a.init_sig)
    for li, Lv in enumerate(levels):
        if f"sig{li}" in zs_.files and zs_[f"sig{li}"].shape == Lv.sig.shape: Lv.sig = zs_[f"sig{li}"].astype(np.float64); log(f"init sig level {li} sum {Lv.sig.sum():.3g}")
    if a.init_tri:
        TI = zs_["TI"].astype(np.int64); lam = zs_["lam"].astype(np.float64); log(f"init tri pool {len(TI)}")
tri_ptr, tri_codes = incidence(n, TI)
CI = np.zeros((0, 3), np.int64); om = np.zeros(0)
if z0 is not None and False: pass
if a.resume:
    _z = np.load(a.resume)
    if "CI" in _z.files: CI = _z["CI"].astype(np.int64); om = _z["om"].astype(np.float64)
    _z.close()
if a.extra_conn:
    E_ = np.load(a.extra_conn)["CI"].astype(np.int64)
    ke = (E_[:, 0]*n + E_[:, 1])*n + E_[:, 2]; kx = (CI[:, 0]*n + CI[:, 1])*n + CI[:, 2] if len(CI) else np.zeros(0, np.int64)
    _, fi = np.unique(ke, return_index=True); E_ = E_[fi]; E_ = E_[~np.isin(ke[fi], kx)]
    CI = np.vstack([CI, E_]) if len(CI) else E_; om = np.r_[om, np.zeros(len(E_))]; log(f"merged {len(E_)} connectivity cuts -> {len(CI)}")
con_ptr, con_codes = incidence(n, CI)
PI = np.zeros((0, 2), np.int64); psi = np.zeros(0)
if a.resume:
    _z = np.load(a.resume)
    if "PI" in _z.files: PI = _z["PI"].astype(np.int64); psi = _z["psi"].astype(np.float64)
    _z.close()
if a.extra_pairs:
    E_ = np.load(a.extra_pairs)["PI"].astype(np.int64); E_ = np.c_[np.minimum(E_[:, 0], E_[:, 1]), np.maximum(E_[:, 0], E_[:, 1])]
    ke = E_[:, 0]*n + E_[:, 1]; kx = PI[:, 0]*n + PI[:, 1] if len(PI) else np.zeros(0, np.int64)
    _, fi = np.unique(ke, return_index=True); E_ = E_[fi]; E_ = E_[~np.isin(ke[fi], kx)]
    PI = np.vstack([PI, E_]) if len(PI) else E_; psi = np.r_[psi, np.zeros(len(E_))]; log(f"merged {len(E_)} pair cuts -> {len(PI)}")
pr_ptr, pr_codes = incidence2(n, PI)

nl = len(levels)
labs = np.ascontiguousarray(np.array([Lv.lab for Lv in levels], np.int64).reshape(nl, n))
Poff = np.r_[0, np.cumsum([Lv.P for Lv in levels])].astype(np.int64)
Wall = sp.vstack([Lv.W for Lv in levels]).tocsr() if nl else sp.csr_matrix((0, n))   # sumP x n, entries d_i
Pcnt = np.array([Lv.P for Lv in levels], np.int64)
Hoff = np.r_[0, np.cumsum([Lv.S if NEST[li] else Lv.P**2 for li, Lv in enumerate(levels)])].astype(np.int64)
# NESTED: per-cluster column range [clo, chi) (global ids) and ppbuf row base (entry (Q,R) at pbase[Q] + R) for pair_all
CLO = np.zeros(int(Poff[-1]), np.int64); CHI = np.zeros(int(Poff[-1]), np.int64); PBASE = np.zeros(int(Poff[-1]), np.int64)
for li, Lv in enumerate(levels):
    o0, o1 = int(Poff[li]), int(Poff[li+1]); Qg = np.arange(o0, o1, dtype=np.int64)
    if NEST[li]:
        lo_, hi_ = Lv.crange(); CLO[o0:o1] = o0 + lo_; CHI[o0:o1] = o0 + hi_; PBASE[o0:o1] = Hoff[li] + Lv.base - o0
    else:
        CLO[o0:o1] = o0; CHI[o0:o1] = o1; PBASE[o0:o1] = Hoff[li] + (Qg - o0)*(o1 - o0) - o0
ppbuf = np.zeros(int(Hoff[-1]))
eabuf = np.zeros(int(Hoff[-1])) if a.ne_split else np.zeros(1)   # fast path: frozen symmetrised edge aggregates EAs (same layout)
lw_all = np.full(nl, float(a.pairlvl_weight))                       # fast path: r_QR = (rq_Q rq_R) * pairlvl_weight
def set_pair_weight(w):
    """V2: change the pair weight everywhere it enters (r_QR = (rq_Q rq_R) * w): Lv.r (canonical sweeps, Kq, dual
    steps), lw_all (pair_all rows), a.pairlvl_weight (level_aug). Kq / pp rows are rebuilt every sweep / batch."""
    a.pairlvl_weight = float(w); lw_all[:] = float(w)
    for Lv in levels:
        if not getattr(Lv, "nested", False): Lv.r = np.outer(Lv.rq, Lv.rq)*float(w)
E_r = np.repeat(np.arange(n), np.diff(A_indptr)); E_c = A_indices; E_w = d[E_r]*d[E_c]       # ordered edge pairs, weight d_i d_j
def edge_agg_ref(Lv, V):
    """EA_QR = sum over ordered edge pairs (i in Q, j in R) of d_i d_j X_ij."""
    x = np.einsum("ij,ij->i", V[E_r], V[E_c])*E_w
    return sp.csr_matrix((x, (Lv.lab[E_r], Lv.lab[E_c])), shape=(Lv.P, Lv.P)).toarray()
_EX = [None]
def edge_agg(Lv, V):
    """edge_agg_ref without the two m x k gathers: x by a parallel numba pass, then the duplicates summed sequentially
    in CSR edge order (= the order scipy's coo->csr->dense uses), so only the k-term dot products round differently."""
    if _EX[0] is None: _EX[0] = np.empty(len(A_indices))
    x = _EX[0]; edge_vals(A_indptr, A_indices, E_w, np.ascontiguousarray(V), x)
    EA = np.zeros((Lv.P, Lv.P)); _ea_acc(A_indptr, A_indices, x, Lv.lab, EA); return EA
def edge_S_apply(X, sigs):
    """sum_l (edge part of S_l) X, edge part entries d_i d_j sig_l[Q(i),Q(j)] on edge pairs."""
    out = np.zeros_like(X)
    for Lv, s_ in zip(levels, sigs):
        Ae = sp.csr_matrix((E_w*s_[Lv.lab[E_r], Lv.lab[E_c]], (E_r, E_c)), shape=(n, n)); out += Ae@X
    return out

# ----------------------------------------------------------------------------- operators
def AX(X):
    X = np.ascontiguousarray(X, dtype=np.float64); out = np.empty_like(X); spmm(A_indptr, A_indices, A_data, X, out); return out
def BV_of(X): return AX(X) - np.outer(d, d@X)/L
def CV(X, sigs, w):
    out = BV_of(X)
    for Lv, s in zip(levels, sigs):
        Y = Lv.W@X
        if getattr(Lv, "nested", False): out += d[:, None]*nest_apply(Lv, s, Y)[Lv.lab] - (s[Lv.diag_idx][Lv.lab]*d**2)[:, None]*X
        else: out += d[:, None]*(s@Y)[Lv.lab] - (np.diag(s)[Lv.lab]*d**2)[:, None]*X
    for Lv in vcl: out += VCO.vc_apply(Lv, Lv.sig, X, d)
    for Lv in vcsl: out += VCS.vcs_apply(Lv, Lv.sig, X, d)
    if len(w): tri_apply(TI, w, tri_ptr, tri_codes, np.ascontiguousarray(X), out)
    if a.nbr_cut and mu.any(): out += 0.5*(mu[:, None]*(A@X) + A@(mu[:, None]*X))
    if len(om): conn_apply(CI, om, con_ptr, con_codes, np.ascontiguousarray(X), out)
    if len(psi): pair_apply(PI, psi, pr_ptr, pr_codes, np.ascontiguousarray(X), out)
    if a.ne_split and nl: out -= edge_S_apply(X, [0.5*(s_+s_.T) for s_ in sigs])
    return out
def pairH(Lv, Y):
    H = Y@Y.T; H[np.diag_indices(Lv.P)] -= Lv.dsq; return 0.5*(H+H.T)
def levelH(Lv, X):
    """constraint values h_QR(X) of a level (non-edge part only with --ne_split), as in augmented()."""
    H = pairH(Lv, Lv.W@X)
    if a.ne_split: EA = edge_agg(Lv, X); H = H - 0.5*(EA + EA.T)
    return H

def _level_state(Lv, Y, rho, V=None):
    """Full (re)build of a level's per-sweep state: H, effective multipliers pp, M = pp Y (+ edge aggregate EA).
    Fast bookkeeping: the 4th entry is the SYMMETRISED edge aggregate EAs = 0.5(EA+EA^T) (frozen within a sweep;
    0.5*(EA[T,:]+EA[:,T].T) == EAs[T,:] bit for bit, so per batch it is a contiguous row gather); it is only
    built when rho > 0 (plain-mixing sweeps never read it)."""
    if rho > 0:
        EAs = None
        if a.ne_split and V is not None: EA = edge_agg_ref(Lv, V); EAs = 0.5*(EA + EA.T); del EA
        H = pairH(Lv, Y)
        if EAs is not None: H = H - EAs
        pp = np.maximum(0.0, Lv.sig - rho*Lv.r*H)
        if not a.slow_bk: H = None                      # H is never read inside a sweep (fast bookkeeping)
    else: H = None; pp = Lv.sig.copy(); EAs = None
    return [H, pp, pp@Y, EAs]

def kq_spec(act, Y, rho):
    """V2 --kq_mode spec (alm_mech): 2 rho lambda_max(sum_R act_QR Y_R Y_R^T + act_QQ Y_Q Y_Q^T) per Q."""
    P, kk = Y.shape; YY = np.einsum("rk,rl->rkl", Y, Y)
    G = (act@YY.reshape(P, kk*kk)).reshape(P, kk, kk) + np.diag(act)[:, None, None]*YY
    return 2*rho*np.linalg.eigvalsh(G)[:, -1]
def _kq(Lv, st, Y, rho):
    if rho <= 0: return np.zeros(Lv.P)
    act = (st[1] > 0)*Lv.r
    if a.kq_mode == "spec": return kq_spec(act, Y, rho)
    y2 = (Y*Y).sum(1); return 2*rho*(act@y2 + np.diag(act)*y2)

def nest_kq(Lv, Y, rho, out): _nest_kq(Lv, Y, rho, float(a.pairlvl_weight), out, kq_level)
def nest_aug(Lv, Y, rho): return _nest_aug(Lv, Y, rho, float(a.pairlvl_weight), level_aug)

TPROF = [0.0]
TVC = [0.0, 0.0, 0.0, 0.0]        # VCS time: sweep state, pre, post, augmented
CANON_SWEEPS = os.environ.get("ALM_CANON_SWEEPS", "0") == "1"      # A/B: force the canonical (merged) sweeps
_mark = [None]; _inc = [None, None]
def Ysum(V):
    """All levels' cluster sums Wall @ V (numba, parallel per-thread partials)."""
    out = np.zeros((int(Poff[-1]), V.shape[1]))
    if nl: cluster_sums(labs, Poff, d, V, int(min(16, numba.get_num_threads())), out)
    return out

def sweeps(V, nsw, rho, rt, damp):
    """FAST sweeps (perf/): same iteration as sweeps_canon with pair_refresh = 1 -- batched-parallel GS with exact
    current cluster-pair aggregates before every batch. Instead of maintaining the full P x P state pp, M after each
    batch, the rows M[Q], pp_QQ (and with --ne_split the whole pp row Q, written into ppbuf) are computed fresh from
    the current Y for the clusters Q of the NEXT batch only (everything batch_step reads). Kq and the --ne_split edge
    aggregate EAs are frozen per sweep exactly as in sweeps_canon. --pair_refresh != 1 / --slow_bk 1 /
    ALM_CANON_SWEEPS=1 -> sweeps_canon."""
    if a.pair_refresh != 1 or a.slow_bk or CANON_SWEEPS: return sweeps_canon(V, nsw, rho, rt, damp)
    kk = V.shape[1]; z = d@V; sP = int(Poff[-1])
    if _mark[0] is None or len(_mark[0]) != sP: _mark[0] = np.zeros(sP, np.int64)
    mark = _mark[0]; stamp = int(mark.max()) if sP else 0
    Mall = np.zeros((sP, kk)); pdiag = np.zeros(sP); nes = int(bool(a.ne_split) and nl > 0)
    for _ in range(nsw):
        Ycat = Ysum(V)
        if nl:                                            # sig row-CSR, concatenated with global ids (one parallel region)
            scsr = [nest_sig_csr(Lv.blk_of, Lv.boff, Lv.base, Lv.sig, 0) if NEST[li_] else sig_csr(np.ascontiguousarray(Lv.sig)) for li_, Lv in enumerate(levels)]
            sg_ptr = np.r_[0, np.cumsum(np.concatenate([np.diff(c[0]) for c in scsr]))].astype(np.int64)
            sg_idx = np.concatenate([c[1] + Poff[li] for li, c in enumerate(scsr)]).astype(np.int64)
            sg_val = np.concatenate([c[2] for c in scsr]); rq_all = np.concatenate([Lv.rq for Lv in levels])
            dsq_all = np.concatenate([Lv.dsq for Lv in levels])
        EAl = [None]*nl
        if nes:
            for li, Lv in enumerate(levels):
                if rho > 0:                               # frozen symmetrised edge aggregate of this sweep
                    EA = edge_agg(Lv, V); EAl[li] = 0.5*(EA + EA.T); eabuf[Hoff[li]:Hoff[li+1]] = EAl[li].ravel()
                else:                                     # plain mixing: the kernel's edge correction uses pp = sig
                    ppbuf[Hoff[li]:Hoff[li+1]] = Lv.sig.ravel()
        order = rng.permutation(n).astype(np.int64)
        Kq = np.zeros(sP)                                 # step-size curvature: once per sweep (canonical _kq)
        if rho > 0:
            for li, Lv in enumerate(levels):
                if NEST[li]: nest_kq(Lv, Ycat[Poff[li]:Poff[li+1]], float(rho), Kq[Poff[li]:Poff[li+1]]); continue
                Y = Ycat[Poff[li]:Poff[li+1]]; H = pairH(Lv, Y)
                if EAl[li] is not None: H = H - EAl[li]
                if a.kq_mode == "spec": Kq[Poff[li]:Poff[li+1]] = kq_spec((Lv.sig - rho*Lv.r*H > 0)*Lv.r, Y, rho)
                else: kq_level(H, np.ascontiguousarray(Lv.sig), Lv.r, (Y*Y).sum(1), float(rho), Kq[Poff[li]:Poff[li+1]])
        if a.nbr_cut and rho > 0:
            rn = rho*a.nbr_weight; sn = np.einsum("ij,ij->i", AX(V), V) - 1.0; pn = np.where(nb_ok, np.maximum(0.0, mu - rn*sn), 0.0)
        else: rn = 0.0; pn = mu if a.nbr_cut else np.zeros(n)
        use_pn = bool(np.any(pn != 0.0))
        if _inc[0] is not tri_codes:                      # incidence-ordered triangle records (rebuilt with the pool)
            _inc[1] = np.empty((len(tri_codes), 3), np.int32 if n < 2**31 else np.int64); inc_abc(TI, tri_codes, _inc[1]); _inc[0] = tri_codes
        IABC = _inc[1]; LINC = np.empty(len(tri_codes)); inc_lam(tri_codes, lam, LINC)
        DD = np.empty((len(order)//max(1, a.nbatch) + 2, kk))
        Yvc = [VCO.vc_state(Lv, V, d, rho, wvc()) for Lv in vcl]
        tq_ = time.time(); Yvs = [VCS.vcs_state(Lv, V, d, rho, wvc(), VCS_EXACT) for Lv in vcsl]; TVC[0] += time.time() - tq_
        GX = np.zeros((DD.shape[0], kk)); KX = np.zeros(DD.shape[0]); GX1 = np.zeros_like(GX); KX1 = np.zeros_like(KX)
        usevc_ = bool(vcl) or bool(vcsl); y2d_ = np.zeros(1)
        for rows in np.array_split(order, a.nbatch):
            if usevc_:
                nr_ = len(rows); GX[:nr_] = 0.0; KX[:nr_] = 0.0; Vold_ = V[rows]
                for Lv, Yv in zip(vcl, Yvc):
                    VCO.vc_pre(rows, V, d, Lv.lab, Yv, np.einsum("ij,ij->i", Yv, Yv), Lv.sig, Lv.sigt, Lv.Z, Lv.cnt, Lv.rq, wvc(), float(rho), float(a.vc_kap2), GX1, KX1)
                    GX[:nr_] += GX1[:nr_]; KX[:nr_] += KX1[:nr_]
                tq_ = time.time()
                for Lv, Yv in zip(vcsl, Yvs):
                    VCS.vcs_pre(rows, V, d, Lv.lab, Yv, np.einsum("ij,ij->i", Yv, Yv), True, Lv.ptr, Lv.col, Lv.sig, Lv.stZ, Lv.own, Lv.Z, Lv.cnt, Lv.rq, wvc(), float(rho), float(a.vc_kap2), GX, KX)
                TVC[1] += time.time() - tq_
            if nl:
                stamp += 1; items, ilev = touched_all(rows, labs, Poff, mark, stamp)
                with numba.parallel_chunksize(2):
                    pair_all(items, ilev, Ycat, CLO, CHI, PBASE, sg_ptr, sg_idx, sg_val, rq_all, dsq_all, lw_all, float(rho), Mall, pdiag, int(nes and rho > 0), eabuf, ppbuf)
            tb_ = time.time()
            batch_step6(rows, V, A_indptr, A_indices, A_data, d, z, float(L), labs, Poff, Mall, pdiag, Kq, IABC, LINC, tri_ptr, tri_codes, float(rt), float(damp), pn, float(rn), CI, om, con_ptr, con_codes, float(rho*a.conn_weight), PI, psi, pr_ptr, pr_codes, float(rho*a.pair_weight), nes, ppbuf, Hoff, Pcnt, use_pn, 32, DD, usevc_, GX, KX)
            TPROF[0] += time.time() - tb_
            if len(rows) >= 4096: _acc_par(rows, DD, z, labs, Poff, Ycat)
            else: _acc_rows(rows, DD, z, labs, Poff, Ycat)
            for Lv, Yv in zip(vcl, Yvc):                  # VC: cluster sums, then Z / cnt / stored multipliers of moved rows
                np.add.at(Yv, Lv.lab[rows], DD[:len(rows)])
                SN = np.empty((len(rows), Lv.P)); VCO.vc_post_rows(rows, V, d, Lv.lab, Yv, Lv.sig, Lv.rq, wvc(), float(rho), SN)
                VCO.vc_post_Z(rows, V, Vold_, d, Lv.sigt, SN, Lv.Z, Lv.cnt); VCO.vc_store(rows, SN, Lv.sigt)
            tq_ = time.time()
            for Lv, Yv in zip(vcsl, Yvs):                 # VCS: cluster sums (batch order), then fused SN / Z / cnt / stZ update
                VCS.vcs_yupd(rows, Lv.lab, DD, Yv)
                VCS.vcs_post(rows, V, Vold_, d, Lv.lab, Yv, Lv.ptr, Lv.col, Lv.sig, Lv.stZ, Lv.rq, wvc(), float(rho), Lv.Z, Lv.cnt, *Lv.towner())
            TVC[2] += time.time() - tq_
    return V

def sweeps_canon(V, nsw, rho, rt, damp):
    """CANONICAL (alm_big_merged.py) sweeps, verbatim: nsw batched-parallel GS sweeps at multipliers (sig, lam), penalty rho (0 = plain mixing).
    Level state (Y, H, pp, M) is updated incrementally after each batch: only clusters T touched by the batch
    change, so the per-batch cost is O(|T| P k) instead of O(P^2 k); everything is rebuilt once per sweep."""
    z = d@V
    for _ in range(nsw):
        Ys = [Lv.W@V for Lv in levels]; sts = [_level_state(Lv, Y, rho, V) for Lv, Y in zip(levels, Ys)]
        for li_, st in enumerate(sts):                       # pp lives in the shared flat buffer (kernel reads it)
            view = ppbuf[Hoff[li_]:Hoff[li_+1]].reshape(Pcnt[li_], Pcnt[li_]); view[:] = st[1]; st[1] = view
        order = rng.permutation(n).astype(np.int64)
        Kq = np.concatenate([_kq(Lv, st, Y, rho) for Lv, st, Y in zip(levels, sts, Ys)]) if nl else np.zeros(0)   # step-size curvature: once per sweep
        Yref = [Y.copy() for Y in Ys]; touched = []; bcount = 0
        if a.nbr_cut and rho > 0:
            rn = rho*a.nbr_weight; sn = np.einsum("ij,ij->i", A@V, V) - 1.0; pn = np.where(nb_ok, np.maximum(0.0, mu - rn*sn), 0.0)
        else: rn = 0.0; pn = mu if a.nbr_cut else np.zeros(n)
        for rows in np.array_split(order, a.nbatch):
            Mall = np.ascontiguousarray(np.concatenate([st[2] for st in sts])) if nl else np.zeros((0, k))
            pdiag = np.concatenate([np.diag(st[1]).copy() for st in sts]) if nl else np.zeros(0)
            Vold = V[rows]
            tb_ = time.time()
            batch_step(rows, V, A_indptr, A_indices, A_data, d, z, float(L), labs, Poff, Mall, pdiag, Kq, TI, lam, tri_ptr, tri_codes, float(rt), float(damp), pn, float(rn), CI, om, con_ptr, con_codes, float(rho*a.conn_weight), PI, psi, pr_ptr, pr_codes, float(rho*a.pair_weight), int(a.ne_split), ppbuf, Hoff, Pcnt)
            TPROF[0] += time.time() - tb_
            dd = d[rows][:, None]*(V[rows] - Vold)
            z += dd.sum(0); bcount += 1
            for Lv, Y in zip(levels, Ys): np.add.at(Y, Lv.lab[rows], dd)
            touched.append(rows)
            if bcount % a.pair_refresh and bcount < a.nbatch: continue
            rr_ = np.concatenate(touched); touched.clear()
            for Lv, Y, st, Yr in zip(levels, Ys, sts, Yref):
                T = np.unique(Lv.lab[rr_]); dYT = Y[T] - Yr[T]
                YoT = Yr[T].copy(); Yr[T] = Y[T]
                if rho > 0 and a.slow_bk:                     # MAIN's original bookkeeping (reference)
                    H, pp, M, EAs = st
                    HT = Y[T]@Y.T; HT[np.arange(len(T)), T] -= Lv.dsq[T]
                    if EAs is not None: HT -= EAs[T, :]             # edge part frozen within a sweep
                    H[T, :] = HT; H[:, T] = HT.T
                    oldcols = pp[:, T].copy()
                    pT = np.maximum(0.0, Lv.sig[T, :] - rho*Lv.r[T, :]*HT)
                    pp[T, :] = pT; pp[:, T] = pT.T
                    M += pp[:, T]@Y[T] - oldcols@YoT
                    M[T] = pp[T, :]@Y
                elif rho > 0:
                    # (fast bookkeeping, from algo/) H is never read inside a sweep -> not maintained; pp is symmetric,
                    # so the old columns are the old rows (contiguous gather) and the M update uses transposed BLAS
                    # operands. pp is the VIEW into ppbuf: it is only updated in place (the ne_split kernel reads it).
                    _, pp, M, EAs = st
                    YT = Y[T]
                    HT = YT@Y.T; HT[np.arange(len(T)), T] -= Lv.dsq[T]
                    if EAs is not None: HT -= EAs[T, :]             # edge part frozen within a sweep
                    oldrows = pp[T, :]                                # fancy index -> copy
                    pT = np.maximum(0.0, Lv.sig[T, :] - rho*Lv.r[T, :]*HT)   # r includes --pairlvl_weight
                    pp[T, :] = pT; pp[:, T] = pT.T
                    M += pT.T@YT - oldrows.T@YoT
                    M[T] = pT@Y
                else:
                    st[2] += Lv.sig[:, T]@dYT
    return V

def augmented(V, rho):
    """Canonical augmented() with the perf/ fused numba passes (quadA, level_aug, tri_aug); per level it returns the
    violation count #(h < -1e-9) instead of the H matrix (the only use of H in the main loop)."""
    zz = d@V; val0 = quadA(A_indptr, A_indices, A_data, V) - float(zz@zz)/L; val = val0; Hs = []; ps = []
    Yall = Ysum(V)
    for li, Lv in enumerate(levels):
        if NEST[li]:
            tot, nv, pp = nest_aug(Lv, Yall[Poff[li]:Poff[li+1]], float(rho)); val -= tot; Hs.append(nv); ps.append(pp); continue
        Y = Yall[Poff[li]:Poff[li+1]]; H = Y@Y.T; H[np.diag_indices(Lv.P)] -= Lv.dsq; pp = np.empty_like(H)
        if a.ne_split: EA = edge_agg(Lv, V); EAs = 0.5*(EA + EA.T); del EA
        else: EAs = np.zeros((1, 1))
        tot, nv = level_aug(H, np.ascontiguousarray(Lv.sig), Lv.rq, float(a.pairlvl_weight), float(rho), pp, int(bool(a.ne_split)), EAs)
        val -= tot; Hs.append(nv); ps.append(pp)
    rt = rho*a.tri_weight; gt = np.empty(len(TI)); pt = np.empty(len(TI))
    if len(TI): tot, _ = tri_aug(TI, V, lam, float(rt), gt, pt); val -= tot
    global pn_new, pc_new, pp_new, vc_new, vc_nv, vcs_new, vcs_nv
    vcs_new = []; vcs_nv = 0; tq_ = time.time()
    for Lv in vcsl:
        Yv = np.ascontiguousarray(Lv.W@V); ppv = np.empty(Lv.nnz); slk = np.empty(Lv.nnz, np.uint8)
        if len(Lv.stZ) != Lv.nnz: Lv.stZ = np.empty(Lv.nnz)
        tot, nv = VCS.vcs_aug(Lv, V, d, Yv, wvc(), float(rho), ppv, slk, Lv.stZ); Lv.st_ok = None; val -= tot; vcs_new.append((ppv, slk)); vcs_nv += nv
    TVC[3] += time.time() - tq_
    vc_new = []; vc_nv = 0
    for Lv in vcl:
        Yv = np.ascontiguousarray(Lv.W@V); ppv = np.empty_like(Lv.sig)
        tot, nv = VCO.vc_aug(V, d, Lv.lab, Yv, Lv.sig, Lv.rq, wvc(), float(rho), ppv); val -= tot; vc_new.append(ppv); vc_nv += nv
    if len(PI):
        rp_ = rho*a.pair_weight; xv = pair_val(PI, V); pp_new = np.maximum(0.0, psi - rp_*xv)
        val -= float(((pp_new*pp_new - psi*psi)/(2*rp_)).sum())
    else: pp_new = psi
    if len(CI):
        rc = rho*a.conn_weight; cv = conn_val(CI, V); pc_new = np.maximum(0.0, om - rc*cv)
        val -= float(((pc_new*pc_new - om*om)/(2*rc)).sum())
    else: pc_new = om
    if a.nbr_cut:
        rn = rho*a.nbr_weight; sn = np.einsum("ij,ij->i", AX(V), V) - 1.0; pn_new = np.where(nb_ok, np.maximum(0.0, mu - rn*sn), 0.0)
        val -= float(((pn_new*pn_new - mu*mu)/(2*rn)).sum())
    return val + C0, Hs, gt, ps, pt, val0 + C0

EIGV = [None]          # last (eigenvalues, eigenvectors) of Diag(gd+t)-C: warm start for eigsh/LOBPCG, directions for --est_escape
def estimate(V, nmix=None):
    W = sweeps(V.copy(), a.est_mix if nmix is None else nmix, 0.0, 0.0, a.est_damp)
    W = W.astype(np.float32).astype(np.float64)
    sigs = [Lv.sig for Lv in levels]
    gd = np.einsum("ij,ij->i", CV(W, sigs, lam), W)
    # per-vertex shift t >= 0 with Diag(gd + t) - C ~ PSD:  q q^T <= Diag(|q| ||q||_1) for each negative eigenpair,
    # then a uniform remainder n*max(0, -lmin(M + Diag t)).  Cost sum(t) = sum_j |l_j| ||q_j||_1^2 << n|l_min| when localized.
    t = np.zeros(n)
    def mv(x, tt): x = x.reshape(-1); return (gd + tt)*x - CV(x.reshape(-1, 1), sigs, lam)[:, 0]
    for rnd in range(3 if a.shift_p > 0 else 0):
        op = sla.LinearOperator((n, n), matvec=lambda x, tt=t.copy(): mv(x, tt), dtype=float)
        ev, Q = sla.eigsh(op, k=a.shift_p, which="SA", tol=a.eig_tol, maxiter=20000, ncv=max(48, 3*a.shift_p))
        neg = ev < 0
        if not neg.any(): break
        for j in np.flatnonzero(neg):
            q = Q[:, j]; t += 1.02*abs(ev[j])*np.abs(q)*np.abs(q).sum()
    op = sla.LinearOperator((n, n), matvec=lambda x: mv(x, t), dtype=float)
    v0_ = EIGV[0][1][:, int(np.argmin(EIGV[0][0]))].copy() if (EIGV[0] is not None and EIGV[0][1].shape[0] == n) else None   # warm start (previous eigvec)
    EIGV[0] = None
    try:
        ev_, Q_ = sla.eigsh(op, k=max(1, a.est_escape), which="SA", tol=a.eig_tol, maxiter=a.eig_maxiter, ncv=32, v0=v0_)
        lmin = float(ev_.min()); EIGV[0] = (ev_, Q_)
    except sla.ArpackNoConvergence as e_:
        lmin = float(np.min(e_.eigenvalues)) if len(e_.eigenvalues) else float("nan")
        if len(e_.eigenvalues): EIGV[0] = (e_.eigenvalues, e_.eigenvectors)
        if not np.isfinite(lmin):        # fallback: a few LOBPCG iterations (Ritz value; an estimate like everything here)
            X0 = np.random.default_rng(1).standard_normal((n, 4))
            if v0_ is not None: X0[:, 0] = v0_
            ev_, Xl_ = sla.lobpcg(op, X0, largest=False, maxiter=60, tol=1e-3); lmin = float(np.min(ev_)); EIGV[0] = (ev_, Xl_)
        log("eigsh not converged; lmin estimate", lmin)
    return (C0 + lam.sum() - mu.sum() + gd.sum() + t.sum() + n*max(0.0, -lmin))/L, lmin, W, (C0 + lam.sum() - mu.sum() + gd.sum())/L

def escape(W):
    """append theta*q for each negative eigenvector q of Diag(gd)-C (Burer-Monteiro saddle escape), renormalize rows."""
    if EIGV[0] is None: return W
    ev_, Q_ = EIGV[0]; neg = np.flatnonzero(ev_ < 0)
    if not len(neg): return W
    cols = [0.3*Q_[:, j]/np.abs(Q_[:, j]).max() for j in neg]
    W = np.c_[W, np.array(cols).T]; W /= np.linalg.norm(W, axis=1, keepdims=True); return np.ascontiguousarray(W)

def save(path, W):
    dd = {"TI": TI, "lam": lam, "V": W.astype(np.float32), "nlev": nl, "mu": mu, "CI": CI, "om": om, "PI": PI, "psi": psi}
    for li, Lv in enumerate(levels):
        if NEST[li]:
            dd[f"lab{li}"] = Lv.lab; dd[f"sig{li}"] = nest_symmetrize(Lv, Lv.sig); dd[f"dense{li}"] = 0; dd[f"pair{li}"] = 1
            dd[f"nested{li}"] = 1; dd[f"blk{li}"] = Lv.blk_of
        else:
            dd[f"lab{li}"] = Lv.lab; dd[f"sig{li}"] = 0.5*(Lv.sig+Lv.sig.T); dd[f"dense{li}"] = 1; dd[f"pair{li}"] = 1
    dd["nvc"] = len(vcl)
    for m, Lv in enumerate(vcl): dd[f"vclab{m}"] = Lv.lab; dd[f"vcsig{m}"] = Lv.sig.astype(np.float32)
    dd["nvcs"] = len(vcsl)
    for m, Lv in enumerate(vcsl):
        dd[f"vcslab{m}"] = Lv.lab; dd[f"vcsptr{m}"] = Lv.ptr; dd[f"vcscol{m}"] = Lv.col; dd[f"vcssig{m}"] = Lv.sig.astype(np.float32); dd[f"vcsfull{m}"] = int(Lv.full)
    tmp = path.replace(".npz", ".tmp.npz"); np.savez(tmp, **dd); os.replace(tmp, path)

stop = [False]
def _stop(*_): stop[0] = True
signal.signal(signal.SIGTERM, _stop); signal.signal(signal.SIGINT, _stop)

# ----------------------------------------------------------------------------- diagnostic dump
if a.diag_out:
    W = sweeps(V.copy(), a.est_mix, 0.0, 0.0, a.est_damp)
    sigs = [Lv.sig for Lv in levels]; CVW = CV(W, sigs, lam); gd = np.einsum("ij,ij->i", CVW, W)
    tri_share = np.zeros(n)
    if len(lam): np.add.at(tri_share, TI.ravel(), np.repeat(lam/3.0, 3))
    BW = BV_of(W); gB = np.einsum("ij,ij->i", BW, W)                    # pure-objective part of the row value
    out = dict(gd=gd, gB=gB, tri_share=tri_share, mu=mu, d=d, C0=C0, L=L, deg_r=np.asarray(A.sum(1)).ravel())
    if a.diag_labels:
        lab = np.unique(np.load(a.diag_labels), return_inverse=True)[1]; P = lab.max()+1
        volc = np.bincount(lab, weights=d, minlength=P)
        Ain = np.asarray(sp.csr_matrix((A.data*(lab[A.indices] == np.repeat(lab, np.diff(A.indptr))), A.indices, A.indptr), shape=A.shape).sum(1)).ravel()
        out.update(q=Ain - d*volc[lab]/L, lab=lab)
    np.savez(a.diag_out, **out)
    log(f"diag: dual part (C0+sum lam+sum g)/L = {(C0 + lam.sum() - mu.sum() + gd.sum())/L:.6f}; objective-only sum gB/L = {gB.sum()/L:.6f}"
        + (f"; partition (C0+sum q)/L = {(C0 + out['q'].sum())/L:.6f}" if a.diag_labels else ""))
    sys.exit(0)

# ----------------------------------------------------------------------------- est_only diagnostics (from algo/)
if a.est_only:
    Wq = V.copy(); te = time.time()
    for r_ in range(a.est_only):
        ub, lmin, Wq, dv = estimate(Wq)
        log(f"est_only round {r_} EST {ub:.6f} (dv {dv:.6f}) lmin {lmin:.2e} k {Wq.shape[1]} [{time.time()-te:.0f}s]")
        if a.est_escape: Wq = escape(Wq)
        if os.environ.get("EST_DIAG"):
            new_, val_ = separate(A_indptr, A_indices, Wq, a.tri_thresh, a.tri_maxh, 1e-3, 10**8)
            kn_ = (new_[:, 0]*n+new_[:, 1])*n+new_[:, 2]; ex_ = (TI[:, 0]*n+TI[:, 1])*n+TI[:, 2]; inp = np.isin(kn_, ex_)
            gt_ = 1.0 - tri_tau(TI, Wq) if len(TI) else np.zeros(0)
            hv = [float(np.minimum(0, Lv.h(Wq)).sum()) for Lv in levels]
            log(f"   diag: sep(W) {len(new_)} viol tri (sum {val_.sum():.0f}, >0.1: {(val_>0.1).sum()}), of which in pool {inp.sum()}; "
                f"new-not-pooled viol sum {val_[~inp].sum():.0f}; pool viol>0.05 {(gt_<-0.05).sum()} sum {(-np.minimum(gt_,0)).sum():.0f}; "
                f"pool lam>0 {(lam>0).sum()} sum lam {lam.sum():.0f}; pair neg-h sums {['%.3g'%x for x in hv]}")
    if a.ls_etas:        # line search on d along the (projected) subgradient at the polished maximizer W
        sig0 = [Lv.sig.copy() for Lv in levels]; lam0 = lam.copy()
        Hs0 = [levelH(Lv, Wq) for Lv in levels]; gt0 = 1.0 - tri_tau(TI, Wq) if len(TI) else np.zeros(0)
        # multiplier payments at W: sum sig*h (pairs) and sum lam*(1-tau) (triangles), in units of Q
        log(f"   payments at W: pairs {[float((Lv.sig*H).sum()/L) for Lv, H in zip(levels, Hs0)]} tri {float((lam*gt0).sum()/L):.5f}")
        for mode in a.ls_mode.split(","):
            for eta in [float(x) for x in a.ls_etas.split(",")]:
                for Lv, s0, H in zip(levels, sig0, Hs0):
                    g_ = np.maximum(H, 0) if mode == "shrink" else (np.minimum(H, 0) if mode == "grow" else H)
                    Lv.sig = np.maximum(0.0, s0 - eta*Lv.r*g_)
                g_ = np.maximum(gt0, 0) if mode == "shrink" else (np.minimum(gt0, 0) if mode == "grow" else gt0)
                if a.ls_what == "pairs": lam = lam0.copy()
                else: lam = np.maximum(0.0, lam0 - eta*a.tri_weight*g_)
                if a.ls_what == "tri":
                    for Lv, s0 in zip(levels, sig0): Lv.sig = s0.copy()
                W2 = Wq.copy()
                for r_ in range(a.ls_rounds): ub, lmin, W2, dv = estimate(W2)
                log(f"   ls {mode} {a.ls_what} eta {eta:g}: EST {ub:.6f} dv {dv:.6f} lmin {lmin:.2e} sum lam {lam.sum():.0f}")
        for Lv, s0 in zip(levels, sig0): Lv.sig = s0
        lam = lam0
    if a.ls_add:        # add triangles violated at W with initial multiplier eta*viol (pool multipliers unchanged)
        TI0 = TI.copy(); lam0 = lam.copy()
        new_, val_ = separate(A_indptr, A_indices, Wq, a.tri_thresh, a.tri_maxh, 1e-3, int(a.ls_add_n*1.3))
        kn_ = (new_[:, 0]*n+new_[:, 1])*n+new_[:, 2]; ex_ = (TI0[:, 0]*n+TI0[:, 1])*n+TI0[:, 2]; nw = ~np.isin(kn_, ex_)
        new_, val_ = new_[nw], val_[nw]; o_ = np.argsort(-val_)[:a.ls_add_n]; new_, val_ = new_[o_], val_[o_]
        log(f"   add: {len(new_)} new W-violated triangles (viol mean {val_.mean():.3f})")
        for eta in [float(x) for x in a.ls_add.split(",")]:
            TI = np.vstack([TI0, new_]); lam = np.r_[lam0, eta*val_]; tri_ptr, tri_codes = incidence(n, TI)
            W2 = Wq.copy()
            for r_ in range(a.ls_rounds): ub, lmin, W2, dv = estimate(W2)
            log(f"   add eta {eta:g}: EST {ub:.6f} dv {dv:.6f} lmin {lmin:.2e} sum lam {lam.sum():.0f}")
    sys.exit(0)

# ----------------------------------------------------------------------------- main loop
if a.init_mix: V = sweeps(V, a.init_mix, 0.0, 0.0, a.damp)
def mobility_ratio(V, w=1.0):
    """(alm_mech) volume-weighted mean over rows of kappa_i / |g_i^B| at pair weight w (rho, all pairs active, trace
    rule): kappa_i = d_i^2 Kq_Q(i) summed over levels, g_i^B = (B V)_i (objective row gradient)."""
    gB = np.linalg.norm(A@V - np.outer(d, d@V)/L, axis=1); kap = np.zeros(n)
    for Lv in levels:
        if getattr(Lv, "nested", False):          # NESTED: block by block, same formula as the dense level per block
            Y = Lv.W@V; y2 = (Y*Y).sum(1); kq_ = np.empty(Lv.P)
            for b in range(Lv.nb):
                lo, hi = Lv.boff[b], Lv.boff[b+1]; r_ = np.outer(Lv.rq[lo:hi], Lv.rq[lo:hi])*w
                kq_[lo:hi] = 2*a.rho*(r_@y2[lo:hi] + np.diag(r_)*y2[lo:hi])
            kap += d**2*kq_[Lv.lab]; continue
        r_ = np.outer(Lv.rq, Lv.rq)*w; Y = Lv.W@V; y2 = (Y*Y).sum(1); kap += d**2*(2*a.rho*(r_@y2 + np.diag(r_)*y2))[Lv.lab]
    return float(np.sum(d*kap/np.maximum(gB, 1e-9))/d.sum())
W_AUTO = None                                     # computed weight (pw_auto / pw_rule): phase2_w is a factor on it
if (a.pw_auto > 0 or a.pw_rule) and nl:
    w0_ = float(a.pairlvl_weight); R0_ = mobility_ratio(V, w0_); R1_ = R0_/w0_
    if a.pw_auto > 0:
        W_AUTO = w0_*(a.pw_auto/R0_)
        log(f"pw_auto: mobility ratio at current weight {R0_:.4g} (R(1) {R1_:.6g}) -> pairlvl_weight {W_AUTO:.4g} ({W_AUTO:.6g})")
    else:
        pr_ = [] if a.pw_rule in ("default", "on", "1") else a.pw_rule.split(",")
        c1_ = float(pr_[0]) if len(pr_) > 0 and pr_[0] else 0.1; c2_ = float(pr_[1]) if len(pr_) > 1 and pr_[1] else 0.05
        n2L_ = n*float(n)/L; wm_ = c1_/R1_; ws_ = c2_/(a.rho*n2L_); W_AUTO = min(wm_, ws_)
        log(f"pw_rule c1 {c1_:g} c2 {c2_:g}: R(1) {R1_:.6g} (R at w {w0_:g}: {R0_:.4g}), n {n} L {L} n^2/L {n2L_:.4g} rho {a.rho:g} | "
            f"w_mob = c1/R {wm_:.6g}, w_step = c2/(rho n^2/L) {ws_:.6g} -> pairlvl_weight {W_AUTO:.6g} ({'mob' if wm_ <= ws_ else 'step'})")
    set_pair_weight(W_AUTO)
PH2 = [False]
log(f"[C0/L {C0/L:.6f}] n {n} m {m} k {k} levels {[Lv.P for Lv in levels]} ntri {len(lam)} rho {a.rho} nbatch {a.nbatch}")
hist = []; best = np.inf; rho = a.rho
Lc, Hs, gt, ps, pt, prim = augmented(V, rho)
log(f"start L {Lc/L:.6f} primal {prim/L:.6f}")
tsweep = 0.0; tsep = 0.0; test = 0.0; tvcs = [0.0]
tk = 1.0; Lprev = np.inf; prev_sig = None; prev_lam = None; anc_sig = None; anc_lam = None
for it in range(1, a.outer+1):
    if not PH2[0] and ((a.phase2_at > 0 and time.time()-t0 > a.phase2_at) or (a.phase2_at_it > 0 and it > a.phase2_at_it)):
        PH2[0] = True; w2_ = a.phase2_w*W_AUTO if W_AUTO is not None else a.phase2_w
        set_pair_weight(w2_); a.inner = a.phase2_inner; a.tri_every = a.phase2_tri_every; a.est_every = a.phase2_est_every
        log(f"PHASE2 at outer {it}: inner {a.inner}, pairlvl_weight {a.pairlvl_weight:.4g}, tri_every {a.tri_every}, est_every {a.est_every}")
    ts = time.time(); V = sweeps(V, a.inner, rho, rho*a.tri_weight, a.damp); tsweep += time.time()-ts
    Lc, Hs, gt, ps, pt, prim = augmented(V, rho)
    nviol = 0
    for Lv, H, pp in zip(levels, Hs, ps): nviol += int(H); Lv.sig = pp          # H = violation count (fused augmented)
    if a.nbr_cut: mu = pn_new
    om = pc_new
    psi = pp_new
    for Lv, pv in zip(vcl, vc_new): Lv.sig = pv
    for Lv, (pv, sk) in zip(vcsl, vcs_new):
        Lv.sig = pv; VCS.age_update(Lv.age, sk); Lv.st_ok = (float(rho), Lv.nnz)     # stZ (from augmented) valid for the next sweep
    vcs_msg = ""
    if vcsl and not stop[0] and it < a.outer and (it % a.vcs_every == 0 or it == a.vcs_first):
        for Lv in vcsl:
            if Lv.full: continue
            ts = time.time(); nz0 = Lv.nnz
            keep = (Lv.age < a.vcs_drop) if a.vcs_drop > 0 else np.ones(Lv.nnz, bool)
            room = max(0, a.vcs_max - int(keep.sum()))
            ni, nR, sc, stv = VCS.separate(Lv, V, d, wvc(), max(1, min(a.vcs_add, room)), a.vcs_tol, a.vcs_score)
            if room == 0: ni, nR = ni[:0], nR[:0]
            VCS.update_ws(Lv, keep, ni, nR); tvcs[0] += time.time() - ts
            vcs_msg += (f" vcs[{Lv.name}] nnz {nz0}->{Lv.nnz} (-{nz0-int(keep.sum())} +{len(ni)}) viol {stv['nviol']:.4g} (ws {stv['nviol_ws']:.4g}) "
                        f"mass/L2 {stv['mass']/L**2:.4g} (ws {stv['mass_ws']/L**2:.4g}) [{time.time()-ts:.1f}s]")
    lam = pt; tviol = float(max(0.0, -gt.min())) if len(gt) else 0.0
    anc_sig = [pp for pp in ps]; anc_lam = pt                       # un-extrapolated multipliers (used for estimates)
    sep_now = a.tri and it % a.tri_every == 0 and not stop[0]
    if a.mom > 0:
        if Lc > Lprev: tk = 1.0                                        # adaptive restart
        tk1 = 0.5*(1+math.sqrt(1+4*tk*tk)); beta = min(a.mom, (tk-1)/tk1); tk = tk1
        if prev_sig is not None and beta > 0:
            for Lv, pp, q in zip(levels, ps, prev_sig): Lv.sig = np.maximum(0.0, pp + beta*(pp - q))
            if not sep_now and prev_lam is not None and len(prev_lam) == len(pt): lam = np.maximum(0.0, pt + beta*(pt - prev_lam))
        prev_sig = ps; prev_lam = None if sep_now else pt
    Lprev = Lc
    msg = ""
    if sep_now:
        ts = time.time()
        Vs = sweeps(V.copy(), a.sep_polish, 0.0, 0.0, a.est_damp) if a.sep_polish > 0 else V   # separate on the Lagrangian maximizer (before the purge: incidence must match TI)
        keep = lam > 0; TI = TI[keep]; lam = lam[keep]
        room = max(0, a.tri_max - len(TI))
        if room:
            if a.dual_step > 0 and a.sep_polish > 0 and a.dual_mode == "full":      # projected subgradient step on d(y) at the (approx.) Lagrangian maximizer
                for Lv in levels:
                    Hs_ = levelH(Lv, Vs); Lv.sig = np.maximum(0.0, Lv.sig - a.dual_step*rho*Lv.r*Hs_)
            if a.tri1_add:          # fused one-pass separation (perf/): outputs identical to separate + separate1
                (new, val), (n1, v1) = separate_both(A_indptr, A_indices, Vs, a.tri_thresh, a.tri_maxh, a.tri1_maxk, a.tri1_hub, a.tri1_cap, 1e-3, int(min(a.tri_add, room)*1.3)+10, int(a.tri1_add*1.3)+10)
            else:
                new, val = separate(A_indptr, A_indices, Vs, a.tri_thresh, a.tri_maxh, 1e-3, int(min(a.tri_add, room)*1.3)+10)
            if a.tri1_add:
                if len(n1): o1 = np.argsort(-v1)[:a.tri1_add]; new = np.vstack([new, n1[o1]]); val = np.r_[val + 10.0, v1[o1]]   # keep both budgets
            if len(new):
                kn = (new[:, 0]*n+new[:, 1])*n+new[:, 2]; ex = np.sort((TI[:, 0]*n+TI[:, 1])*n+TI[:, 2])
                pos = np.searchsorted(ex, kn); isnew = np.ones(len(kn), bool); ok = pos < len(ex); isnew[ok] = ex[pos[ok]] != kn[ok]
                new, val, kn = new[isnew], val[isnew], kn[isnew]
                _, fi = np.unique(kn, return_index=True); new, val = new[fi], val[fi]
                o = np.argsort(-val)[:min(a.tri_add + a.tri1_add, room)]; new = new[o]
                TI = np.vstack([TI, new]); lam = np.r_[lam, np.zeros(len(new))]
            msg = f" +{len(new)}tri"
            if a.dual_step > 0 and a.sep_polish > 0 and len(TI):
                g_ = 1.0 - tri_tau(TI, Vs)
                if a.dual_mode == "tri_shrink": lam = np.maximum(0.0, lam - a.dual_step*np.maximum(g_, 0.0))   # absolute eta
                else: lam = np.maximum(0.0, lam - a.dual_step*rho*a.tri_weight*g_)
        tri_ptr, tri_codes = incidence(n, TI)
        if a.pair_add:
            keepp = psi > 0; PI = PI[keepp]; psi = psi[keepp]
            roomp = max(0, a.pair_max - len(PI))
            if roomp:
                newp = separate_pairs(Vs, d, min(a.pair_hubs, n), min(a.pair_add, roomp))
                if len(newp):
                    kn = newp[:, 0]*n + newp[:, 1]; kx = np.sort(PI[:, 0]*n + PI[:, 1]) if len(PI) else np.zeros(0, np.int64)
                    newp = newp[~np.isin(kn, kx)]
                    PI = np.vstack([PI, newp]) if len(PI) else newp; psi = np.r_[psi, np.zeros(len(newp))]
                msg += f" +{len(newp)}pairs"
            pr_ptr, pr_codes = incidence2(n, PI)
        tsep += time.time()-ts
    rec = dict(outer=it, t=round(time.time()-t0, 1), Laug=Lc/L, primal=prim/L, nviol=nviol, tviol=tviol, ntri=int(len(lam)))
    if vcsl: rec.update(vcs_nnz=[Lv.nnz for Lv in vcsl], vcs_viol=int(vcs_nv), vcs_sig=float(sum(float(Lv.sig.sum()) for Lv in vcsl)))
    if vcl: msg += f" vcviol {vc_nv} vcsig {sum(float(Lv.sig.sum()) for Lv in vcl):.4g}"
    if vcsl: msg += f" vcsviol {vcs_nv} vcssig {sum(float(Lv.sig.sum()) for Lv in vcsl):.4g} nnz {[Lv.nnz for Lv in vcsl]} act {[int((Lv.sig > 0).sum()) for Lv in vcsl]} [vcs sep {tvcs[0]:.0f}s state/pre/post/aug {TVC[0]:.0f}/{TVC[1]:.0f}/{TVC[2]:.0f}/{TVC[3]:.0f}s]" + vcs_msg
    line = f"it {it} L {Lc/L:.6f} prim {prim/L:.6f} viol {nviol} tviol {tviol:.3f} ntri {len(lam)}{msg} [sweep {tsweep:.0f}s (rowsteps {TPROF[0]:.0f}s) sep {tsep:.0f}s est {test:.0f}s]"
    last = stop[0] or it == a.outer or time.time()-t0 > a.time_limit
    if a.est_every > 0 and (it % a.est_every == 0 or last):
        ts = time.time(); cur_sig = [Lv.sig for Lv in levels]; cur_lam = lam
        if a.mom > 0 and len(anc_lam) == len(lam):
            for Lv, s_ in zip(levels, anc_sig): Lv.sig = s_
            lam = anc_lam
        if a.save_last: save(a.save.replace(".npz", "_last.npz"), V)        # checkpoint before the (slow) estimate
        ub, lmin, W, dv = estimate(V); test += time.time()-ts; rec.update(est=ub, dv=dv, lmin=lmin)
        if ub < best: best = ub; save(a.save, W); line += " *"
        if a.save_last: save(a.save.replace(".npz", "_last.npz"), W)
        for Lv, s_ in zip(levels, cur_sig): Lv.sig = s_
        lam = cur_lam
        line += f" EST {ub:.6f} (dv {dv:.6f}) best {best:.6f} lmin {lmin:.1e}"
    if (last and a.est_every <= 0) or (a.ckpt_every > 0 and it % a.ckpt_every == 0 and not (a.est_every > 0 and it % a.est_every == 0)):
        save(a.save.replace(".npz", "_last.npz"), V)               # checkpoint without an estimate (est_every 0 / --ckpt_every)
    log(line); hist.append(rec)
    if it % 20 == 0 or last: json.dump(hist, open(a.save.replace(".npz", ".history.json"), "w"))
    if last: break
if a.final_polish > 0:
    src = a.save.replace(".npz", "_last.npz") if a.save_last and os.path.exists(a.save.replace(".npz", "_last.npz")) else a.save
    zz = np.load(src); TI = zz["TI"].astype(np.int64); lam = zz["lam"]; V = aligned(zz["V"].astype(np.float64))
    for li, Lv in enumerate(levels): Lv.sig = zz[f"sig{li}"].astype(np.float64)
    for m, Lv in enumerate(vcl): Lv.sig = zz[f"vcsig{m}"].astype(np.float64)
    for m, Lv in enumerate(vcsl): Lv.full = bool(int(zz[f"vcsfull{m}"])); Lv.set_ws(zz[f"vcsptr{m}"], zz[f"vcscol{m}"], zz[f"vcssig{m}"].astype(np.float64))
    zz.close(); tri_ptr, tri_codes = incidence(n, TI); a.est_mix = a.final_batch; tp = time.time()
    log(f"final polish of {src}")
    while time.time() - tp < a.final_polish and not stop[0]:
        ub, lmin, W, dv = estimate(V); V = W
        if ub < best: best = ub; save(a.save, W)
        if a.est_escape: V = escape(V)
        log(f"final polish EST {ub:.6f} (dv {dv:.6f}) best {best:.6f} lmin {lmin:.1e}")
log(f"BEST {best:.6f}  (estimate only)")
