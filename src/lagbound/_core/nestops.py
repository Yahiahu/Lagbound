"""Nested (block-diagonal) cluster-pair level for alm_nested.py / est_dual_nested.py.

Fine labels f refine parent labels p (enforced by intersecting: fine cluster = (p, f) pair). Fine clusters are numbered
so that the clusters of one parent are contiguous (sorted by (parent, fine)); block b = parent b covers global-in-level
cluster ids boff[b] .. boff[b+1]-1, Pc = boff[b+1]-boff[b]. Constraints: h_QR >= 0 for Q, R in the SAME block only
(including Q = R). Multipliers are stored flat: block b is a dense Pc x Pc row-major matrix at sig[soff[b]:soff[b+1]];
entry (Q, R) of block b lives at sig[base[Q] + R] with base[Q] = soff[b] + (Q - boff[b]) Pc - boff[b].
Mathematically this is a dense fine level whose cross-block multipliers are fixed at 0 (so the dual bound stays valid for
any sig >= 0). Memory/compute sum_b Pc^2 instead of P^2.

Blocks with Pc >= BIG use the SAME numpy/BLAS formulas and dense numba kernels as alm_big_v2's dense levels on the block
(so a single-parent nested level reproduces a dense level bit for bit); smaller blocks use loop kernels below.
"""
import os
import numpy as np, scipy.sparse as sp
from numba import njit, prange

BIG = int(os.environ.get("ALM_NEST_BIG", "48"))


class NestedLevel:
    def __init__(self, fine, parent, d):
        fine = np.asarray(fine).astype(np.int64); parent = np.asarray(parent).astype(np.int64)
        n = len(fine); assert len(parent) == n
        pa = np.unique(parent, return_inverse=True)[1]
        fi = np.unique(fine, return_inverse=True)[1]
        key = pa*(int(fi.max()) + 1) + fi                   # intersect: (parent, fine) pairs, sorted by parent
        uk, lab = np.unique(key, return_inverse=True)
        self.n = n; self.lab = lab.astype(np.int64); self.P = int(len(uk)); self.d = d
        blk_of = (uk // (int(fi.max()) + 1)).astype(np.int64)      # parent of each fine cluster (non-decreasing)
        _, blk_of = np.unique(blk_of, return_inverse=True); self.blk_of = blk_of.astype(np.int64)
        self.nb = int(blk_of.max()) + 1
        pc = np.bincount(blk_of, minlength=self.nb).astype(np.int64)
        self.boff = np.r_[0, np.cumsum(pc)].astype(np.int64)
        self.soff = np.r_[0, np.cumsum(pc*pc)].astype(np.int64)
        Q = np.arange(self.P, dtype=np.int64); b = blk_of
        self.base = (self.soff[b] + (Q - self.boff[b])*pc[b] - self.boff[b]).astype(np.int64)
        self.diag_idx = (self.base + Q).astype(np.int64)
        self.pc = pc; self.S = int(self.soff[-1])
        self.big = np.flatnonzero(pc >= BIG).astype(np.int64)
        small_mask = pc[blk_of] < BIG
        self.small_cl = np.flatnonzero(small_mask).astype(np.int64)        # clusters in small blocks
        self.W = sp.csr_matrix((d, (self.lab, np.arange(n))), shape=(self.P, n))
        self.dsq = np.bincount(self.lab, weights=d*d, minlength=self.P)
        self.sig = np.zeros(self.S); self.nested = True; self.r = None; self.pair = True; self.dense = False

    def block(self, arr, b):
        """view of block b of a flat array as a Pc x Pc matrix"""
        p = int(self.pc[b]); return arr[self.soff[b]:self.soff[b+1]].reshape(p, p)

    def crange(self):
        """per-cluster [lo, hi) column ranges (level-local ids)"""
        return self.boff[self.blk_of], self.boff[self.blk_of + 1]

    def sigdiag(self, sig=None):
        return (self.sig if sig is None else sig)[self.diag_idx]


# ----------------------------------------------------------------------------- small-block kernels
@njit(parallel=True, cache=True)
def nest_kq_small(cl, blk_of, boff, base, Y, sig, rq, dsq, w, rho, out):
    """Kq for clusters cl (local ids) in small blocks: = kq_level restricted to the block (trace rule):
    2 rho (sum_{R in blk, pp_QR > 0} r_QR y2_R + [pp_QQ > 0] r_QQ y2_Q),  r = (rq_Q rq_R) w,  pp = sig - rho r h."""
    k = Y.shape[1]
    for t in prange(cl.shape[0]):
        Q = cl[t]; b = blk_of[Q]; lo = boff[b]; hi = boff[b+1]; s = 0.0
        for R in range(lo, hi):
            h = 0.0; y2 = 0.0
            for c in range(k):
                h += Y[Q, c]*Y[R, c]; y2 += Y[R, c]*Y[R, c]
            if R == Q: h -= dsq[Q]
            rr = (rq[Q]*rq[R])*w
            p = sig[base[Q] + R] - rho*rr*h
            if p > 0.0: s += rr*y2
        hq = 0.0; yq = 0.0                                # diagonal term (as kq_level: separate pass)
        for c in range(k):
            hq += Y[Q, c]*Y[Q, c]
        yq = hq; hq -= dsq[Q]; rqq = (rq[Q]*rq[Q])*w
        dg = rqq*yq if sig[base[Q] + Q] - rho*rqq*hq > 0.0 else 0.0
        out[Q] = 2*rho*(s + dg)


@njit(parallel=True, cache=True)
def nest_aug_small(cl, blk_of, boff, base, Y, sig, rq, dsq, w, rho, pp):
    """level_aug for clusters cl in small blocks: pp = max(0, sig - rho r h) (flat), returns
    (sum (pp^2 - sig^2)/(2 rho r), #(h < -1e-9))."""
    k = Y.shape[1]; tot = 0.0; nv = 0
    for t in prange(cl.shape[0]):
        Q = cl[t]; b = blk_of[Q]; lo = boff[b]; hi = boff[b+1]; s = 0.0; cnt = 0
        for R in range(lo, hi):
            h = 0.0
            for c in range(k): h += Y[Q, c]*Y[R, c]
            if R == Q: h -= dsq[Q]
            rr = rho*((rq[Q]*rq[R])*w); sg = sig[base[Q] + R]
            p = sg - rr*h
            if p < 0.0: p = 0.0
            pp[base[Q] + R] = p; s += (p*p - sg*sg)/(2*rr)
            if h < -1e-9: cnt += 1
        tot += s; nv += cnt
    return tot, nv


@njit(parallel=True, cache=True)
def nest_apply_small(cl, blk_of, boff, base, Y, sig, M):
    """M[Q] = sum_{R in blk(Q)} sig_QR Y_R for clusters cl."""
    k = Y.shape[1]
    for t in prange(cl.shape[0]):
        Q = cl[t]; b = blk_of[Q]; lo = boff[b]; hi = boff[b+1]
        for c in range(k): M[Q, c] = 0.0
        for R in range(lo, hi):
            s_ = sig[base[Q] + R]
            if s_ != 0.0:
                for c in range(k): M[Q, c] += s_*Y[R, c]


@njit(parallel=True, cache=True)
def nest_sym(blk_of, boff, base, sig, out):
    """out = per-block 0.5 (S + S^T) (same arithmetic as dense 0.5*(sig+sig.T))."""
    P = blk_of.shape[0]
    for Q in prange(P):
        b = blk_of[Q]
        for R in range(boff[b], boff[b+1]):
            out[base[Q] + R] = 0.5*(sig[base[Q] + R] + sig[base[R] + Q])


@njit(cache=True)
def nest_sig_csr(blk_of, boff, base, sig, goff):
    """row-CSR of the block-diagonal sig (column ids + goff), columns ascending (= sig_csr of the dense equivalent)."""
    P = blk_of.shape[0]; ptr = np.zeros(P+1, np.int64)
    for Q in range(P):
        b = blk_of[Q]; c = 0
        for R in range(boff[b], boff[b+1]):
            if sig[base[Q] + R] != 0.0: c += 1
        ptr[Q+1] = ptr[Q] + c
    idx = np.empty(ptr[P], np.int64); val = np.empty(ptr[P])
    for Q in range(P):
        b = blk_of[Q]; m = ptr[Q]
        for R in range(boff[b], boff[b+1]):
            v = sig[base[Q] + R]
            if v != 0.0: idx[m] = R + goff; val[m] = v; m += 1
    return ptr, idx, val


# ----------------------------------------------------------------------------- level operations (numpy + kernels)
def nest_apply(Lv, sig, Y):
    """blockdiag(sig) @ Y  (P x k).  Big blocks: dense BLAS sig_b @ Y_b (as dense s@Y)."""
    M = np.empty_like(Y)
    for b in Lv.big:
        lo, hi = Lv.boff[b], Lv.boff[b+1]; M[lo:hi] = Lv.block(sig, b)@Y[lo:hi]
    if len(Lv.small_cl): nest_apply_small(Lv.small_cl, Lv.blk_of, Lv.boff, Lv.base, np.ascontiguousarray(Y), sig, M)
    return M


def nest_symmetrize(Lv, sig):
    out = np.empty_like(sig); nest_sym(Lv.blk_of, Lv.boff, Lv.base, sig, out); return out


def nest_kq(Lv, Y, rho, w, out, kq_level):
    """NESTED Kq (trace rule) = kq_level restricted to each parent block; big blocks via the dense code path verbatim
    (pairH on the block, kq_level with r_b = outer(rq_b, rq_b) w)."""
    for b in Lv.big:
        lo, hi = Lv.boff[b], Lv.boff[b+1]; Yb = Y[lo:hi]
        H = Yb@Yb.T; H[np.diag_indices(hi-lo)] -= Lv.dsq[lo:hi]; H = 0.5*(H+H.T)
        rb = np.outer(Lv.rq[lo:hi], Lv.rq[lo:hi])*w
        kq_level(H, np.ascontiguousarray(Lv.block(Lv.sig, b)), rb, (Yb*Yb).sum(1), float(rho), out[lo:hi])
    if len(Lv.small_cl): nest_kq_small(Lv.small_cl, Lv.blk_of, Lv.boff, Lv.base, np.ascontiguousarray(Y), Lv.sig, Lv.rq, Lv.dsq, w, float(rho), out)

def nest_aug(Lv, Y, rho, w, level_aug):
    """NESTED augmented() level term: (sum (pp^2-sig^2)/(2 rho r), #(h < -1e-9), new flat pp); big blocks via level_aug."""
    pp = np.empty_like(Lv.sig); tot = 0.0; nv = 0
    for b in Lv.big:
        lo, hi = Lv.boff[b], Lv.boff[b+1]; Yb = Y[lo:hi]
        H = Yb@Yb.T; H[np.diag_indices(hi-lo)] -= Lv.dsq[lo:hi]; ppb = np.empty_like(H)
        t_, n_ = level_aug(H, np.ascontiguousarray(Lv.block(Lv.sig, b)), Lv.rq[lo:hi], w, float(rho), ppb, 0, np.zeros((1, 1)))
        Lv.block(pp, b)[...] = ppb; tot += t_; nv += n_
    if len(Lv.small_cl):
        t_, n_ = nest_aug_small(Lv.small_cl, Lv.blk_of, Lv.boff, Lv.base, np.ascontiguousarray(Y), Lv.sig, Lv.rq, Lv.dsq, w, float(rho), pp)
        tot += t_; nv += n_
    return tot, nv, pp
