"""Vertex–cluster (VC) nonnegativity level for alm_vc.py / est_dual_vc.py.

For a labelling R(.) with P clusters, every vertex i and cluster R:
    g_iR(X) = d_i sum_{j in R, j != i} d_j X_ij >= 0          (holds for every partition matrix X)
With X = V V^T, unit rows, Y_R = sum_{j in R} d_j v_j:   g_iR = d_i (v_i.Y_R - [R = R(i)] d_i).
Multipliers sig (n x P) >= 0. Lagrangian term sum_iR sig_iR g_iR(X) = <S_vc, X>,
    S_vc[i, j] = d_i d_j (sig_{i,R(j)} + sig_{j,R(i)}) / 2   (i != j),  zero diagonal,
so the dual matrix C gets + S_vc and the bound UB = (C0 + sum lam + max_X <C, X>)/L stays valid for ANY sig >= 0.
PHR: penalty rho r_iR, r_iR = wv * rq_R / d_i^2 (rq_R = |R| / vol_R^2, the pair levels' rq; wv = vc_weight * pair weight):
    pp = max(0, sig - rho r g)  (multiplier update),  augmented term -sum (pp^2 - sig^2) / (2 rho r).
Row step (half-gradient convention of the solver, g_i = sum_j C_ij v_j):
    GX_i = 1/2 d_i (sum_R st_iR Y_R - st_{i,R(i)} d_i v_i) + 1/2 d_i (Z_{R(i)} - stZ_{i,R(i)} d_i v_i),
    st = fresh effective multipliers of row i (current v_i, batch-frozen Y), Z_R = sum_j stZ_jR d_j v_j with stZ the
    effective multipliers stored per row (refreshed at the start of every sweep and for every moved row after its
    batch: Z, cnt maintained incrementally).
    curvature (full Hessian trace): rho d_i^2 sum_{R: st>0} r_iR |Y_R - [R=R(i)] d_i v_i|^2
                                    + kap2 * rho d_i^2 wv rq_{R(i)} (cnt_{R(i)} - [stZ_{i,R(i)} > 0]),
    cnt_R = #{j: stZ_jR > 0} (the transposed terms g_{j,R(i)}, j != i, are linear in v_i with Hessian
    rho r_{j,R(i)} d_j^2 d_i^2 v_j v_j^T, trace rho wv rq_{R(i)} d_i^2 each).
"""
import numpy as np, scipy.sparse as sp
from numba import njit, prange


class VCLevel:
    def __init__(self, lab, d):
        self.lab = np.unique(lab, return_inverse=True)[1].astype(np.int64); self.P = int(self.lab.max()) + 1
        self.n = len(lab); n = self.n
        self.W = sp.csr_matrix((d, (self.lab, np.arange(n))), shape=(self.P, n))
        cnt = np.bincount(self.lab, minlength=self.P).astype(float); vol = np.maximum(np.bincount(self.lab, weights=d, minlength=self.P), 1.0)
        self.rq = cnt/vol**2
        self.sig = np.zeros((n, self.P)); self.sigt = None; self.Z = None; self.cnt = None


@njit(parallel=True, cache=True)
def vc_sigt(V, d, lab, Y, sig, rq, wv, rho, out):
    """out = effective multipliers max(0, sig - rho r g) at (V, Y) (rho = 0: sig)."""
    n, k = V.shape; P = Y.shape[0]
    for i in prange(n):
        di = d[i]; Ri = lab[i]
        if rho <= 0.0 or di <= 0.0:
            for R in range(P): out[i, R] = sig[i, R]
            continue
        for R in range(P):
            s = 0.0
            for c in range(k): s += V[i, c]*Y[R, c]
            g = di*(s - di) if R == Ri else di*s
            p = sig[i, R] - rho*(wv*rq[R]/(di*di))*g
            out[i, R] = p if p > 0.0 else 0.0


@njit(parallel=True, cache=True)
def _vc_aug_rows(V, d, lab, Y, sig, rq, wv, rho, pp, trow, nrow):
    n, k = V.shape; P = Y.shape[0]
    for i in prange(n):
        di = d[i]; Ri = lab[i]; acc = 0.0; cc = 0
        for R in range(P):
            s = 0.0
            for c in range(k): s += V[i, c]*Y[R, c]
            g = di*s
            if R == Ri: g = di*(s - di)
            sg = sig[i, R]; p = sg
            if di > 0.0:
                rr = rho*(wv*rq[R]/(di*di)); p = max(0.0, sg - rr*g)
                acc += (p*p - sg*sg)/(2*rr)
            pp[i, R] = p
            if g < -1e-9: cc += 1
        trow[i] = acc; nrow[i] = cc


def vc_aug(V, d, lab, Y, sig, rq, wv, rho, pp):
    """augmented() VC term: pp = max(0, sig - rho r g); returns (sum (pp^2 - sig^2)/(2 rho r), #(g < -1e-9))."""
    trow = np.zeros(V.shape[0]); nrow = np.zeros(V.shape[0], np.int64)
    _vc_aug_rows(V, d, lab, Y, sig, rq, float(wv), float(rho), pp, trow, nrow)
    return float(trow.sum()), int(nrow.sum())


@njit(parallel=True, cache=True)
def vc_pre(rows, V, d, lab, Y, Y2, sig, stZ, Z, cnt, rq, wv, rho, kap2, GX, KX):
    """per batch row (before its step): GX = VC half-gradient, KX = VC curvature (see module doc)."""
    k = V.shape[1]; P = Y.shape[0]
    for idx in prange(rows.shape[0]):
        i = rows[idx]; di = d[i]; Ri = lab[i]; acc = np.zeros(k); kap = 0.0; pown = 0.0
        for R in range(P):
            s = 0.0; y2 = Y2[R]                               # Y2 = |Y_R|^2 (per batch)
            for c in range(k): s += V[i, c]*Y[R, c]
            if rho > 0.0 and di > 0.0:
                g = di*(s - di) if R == Ri else di*s
                rr = wv*rq[R]/(di*di); p = sig[i, R] - rho*rr*g
                if p > 0.0:
                    if R == Ri: y2 = y2 - 2*di*s + di*di          # |Y_R - d_i v_i|^2
                    kap += rho*rr*di*di*y2
                else: p = 0.0
            else: p = sig[i, R]
            if R == Ri: pown = p
            if p != 0.0:
                for c in range(k): acc[c] += p*Y[R, c]
        sz = stZ[i, Ri]
        for c in range(k):
            GX[idx, c] = 0.5*di*(acc[c] - pown*di*V[i, c]) + 0.5*di*(Z[Ri, c] - sz*di*V[i, c])
        if rho > 0.0:
            kap += kap2*rho*di*di*wv*rq[Ri]*(cnt[Ri] - (1.0 if sz > 0.0 else 0.0))
        KX[idx] = kap


@njit(parallel=True, cache=True)
def vc_post_rows(rows, V, d, lab, Y, sig, rq, wv, rho, SN):
    """new effective multipliers of the moved rows (new v_i, current Y)."""
    k = V.shape[1]; P = Y.shape[0]
    for idx in prange(rows.shape[0]):
        i = rows[idx]; di = d[i]; Ri = lab[i]
        for R in range(P):
            if rho > 0.0 and di > 0.0:
                s = 0.0
                for c in range(k): s += V[i, c]*Y[R, c]
                g = di*(s - di) if R == Ri else di*s
                p = sig[i, R] - rho*(wv*rq[R]/(di*di))*g
                SN[idx, R] = p if p > 0.0 else 0.0
            else: SN[idx, R] = sig[i, R]


@njit(parallel=True, cache=True)
def vc_post_Z(rows, V, Vold, d, stZ, SN, Z, cnt):
    """Z_R += d_i (SN_iR v_i,new - stZ_iR v_i,old), cnt_R += [SN>0] - [stZ>0] for the moved rows (parallel over R)."""
    k = V.shape[1]; P = Z.shape[0]
    for R in prange(P):
        for idx in range(rows.shape[0]):
            i = rows[idx]; di = d[i]; sn = SN[idx, R]; so = stZ[i, R]
            if sn != 0.0 or so != 0.0:
                for c in range(k): Z[R, c] += di*(sn*V[i, c] - so*Vold[idx, c])
            if sn > 0.0: cnt[R] += 1.0
            if so > 0.0: cnt[R] -= 1.0


@njit(parallel=True, cache=True)
def vc_store(rows, SN, stZ):
    P = stZ.shape[1]
    for idx in prange(rows.shape[0]):
        i = rows[idx]
        for R in range(P): stZ[i, R] = SN[idx, R]


def vc_apply(Lv, sig, X, d):
    """S_vc @ X for a VC level with multipliers sig (n x P)."""
    YX = Lv.W@X; own = sig[np.arange(len(d)), Lv.lab]
    ZX = (sig*d[:, None]).T@X
    return 0.5*d[:, None]*(sig@YX + ZX[Lv.lab]) - (own*d**2)[:, None]*X


def vc_state(Lv, V, d, rho, wv):
    """start-of-sweep state: Y, stZ (effective multipliers), Z = stZ^T (d v), cnt."""
    Y = np.ascontiguousarray(Lv.W@V)
    if Lv.sigt is None or Lv.sigt.shape != Lv.sig.shape: Lv.sigt = np.empty_like(Lv.sig)
    vc_sigt(V, d, Lv.lab, Y, Lv.sig, Lv.rq, float(wv), float(rho), Lv.sigt)
    Lv.Z = np.ascontiguousarray(Lv.sigt.T@(d[:, None]*V)); Lv.cnt = (Lv.sigt > 0).sum(0).astype(np.float64)
    return Y
