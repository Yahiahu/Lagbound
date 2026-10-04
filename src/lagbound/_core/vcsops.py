"""SPARSE active-set vertex-cluster (VC) nonnegativity level for alm_vcs.py / est_dual_vcs.py (Oct 1 2026).

Same constraint, penalty and Lagrangian as nested/vcops.py (dense n x P):
    g_iR(X) = d_i sum_{j in R, j != i} d_j X_ij >= 0,  unit rows: g_iR = d_i (v_i.Y_R - [R = R(i)] d_i),  Y_R = sum_{j in R} d_j v_j
    r_iR = wv rq_R / d_i^2 (wv = vc_weight * pairlvl_weight, rq_R = |R|/vol_R^2),  PHR p = max(0, sig - rho r g)
    S_vc[i,j] = d_i d_j (sig_{i,R(j)} + sig_{j,R(i)})/2 (i != j), zero diagonal; bound valid for any sig >= 0.
but only the entries (i, R) of a WORKING SET carry a multiplier / penalty (all others: constraint not in the
Lagrangian, sig = 0). Working set = row-CSR (ptr: n+1 int64, col: int32 sorted inside each row), per entry:
sig (float64), stZ (stored effective multiplier of the current sweep), age (uint8: consecutive outers with new
sig = 0 and g > 0), row (int32). Z rebuilds scan in row order with thread t owning R = t mod T (vcs_zinit_rows; same
per-R order as the column-view reference vcs_zinit). Memory ~33 B/entry persistent (+ 9 B/entry transient in augmented).
With the working set = all n x P entries (sorted) the arrays are exactly the dense row-major arrays, and every kernel
below does the dense code's arithmetic in the dense code's order (validation: alm_vcs --vcs_full == alm_vc --vc_labels).

Row step (half-gradient, as vcops):
    GX_i = 1/2 d_i (sum_{R in ws(i)} p_iR Y_R - p_{i,R(i)} d_i v_i) + 1/2 d_i (Z_{R(i)} - stZ_{i,R(i)} d_i v_i)
    Z_R  = sum_{j: R in ws(j)} stZ_jR d_j v_j  (built per sweep, updated incrementally after every batch)
    KX_i = rho d_i^2 sum_{R in ws(i), p > 0} r_iR |Y_R - [R=R(i)] d_i v_i|^2 + kap2 rho d_i^2 wv rq_{R(i)} (cnt_{R(i)} - [stZ_{i,R(i)} > 0])
Post-batch update (vcs_post): fused "new effective multipliers of the moved rows" + Z/cnt update + store, parallel
over R-classes (R mod T) so every Z_R is owned by one thread and summed in batch-row order (= dense vc_post_Z).
Separation (vcs_sep_rows): G = V_chunk Y^T by BLAS, then per row the violated entries NOT in the working set with
mean-X violation xbar = s'/vol(R minus i) < -tol, scored by
    pen : wv rq_R s'^2 (= r g^2: first-order decrease of the augmented Lagrangian if the entry is added)
    mean: -xbar         mass: -d_i s' (= -g)
"""
import numpy as np, scipy.sparse as sp
from numba import njit, prange
import numba


class VCSLevel:
    def __init__(self, lab, d, full=False, name=""):
        self.lab = np.unique(lab, return_inverse=True)[1].astype(np.int64); self.P = int(self.lab.max()) + 1
        self.n = len(lab); n = self.n; self.full = bool(full); self.name = name
        self.W = sp.csr_matrix((d, (self.lab, np.arange(n))), shape=(self.P, n))
        cnt = np.bincount(self.lab, minlength=self.P).astype(float)
        self.vol = np.maximum(np.bincount(self.lab, weights=d, minlength=self.P), 1.0)
        self.rq = cnt/self.vol**2
        self.Z = None; self.cnt = None; self.stZ = None
        if full: self.set_ws(np.arange(n+1, dtype=np.int64)*self.P, np.tile(np.arange(self.P, dtype=np.int32), n))
        else: self.set_ws(np.zeros(n+1, np.int64), np.zeros(0, np.int32))

    @property
    def nnz(self): return int(self.ptr[-1])

    def set_ws(self, ptr, col, sig=None, age=None):
        self.ptr = np.ascontiguousarray(ptr, np.int64); self.col = np.ascontiguousarray(col, np.int32); nz = len(self.col)
        self.sig = np.zeros(nz) if sig is None else np.ascontiguousarray(sig, np.float64)
        self.age = np.zeros(nz, np.uint8) if age is None else np.ascontiguousarray(age, np.uint8)
        self.row = np.repeat(np.arange(self.n, dtype=np.int32), np.diff(self.ptr))
        self.stZ = np.empty(nz); self.st_ok = None
        self.own = own_entry(self.ptr, self.col, self.lab)      # entry id of (i, R(i)) or -1

    def nbytes(self):
        return sum(x.nbytes for x in (self.ptr, self.col, self.sig, self.age, self.row, self.stZ, self.own))

    def towner(self):
        """(T, owner thread of each cluster: (R // 8) mod T) for the R-partitioned kernels (any owner map gives the same per-R order)."""
        T = int(numba.get_num_threads())
        if getattr(self, "_town", None) is None or self._town[0] != T: self._town = (T, ((np.arange(self.P)//8) % T).astype(np.int32))   # blocks of 8: no false sharing on cnt
        return self._town

    @property
    def cptr(self): return col_view(self.col, self.P)[0]

    @property
    def cperm(self): return col_view(self.col, self.P)[1]

    def csr(self, sig=None):
        return sp.csr_matrix((self.sig if sig is None else sig, self.col, self.ptr), shape=(self.n, self.P))


@njit(cache=True)
def col_view(col, P):
    nz = col.shape[0]; cptr = np.zeros(P+1, np.int64)
    for e in range(nz): cptr[col[e]+1] += 1
    for R in range(P): cptr[R+1] += cptr[R]
    pos = cptr[:-1].copy(); perm = np.empty(nz, np.int64)
    for e in range(nz):
        R = col[e]; perm[pos[R]] = e; pos[R] += 1
    return cptr, perm


@njit(parallel=True, cache=True)
def own_entry(ptr, col, lab):
    n = ptr.shape[0] - 1; out = np.full(n, -1, np.int64)
    for i in prange(n):
        Ri = lab[i]
        for e in range(ptr[i], ptr[i+1]):
            if col[e] == Ri: out[i] = e; break
    return out


@njit(parallel=True, cache=True)
def vcs_sigt(V, d, lab, Y, ptr, col, sig, rq, wv, rho, out):
    """out[e] = effective multiplier max(0, sig - rho r g) at (V, Y) for every working-set entry (rho = 0: sig)."""
    n, k = V.shape
    for i in prange(n):
        di = d[i]; Ri = lab[i]
        if rho <= 0.0 or di <= 0.0:
            for e in range(ptr[i], ptr[i+1]): out[e] = sig[e]
            continue
        for e in range(ptr[i], ptr[i+1]):
            R = col[e]; s = 0.0
            for c in range(k): s += V[i, c]*Y[R, c]
            g = di*(s - di) if R == Ri else di*s
            p = sig[e] - rho*(wv*rq[R]/(di*di))*g
            out[e] = p if p > 0.0 else 0.0


@njit(parallel=True, cache=True)
def vcs_zinit(V, d, row, cptr, cperm, stZ, Z, cnt):
    """Z_R = sum_{entries e in column R} stZ_e d_i v_i (rows ascending), cnt_R = #(stZ_e > 0)."""
    P, k = Z.shape
    for R in prange(P):
        for c in range(k): Z[R, c] = 0.0
        cc = 0.0
        for q in range(cptr[R], cptr[R+1]):
            e = cperm[q]; s = stZ[e]
            if s != 0.0:
                i = row[e]; f = d[i]
                for c in range(k): Z[R, c] += s*(f*V[i, c])
            if s > 0.0: cc += 1.0
        cnt[R] = cc


@njit(parallel=True, cache=True)
def vcs_pre(rows, V, d, lab, Y, Y2, useY2, ptr, col, sig, stZ, own, Z, cnt, rq, wv, rho, kap2, GX, KX):
    """per batch row (before its step): GX += VC half-gradient, KX += VC curvature (module doc)."""
    k = V.shape[1]
    for idx in prange(rows.shape[0]):
        i = rows[idx]; di = d[i]; Ri = lab[i]; acc = np.zeros(k); kap = 0.0; pown = 0.0
        for e in range(ptr[i], ptr[i+1]):
            R = col[e]; s = 0.0
            for c in range(k): s += V[i, c]*Y[R, c]
            if rho > 0.0 and di > 0.0:
                g = di*(s - di) if R == Ri else di*s
                rr = wv*rq[R]/(di*di); p = sig[e] - rho*rr*g
                if p > 0.0:
                    if useY2: y2 = Y2[R]
                    else:
                        y2 = 0.0
                        for c in range(k): y2 += Y[R, c]*Y[R, c]
                    if R == Ri: y2 = y2 - 2*di*s + di*di          # |Y_R - d_i v_i|^2
                    kap += rho*rr*di*di*y2
                else: p = 0.0
            else: p = sig[e]
            if R == Ri: pown = p
            if p != 0.0:
                for c in range(k): acc[c] += p*Y[R, c]
        eo = own[i]; sz = stZ[eo] if eo >= 0 else 0.0
        for c in range(k):
            GX[idx, c] += 0.5*di*(acc[c] - pown*di*V[i, c]) + 0.5*di*(Z[Ri, c] - sz*di*V[i, c])
        if rho > 0.0:
            kap += kap2*rho*di*di*wv*rq[Ri]*(cnt[Ri] - (1.0 if sz > 0.0 else 0.0))
        KX[idx] += kap


@njit(cache=True)
def vcs_yupd(rows, lab, DD, Y):
    """Y[lab[i]] += DD[idx] in batch order (= np.add.at)."""
    k = Y.shape[1]
    for idx in range(rows.shape[0]):
        R = lab[rows[idx]]
        for c in range(k): Y[R, c] += DD[idx, c]


@njit(parallel=True, cache=True)
def vcs_zinit_rows(V, d, ptr, col, stZ, Z, cnt, T, town):
    """= vcs_zinit (same per-R summation order: rows ascending) by a row-order scan; thread t owns R = t mod T."""
    P, k = Z.shape
    for t in prange(T):
        for R in range(P):                         # zero exactly the clusters this thread owns (no race with other owners)
            if town[R] == t:
                cnt[R] = 0.0
                for c in range(k): Z[R, c] = 0.0
        n = ptr.shape[0] - 1
        for i in range(n):
            f = d[i]
            for e in range(ptr[i], ptr[i+1]):
                R = col[e]
                if town[R] != t: continue
                s = stZ[e]
                if s != 0.0:
                    for c in range(k): Z[R, c] += s*(f*V[i, c])
                if s > 0.0: cnt[R] += 1.0


@njit(parallel=True, cache=True)
def vcs_post(rows, V, Vold, d, lab, Y, ptr, col, sig, stZ, rq, wv, rho, Z, cnt, T, town):
    """moved rows: new effective multipliers SN (new v_i, current Y), Z_R += d_i (SN v_i,new - stZ v_i,old),
    cnt_R += [SN>0] - [stZ>0], stZ = SN. Thread t owns the clusters R = t mod T."""
    k = V.shape[1]
    for t in prange(T):
        for idx in range(rows.shape[0]):
            i = rows[idx]; di = d[i]; Ri = lab[i]
            for e in range(ptr[i], ptr[i+1]):
                R = col[e]
                if town[R] != t: continue
                if rho > 0.0 and di > 0.0:
                    s = 0.0
                    for c in range(k): s += V[i, c]*Y[R, c]
                    g = di*(s - di) if R == Ri else di*s
                    p = sig[e] - rho*(wv*rq[R]/(di*di))*g
                    sn = p if p > 0.0 else 0.0
                else: sn = sig[e]
                so = stZ[e]
                if sn != 0.0 or so != 0.0:
                    for c in range(k): Z[R, c] += di*(sn*V[i, c] - so*Vold[idx, c])
                if sn > 0.0:
                    if not so > 0.0: cnt[R] += 1.0
                elif so > 0.0: cnt[R] -= 1.0
                stZ[e] = sn


@njit(parallel=True, cache=True)
def _post_sn(rows, V, d, lab, Y, ptr, col, sig, rq, wv, rho, off, SN):
    k = V.shape[1]
    for idx in prange(rows.shape[0]):
        i = rows[idx]; di = d[i]; Ri = lab[i]; o = off[idx]
        for e in range(ptr[i], ptr[i+1]):
            R = col[e]
            if rho > 0.0 and di > 0.0:
                s = 0.0
                for c in range(k): s += V[i, c]*Y[R, c]
                g = di*(s - di) if R == Ri else di*s
                p = sig[e] - rho*(wv*rq[R]/(di*di))*g
                SN[o] = p if p > 0.0 else 0.0
            else: SN[o] = sig[e]
            o += 1


@njit(parallel=True, cache=True)
def _post_z(rows, V, Vold, d, ptr, col, stZ, Z, cnt, T, town, off, SN):
    k = V.shape[1]
    for t in prange(T):
        for idx in range(rows.shape[0]):
            i = rows[idx]; di = d[i]; o = off[idx] - ptr[i]
            for e in range(ptr[i], ptr[i+1]):
                R = col[e]
                if town[R] != t: continue
                sn = SN[o + e]; so = stZ[e]
                if sn != 0.0 or so != 0.0:
                    for c in range(k): Z[R, c] += di*(sn*V[i, c] - so*Vold[idx, c])
                if sn > 0.0:
                    if not so > 0.0: cnt[R] += 1.0
                elif so > 0.0: cnt[R] -= 1.0


@njit(parallel=True, cache=True)
def _post_store(rows, ptr, off, SN, stZ):
    for idx in prange(rows.shape[0]):
        i = rows[idx]; o = off[idx] - ptr[i]
        for e in range(ptr[i], ptr[i+1]): stZ[e] = SN[o + e]


def vcs_post2(rows, V, Vold, d, lab, Y, ptr, col, sig, stZ, rq, wv, rho, Z, cnt, T, town):
    """= vcs_post (identical arithmetic) in two passes: SN per row in parallel, then the R-partitioned Z/cnt/stZ update."""
    ln = ptr[rows+1] - ptr[rows]; off = np.zeros(len(rows), np.int64); np.cumsum(ln[:-1], out=off[1:])
    SN = np.empty(int(ln.sum()))
    _post_sn(rows, V, d, lab, Y, ptr, col, sig, rq, wv, rho, off, SN)
    _post_z(rows, V, Vold, d, ptr, col, stZ, Z, cnt, T, town, off, SN)        # reads stZ only (no false sharing)
    _post_store(rows, ptr, off, SN, stZ)


@njit(parallel=True, cache=True)
def _vcs_aug_rows(V, d, lab, Y, ptr, col, sig, rq, wv, rho, pp, trow, nrow, slack, stn, dost):
    n, k = V.shape
    for i in prange(n):
        di = d[i]; Ri = lab[i]; acc = 0.0; cc = 0
        for e in range(ptr[i], ptr[i+1]):
            R = col[e]; s = 0.0
            for c in range(k): s += V[i, c]*Y[R, c]
            g = di*s
            if R == Ri: g = di*(s - di)
            sg = sig[e]; p = sg
            if di > 0.0:
                rr = rho*(wv*rq[R]/(di*di)); p = max(0.0, sg - rr*g)
                acc += (p*p - sg*sg)/(2*rr)
            pp[e] = p
            if dost:                                   # next sweep's start-of-sweep effective multiplier (same V, Y, rho)
                if rho <= 0.0 or di <= 0.0: stn[e] = p
                else:
                    q = p - rho*(wv*rq[R]/(di*di))*g
                    stn[e] = q if q > 0.0 else 0.0
            slack[e] = 1 if (p == 0.0 and g > 0.0) else 0
            if g < -1e-9: cc += 1
        trow[i] = acc; nrow[i] = cc


def vcs_aug(Lv, V, d, Y, wv, rho, pp, slack, stn=None):
    """augmented() term over the working set: pp = max(0, sig - rho r g); returns (sum (pp^2-sig^2)/(2 rho r), #(g<-1e-9)).
    stn (optional): also write max(0, pp - rho r g), i.e. the next sweep's stZ if sig <- pp and V is unchanged."""
    trow = np.zeros(V.shape[0]); nrow = np.zeros(V.shape[0], np.int64)
    _vcs_aug_rows(V, d, Lv.lab, Y, Lv.ptr, Lv.col, Lv.sig, Lv.rq, float(wv), float(rho), pp, trow, nrow, slack,
                  stn if stn is not None else np.zeros(1), stn is not None)
    return float(trow.sum()), int(nrow.sum())


@njit(cache=True)
def age_update(age, slack):
    for e in range(age.shape[0]):
        if slack[e]:
            if age[e] < 255: age[e] += 1
        else: age[e] = 0


def vcs_state(Lv, V, d, rho, wv, exact=False):
    """start-of-sweep state: Y, stZ (effective multipliers), Z, cnt."""
    Y = np.ascontiguousarray(Lv.W@V)
    if getattr(Lv, "st_ok", None) is not None and Lv.st_ok == (float(rho), Lv.nnz) and len(Lv.stZ) == Lv.nnz:
        pass                                       # stZ precomputed by the last augmented() at this V (alm_vcs main loop)
    else:
        if Lv.stZ is None or len(Lv.stZ) != Lv.nnz: Lv.stZ = np.empty(Lv.nnz)
        vcs_sigt(V, d, Lv.lab, Y, Lv.ptr, Lv.col, Lv.sig, Lv.rq, float(wv), float(rho), Lv.stZ)
    Lv.st_ok = None
    if exact and Lv.nnz == Lv.n*Lv.P:            # validation: dense BLAS product as in vcops.vc_state
        S_ = Lv.stZ.reshape(Lv.n, Lv.P)
        Lv.Z = np.ascontiguousarray(S_.T@(d[:, None]*V)); Lv.cnt = (S_ > 0).sum(0).astype(np.float64)
    else:
        Lv.Z = np.empty((Lv.P, V.shape[1])); Lv.cnt = np.empty(Lv.P)
        vcs_zinit_rows(V, d, Lv.ptr, Lv.col, Lv.stZ, Lv.Z, Lv.cnt, *Lv.towner())
    return Y


def vcs_apply(Lv, sig, X, d):
    """S_vc @ X for a sparse VC level (matrix-free): 1/2 d (Sig (W X) + (Sig^T (d X))[lab]) - own sig d^2 X."""
    S = Lv.csr(sig); YX = Lv.W@X; ZX = S.T@(d[:, None]*X)
    own = np.where(Lv.own >= 0, sig[np.maximum(Lv.own, 0)], 0.0) if len(sig) else np.zeros(len(d))
    return 0.5*d[:, None]*(S@YX + ZX[Lv.lab]) - (own*d**2)[:, None]*X


# ----------------------------------------------------------------------------- separation
@njit(parallel=True, cache=True)
def _sep_rows(G, rowids, d, lab, vol, rq, wv, ptr, col, tol, thr, mode, out_cnt, out_i, out_R, out_s, out_base, write, stats):
    """rows i0..i0+nr-1 of G = V Y^T: count (write=0) / write (write=1) violated non-working-set entries with score > thr.
    stats[r, :] = (#violated all, violation mass all (sum -g), #violated in ws, mass in ws)."""
    nr, P = G.shape
    for r in prange(nr):
        i = rowids[r]; di = d[i]; Ri = lab[i]; q = ptr[i]; qe = ptr[i+1]; cnt = 0; o = out_base[r] if write else 0
        nv = 0.0; mv = 0.0; nw = 0.0; mw = 0.0
        for R in range(P):
            inws = False
            while q < qe and col[q] < R: q += 1
            if q < qe and col[q] == R: inws = True
            s = G[r, R]; vR = vol[R]
            if R == Ri: s = s - di; vR = vR - di
            if vR <= 0.0: continue
            xb = s/vR
            if xb >= -tol: continue
            nv += 1.0; mv += -di*s
            if inws:
                nw += 1.0; mw += -di*s; continue
            if mode == 0: sc = wv*rq[R]*s*s
            elif mode == 1: sc = -xb
            else: sc = -di*s
            if sc <= thr: continue
            if write:
                out_i[o] = i; out_R[o] = R; out_s[o] = sc; o += 1
            cnt += 1
        if not write:
            out_cnt[r] = cnt
            stats[r, 0] = nv; stats[r, 1] = mv; stats[r, 2] = nw; stats[r, 3] = mw


@njit(parallel=True, cache=True)
def _sep_rows1(G, rowids, d, lab, vol, rq, wv, ptr, col, tol, thr, mode, m, out_cnt, out_R, out_s, stats):
    """one pass: like _sep_rows (count + stats) but writes the first m candidates of row r into slots r*m.. (rows with
    out_cnt > m are redone exactly by the caller)."""
    nr, P = G.shape
    for r in prange(nr):
        i = rowids[r]; di = d[i]; Ri = lab[i]; q = ptr[i]; qe = ptr[i+1]; cnt = 0; o = r*m
        nv = 0.0; mv = 0.0; nw = 0.0; mw = 0.0
        for R in range(P):
            s = G[r, R]; vR = vol[R]
            if R == Ri: s = s - di; vR = vR - di
            if vR <= 0.0: continue
            xb = s/vR
            if xb >= -tol: continue
            while q < qe and col[q] < R: q += 1
            nv += 1.0; mv += -di*s
            if q < qe and col[q] == R:
                nw += 1.0; mw += -di*s; continue
            if mode == 0: sc = wv*rq[R]*s*s
            elif mode == 1: sc = -xb
            else: sc = -di*s
            if sc <= thr: continue
            if cnt < m:
                out_R[o + cnt] = R; out_s[o + cnt] = sc
            cnt += 1
        out_cnt[r] = cnt
        stats[r, 0] = nv; stats[r, 1] = mv; stats[r, 2] = nw; stats[r, 3] = mw


def separate(Lv, V, d, wv, nadd, tol=1e-4, mode="pen", chunk_bytes=64*2**20):
    """returns (rows i, cols R, scores) of the <= nadd best violated entries not in the working set, + stats dict."""
    n, k = V.shape; P = Lv.P; Y = np.ascontiguousarray(Lv.W@V); md = {"pen": 0, "mean": 1, "mass": 2}[mode]
    ch = int(max(64, min(n, chunk_bytes//(8*P)))); thr = 0.0
    bi = [np.zeros(0, np.int64)]; bR = [np.zeros(0, np.int32)]; bs = [np.zeros(0)]; nb = 0
    st = np.zeros(4)
    if n > 20*ch:                                         # initial threshold from a row sample (keeps the buffer small)
        bl = max(8, ch//20); blk = np.linspace(0, n - bl, 20).astype(np.int64); ids = np.concatenate([np.arange(b, b+bl) for b in blk])   # ~ one chunk
        G = V[ids]@Y.T; cnt = np.zeros(len(ids), np.int64); stats = np.zeros((len(ids), 4)); dum = np.zeros(1, np.int64)
        _sep_rows(G, ids, d, Lv.lab, Lv.vol, Lv.rq, float(wv), Lv.ptr, Lv.col, float(tol), 0.0, md, cnt, dum, np.zeros(1, np.int32), np.zeros(1), dum, 0, stats)
        tot = int(cnt.sum()); want = int(2.0*nadd*len(ids)/n)
        if tot > want > 0:
            base = np.r_[0, np.cumsum(cnt)[:-1]].astype(np.int64); oi = np.empty(tot, np.int64); oR = np.empty(tot, np.int32); os_ = np.empty(tot)
            _sep_rows(G, ids, d, Lv.lab, Lv.vol, Lv.rq, float(wv), Lv.ptr, Lv.col, float(tol), 0.0, md, cnt, oi, oR, os_, base, 1, stats)
            thr = float(np.partition(os_, tot - want)[tot - want])
        del G
    for i0 in range(0, n, ch):
        G = V[i0:i0+ch]@Y.T; nr = G.shape[0]; ids = np.arange(i0, i0+nr, dtype=np.int64)
        cnt = np.zeros(nr, np.int64); stats = np.zeros((nr, 4)); M1 = 32
        bufR = np.empty(nr*M1, np.int32); bufs = np.empty(nr*M1)
        _sep_rows1(G, ids, d, Lv.lab, Lv.vol, Lv.rq, float(wv), Lv.ptr, Lv.col, float(tol), float(thr), md, M1, cnt, bufR, bufs, stats)
        st += stats.sum(0); tot = int(cnt.sum())
        if tot:
            ok = cnt <= M1; c_ = np.minimum(cnt, M1)
            sl = (np.arange(M1)[None, :] < c_[:, None]) & ok[:, None]
            oi = np.repeat(ids, M1).reshape(nr, M1)[sl]; oR = bufR.reshape(nr, M1)[sl]; os_ = bufs.reshape(nr, M1)[sl]
            bi.append(oi); bR.append(oR); bs.append(os_); nb += len(oi)
            big = np.flatnonzero(~ok)
            if len(big):                                  # rows with more than M1 candidates: exact two-pass on those rows
                Gb = np.ascontiguousarray(G[big]); cb = cnt[big].copy(); stb = np.zeros((len(big), 4))
                base = np.r_[0, np.cumsum(cb)[:-1]].astype(np.int64); tb = int(cb.sum())
                oi = np.empty(tb, np.int64); oR = np.empty(tb, np.int32); os_ = np.empty(tb)
                _sep_rows(Gb, ids[big], d, Lv.lab, Lv.vol, Lv.rq, float(wv), Lv.ptr, Lv.col, float(tol), float(thr), md, cb, oi, oR, os_, base, 1, stb)
                bi.append(oi); bR.append(oR); bs.append(os_); nb += tb
        if nb > 2*nadd:                                   # prune the candidate buffer to the nadd best, raise thr
            ai = np.concatenate(bi); aR = np.concatenate(bR); as_ = np.concatenate(bs)
            sel = np.argpartition(-as_, nadd-1)[:nadd]; thr = float(as_[sel].min())
            bi = [ai[sel]]; bR = [aR[sel]]; bs = [as_[sel]]; nb = nadd
        del G
    ai = np.concatenate(bi); aR = np.concatenate(bR); as_ = np.concatenate(bs)
    if len(as_) > nadd: sel = np.argpartition(-as_, nadd-1)[:nadd]; ai, aR, as_ = ai[sel], aR[sel], as_[sel]
    return ai, aR, as_, dict(nviol=st[0], mass=st[1], nviol_ws=st[2], mass_ws=st[3])


@njit(cache=True)
def _merge(n, ptr, col, sig, age, keep, ni, nR, out_ptr, out_col, out_sig, out_age):
    """merge kept old entries (row-sorted) with new entries (sorted by (i, R), disjoint from old) -> new CSR."""
    q = 0; nn = ni.shape[0]; o = 0
    for i in range(n):
        out_ptr[i] = o; e = ptr[i]; ee = ptr[i+1]
        while True:
            while e < ee and not keep[e]: e += 1
            hasold = e < ee; hasnew = q < nn and ni[q] == i
            if not hasold and not hasnew: break
            if hasold and (not hasnew or col[e] < nR[q]):
                out_col[o] = col[e]; out_sig[o] = sig[e]; out_age[o] = age[e]; e += 1
            else:
                out_col[o] = nR[q]; out_sig[o] = 0.0; out_age[o] = 0; q += 1
            o += 1
    out_ptr[n] = o


def update_ws(Lv, keep, ni, nR):
    """drop entries with keep = False, add new entries (ni, nR) with sig 0."""
    o = np.lexsort((nR, ni)); ni = np.ascontiguousarray(ni[o], np.int64); nR = np.ascontiguousarray(nR[o], np.int32)
    nz = int(keep.sum()) + len(ni)
    ptr = np.empty(Lv.n+1, np.int64); col = np.empty(nz, np.int32); sig = np.empty(nz); age = np.empty(nz, np.uint8)
    _merge(Lv.n, Lv.ptr, Lv.col, Lv.sig, Lv.age, keep, ni, nR, ptr, col, sig, age)
    assert ptr[-1] == nz
    Lv.set_ws(ptr, col, sig, age)
