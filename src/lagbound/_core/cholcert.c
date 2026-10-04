/* cholcert.c -- memory-lean CHOLMOD supernodal LL^T for certify_lifted_vc.py (backend "cholcert").
 * Factors A (CSC, lower triangle, int64 indices) in NATURAL order (caller pre-permutes), keeps the factor
 * supernodal (no simplicial / scipy copies) and returns only what the rigorous price needs:
 *   out[0] ok (1 = factorization completed, all diagonals > 0 and finite)
 *   out[1] nnz(L) (structural, incl. amalgamation zeros)     out[2] k = max(max column count, max row count)
 *   out[3] S = sum_j (sum_i |L_ij|)^2  (= || |L|^T 1 ||^2)    out[4] min diag(L)   out[5] max diag(L)
 *   out[6] CHOLMOD flop count                                 out[7] cholmod status   out[8] L->minor
 *   out[9] Sw = sum_j (sum_i w_i |L_ij|)^2, w_i = sqrt(gamma-ish(rowcount_i + 4)): per-entry backward-error price
 * Higham Thm 10.3 (any summation order, blocked BLAS): L L^T = A + dA, |dA| <= gamma_{k+c} |L||L|^T, so
 * sum_kl |dA_kl| <= gamma * S  (python side uses gamma(k+4)).
 * build: gcc -O2 -shared -fPIC cholcert.c -o libcholcert.so -I/usr/include/suitesparse -lcholmod -lsuitesparseconfig -lm
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "cholmod.h"

static void strict_relax(cholmod_common *c)   /* CHOLCERT_STRICT=1: fundamental supernodes only (no relaxed zeros) */
{
    const char *e = getenv("CHOLCERT_STRICT");
    if (e && e[0] == '1') { for (int t = 0; t < 3; t++) { c->nrelax[t] = 0; c->zrelax[t] = 0.0; } }
}

int cholcert(long n, long *Ap, long *Ai, double *Ax, double *out)
{
    cholmod_common c; cholmod_l_start(&c);
    c.nmethods = 1; c.method[0].ordering = CHOLMOD_NATURAL; c.postorder = 0;
    c.supernodal = CHOLMOD_SUPERNODAL; c.final_asis = 1; c.quick_return_if_not_posdef = 1; strict_relax(&c);
    cholmod_sparse A; memset(&A, 0, sizeof(A));
    A.nrow = n; A.ncol = n; A.nzmax = Ap[n]; A.p = Ap; A.i = Ai; A.x = Ax; A.nz = NULL; A.z = NULL;
    A.stype = -1; A.itype = CHOLMOD_LONG; A.xtype = CHOLMOD_REAL; A.dtype = CHOLMOD_DOUBLE; A.sorted = 1; A.packed = 1;
    for (int t = 0; t < 10; t++) out[t] = 0.0;
    cholmod_factor *L = cholmod_l_analyze(&A, &c);
    if (!L) { out[7] = c.status; cholmod_l_finish(&c); return -1; }
    out[6] = c.fl;
    long *Perm = (long *) L->Perm;
    for (long j = 0; j < n; j++) if (Perm[j] != j) { out[7] = -100; cholmod_l_free_factor(&L, &c); cholmod_l_finish(&c); return -2; }
    cholmod_l_factorize(&A, L, &c);
    out[7] = c.status; out[8] = (double) L->minor;
    if (c.status != CHOLMOD_OK || L->minor < (size_t) n || !L->is_super || !L->is_ll) {
        cholmod_l_free_factor(&L, &c); cholmod_l_finish(&c); return 1;
    }
    long nsuper = L->nsuper, *super = (long *) L->super, *pi = (long *) L->pi, *px = (long *) L->px, *si = (long *) L->s;
    double *x = (double *) L->x;
    long *rowcnt = (long *) calloc(n, sizeof(long));
    double S = 0.0, Sw = 0.0, dmin = INFINITY, dmax = 0.0, nnz = 0.0; long kmax = 0; int ok = 1;
    for (long s = 0; s < nsuper; s++) {             /* pass 1: row counts, diagonal checks */
        long k1 = super[s], k2 = super[s+1], psi = pi[s], nsrow = pi[s+1] - pi[s], nscol = k2 - k1, psx = px[s];
        for (long cc = 0; cc < nscol; cc++) {
            double *col = x + psx + cc*nsrow, dg = col[cc];
            if (!(dg > 0.0) || !isfinite(dg)) ok = 0;
            if (dg < dmin) dmin = dg; if (dg > dmax) dmax = dg;
            for (long r = cc; r < nsrow; r++) rowcnt[si[psi + r]]++;
            long cnt = nsrow - cc; if (cnt > kmax) kmax = cnt; nnz += (double) cnt;
        }
    }
    for (long j = 0; j < n; j++) if (rowcnt[j] > kmax) kmax = rowcnt[j];
    /* pass 2: entry (i,i') of L L^T is a sum of <= min(rc_i, rc_i') products, gamma(min(a,b)+4) <= w_i w_i',
       w_i = sqrt((rc_i + 4) u / (1 - (kmax + 4) u))  ->  sum |dA| <= sum_j (sum_i w_i |L_ij|)^2 = Sw */
    double u = ldexp(1.0, -53), den = 1.0 - (kmax + 4.0)*u;
    for (long s = 0; s < nsuper; s++) {
        long k1 = super[s], k2 = super[s+1], psi = pi[s], nsrow = pi[s+1] - pi[s], nscol = k2 - k1, psx = px[s];
        for (long cc = 0; cc < nscol; cc++) {
            double *col = x + psx + cc*nsrow, sum = 0.0, sumw = 0.0;
            for (long r = cc; r < nsrow; r++) {
                double v = fabs(col[r]); if (!isfinite(v)) ok = 0;
                sum += v; sumw += v*sqrt((rowcnt[si[psi + r]] + 4.0)*u/den);
            }
            S += sum*sum; Sw += sumw*sumw;
        }
    }
    free(rowcnt);
    out[0] = ok; out[1] = nnz; out[2] = (double) kmax; out[3] = S; out[9] = Sw; out[4] = dmin; out[5] = dmax;
    cholmod_l_free_factor(&L, &c); cholmod_l_finish(&c);
    return ok ? 0 : 2;
}

/* symbolic only: out[0] xsize (doubles in supernodal L->x, incl. rectangle padding), out[1] ssize, out[2] nsuper,
   out[3] flop count, out[4] lnz (CHOLMOD estimate), out[5] status */
int cholsym(long n, long *Ap, long *Ai, double *out)
{
    cholmod_common c; cholmod_l_start(&c);
    c.nmethods = 1; c.method[0].ordering = CHOLMOD_NATURAL; c.postorder = 0; c.supernodal = CHOLMOD_SUPERNODAL; strict_relax(&c);
    cholmod_sparse A; memset(&A, 0, sizeof(A));
    A.nrow = n; A.ncol = n; A.nzmax = Ap[n]; A.p = Ap; A.i = Ai; A.x = NULL; A.nz = NULL; A.z = NULL;
    A.stype = -1; A.itype = CHOLMOD_LONG; A.xtype = CHOLMOD_PATTERN; A.dtype = CHOLMOD_DOUBLE; A.sorted = 1; A.packed = 1;
    for (int t = 0; t < 6; t++) out[t] = 0.0;
    cholmod_factor *L = cholmod_l_analyze(&A, &c);
    out[5] = c.status;
    if (!L) { cholmod_l_finish(&c); return -1; }
    out[0] = (double) L->xsize; out[1] = (double) L->ssize; out[2] = (double) L->nsuper; out[3] = c.fl; out[4] = c.lnz;
    cholmod_l_free_factor(&L, &c); cholmod_l_finish(&c);
    return 0;
}
