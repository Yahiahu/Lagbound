# brk/verify — independent verifier of the modularity UB certificates (Oct 2 2026)

Files (laptop + VM `~/sdp2/tol5/brk/verify/`, logs/dumps in VM `runs/`):
* `verify_cert.py` — the verifier. Imports nothing from the certifier / estimator / ALM code.
* `certify_lifted_vc_dump.py` — copy of `../certvc/certify_lifted_vc.py`, identical numerics, plus `--dump_cert FILE.npz`
  (saves exactly what was certified: multipliers after tri-merge/filter and vc_drop, g, sdiag = shift*w, mu_c, lifted tree
  link_p/link_vol/link_ch/link_cf/root, cluster -> node/coef maps per pair and VC level, certifier UB + json).
* `test_verify_bf.py` — brute-force validation; `scr_verify.sh` — 25k screening re-certify (+dump) + verify.
* (Oct 2, later) `verify_cert.py --method sn`: supernodal multithreaded BLAS-3 Cholesky PSD proof (section "Method sn"
  below); `test_verify_bf.py` now also runs/checks sn; `dev/test_sn_unit.py` (synthetic, all code paths), `dev/scr_sn.sh`
  (25k ldl vs sn on lrcmp6), `dev/scr_tamper.sh`, `cs_verify_sn.sh` (Citeseer).  Pre-sn originals: `dev/*_orig.py` (VM).

## What is checked (from the instance files + the dump only)
1. Instance: simple graph, integer weights, sum(vol) = L_orig, sum(vol) - 2 sum(w) = C0 (contraction bookkeeping),
   vol >= weighted degree; dump's n, L, C0, vol must equal these.
2. Multipliers finite and >= 0, sig symmetric, indices in range, no repeated vertex in a constraint. Reports whether
   mu/om (neighbour / connectivity: valid only for OPTIMAL partitions) are used; all certificates below use none
   (only triangles, pair-level and VC nonnegativity: valid for every partition).
3. Lifted tree: each aux node defined by exactly one link, acyclic (strong components), child coefficients = child
   volume, vol_p = sum of coefficients; exact integer leaf vectors S_p (fixed point S = E + G S) with sum S_p = vol_p
   (so ||x_p|| <= 1), S_root = d, S_node(R) = d * 1_R for every cluster of every level.
4. Mt assembled independently (lower triangle, one term per unordered pair) + price_asm = sum gamma(h+5) sum|terms|.
5. Identity H^T (Mt - Diag(sdiag)) H = Diag(g) - C(y), C from a separate vertex-space implementation (dense for
   n <= 3000, else 6 random vectors): rel err ~1e-16 on all instances.
6. PSD: `--method ldl` own numba up-looking simplicial LDL^T (no CHOLMOD/BLAS) with a per-entry RUNNING residual bound:
   entry (k,r): |A - L D L^T|_kr <= gamma(h+3) (|A_kr| + sum_i |L_ri y_i|) + u |y_r|  (y_i = L_ki D_i before the division,
   h = |row pattern k| + 1; diagonal analogous without the u|y| term); D > 0 required; price = full-matrix sum.
   `--method eig` (N <= 15000): dense eigh + Weyl bound with ||E||_F, ||V^T V - I||_F (products bounded by gamma(N+2)|.||.|);
   valid but loose (N |lambda_lb| ~ 2e-4 on flixgiant, dominated by the large link entries) -> cross-check only.
   `--method sn`: supernodal Cholesky, BLAS-3 DGEMM + own kernels, a-priori componentwise bound (see "Method sn").
   `--method all` = ldl + eig + sn, `ldlsn` = ldl + sn.  UB = min over the methods that passed.
7. UB = (C0 + sum lam - sum mu + sum g + sum sdiag + price_asm + price_psd)/L in exact fractions, rounded up.
8. `--orig DIR --labels LAB`: original graph -> reduced instance bookkeeping exact (vol = summed degrees, C0 = 2 #inside,
   weights = edge counts), and Q_orig(labels) == (C0 + <B', X'>)/L as exact fractions.

## Validation
* Brute force (laptop, `test_verify_bf.py` seeds 2/5/7: 73 instances; original graphs n0 <= 11 with leaves + true twins ->
  reduce_exact.py -> reduced n 3..9, N <= 31; random multipliers incl. mu/om on 2/3, polished or random g; certifier copy CLI):
  all PASS (ldl and eig); |UB_ver - UB_cert| <= 2.1e-10; UB >= max over ALL partitions of the Lagrangian computed from the
  constraint definitions (min margin 5.7e-5 on polished = tight instances); without mu/om every partition has
  Lagrangian >= Q and UB >= Q*; Q*(original, all partitions) == max reduced objective (exact ints) every time; exact-
  fraction residual sum|A - L D L^T| of the verifier's own factor <= 4.0% of its running bound; tampering: sdiag decrease
  69/69, corrupted link coefficient / swapped cluster node / negative lam 69/69 each REJECTED; random g decreases rejected
  61/69 and 39/69 (the accepted ones still gave UB >= max Lagrangian: sound, only slack consumed).
* flixgiant__rx (n 1269, N 4775, 97k tri, 37.9k VC, 3 pair levels; ckpt vcsparse/val/eqS_last, est W, --shift_est):
  certifier 0.878743622 -> VERIFIED 0.878743655 (ldl, price_psd 5.6e-8, 3.6 s); eig cross-check 0.87893 (loose, valid).
  Same ckpt, other g: dense-lmin shift 0.878721910 -> 0.878721943; rgd 2000: 0.878326357 -> 0.878326391; raw V 0.883444 ok.
  Reduction: Q_orig(lb_labels) = 44077917/56685841 = 0.7775825 == reduced value exactly (LB 0.777582).
* 25k screening (lrcmp2, `scr_verify.sh`: certifier copy re-run with the original args + dump; the current certifier gives
  slightly lower UBs than the stored jsons because of the newer row-weighted rounding price; same g/shift):
  | instance | N | nnz(L) ver | cert UB (stored / rerun) | VERIFIED | verify time, RSS | Q(leiden) exact |
  |---|---|---|---|---|---|---|
  | dblp | 45.5k | 2.86e7 | 0.883639 / 0.88363544 | 0.88363570 | 105 s, 1.0 GB | 0.842175 |
  | as_skitter | 71.2k | 1.29e7 | 0.883935 / 0.88392810 | 0.88392846 | 19 s, 0.9 GB | 0.851923 |
  | amazon | 83.5k | 3.74e7 | 0.945624 / 0.94561789 | 0.94561824 | 100 s, 1.5 GB | 0.913165 |
  | us_patents | 83.3k | 5.47e7 | 0.872454 / 0.87244647 | 0.87244690 | 196 s, 1.4 GB | 0.853919 |
  Identity rel err <= 1.3e-16; reduction bookkeeping + Q identity exact on all four.
* coPapersCiteseer__rx: SKIPPED — no certificate exists (only est jsons; ckpt has 13.4M tri + 140M VC entries, est UB 0.9654).
## Limits
LDL is simplicial, single-threaded: ~0.75 GFlop/s (flops = sum colcount^2), memory 16 B/nnz(L) + ~100 B per assembled
term. Practical: flops <= ~3e12 (1 h), nnz(L) <= ~1.5e9 in 30 GB. Covers the 25k screening and fx-scale pieces up to
~1e12 flops; NOT fxp4 (3.2e13 flops, ~12 h) or hollywood (1.4e15). eig method: N <= 15000. Bigger -> `--method sn`
(below): ~1.2e11 sum-colcount^2 per second on lrcmp6 (12 vCPU), memory 8 B per stored L entry (+ ~2% padding) + ~18 GB
fixed for a Citeseer-size assembly.

## Command
    DATA=~/sdp2/data; export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
    # 1. certificate dump (same args as the original certification + --dump_cert)
    ~/env/bin/python brk/verify/certify_lifted_vc_dump.py TAG CK.npz --initV W.npz --shift_est EST.json --bisect 3 --dump_cert CERT.npz
    # 2. verify (exit code 0 = VERIFIED)
    ~/env/bin/python brk/verify/verify_cert.py $DATA/TAG CERT.npz --method ldl [--orig $DATA/TAG_without__rx --labels LAB.npy] --out R.json
    # big certificates: --method sn with NUMBA_NUM_THREADS=OPENBLAS_NUM_THREADS=#cores (options --sn_W 256 --sn_nb 64 --sn_max_GB 120)

## Method `sn` (Oct 2 2026): supernodal BLAS-3 Cholesky with an a-priori componentwise bound
Route (b) of the plan: a floating-point Cholesky A + dA = L L^T (L = computed factor, exactly PSD product) with a
rigorous ENTRYWISE bound on dA, summed over the full matrix (|Xt_kl| <= 1 makes the entrywise sum the right price; no
N * ||.||_2 scaling as in a Rump-type diagonal shift).

**Theorem used.** Let A (N x N, binary64, symmetric) be the assembled fl(Mt) in a symmetric order (any order is valid).
Suppose an algorithm produces finite l_jj > 0 and l_ij (i > j) with
    l_jj = fl(sqrt(s_jj)),   s_jj = a_jj - sum_{k in K_jj} l_jk^2,          s_jj > 0 (tested as `not (s > 0)` -> reject, catches NaN)
    l_ij = fl(s_ij / l_jj),  s_ij = a_ij - sum_{k in K_ij} l_ik l_jk         (true division, no reciprocal)
where each s is evaluated in floating point in ANY order (any parenthesisation, partial sums stored anywhere, products
rounded or fused (FMA)), and K_ij contains every k < j for which l_ik l_jk is not an exact zero (extra exact-zero products
are allowed).  Let h_ij = #{k < j : l_ik, l_jk structurally nonzero}.  Then, without underflow,
    |A - L L^T|_ij <= gamma(h_ij + 2) (|L| |L|^T)_ij ,     gamma(k) = k u / (1 - k u),  u = 2^-53.
Proof: Higham, *Accuracy and Stability of Numerical Algorithms* (2nd ed., SIAM 2002), Lemma 8.4 ("no matter what the
order of evaluation") and the proof of Thm 10.3 (Cholesky: |dA| <= gamma(n+1)|R^T||R|), with n replaced by the number of
products actually summed.  Any-order detail: in a summation tree with leaves c = a_ij and the m products, the factor
(1+d) of an internal node multiplies every leaf below it; dividing the identity by the path factor of c, the common
ancestors cancel, so each product carries at most (its own rounding) + (#internal nodes on the leaf-to-leaf path) <= m
factors (1+d)^{+-1}; the division (or sqrt, squared) adds <= 2 more, giving 1 + theta(m+2) (Higham Lemma 3.1).  An FMA
node only removes a product rounding.  We use gamma(h+3) (one spare rounding).
Structure: h_ij <= min(rc_i, rc_j), rc_i = # strictly-lower structural nonzeros in row i of the Cholesky factor of A's
pattern (computed by the verifier from A's pattern: etree + row-subtree count `_colrowcounts`).  Entries stored by
supernode amalgamation ("padding") are EXACT zeros: a padded (i,j) has a_ij = 0 and for every k one factor of l_ik l_jk is
padded -> 0 by induction; so padding adds no products to h.  With gamma(min(a,b)) <= sqrt(gamma(a) gamma(b)):
    sum_{i,j} |A - L L^T|_ij <= sum_k ( sum_i sg_i |l_ik| )^2 ,   sg_i = sqrt(gamma(rc_i + 3))       (O(nnz L) to evaluate)
Underflow / flush-to-zero (either may occur inside BLAS): every rounding then has an extra absolute error <= 2^-1022
(additions too under FTZ); per entry this adds <= 4(2h+3) 2^-1022 + 2 |l_jj| 2^-1022, so we add
    under = 2 (2 store) (4 (2 hmax + 3) + 2 max|l|) 2^-1022            (store = stored entries incl. padding; ~1e-290)
Overflow: an inf/NaN in any partial result that reaches L cannot be cancelled, so "all stored entries of L finite" (checked)
excludes it.  price_psd = 1.001 (sum_k (...)^2 + under) (the 1.001 covers the float evaluation of the sums, rel. err.
< 1e-6).  Then <A, Xt> = <L L^T, Xt> - <dA, Xt> >= -price_psd for every PSD Xt with |Xt_kl| <= 1, and step 7 of the
verifier adds price_psd to the UB exactly as for `ldl`.

**Implementation and the hypotheses.**
* Ordering: METIS ND on non-dense rows, dense rows (deg > 1000) last (same as `ldl`), then an etree postorder. Untrusted.
* Symbolic: etree, column/row counts (own numba), fundamental supernodes, CHOLMOD-style relaxed amalgamation of the
  adjacent child (nrelax 4/16/48, zrelax 0.8/0.1/0.05), split to width <= W = 256.  Untrusted for soundness, but checked
  for consistency: every supernode's computed row structure has exactly cnt(last column) rows; every entry of A and every
  update row is found in its target's structure, otherwise verification is REFUSED (an entry cannot be dropped silently).
* Numeric, right-looking, one supernode after the other (parallelism inside the kernels):
  own numba kernels (no fastmath) for the diagonal sub-blocks (width nb = 64: sqrt, true division, pivot test) and the
  triangular solve of the rows below (row-parallel, each entry a - sum_k ... then / l_jj); everything else is DGEMM from
  scipy's OpenBLAS 0.3.30 (scipy-openblas32, called through `scipy.linalg.cython_blas` with pointers + leading dimension):
  (i) trailing updates inside a supernode, (ii) updates of ancestor supernodes, either directly in place
  (C := C - A B^T, beta = 1, when the target rows are contiguous) or into a temporary T := A B^T (beta = 0) that our
  scatter subtracts from the target.  alpha = +-1 and beta in {0,1} are exact.  Upper triangles of diagonal blocks are
  scratch (written by full-rectangle GEMMs, never read).
* Arithmetic assumptions (stated for reviewers): IEEE binary64, round-to-nearest (the process default; nothing changes
  MXCSR); DGEMM is CONVENTIONAL: each output entry is fl of c_ij + sum over the exact set of products in some order
  (blocking, threading, FMA, vector accumulators all allowed; Strassen-type fast multiplication NOT; OpenBLAS never uses
  it).  No other BLAS/LAPACK routine (no dpotrf/dtrsm whose internal reciprocal/inverse use would need extra analysis).
* Hence every final s_ij is one summation tree over {a_ij} U {-l_ik l_jk} U {exact zeros}; that EVERY product appears
  exactly once is the algorithm's correctness, validated by (1) exact rational residuals, (2) agreement with dense
  Cholesky to 1e-15 on synthetic matrices forcing every code path (W, nb down to 2/1, scatter chunk budget 5),
  (3) an a-posteriori random-sign test run on EVERY certificate: |A z - L (L^T z)| <= sg o (|L| (|L|^T sg)) + gamma terms
  must hold componentwise (if it fails the verifier refuses; it is a bug detector, not part of the proof).
* References: N.J. Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., SIAM 2002 (Lemma 3.1, Sec. 3.1,
  Lemma 8.4, Thm 10.3, Sec. 2.1 eq. (2.8) for underflow); S.M. Rump, "Verification of positive definiteness", BIT 46
  (2006) 433-452 (the same floating-point Cholesky backward-error idea, used there with a diagonal shift for a spectral
  statement; here the entrywise sum suffices because |Xt| <= 1).  Route (a) (Rump shift) was not used.

**Validation of sn (lrcmp6; all runs below used verify_cert.py md5 766baaee3fe040964738cd09bcbb4f32; afterwards only the module docstring was edited -> md5 da527430ba73888e46b35147bedd843c, `diff` = docstring lines only).**
* `dev/test_sn_unit.py`: 9 synthetic sparse SPD matrices (N 40..4000, hub rows) incl. W/nb down to 4/2, 5/3 and scatter
  chunk budget 7 (forces splitting, direct GEMM, chunked scatter): factor == numpy dense Cholesky to <= 2.1e-15 rel.;
  exact rational sum|A - L L^T| <= 3.5..4.7% of the sn price (N <= 70); random-sign check ratio <= 8e-3; 3 indefinite
  matrices rejected (status 1).
* Brute force `test_verify_bf.py` seeds 2/5/7 (69 valid instances): sn PASSES on all, |UB_sn - UB_ldl| <= 3.0e-10,
  UB_sn >= max Lagrangian over all partitions; exact rational residual of the sn factor <= 10.8% of its a-priori price
  for (W, nb) = (256,64), (2,1), (3,2) (every column its own / split supernode); tampering: sn rejects EXACTLY the same
  cases as ldl (sdiag 69/69, coef 69/69, node 69/69, neg 69/69, g 61/69, gall 42/69; accepted ones UB >= max Lagrangian).
* 25k screening (dumps regenerated on lrcmp6 with the scr_verify.sh args, `dev/scr_sn.sh`, 6 threads; ldl reproduces
  the lrcmp2 values to 1e-9):
  | instance | cert UB (rerun) | VERIFIED ldl | VERIFIED sn | sn price_psd | sn factor | verify total (ldl+sn), RSS |
  |---|---|---|---|---|---|---|
  | dblp | 0.883635440 | 0.883635699 | 0.883635431 | 8.5e-08 | 3.0 s (ldl 68 s) | 80 s, 1.4 GB |
  | as_skitter | 0.883928103 | 0.883928460 | 0.883928097 | 9.2e-08 | 1.6 s (ldl 6.5 s) | 18 s, 1.1 GB |
  | amazon | 0.945617894 | 0.945618238 | 0.945617883 | 1.0e-07 | 4.0 s | 88 s, 2.3 GB |
  | us_patents | 0.872446472 | 0.872446899 | 0.872446458 | 1.4e-07 | 5.4 s | 155 s, 2.2 GB |
  (sn price ~4x below ldl's running bound: no |A| term and per-row gamma.)  Tampering sdiag*0.5 and sdiag*0.9 on all four:
  ldl AND sn REJECT 8/8 (pivot failure in the dense tail, same fraction for both).
* Citeseer tampering (sn, `dev/cs_tamper*.sh`): sdiag*0.5 -> REJECTED (pivot failure at column 434,877 of 434,880,
  34 min); sdiag*0.9 -> ACCEPTED with VERIFIED UB 0.965570203 (sound: the price is charged on the tampered, smaller
  shift; it shows the certifier's `--margin 1.5` shift has >= 10% slack, i.e. a ~6.5e-5 tighter Citeseer UB is available
  from the same multipliers).

**coPapersCiteseer__rx (lrcmp6, `cs_verify_sn.sh`, 12 threads, Oct 2 19:21-19:57 UTC).** Dump `runs/citeseer_cert.npz`
(md5 7126b286cb09646b2965a751a3608d65; certifier rerun UB 0.965635665, its price_round 1.08e-6).
N 434,880, tri 13.08M, VC entries 124.8M, nterms 171M, nnz lower(Mt) 139.7M, price_asm 7.7e-10, identity rel err 1.85e-16.
sn: METIS-ND + 64,327 dense last + postorder; nnz(L) 4.147e9 (= certifier's symbolic count), 101,910 supernodes, storage
4.213e9 (31.4 GB, padding 1.6%), sum colcount^2 2.133e14, max row count 203,489, max col 76,719.
Factor 1784 s = DGEMM 608 s (2.27e14 flop, 374 GFlop/s) + scatter 927 s (4.1e11 entries) + own panel kernels 206 s +
plan 42 s; min pivot l_jj 0.386, max|l| 5.7e5, all finite; random-sign check ratio 2.0e-4 (<= 1 required).
price_psd = 31.71 (+ underflow 1e-291) -> 9.897e-7 / L.
**VERIFIED UB 0.965635578** (certifier 0.965635665; dv + shift 0.965634588).  Reduction check: bookkeeping ok, labels
`coPapersCiteseer__rx/leiden_r1_s1.npy`: Q_orig = 474635568999419/514352776716800 = 0.9227821652 == reduced, exact.
Gap (UB - Q)/Q = 4.644%.  Wall 36:18 (load+checks 290 s, factor 1784 s, bound + check 80 s), peak RSS 50.5 GB, 674% CPU.
Files: `runs/citeseer_verify.json`, `runs/citeseer_verify.log`.

**Scaling / projection.** On lrcmp6 the factor runs at ~1.2e11 (sum colcount^2)/s; time split GEMM 34% / scatter 52% /
own kernels 12%.  hollywood-2009 (nnz(L) ~4e10, flops 1.4e15-4e15): storage ~41e9 x 8 B = 330 GB (+~2%) + assembly
(~2x Citeseer's 18 GB, scaled by #terms) -> ~400 GB peak: fits a 600 GB machine; time at lrcmp6 speed 3.3-9.4 h
(linear in flops); on a 600 GB machine with >= 32 physical cores the GEMM part scales with cores and the scatter
(prange, memory-bound) with bandwidth -> plausibly 1-4 h.  Not tested beyond Citeseer; the scatter share depends on the
structure (a larger dense tail shifts work to GEMM).

**Caveats a reviewer may raise.**
1. Trust in the BLAS: soundness assumes conventional DGEMM (any order/FMA/threads; no Strassen) and IEEE binary64
   round-to-nearest.  Only DGEMM is used; cross-check possible by re-running with another conventional BLAS (reference
   BLAS / MKL / BLIS) and comparing UBs.
2. Algorithmic correctness (every product exactly once) is validated (exact residuals, dense comparison, random-sign test
   on every run), not formally proved; the ldl method (no BLAS at all) agrees to <= 4.4e-7 in UB on the 25k certificates
   and 3e-10 on brute force but cannot run Citeseer (~80 h).
3. The a-priori bound is an upper bound (exact residuals are 4-11% of it); the price (9.9e-7) is ~0.002% of the
   Citeseer gap, so tightness is immaterial.
4. The verifier shares numpy/scipy/numba with the certifier but none of its code; the certifier factors with CHOLMOD
   (cholcert, different OpenBLAS build from ~/local/ob); the verifier uses scipy-openblas32 0.3.30 DGEMM.  The ordering
   heuristic (ND + dense last) is similar to the certifier's but is untrusted (any symmetric permutation is valid).
5. The random-sign check is probabilistic and is NOT part of the proof (it only refuses on inconsistency).
