# Provenance of `src/lagbound/_core`

Source paths are relative to the research directory (`decomp_research_2026-09-29`) or, where noted, to the copies
pulled from the solver VMs. Hashes are the first 16 hex digits of SHA-256 of the file as shipped here.

| file | SHA-256 | taken from | edits |
|---|---|---|---|
| `__init__.py` | `e3b0c44298fc1c14` | new | empty package marker |
| `aggops.py` | `201a5e452c435880` | aggops.py | unchanged |
| `alm_vcs.py` | `81916424d30f6e48` | tol5/vcsparse/alm_vcs.py | Nov 18 2026: personal absolute data-path fallback replaced by `data/`; `DATA_DIR` override unchanged (previous hash `21f7f7a445aaa6a0`); `level_aug` collects per-row totals before reduction to avoid Numba issue 10132 (same real-arithmetic formula; floating-point summation order may differ) |
| `certify_lifted_vc.py` | `72467204e88a18fa` | tol5/brk/verify/certify_lifted_vc_dump.py | adds `--dump_only` (write the certificate at a given shift without factorizing), as in the copy used to dump the Hollywood-2009 and Hollywood-2011 certificates. Nov 14 2026: the final total is accumulated in exact rationals and rounded up once (`certified_total`; it was nested `math.fsum` + one `nextafter`, which can fall about 2^-53 (\|const\| + \|sum g\|) below the exact sum), and checkpoints with neighbour/connectivity (mu/om) multipliers are refused. Allowances, factorization and shift search unchanged (was `065e756da6dae910`). Nov 18 2026: personal absolute data-path fallback replaced by `data/`; `DATA_DIR` override unchanged (previous hash `1933f77e77e3bf88`) |
| `cholcert.c` | `74cd3733454f12c3` | tol5/brk/certvc/cholcert.c | unchanged |
| `est_dual_vcs.py` | `7b20b1f11aefa548` | tol5/vcsparse/est_dual_vcs_esc.py (rank-escape version, deployed on the VMs Oct 2) | line 330: `X = np.asarray(X, dtype=np.float64).reshape(n, -1)` instead of `X = X.reshape(n, -1)`. Nov 18 2026: personal absolute data-path fallback replaced by `data/`; `DATA_DIR` override unchanged (previous hash `847ab854b2b62af8`) |
| `hw_fix.py` | `a7c0bf63754a623a` | tol5/brk/certvc/hw_fix.py | imports the certifier from its own directory and reads the data path from `DATA_DIR` instead of a fixed server path |
| `improve2.py` | `6074acd1c40bc686` | big/lbimp/improve2.py (copy from the VMs) | line 80: skips the registry's previous-best start when there is none |
| `lbutil.py` | `5f26c39184207928` | lb_improve/lbutil.py | the hard-coded registry of research graphs and paths is replaced by a lookup under `DATA_DIR`; output goes to `LAGBOUND_OUT` |
| `leiden_one.py` | `455cf2ed74d54f3f` | big/leiden_one.py (copy from the solver VM) | unchanged; the older tol5 copy has a different command line |
| `nestops.py` | `96260879281630f2` | tol5/nested/nestops.py | unchanged |
| `reduce_exact.py` | `ee611ccc2e6b2016` | tol5/reduce_exact.py | line 48: the log line prints max weight 0 when no edge is left between groups (it crashed on complete graphs); reduction unchanged |
| `vcops.py` | `ff77515b720c8d3b` | tol5/nested/vcops.py | unchanged |
| `vcsops.py` | `df8ce02f2d8e8eff` | tol5/vcsparse/vcsops.py | unchanged |
| `verify_cert.py` | `9c3be78abbacba6d` | tol5/brk/verify/verify_cert.py | Oct 22 2026 (47afdca): contraction-eligibility check (`contraction_witness`, `check_contractions`) recomputed from the original graph. Nov 14 2026: certificates with neighbour/connectivity (mu/om) multipliers are refused instead of reported; every error allowance passes `accept_price`, an exact-rational check that the charged value fl(1.001 x computed) dominates the allowance after its own evaluation error (gamma of the rounding count, plus gradual underflow); the assembly check requires at most 1.6e8 terms per entry. Factorizations and the allowance formulas are unchanged (research copy `abbc78cf9b2c6d9e`); Nov 18 2026: document canonical package import for cached functions; verifier calculations unchanged |

What this does and does not establish: the shipped files match the research copies above, with the listed edits.
The run notes record that the VM code matched these copies at the Oct 1 launch and which later updates were
deployed (rank escape, the dump option), but per-run hashes were not stored, so a byte-level match with what each
VM executed cannot be proven after the fact. The certificates
themselves do not depend on this where they were replayed: all eight reported certificates were re-checked by the
independent verifier from their stored proof files (ogbn-products on Oct 12-13 2026, 13.1 h, 481 GiB). The original proof
files and checkpoints are unavailable: some cloud copies were deleted and remaining inaccessible copies have not
been recovered. The release archive distinguishes full replay records, console excerpts and session notes.
The released code adds checks after the historical runs; those changes are not represented as fresh large-network replays.

The wrapper (`pipeline.py`, `api.py`) passes the production flags listed in the README. Per-graph settings
(cluster levels, triangle filter, certificate margin and bisection) are parameters; the README lists the
values used for each reported graph.

Wrapper changes after the Oct 28 2026 check against the 155 MaxSAT-proven optima (Ansótegui et al., ICCS 2025):
the solver gets at least 8 row batches (`min(n, max(8, n // 64))` below 32,768 reduced nodes, 512 above, as
before), because a single batch is a Jacobi sweep that locked antipodal rows on 15 small reduced instances and let
the bound diverge; and instances the reductions solve completely (no edge left between groups) return the exact
optimum, checked by the verifier's reduction check, instead of running the solver. Instances with 512 or more
reduced nodes, which include all eight reported graphs, get the same batch count as before.

November 18, 2026 wrapper correction: reduction-only verification imports
`lagbound._core.verify_cert` consistently, preventing caches created under the bare name `verify_cert`
from failing when loaded through the package. The test suite exercises subprocess and package use with a shared cache.
