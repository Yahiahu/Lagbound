# Provenance of `src/lagbound/_core`

Source paths are relative to the research directory (`decomp_research_2026-09-29`) or, where noted, to the copies
pulled from the solver VMs. Hashes are the first 16 hex digits of SHA-256 of the file as shipped here.

| file | SHA-256 | taken from | edits |
|---|---|---|---|
| `__init__.py` | `e3b0c44298fc1c14` | new | empty package marker |
| `aggops.py` | `201a5e452c435880` | aggops.py | unchanged |
| `alm_vcs.py` | `21f7f7a445aaa6a0` | tol5/vcsparse/alm_vcs.py | unchanged |
| `certify_lifted_vc.py` | `065e756da6dae910` | tol5/brk/verify/certify_lifted_vc_dump.py | adds `--dump_only` (write the certificate at a given shift without factorizing), as in the copy used to dump the Hollywood-2009 and Hollywood-2011 certificates; certification arithmetic unchanged |
| `cholcert.c` | `74cd3733454f12c3` | tol5/brk/certvc/cholcert.c | unchanged |
| `est_dual_vcs.py` | `847ab854b2b62af8` | tol5/vcsparse/est_dual_vcs_esc.py (rank-escape version, deployed on the VMs Oct 2) | line 330: `X = np.asarray(X, dtype=np.float64).reshape(n, -1)` instead of `X = X.reshape(n, -1)` |
| `hw_fix.py` | `a7c0bf63754a623a` | tol5/brk/certvc/hw_fix.py | imports the certifier from its own directory and reads the data path from `DATA_DIR` instead of a fixed server path |
| `improve2.py` | `6074acd1c40bc686` | big/lbimp/improve2.py (copy from the VMs) | line 80: skips the registry's previous-best start when there is none |
| `lbutil.py` | `5f26c39184207928` | lb_improve/lbutil.py | the hard-coded registry of research graphs and paths is replaced by a lookup under `DATA_DIR`; output goes to `LAGBOUND_OUT` |
| `leiden_one.py` | `455cf2ed74d54f3f` | big/leiden_one.py (copy from the solver VM) | unchanged; the older tol5 copy has a different command line |
| `nestops.py` | `96260879281630f2` | tol5/nested/nestops.py | unchanged |
| `reduce_exact.py` | `000198893c0a0301` | tol5/reduce_exact.py | unchanged |
| `vcops.py` | `ff77515b720c8d3b` | tol5/nested/vcops.py | unchanged |
| `vcsops.py` | `df8ce02f2d8e8eff` | tol5/vcsparse/vcsops.py | unchanged |
| `verify_cert.py` | `abbc78cf9b2c6d9e` | tol5/brk/verify/verify_cert.py | unchanged |

What this does and does not establish: the shipped files match the research copies above, with the listed edits.
The run notes record that the VM code matched these copies at the Oct 1 launch and which later updates were
deployed (rank escape, the dump option), but per-run hashes were not stored, so a byte-level match with what each
VM executed cannot be proven after the fact. The certificates
themselves do not depend on this where they were replayed: seven of the eight reported certificates were
re-checked by the independent verifier from their stored proof files. The ogbn-products certificate was not
replayed.

The wrapper (`pipeline.py`, `api.py`) passes the production flags listed in the README. Per-graph settings
(cluster levels, triangle filter, certificate margin and bisection) are parameters; the README lists the
values used for each reported graph.
