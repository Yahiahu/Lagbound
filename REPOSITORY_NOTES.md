# Repository corrections and open reconciliation

Reviewed on November 18, 2026.

## Authorship and citation

The authors are **Yahia Hussein and Samin Aref**. The initial October 4, 2026 commit
(`b1c1b8e00196607fce0909c439460dd490be14fe`) contained an incorrect author name in the README citation and
copyright notice. The Git commit itself already identifies Yahia Hussein. Commit history does not establish
why the incorrect name was inserted into the file contents.

The current branch corrects the README, license attribution, and package metadata, and adds `CITATION.cff`.
The citation now identifies this software repository rather than claiming an unverified preprint title.
The license terms remain MIT.

The existing v0.1.1 tag, its automatically generated source archives, and the source tarball inside
`atris_artifact_2026-11-14.zip` retain the old file contents. They are historical snapshots. Use the current
branch and its citation for the corrected attribution. The archived research evidence and its checksums
have not been replaced.

## Historical results and the current manuscript

The repository's `CERTIFIED` lookup reproduces the bounds in `table1_bounds.csv` from the November 14 release
archive. The current manuscript, *Atris Algorithm for Large-Scale Modularity Maximization with Approximation
Guarantees*, has different upper bounds for three networks in the draft reviewed on November 18.

| Network | v0.1.1 archive and lookup U | Current manuscript U |
|---|---:|---:|
| web-Stanford | 0.946186 | 0.946599 |
| eu-2005 | 0.960156 | 0.959266 |
| web-NotreDame | 0.971975 | 0.973098 |

The three manuscript values were not located in the release archive's surviving text records during this review.
Their provenance needs to be reconciled with the corresponding graph, run, and verification records before
changing the stored lookup. This review does not establish which set should be used in the final manuscript.
The other five displayed upper bounds agree.

The README now labels its table as historical and uses `100 * (U - L) / U`, rounded upward to two decimal places,
for its "within optimal" percentages. The public API's existing `gap` and `Score.improvement` use the partition
score as denominator and retain their existing meaning. These quantities must not be interchanged.

The original eight large-network certificate files and checkpoints are unavailable. The release contains
surviving verification records, including a full ogbn-products replay JSON/log, but cannot independently replay
the eight original proofs. Tests on small graphs do not establish the validity of those historical results.

## Other corrections

- The README identifies the project as Atris. The import and command names remain `lagbound` for compatibility.
- Installation, homepage, repository, and release links use `Yahiahu/atris`.
- The three core scripts that defaulted to a personal absolute data path now default to `data/`. The wrapper
  continues to provide `DATA_DIR` explicitly. Their current hashes and edits are recorded in `PROVENANCE.md`.
- The exported Python version matches the package metadata at `0.1.1`.
- The installation example includes NetworkX, which the Python example requires.
- The README distinguishes the search time budget from total runtime and explains that the low-rank estimate
  needs certification before it is accepted as an upper bound.
- A nested conditional reduction in the optimizer failed to compile with Numba 0.68.1 and Python 3.12.
  The kernel now collects independent row totals before summing them, retaining parallel row evaluation.
  This addresses the failure pattern reported in [Numba issue 10132](https://github.com/numba/numba/issues/10132).
  Floating-point summation order can change, so identical optimization trajectories are not promised.
- The reduction-only verification subprocess now imports the verifier by its package name. Previously it
  cached functions under the bare name `verify_cert`, which a later package import could not resolve.
  A regression test exercises both entry points with a shared cache.

This is an attribution, documentation, packaging, and runtime compatibility review. It is not a new proof audit or a rerun
of the large-network experiments.

## Validation

On November 18, 2026, all 35 tests passed on macOS with Python 3.12.14, NumPy 2.5.5, SciPy 1.18.1,
Numba 0.68.1, igraph 1.0.0, and PyMetis 2025.2.2. The 11 warnings describe intentional input normalization
in the graph tests, including ignored edge weights and removed self-loops.

`CITATION.cff` passes the Citation File Format 1.2.0 schema. The wheel builds successfully and contains the
correct author names, license attribution, version, and repository links. All recorded core-file SHA-256
prefixes match the current source. The eight README rows match the archived CSV, and their percentages
were checked using decimal arithmetic with upward rounding.
