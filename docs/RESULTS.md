# Results reported in the Atris manuscript

These tables follow the November 18, 2026 draft of *Atris Algorithm for Large-Scale Modularity Maximization with
Approximation Guarantees*, by Yahia Hussein and Samin Aref. They describe the manuscript's experiments. The
package's default lower-bound heuristic is Leiden with ensemble refinement, whereas the large-network results
below use externally generated VieClus partitions.

## Selecting the lower-bound heuristic

A stronger partition raises the feasible lower bound and tightens the approximation guarantee without a new
upper-bound computation. The comparison covers 60 instances with known optimal modularity, two additional real
networks, and 38 synthetic networks, with up to 128,000 nodes. All inputs are simple undirected graphs.

Each run has a five-minute limit and a 1,200 MiB memory limit, with two threads where supported and one fixed
random seed. Different internal stopping rules mean that these resource limits are not an equal-runtime
comparison. Skipped and unsuccessful runs are not counted as matches.

| Method | Returned (of 100) | Optimum matched (of 60) | Best or tied (of 100) |
|---|---:|---:|---:|
| VieClus | 100 | 60 | 100 |
| Combo | 81 | 44 | 58 |
| GNN-2500 | 70 | 34 | 38 |
| Leiden | 100 | 31 | 42 |
| GNN-100 | 95 | 30 | 32 |
| LN (RB-config) | 100 | 29 | 39 |
| Louvain | 100 | 23 | 25 |
| EdMot-Louvain | 98 | 21 | 23 |
| CNM | 99 | 20 | 20 |
| Belief | 72 | 11 | 11 |
| Paris | 100 | 9 | 9 |

An optimum or best score is matched within `1e-9` in modularity. Best scores are computed over available results
on each network. GNN-2500 returned results on 59 of the 60 reference instances. VieClus's complete coverage and
high modularity under these conditions motivate its use for the large-network experiments. These observations
do not establish that it will be optimal on other networks.

## Approximation guarantees on large networks

For each network, a VieClus partition supplies `L`. A verified upper bound `U` yields the statement that the
partition is **within `100 * (U - L) / U` percent of optimal modularity**. The manuscript reports the following
values, with `L` rounded downward, `U` rounded upward, and percentages rounded upward to two decimal places.

| Network | Nodes (millions) | Edges (millions) | L | U | Atris within optimal (%) | Triangle-only LR within optimal (%) |
|---|---:|---:|---:|---:|---:|---:|
| web-Stanford | 0.282 | 1.993 | 0.928191 | 0.946599 | 1.95 | 3.93 |
| eu-2005 | 0.863 | 16.138 | 0.941574 | 0.959266 | 1.85 | 4.59 |
| web-NotreDame | 0.326 | 1.090 | 0.945390 | 0.973098 | 2.85 | 3.58 |
| ogbn-products | 2.401 | 61.859 | 0.883424 | 0.917365 | 3.70 | 11.55 |
| Hollywood-2011 | 1.985 | 114.493 | 0.759931 | 0.792885 | 4.16 | 22.61 |
| coPapersCiteseer | 0.434 | 16.037 | 0.922867 | 0.965637 | 4.43 | 6.62 |
| Hollywood-2009 | 1.107 | 56.376 | 0.754784 | 0.790501 | 4.52 | 22.31 |
| coPapersDBLP | 0.540 | 15.246 | 0.867665 | 0.912993 | 4.97 | 12.64 |

The upper bounds for the first three networks differ from the v0.1.1 archive and the package's `CERTIFIED` lookup.
Their manuscript values have not been located in the surviving release records. The [reconciliation notes](../REPOSITORY_NOTES.md#historical-results-and-the-current-manuscript)
record the differences and preserve the archived values. The original eight certificate files are unavailable,
so these historical results cannot be independently replayed from this release.

The large-network VieClus runs used 12 parallel processes, seed zero, and a requested internal time limit of
3,600 seconds. Actual runtimes were 1.0–2.1 hours. Their partitions are held fixed when comparing upper-bound
methods. The triangle-only Lagrangian baseline uses a 20-hour budget. The comparison concerns the reported
configurations and budgets, with a tighter Atris guarantee on all eight networks.

The manuscript also compares numerical objectives from a basic SDP. Those objectives are unverified and are not
established modularity upper bounds, so they are not used to report approximation guarantees here.

## Available records

The [v0.1.1 GitHub release](https://github.com/Yahiahu/atris/releases/tag/v0.1.1) contains saved partitions,
experimental records, recovered verification evidence, resource tables, numerical audits, and checksums.
For the large-network certificates, ogbn-products retains a full replay JSON/log. Other networks have historical
console excerpts or research notes. These records do not replace the missing certificate files.

The package's small-graph tests and the release's newly generated karate-club certificate provide separate
checks of the software. They do not re-verify the historical large-network results. See the
[README](../README.md#research-files-and-limitations) and [repository notes](../REPOSITORY_NOTES.md) for the
remaining limitations.
