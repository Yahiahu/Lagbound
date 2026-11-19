# Small-network study, November 18, 2026

These are results from the original research runs, separate from the manuscript's eight large networks.
They combine the v0.1.1 SDP pipeline with a separately run triangle LP and external VieClus partitions.
The optional package integration is new in version 0.1.2; the entire study has not been rerun with that integration.

The 155 networks have 10–1,461 nodes and reference optimal modularities from the
[Ansótegui et al. MaxSAT benchmark](https://doi.org/10.1007/978-3-031-97629-2_3).
The records contain 152 positive optimal modularities and three complete graphs with optimum zero.
Known optima were used for subsequent comparisons, not to compute the partitions or bounds.

## Recorded results

| Statistic | Atris v0.1.1 alone | Minimum of SDP and triangle-LP bounds |
|---|---:|---:|
| Median percentage | 0.41% | 0.02% |
| Mean percentage | 1.22% | 0.37% |
| Worst percentage | 8.35% | 6.04% |
| Within 1% of optimal | 100/152 | 138/152 |
| Within 1.85% of optimal | 115/152 | 143/152 |
| Within 4.97% of optimal | 145/152 | 150/152 |

Percentages use `100*(U-L)/U`. Summary statistics use nearest rounding, while threshold counts use exact rational
comparisons. The [recorded scalars](../experiments/small_networks/recorded_155.csv) retain exact partition scores,
exact triangle bounds, and exact binary representations of the reported SDP bounds. The original script used
partition scores rounded down to nine decimal places and triangle bounds rounded up to nine places, with subsequent
floating-point conversion. Using the stored exact rationals instead avoids that conversion issue and leaves the
displayed summary statistics and these threshold counts unchanged.

Recompute the summary without an optimization solver:

```bash
python experiments/small_networks/summarize.py
```

All 155 recorded partition scores equal the reference optima. Both stored upper bounds on each network are at least
the reference optimum under exact scalar comparison. This is a consistency check, not independent proof replay.
The original collection did not retain the VieClus memberships or the complete triangle dual certificates, and
the SDP proof files are not present in this collection. Consequently, these records alone cannot independently
establish either the recorded partitions or the bounds. New package runs save triangle proof data and supplied
incumbents so those checks can be performed.

The earlier research summary called 60 positive-score cases "proven optimal" using a percentage tolerance.
That label is not retained here. A small positive gap is not exact equality. The new verifier reports optimality
only when the replayed rational upper bound equals the exactly recomputed score of its saved partition.

## Configuration and timing

The research runs used a 30-second VieClus request per network with seed zero. SDP upper bounds used the v0.1.1 defaults,
including a 3,600-second search budget and a 3,000-iteration cap per phase, with independent verification requested.
The triangle LP used one solver thread, an 8 GiB allocation estimate budget, and a configured 600-second search
budget. The SDP and LP components were run separately, and their smaller accepted bound was selected afterwards.

The recorded component runtimes across all 155 networks are:

| Component | Median (seconds) | Maximum (seconds) |
|---|---:|---:|
| VieClus | 30.01 | 30.06 |
| SDP upper-bound computation | 68.10 | 607.69 |
| Triangle-LP tightening | 0.89 | 501.66 |

The per-network timings are included in the recorded CSV and reproduced by `summarize.py`. These separately
recorded components do not provide a measured end-to-end runtime or a comparison on common hardware.

The LP reported numerical convergence on
143 instances and a solver time-limit status on 12. Those 12 runs ended after approximately 306–502 seconds,
not after 600 seconds of wall time. The research script reset HiGHS's time limit without accounting for accumulated
solver time. The package integration accounts for cumulative solver time. Therefore, trajectories and final
bounds may differ when rerun. Exact proof construction and verification also add runtime beyond the search budget.

## Cattle and high_tech_company

| Network | Nodes | Edges | Density | Recorded Q* | SDP U | U − Q* | Within optimal, rounded upward |
|---|---:|---:|---:|---:|---:|---:|---:|
| cattle | 28 | 205 | 54.23% | 0.10216538 | 0.10873160 | 0.00656622 | 6.04% |
| high_tech_company | 21 | 159 | 75.71% | 0.05464578 | 0.05811568 | 0.00346990 | 5.98% |

The recorded feasible scores match the known optima, so the remaining gaps come from upper bounds above the optimum.
The low modularity values make these small absolute differences appreciable in relative terms. The triangle LP
reported numerical convergence on both graphs, but its weaker bounds give approximately 27.41% and 22.09% gaps
(nearest rounding). It therefore does not improve the retained SDP bounds on these cases.

The SDP runs took about 40 and 38 seconds, and their certified bounds differ from their internal estimates by about
0.000596 and 0.000273. Neither short runtime nor agreement between these estimates proves that the SDP relaxation
has been optimized globally. An iteration limit can end search before its time budget is exhausted. The data do not
establish an irreducible SDP integrality gap, nor do they show that graph density caused the difficulty. In particular,
density alone does not imply that almost all modularity-matrix entries are near zero.

The comparison concerns the tested triangle LP, not every relaxation-based method. It illustrates the distinction
between finding an optimal partition and tightly certifying its quality. No branching or exact-solver fallback is
introduced in this release.
