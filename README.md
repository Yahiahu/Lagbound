# Atris

**Authors** — Yahia Hussein and Samin Aref.

Atris computes upper bounds on maximum modularity to assess the quality of graph partitions.
The Python package and command remain named `lagbound` for compatibility.

Community detection tools such as Louvain and Leiden return a partition and its modularity, but not how much
better a partition could be. Atris computes both sides:

- a **lower bound** `L`: the modularity of the best partition it finds (exact, as a fraction);
- an **upper bound** `U`: a value that no partition of the graph can exceed, backed by a certificate that a
  separate verifier re-checks in outward-rounded arithmetic.

For an accepted bound, the true maximum lies in `[L, U]`. When `L > 0`, the partition is within
`100 * (U - L) / U` percent of optimal modularity. The API's `gap` instead reports `(U - L) / L`,
which bounds the relative improvement over the returned partition. These percentages use different denominators.
Putting all nodes in one community gives modularity zero, so the maximum is never negative.

## Historical results from v0.1.1

The following values reproduce `table1_bounds.csv` in the November 14, 2026 release archive.
Three upper bounds differ from the current manuscript. See [the reconciliation notes](REPOSITORY_NOTES.md)
before using this table as a manuscript reference.

| Network | nodes | edges | L | U | Within optimal (%) |
|---|---:|---:|---:|---:|---:|
| web-Stanford | 282K | 2.0M | 0.928191 | 0.946186 | 1.91 |
| eu-2005 | 863K | 16.1M | 0.941574 | 0.960156 | 1.94 |
| web-NotreDame | 326K | 1.1M | 0.945390 | 0.971975 | 2.74 |
| ogbn-products | 2.40M | 61.9M | 0.883424 | 0.917365 | 3.70 |
| Hollywood-2011 | 1.99M | 114.5M | 0.759931 | 0.792885 | 4.16 |
| coPapersCiteseer | 434K | 16.0M | 0.922867 | 0.965637 | 4.43 |
| Hollywood-2009 | 1.11M | 56.4M | 0.754784 | 0.790501 | 4.52 |
| coPapersDBLP | 540K | 15.2M | 0.867665 | 0.912993 | 4.97 |

`L` is the modularity of a partition found by VieClus, rounded down (the library's own `lower_bound` uses Leiden);
`U` is the reported independent replay bound, rounded up. Percentages use `100 * (U - L) / U`, rounded upward
to two decimal places. The surviving records report independent verification of these historical runs.
The original large-network certificates are unavailable, so this release cannot replay them. The archive contains
a full ogbn-products replay JSON/log and excerpts or research notes for the other networks.
The ogbn-products replay took 13.1 h and peaked at about 481 GiB.

## Install

```bash
pip install "lagbound[networkx] @ git+https://github.com/Yahiahu/atris.git"
```

or from a clone:

```bash
pip install -e .
```

Requires Python 3.10+, NumPy, SciPy, Numba, python-igraph, and PyMetis. The installation command above also includes
NetworkX for the examples. Linux and macOS are supported. The first run compiles the Numba kernels, which adds
startup time. Later runs use the cache in `~/.cache/lagbound/numba`.

Optional: `pip install scikit-sparse` (needs SuiteSparse) makes the certificate use CHOLMOD instead of SciPy's
SuperLU. On large graphs this is required in practice.

## Usage

### Python

```python
import networkx as nx
import lagbound

G = nx.karate_club_graph()

res = lagbound.bound(G, time_limit=60, verify=True)
print(res.summary())
# Example output format; the upper bound and runtime depend on the run.
# lower bound  0.419790  (best partition found)
# upper bound  0.420290  (verified)
# gap          0.000500  (0.12%)

res.partition                            # {node: cluster} for the best partition
res.lower.exact                          # Fraction(1277, 3042)
res.upper.certificate                    # path of the stored certificate
```

Only one side:

```python
lb = lagbound.lower_bound(G)                         # best partition, no upper bound
ub = lagbound.upper_bound(G, time_limit=600)         # upper bound only
ub = lagbound.upper_bound(G, verify=True)            # certified, then re-checked by the verifier
ub = lagbound.upper_bound(G, certify=False)          # estimate only (faster, not a proof)
ub = lagbound.upper_bound(G, partition=my_labels)    # start the solver from your own partition
```

`upper_bound` without a partition starts from its own Leiden run, so it does not need the lower bound.
`bound` runs both and passes the lower bound's partition to the solver, which is what the production runs did.

`lagbound.modularity(G, partition, exact=True)` evaluates any partition exactly.

### Scoring a partition

A certified bound belongs to the graph, not to the partition that came with it, so any other partition of the same
graph can be scored against it without running the solver:

```python
s = lagbound.score(G, my_partition, upper=0.420290)            # your own bound U
s = lagbound.score(G, my_partition, network="coPapersDBLP")    # a bound from the table above
print(s.summary())
s.modularity, s.exact                    # Q, float and Fraction
s.improvement                            # (at_least, at_most)
```

`improvement` is the bracket `[max(0, L - Q) / Q, (U - Q) / Q]`, rounded outward: no partition improves on yours by
more than the upper end, and a known partition already improves on it by the lower end (0 without `lower`). Both are
`None` when `Q <= 0`. `lagbound.CERTIFIED` holds the historical v0.1.1 `L` and `U` values for the eight networks above.
It is a stored lookup, not a new verification of the input graph. With `network`, the graph's
node and edge counts are checked first as a sanity check. The caller must establish that the input is the same
preprocessed graph; matching counts alone do not establish graph identity.
Explicit `upper` and `lower` take precedence over the network's values.

On the command line, the partition is a file with one `node cluster` pair per line (the format `--partition-out`
writes):

```bash
lagbound score graph.txt part.txt --network coPapersDBLP
lagbound score graph.txt part.txt --upper 0.912993 --lower 0.867665 --json
```

### Graph input

Any of: a NetworkX or igraph graph, a SciPy sparse or dense NumPy adjacency matrix, an `(m, 2)` array of edges,
or the path to an edge-list file (`u v` per line, `#` and `%` comments, `.gz` allowed). Node names can be any
hashable values. Modularity is computed on the simple, undirected, unweighted graph: directions and weights are
dropped, and self-loops and duplicate edges removed, with a warning when that changes the input. The bounds are
for that graph, not for a weighted or directed version. Isolated nodes cannot change modularity; they are set aside and returned as
singleton clusters.

### Command line

```bash
lagbound graph.txt                                   # both bounds
lagbound graph.txt --lower-only --partition-out part.txt
lagbound graph.txt --upper-only --time-limit 7200 --verify --workdir run1
lagbound graph.txt --json
lagbound score graph.txt part.txt --network web-Stanford   # score a partition against a certified bound
lagbound score graph.txt part.txt --upper 0.42 --lower 0.41
```

`lagbound --help` lists all options.

## Results

`lower_bound` returns a `LowerBound`:

| field | meaning |
|---|---|
| `value`, `exact` | modularity of the partition, float and `Fraction` |
| `membership` | cluster of each non-isolated node, in the order of `as_graph(G).labels` |
| `partition` | `{node: cluster}` over all nodes |

`upper_bound` returns an `UpperBound`:

| field | meaning |
|---|---|
| `value` | the upper bound |
| `status` | `"verified"`, `"certified"`, `"rejected"` (verification was requested and failed), or `"estimate"` |
| `estimate` | the solver's bound estimate before certification |
| `certificate` | path of the certificate file (`.npz`) |
| `details` | estimates, certificate and verification summaries |
| `workdir` | directory with checkpoints and logs of every stage |
| `verify()` | runs the independent verifier on the certificate |

`bound` returns a `BoundResult` with `lower`, `upper`, `absolute_gap` (`U - L`), `gap` (`(U - L) / L`, or `None`
when `L <= 0`), `partition`, and `summary()`.

Only `"certified"` and `"verified"` values are proven bounds. A `"rejected"` value is the certifier's number after the
verifier refused it; do not use it as a bound. An `"estimate"` is the solver's own number and can
be slightly too low.

When the exact reductions leave no edge between the merged groups (a complete graph, for example, collapses to one
node), keeping every group separate is optimal, so the upper bound is the exact optimum. There is then no solver run
and no certificate (`certificate` is `None`, `details["solved_by_reduction"]` is true). With `verify=True` the
verifier's reduction check re-derives every merge from the original graph.

## How it works

Modularity maximization is a clique-partitioning problem over node pairs. Atris bounds it with a semidefinite
relaxation strengthened by triangle inequalities and by constraints on clusters taken from Leiden partitions,
solved by a low-rank augmented Lagrangian method. The low-rank search supplies candidate multipliers and a bound
estimate. Certification corrects that estimate and checks the matrix condition needed for a valid upper bound.
The stages are as follows.

1. **Leiden levels.** Leiden partitions at two resolutions for the cluster-pair rows (`levels`, default 16 and
   64) and one for the node-cluster rows (`vc_level`, default 64), plus the starting partition. They define the
   cluster aggregates that carry most of the bound's strength.
2. **Exact reductions.** Leaves and twin nodes are merged in ways that provably keep the optimum. On
   Hollywood-2011 this halves the number of nodes.
3. **Main phase.** The augmented Lagrangian solver adds violated triangles and cluster constraints, updates
   multipliers, and checkpoints.
4. **Proximal phase.** From the main phase's checkpoint, no new triangles are added and the multipliers are
   updated with four inner sweeps each. Node-cluster separation stays on until its cap, as in the production runs.
   This removes oscillation and lowers the bound further.
5. **Bound estimate.** For the final checkpoints, the dual is evaluated with a refined factor and a rank-escape
   step, and the better checkpoint is kept.
6. **Certificate.** A repaired factor and a sparse lifted factorization prove the bound. Every rounding error is
   priced into the bound with outward rounding, so the certified value is slightly above the estimate.
7. **Verification** (optional). `verify_cert.py` reloads the certificate, checks that each constraint family is
   valid for every partition, checks the exact reductions against the original graph, and re-proves positive
   semidefiniteness with its own factorization. For the reductions it recomputes from the original edge list that
   every contraction is eligible (a degree-one node and its neighbour, or adjacent nodes with identical closed
   neighbourhoods and `deg^2 < 2m`), not only that the bookkeeping is consistent. See `VERIFIER_NOTES.md`.

The lower bound comes from converged Leiden runs followed by an ensemble refinement that contracts the nodes all
good partitions agree on and re-runs Leiden on the result.

## Large graphs

Time and memory grow with the graph. The numbers above took hours on 16 to 96 core machines with hundreds of GB of
RAM, mostly for the certificate. Practical notes:

- `time_limit` controls the main and proximal search budget. Bound estimation, certification, and independent
  verification add time beyond that budget, so it is not a limit on total wall-clock time.
- Give the solver time: `time_limit` of several hours for graphs with tens of millions of edges. 60% goes to the
  main phase and 40% to the proximal phase (`proximal_fraction`).
- Set `workdir` so checkpoints, certificates, and logs are kept; without it they go to a temporary directory.
- `fill_filter=True` (`--fill-filter`) keeps only triangles compatible with the certificate's fill pattern before
  the proximal phase. Hollywood-2009, Hollywood-2011, and ogbn-products were certified this way; their
  certificates did not fit in memory otherwise.
- `verify_max_gb` (`--verify-max-gb`) caps the verifier's factor at 120 GB by default. Hollywood-2009 needed
  142 GB and was verified with 400; Hollywood-2011 peaked at 356 GB.
- `backend="cholcert"` uses a small C helper that keeps the CHOLMOD factor supernodal and saves memory. Build it
  once next to the package:

  ```bash
  cd "$(python -c 'import lagbound, os; print(os.path.dirname(lagbound.__file__))')/_core"
  gcc -O2 -shared -fPIC cholcert.c -o libcholcert.so -I/usr/include/suitesparse -lcholmod -lsuitesparseconfig -lm
  ```

- `threads` sets the solver threads (default: all cores).

### Settings of the reported runs

The defaults are the settings of the coPapersDBLP run. The other graphs used different
cluster levels and certificate settings:

| Network | `levels` | `vc_level` | other settings |
|---|---|---|---|
| coPapersDBLP | 16, 64 | 64 | defaults (margin 2) |
| coPapersCiteseer | 16, 64 | 64 | certificate margin 1.5 |
| Hollywood-2009 | 1, 4 | 4 | `max_clusters=3000`, fill filter; certificate margin 1.5, `bisect=1`, `bisect_lo=0.7`, cholcert |
| Hollywood-2011 | 1, 16 | 16 | fill filter, then a longer proximal phase on the kept triangles; margin 2 |
| ogbn-products | not recorded | not recorded | fill filter, then a proximal phase on the kept triangles |
| com-Orkut | 32, 128 | 128 | estimate only: the certificate needs about 5.7 TB |
| web graphs, eu-2005 | not recorded | not recorded | proximal phase |

"Not recorded" means the setting is not in the run notes kept with the code; the launch scripts for those runs
stayed on the VM disks. Level 1 is the starting partition, which in these runs was the best Leiden partition (the lower bound's).
The cluster counts per level are in the paper's appendix. These settings change the bound you get. The library
does not reproduce two more parts of the runs. There, the bound was estimated every 50 iterations while the solver
ran, and the main phase was stopped by hand once it plateaued. The proximal phase then resumed from a chosen
checkpoint. `upper_bound` instead splits a fixed time budget and estimates the last checkpoint of each phase.
To repeat a run step by step, call the stages of `lagbound.pipeline.Workspace` directly.

## Code layout

```
src/lagbound/
  api.py         lower_bound, upper_bound, bound
  pipeline.py    runs each stage of _core with the production flags
  graph.py       graph input
  modularity.py  exact modularity
  score.py       score a partition against a certified bound
  cli.py         command line
  _core/         the research code that produced the results above
```

`_core` is the research code. `PROVENANCE.md` records its research origins and subsequent changes.
The 0.1.1 release includes the original-graph contraction-eligibility check, exact final scalar accumulation
in the certifier, rejection of unsupported neighbour/connectivity rows, and an exact-rational acceptance check
for the evaluation error of each verifier allowance. Factorization still uses floating-point arithmetic.
Small-instance wrapper fixes handle instances solved completely by reduction and avoid a single parallel
row batch on tiny reduced graphs. These wrapper changes do not change the batch counts of the eight large runs.

## Research files and limitations

The [v0.1.1 research archive on GitHub](https://github.com/Yahiahu/atris/releases/tag/v0.1.1) includes saved partitions,
experimental records, recovered replay evidence, resource tables, numerical audits, and checksums. It also includes
a newly generated, independently verified karate-club certificate and a script to check it, plus a fresh validation
of the archived Notre Dame partition against the public SNAP graph.

The original eight large-network certificates and solver checkpoints are unavailable. Products retains a full replay
JSON/log; the other networks have historical console excerpts and/or session notes. The archive distinguishes these
records. They cannot substitute for the missing proof objects. Some per-run settings are incomplete, so regenerating
certificates requires new search and certification and is not promised to reproduce the historical values exactly.
A graph's node/edge counts alone do not establish graph identity: callers reusing a stored bound must ensure that
the graph, preprocessing and node mapping match the experiment.

The archived source retains the attribution error corrected in the current branch. The correct authors are
Yahia Hussein and Samin Aref. See [the repository notes](REPOSITORY_NOTES.md) for the correction and the
differences between the archived bounds and the current manuscript.

## Tests

```bash
pip install -e ".[test]"
pytest
```

The tests check, on small graphs whose optimum is found by enumeration, that `L <= optimum <= U` and that the
verifier accepts the certificate, and on Zachary's karate club that the lower bound reaches the known optimum.

## Citation

To cite this software repository, use the following entry or GitHub's "Cite this repository" option.

```bibtex
@software{hussein2026atris,
  title  = {{Atris}},
  author = {Hussein, Yahia and Aref, Samin},
  year   = {2026},
  url    = {https://github.com/Yahiahu/atris}
}
```

## License

MIT
