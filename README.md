# Lagbound

Lagbound tells you how far a graph partition can be from the best possible modularity.

Community detection tools such as Louvain and Leiden return a partition and its modularity, but not how much
better a partition could be. Lagbound computes both sides:

- a **lower bound** `L`: the modularity of the best partition it finds (exact, as a fraction);
- an **upper bound** `U`: a value that no partition of the graph can exceed, backed by a certificate that a
  separate verifier re-checks in outward-rounded arithmetic.

The true maximum lies in `[L, U]`, so no partition beats the one returned by more than `U - L`. When `L > 0` the
relative gap `(U - L) / L` says the same as a fraction. (The maximum is never negative, since putting all nodes in
one cluster gives 0.)

The code is the pipeline that produced these results on graphs with up to 117 million edges:

| Network | nodes | edges | L | U | gap |
|---|---:|---:|---:|---:|---:|
| web-Stanford | 282K | 2.0M | 0.928168 | 0.946186 | 1.94% |
| eu-2005 | 863K | 16.1M | 0.941251 | 0.960156 | 2.01% |
| web-NotreDame | 326K | 1.1M | 0.945251 | 0.971975 | 2.83% |
| ogbn-products | 2.40M | 61.9M | 0.883616 | 0.917364 | 3.82% |
| Hollywood-2011 | 1.99M | 114.5M | 0.760396 | 0.792885 | 4.27% |
| coPapersCiteseer | 434K | 16.0M | 0.922782 | 0.965636 | 4.64% |
| Hollywood-2009 | 1.11M | 56.4M | 0.754869 | 0.790502 | 4.72% |
| coPapersDBLP | 540K | 15.2M | 0.867411 | 0.912993 | 5.26% |

All upper bounds are certified; all except ogbn-products were also accepted by the independent verifier.

## Install

```bash
pip install git+https://github.com/Yahiahu/Lagbound.git
```

or from a clone:

```bash
pip install -e .
```

Requires Python 3.10+, NumPy, SciPy, Numba, python-igraph, and PyMetis. Linux and macOS. The first run compiles
the Numba kernels (about a minute); later runs use the cache in `~/.cache/lagbound/numba`.

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
| `status` | `"verified"`, `"certified"`, or `"estimate"` |
| `estimate` | the solver's bound estimate before certification |
| `certificate` | path of the certificate file (`.npz`) |
| `details` | estimates, certificate and verification summaries |
| `workdir` | directory with checkpoints and logs of every stage |
| `verify()` | runs the independent verifier on the certificate |

`bound` returns a `BoundResult` with `lower`, `upper`, `absolute_gap` (`U - L`), `gap` (`(U - L) / L`, or `None`
when `L <= 0`), `partition`, and `summary()`.

Only `"certified"` and `"verified"` values are proven bounds. An `"estimate"` is the solver's own number and can
be slightly too low.

## How it works

Modularity maximization is a clique-partitioning problem over node pairs. Lagbound bounds it with a semidefinite
relaxation strengthened by triangle inequalities and by constraints on clusters taken from Leiden partitions,
solved by a low-rank augmented Lagrangian method. Any dual point gives a valid bound; the solver only has to make
it a good one. The stages:

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
   semidefiniteness with its own factorization. See `VERIFIER_NOTES.md`.

The lower bound comes from converged Leiden runs followed by an ensemble refinement that contracts the nodes all
good partitions agree on and re-runs Leiden on the result.

## Large graphs

Time and memory grow with the graph. The numbers above took hours on 16 to 96 core machines with hundreds of GB of
RAM, mostly for the certificate. Practical notes:

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
  cli.py         command line
  _core/         the research code that produced the results above
```

`_core` is the research code. `PROVENANCE.md` lists each file's SHA-256, the research copy it was taken from, and
every edit. The solver, operators, reductions, and verifier are unchanged. The edits are packaging only: input
paths come from the working directory instead of fixed server paths (`hw_fix.py`, `lbutil.py`, `improve2.py`), and
one array is cast to float in `est_dual_vcs.py` so integer inputs work on small graphs.

The wrapper itself changes two things, both of which only take effect on small graphs:

- **Solver batches.** The solver uses 512 batches on large graphs, as in the runs above. On small graphs it uses
  fewer, because 512 would leave most batches empty.
- **Certificate shift search.** If the search reaches its 40-step limit, `pipeline.py` restarts it from the last
  shift tried. This happens when the starting estimate is far too small, as it can be on tiny graphs.

Apart from those, run settings differ between graphs, as listed in the table above.

## Tests

```bash
pip install -e ".[test]"
pytest
```

The tests check, on small graphs whose optimum is found by enumeration, that `L <= optimum <= U` and that the
verifier accepts the certificate, and on Zachary's karate club that the lower bound reaches the known optimum.

## Citation

```bibtex
@misc{khalil2026lagbound,
  title  = {Certified Upper Bounds for Modularity Maximization on Web-Scale Graphs},
  author = {Khalil, Ashraf and Aref, Samin},
  year   = {2026},
  note   = {Preprint}
}
```

## License

MIT
