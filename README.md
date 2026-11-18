# Atris

**Large-scale modularity maximization with approximation guarantees**

**Authors** — Yahia Hussein and Samin Aref.

Atris returns a community partition together with a guarantee of how close its modularity is to the global optimum.
It combines a heuristic or user-supplied partition with a strengthened semidefinite relaxation and independent
numerical verification. The partition supplies a feasible solution, and the verified upper bound limits how much
better any partition could be.

The current manuscript reports partitions **within 1.85–4.97% of optimal modularity** on eight networks with
0.28–2.4 million nodes and 1.1–114.5 million edges. These results combine high-modularity partitions, guarantees
unavailable from heuristic scores alone, and demonstrated scalability to networks with millions of nodes.
[Reported results](#reported-results) identifies the manuscript values and the outstanding differences from the
archived records.

The Python package and command remain named `lagbound` for compatibility.

## The approximation guarantee

Let `Q*` be the maximum modularity of the input graph. Atris combines:

- a feasible partition `P`, with exact modularity `L = Q(P)` or a conservative lower bound on that score;
- a verified upper bound `U`, which no valid partition of the graph can exceed.

Together, these establish `L <= Q* <= U`. For `L > 0`, the returned partition is **within x% of optimal**, where

```text
x = 100 * (U - L) / U
```

This means `Q(P) >= (1 - x/100) * Q*`. For example, `L = 0.90` and `U = 0.93` establish that the partition is
within 3.23% of optimal, rounding the percentage upward. A smaller percentage is a tighter guarantee.

The guarantee is computed for the particular graph and partition. Atris does not promise the same percentage on
every graph or that any requested tolerance will be reached. The relaxation can remain above the true optimum
even after full optimization. For `L = 0`, this formula provides no nontrivial relative guarantee. Proximity to
optimal modularity also does not establish that the communities are unique or correct for a particular application.

## The algorithm

Algorithm 1 in the manuscript starts with the all-in-one partition, `L = 0`, and the universal bound `U = 1`.
It accepts an optional incumbent partition, optionally runs a partition heuristic, and retains the best exactly
evaluated candidate. A stronger partition raises `L` and improves the guarantee.

Atris then tightens `U` through four steps:

1. **Reduce the graph.** Remove isolated nodes and contract eligible leaves and true twins without changing
   the maximum modularity.
2. **Optimize a strengthened relaxation.** Combine positive semidefiniteness, selected triangle inequalities,
   and aggregate nonnegativity constraints. A Lagrangian method searches over a low-dimensional node embedding
   to make this computation practical on large graphs.
3. **Construct a candidate bound.** Use the optimized constraint multipliers and diagonal values to obtain
   an upper-bound estimate.
4. **Verify and correct the bound.** A fixed low rank can exclude valid partitions, so the search estimate alone
   does not establish an upper bound for every partition. Atris checks a matrix inequality and raises the candidate
   value as needed, accounting for numerical errors. Auxiliary variables express the dense terms sparsely so that
   verification can use a sparse factorization.

The algorithm keeps each improvement in the verified upper bound and stops when the requested percentage is reached
or the computational budget is exhausted. An unsuccessful verification leaves the previously accepted bound intact.
The output is the retained partition, `L`, `U`, the percentage when `L > 0`, and whether a supplied target was met.
Validity of an accepted bound does not require the low-rank search to reach a global optimum.

The current Python wrapper exposes partition search, bound computation, and verification. Its combined `bound()`
call uses Leiden with ensemble refinement. The manuscript's large-network experiments use VieClus partitions,
which can be supplied through the [incumbent workflow](#using-an-incumbent-partition). The wrapper does not yet
automate Algorithm 1's target-based stopping loop or its fallback to the previously accepted bound after rejection.
Inspect the returned status before reporting a guarantee.

## Reported results

The manuscript selects VieClus after comparing 11 heuristic configurations on 100 networks. VieClus returned a
partition on every network and achieved the highest modularity, or tied for it, among the available results on all
100. On the 60 reference instances, it matched the known optima within `1e-9`. The [full comparison](docs/RESULTS.md#selecting-the-lower-bound-heuristic)
records coverage, optimum matches, and best-or-tied counts under the stated budgets. VieClus is an external
heuristic and is not bundled with this Python package.

The eight large-network results reported in the current manuscript are:

| Network | Nodes (millions) | Edges (millions) | Within optimal (%) |
|---|---:|---:|---:|
| web-Stanford | 0.282 | 1.993 | 1.95 |
| eu-2005 | 0.863 | 16.138 | 1.85 |
| web-NotreDame | 0.326 | 1.090 | 2.85 |
| ogbn-products | 2.401 | 61.859 | 3.70 |
| Hollywood-2011 | 1.985 | 114.493 | 4.16 |
| coPapersCiteseer | 0.434 | 16.037 | 4.43 |
| Hollywood-2009 | 1.107 | 56.376 | 4.52 |
| coPapersDBLP | 0.540 | 15.246 | 4.97 |

The manuscript reports a tighter guarantee on every network than the tested triangle-only Lagrangian baseline,
whose percentages range from 3.58% to 22.61%, using the same feasible partitions. See the [full results](docs/RESULTS.md#approximation-guarantees-on-large-networks)
for `L`, `U`, and the per-network comparison.

**Record reconciliation remains open.** The upper bounds for web-Stanford, eu-2005, and web-NotreDame differ from
the v0.1.1 archive and the package's stored lookup. Their manuscript values have not been located in the surviving
release records. The [repository notes](REPOSITORY_NOTES.md#historical-results-and-the-current-manuscript) preserve
both sets of values. The original eight large-network certificates are unavailable, so the release cannot replay
those proofs. These reported results should be read with the [available evidence and limitations](#research-files-and-limitations).

## Install

```bash
pip install "lagbound[networkx] @ git+https://github.com/Yahiahu/atris.git"
```

Or from a clone:

```bash
pip install -e ".[networkx]"
```

Requires Python 3.10+, NumPy, SciPy, Numba, python-igraph, and PyMetis. These installation commands also include
NetworkX for the examples. Linux and macOS are supported. The first run compiles the Numba kernels, which adds
startup time. Later runs use the cache in `~/.cache/lagbound/numba`.

Optional: `pip install scikit-sparse` (needs SuiteSparse) makes the certificate use CHOLMOD instead of SciPy's
SuperLU. On large graphs this is required in practice.

## Usage

### Return a partition and its guarantee

```python
from fractions import Fraction
from math import ceil

import networkx as nx
import lagbound


def within_percent(lower_exact, upper_result):
    """Conservative percentage, after independent verification and for L > 0."""
    if upper_result.status != "verified" or lower_exact <= 0:
        return None
    U = Fraction(upper_result.value)
    if U < lower_exact:
        raise ValueError("The upper bound is below the partition's exact modularity")
    return ceil(10000 * (U - lower_exact) / U) / 100


G = nx.karate_club_graph()
res = lagbound.bound(G, time_limit=60, verify=True, workdir="karate-run")

print(res.partition)  # {node: community} for the returned partition
x = within_percent(res.lower.exact, res.upper)
if x is not None:
    print(f"Within {x:.2f}% of optimal modularity")
else:
    print(f"No verified relative guarantee reported (status={res.upper.status})")
```

The example computes the percentage from the exact rational partition score and rounds upward to two decimal
places. The floating-point `res.lower.value` and the formatted `summary()` are useful for display, but do not
replace `res.lower.exact` in this calculation. The `time_limit` covers the upper-bound search. Partition search,
bound estimation, certification, and verification require additional time.

`res.upper.certificate` gives the certificate path. `verify=True` requests the independent verification used in
the example. Without it, the default is certification without that separate re-check.

### Using an incumbent partition

Supply a partition obtained from VieClus, another heuristic, or an earlier run as a `{node: community}` mapping.
Using `G` and `within_percent` from the example above:

```python
# my_partition is a mapping that covers the nodes of G.
L = lagbound.modularity(G, my_partition, exact=True)
ub = lagbound.upper_bound(
    G, partition=my_partition, lower=float(L),
    time_limit=600, verify=True, workdir="incumbent-run",
)
x = within_percent(L, ub)
if x is not None:
    print(f"The supplied partition is within {x:.2f}% of optimal modularity")
else:
    print(f"No verified relative guarantee reported (status={ub.status})")
```

`upper_bound(partition=...)` uses the supplied partition to initialize the bound search. The partition remains the
feasible solution being assessed. The `lower` argument only affects progress logs. The current `bound()` wrapper
does not accept an incumbent directly, so use this workflow when supplying one.

### Reusing the upper bound

Once verified, `U` applies to every partition of the same graph. A better partition tightens the guarantee without
repeating the upper-bound computation:

```python
new_L = lagbound.modularity(G, improved_partition, exact=True)
new_x = within_percent(new_L, ub)
if new_x is not None:
    print(f"The new partition is within {new_x:.2f}% of optimal modularity")
```

The graph, preprocessing, and node mapping must match those used to compute `ub`.

The separate `score()` API reports possible improvement over a partition:

```python
if ub.status != "verified":
    raise ValueError("Independent verification is required before using this value")
s = lagbound.score(G, improved_partition, upper=ub.value)
print(s.summary())
```

Use this only after accepting `ub` as a bound. `score()` trusts the supplied value and does not verify a certificate.
Its `improvement` is `[max(0, L - Q) / Q, (U - Q) / Q]`, rounded outward, where `Q` is the supplied partition's score
and `L` is an optional better known score. Both entries are `None` when `Q <= 0`.

`lagbound.CERTIFIED` holds the [historical v0.1.1 lookup](REPOSITORY_NOTES.md#archived-bounds-used-by-the-package).
Calling `score(..., network="coPapersDBLP")` checks node and edge counts before using that stored bound. Counts alone
do not establish graph identity, and this lookup is not a new verification. Explicit `upper` and `lower` values
take precedence over the stored values.

### Graph input

Input can be a NetworkX or igraph graph, a SciPy sparse or dense NumPy adjacency matrix, an `(m, 2)` array of edges,
or the path to an edge-list file (`u v` per line, `#` and `%` comments, `.gz` allowed). Node names can be any
hashable values. Modularity is computed on the simple, undirected, unweighted graph. Directions and weights are
dropped, and self-loops and duplicate edges removed. The loader warns about several input changes, including
ignored weights and directions. The guarantee applies to the normalized graph. Isolated nodes cannot change
modularity and are returned as singleton clusters.

### Command line

```bash
# Return a partition, compute an upper bound, and verify it independently.
lagbound graph.txt --verify --partition-out partition.txt --workdir run1

# Run either component separately.
lagbound graph.txt --lower-only --partition-out partition.txt
lagbound graph.txt --upper-only --time-limit 7200 --verify --workdir bound-run

# Inspect structured results or assess a partition against an existing bound.
lagbound graph.txt --verify --json
lagbound score graph.txt partition.txt --upper 0.912993 --lower 0.867665 --json
```

The last command assumes those bounds have already been established for `graph.txt`. A partition file contains
one `node cluster` pair per line. `lagbound --help` lists the available options.

The existing CLI and `BoundResult.gap` report `(U - L) / L`, the possible relative improvement over the returned
partition. The manuscript's **within x% of optimal** uses `100 * (U - L) / U`. The two denominators differ.
The Python example above computes the manuscript's percentage. Neither a displayed gap nor a successful process
exit substitutes for checking the bound's status.

## API results and verification status

`bound()` returns a `BoundResult` containing `partition`, `lower`, `upper`, `absolute_gap`, `gap`, and `summary()`.
Its default runs both components. `lower=False` or `upper=False` selects just one.

| Result | Field | Meaning |
|---|---|---|
| `LowerBound` | `partition` | `{node: community}` for every original node |
| | `exact`, `value` | partition modularity as a `Fraction` and as a float |
| | `membership` | communities of non-isolated nodes, in `as_graph(G).labels` order |
| `UpperBound` | `value`, `status` | candidate or accepted value, with the status below |
| | `estimate` | search estimate before certification |
| | `certificate` | stored certificate path (`.npz`), when a certificate is needed |
| | `details`, `workdir` | stage summaries, checkpoints, and logs |
| | `verify()` | run the independent verifier and return its report |
| `BoundResult` | `absolute_gap` | floating-point `U - L` |
| | `gap` | floating-point `(U - L) / L`, or `None` for `L <= 0` |

| `upper.status` | Meaning |
|---|---|
| `"verified"` | certification succeeded and the independent verifier accepted the result |
| `"certified"` | certification succeeded without the independent re-check |
| `"rejected"` | the independent verifier refused the result, so do not use the value as a bound |
| `"estimate"` | an unverified search estimate, obtained with `certify=False`, with no upper-bound guarantee |

`UpperBound.verify()` returns a verification report. It does not update the object's `status` or `value`.
For the complete checked result, use `verify=True` in the original `bound()` or `upper_bound()` call.

Some instances are solved entirely by the exact reductions. If no edge remains between merged groups, keeping
them separate is optimal. The returned upper bound then rounds that optimum upward, with no solver run or stored
certificate (`certificate=None`, `details["solved_by_reduction"]=True`). With `verify=True`, the reduction checker
re-derives the merges from the original graph.

## Implementation details

The lower-bound component runs Leiden with several seeds, then refines an ensemble by contracting nodes on whose
community assignment good partitions agree and re-running Leiden.

The upper-bound component constructs Leiden groupings at the requested resolutions for aggregate constraints,
applies exact reductions, and runs the low-rank augmented Lagrangian search. A subsequent proximal phase keeps
the triangle set fixed while updating multipliers. Checkpoint evaluation and rank refinement produce candidate
bound parameters. Certification repairs the candidate and uses a sparse lifted matrix to check the required
positive-semidefiniteness condition, with explicit allowances for numerical errors.

The independent verifier checks constraint validity, exact reduction eligibility, and the matrix condition using
its own factorization. For contractions, it recomputes the required degree and neighborhood conditions from the
original graph. See [VERIFIER_NOTES.md](VERIFIER_NOTES.md) for the numerical checks and accepted constraint families.

## Large graphs

Time and memory grow with the graph. The recorded large-network runs took hours on 16 to 96 core machines with
hundreds of GB of RAM, mostly for certification and verification. Practical notes:

- `time_limit` controls the main and proximal search budget. Bound estimation, certification, and independent
  verification add time beyond that budget, so it is not a limit on total wall-clock time.
- Give the solver time: `time_limit` of several hours for graphs with tens of millions of edges. With the default
  `proximal_fraction=0.4`, the main phase is allocated 60% of that budget. The proximal phase uses the time remaining
  after preparation and the main phase, so its actual allocation can be smaller than 40%.
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
stayed on the VM disks. Level 1 denotes the partition used to initialize the upper-bound search. The recorded
searches used Leiden starts, while the manuscript evaluates the final guarantees against VieClus partitions.
Additional experimental details are in the manuscript. These settings change the bound you get. The library
does not reproduce two more parts of the runs. There, the bound was estimated every 50 iterations while the solver
ran, and the main phase was stopped by hand once it plateaued. The proximal phase then resumed from a chosen
checkpoint. `upper_bound` instead splits a fixed time budget and estimates the last checkpoint of each phase.
To repeat a run step by step, call the stages of `lagbound.pipeline.Workspace` directly.

## Code layout

```
src/lagbound/
  api.py         partition search, upper bounds, and the combined approximation result
  pipeline.py    runs each stage of _core with the production flags
  graph.py       graph input
  modularity.py  exact modularity
  score.py       score a partition against a certified bound
  cli.py         command line
  _core/         research optimization and certification routines
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
