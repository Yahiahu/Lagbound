"""Score a partition against a certified upper bound.

A certificate is a property of the graph, not of the partition that came with it, so any other partition of the same
graph can be scored without re-running the solver: if ``U >= Q*`` and ``Q`` is the partition's modularity, no partition
improves on it by more than ``(U - Q) / Q``. A known better partition with modularity ``L`` adds the lower limit
``max(0, L - Q) / Q``, so the improvement still available over the partition lies in

    [max(0, L - Q) / Q,  (U - Q) / Q].

``CERTIFIED`` holds the bounds of the networks in the paper; scoring against one of them checks the node and edge
counts of the graph first. Counts do not establish graph identity: the caller must ensure the input is the same
preprocessed graph used in the experiment.
"""
from __future__ import annotations

import gzip
import math
from dataclasses import dataclass
from fractions import Fraction

from .graph import as_graph
from .modularity import modularity

# Networks of the paper (after removing directions, self-loops, duplicate edges, and isolated nodes). ``upper`` is the
# certified and independently verified bound rounded up; ``lower`` the modularity of the VieClus partition rounded down.
CERTIFIED = {
    "web-Stanford": dict(nodes=281_903, edges=1_992_636, lower=0.928191, upper=0.946186),
    "eu-2005": dict(nodes=862_664, edges=16_138_468, lower=0.941574, upper=0.960156),
    "web-NotreDame": dict(nodes=325_729, edges=1_090_108, lower=0.945390, upper=0.971975),
    "ogbn-products": dict(nodes=2_400_608, edges=61_859_012, lower=0.883424, upper=0.917365),
    "Hollywood-2011": dict(nodes=1_985_306, edges=114_492_816, lower=0.759931, upper=0.792885),
    "coPapersCiteseer": dict(nodes=434_102, edges=16_036_720, lower=0.922867, upper=0.965637),
    "Hollywood-2009": dict(nodes=1_107_243, edges=56_375_711, lower=0.754784, upper=0.790501),
    "coPapersDBLP": dict(nodes=540_486, edges=15_245_729, lower=0.867665, upper=0.912993),
}


@dataclass
class Score:
    """Modularity of a partition and the improvement over it that the bounds allow.

    ``improvement`` is ``(at_least, at_most)``: relative improvement still available over the partition, rounded
    outward. ``at_least`` is 0 without a better known partition; both are ``None`` when ``modularity <= 0``.
    """

    modularity: float
    exact: Fraction
    upper: float
    lower: float | None
    absolute_gap: float
    improvement: tuple
    network: str | None = None

    def summary(self) -> str:
        lines = [f"modularity   {self.modularity:.6f}",
                 f"upper bound  {self.upper:.6f}" + (f"  ({self.network}, certified)" if self.network else "")]
        if self.lower is not None:
            lines.append(f"lower bound  {self.lower:.6f}  (best known partition)")
        lo, hi = self.improvement
        if hi is None:
            lines.append(f"improvement  at most {self.absolute_gap:.6f} in modularity")
        else:
            lines.append(f"improvement  at most {100 * hi:.2f}% (absolute {self.absolute_gap:.6f})")
            if lo:
                lines.append(f"             at least {100 * lo:.2f}% (a better partition is known)")
        return "\n".join(lines)


def _round(x: Fraction, digits: int, up: bool) -> float:
    s = 10 ** digits
    return (math.ceil(x * s) if up else math.floor(x * s)) / s


def score(graph, partition, upper: float | None = None, lower: float | None = None, network: str | None = None) -> Score:
    """Score ``partition`` on ``graph`` against an upper bound.

    Give the bound as ``upper`` (and optionally the modularity ``lower`` of a better known partition), or name one of
    the certified ``network``\\ s, whose bounds are used after a node/edge-count sanity check. The caller must
    establish graph identity independently. Explicit
    ``upper`` / ``lower`` values take precedence over the network's.
    """
    g = as_graph(graph)
    if network is not None:
        if network not in CERTIFIED:
            raise ValueError(f"unknown network {network!r}; certified: {', '.join(CERTIFIED)}")
        ref = CERTIFIED[network]
        if (g.n, g.m) != (ref["nodes"], ref["edges"]):
            raise ValueError(f"graph has {g.n} nodes and {g.m} edges, {network} has {ref['nodes']} and {ref['edges']}: "
                             "the certified bound does not apply")
        upper = ref["upper"] if upper is None else upper
        lower = ref["lower"] if lower is None else lower
    if upper is None:
        raise ValueError("give an upper bound or a certified network")
    if isinstance(partition, dict):
        missing = [name for name in g.labels if name not in partition]
        if missing:
            raise ValueError(f"partition has no cluster for {len(missing)} nodes, e.g. {missing[:3]}")
    q = modularity(g, partition, exact=True)
    U = Fraction(upper)
    if U < q:
        raise ValueError(f"upper bound {upper} is below the partition's modularity {float(q):.6f}: "
                         "the bound belongs to a different graph")
    lo = hi = None
    if q > 0:
        hi = _round((U - q) / q, 6, up=True)
        lo = _round(max(Fraction(0), Fraction(lower) - q) / q, 6, up=False) if lower is not None else 0.0
    return Score(modularity=float(q), exact=q, upper=upper, lower=lower, absolute_gap=_round(U - q, 6, up=True),
                 improvement=(lo, hi), network=network)


def read_partition(path: str) -> dict:
    """Read a ``node cluster`` file (one pair per line, as written by ``lagbound --partition-out``)."""
    opener = gzip.open if path.endswith(".gz") else open
    pairs = []
    with opener(path, "rt") as fh:
        for line in fh:
            parts = line.split()
            if parts and parts[0][0] not in "#%":
                if len(parts) < 2:
                    raise ValueError(f"{path}: expected 'node cluster' per line, got {line.strip()!r}")
                pairs.append((parts[0], parts[1]))
    try:
        return {int(a): b for a, b in pairs}
    except ValueError:
        return dict(pairs)
