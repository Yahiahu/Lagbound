"""Exact replay of triangle-LP certificates, without importing an LP solver.

The embedded graph uses the normalized node order of ``as_graph``. Supplying the
graph to ``verify_triangle_certificate`` checks that identity as well as the proof.
"""
from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction


FORMAT = "atris-triangle-lp-v1"


def upper_float(q: Fraction) -> float:
    """Smallest adjacent float at or above an exact rational value."""
    value = float(q)
    return math.nextafter(value, math.inf) if Fraction(value) < q else value


def graph_digest(n, edges):
    h = hashlib.sha256(f"{n}\n".encode("ascii"))
    for i, j in edges:
        h.update(f"{i} {j}\n".encode("ascii"))
    return h.hexdigest()


def verify_triangle_certificate(path, graph=None) -> dict:
    """Recompute a saved bound and optional partition score using Python integers.

    Without ``graph``, the report applies to the embedded graph only. No solver
    status, floating-point objective, or stored optimality claim is trusted.
    """
    try:
        with open(path) as fh:
            proof = json.load(fh)
        if proof["format"] != FORMAT:
            raise ValueError("unknown triangle certificate format")
        n = proof["graph"]["nodes"]
        if type(n) is not int or n < 2:
            raise ValueError("invalid node count")
        edges = proof["graph"]["edges"]
        seen, degrees = set(), [0] * n
        for edge in edges:
            if len(edge) != 2:
                raise ValueError("invalid edge")
            i, j = edge
            if type(i) is not int or type(j) is not int or not 0 <= i < j < n or (i, j) in seen:
                raise ValueError("graph must be simple and canonically ordered")
            seen.add((i, j))
            degrees[i] += 1
            degrees[j] += 1
        ordered = sorted(seen)
        if not ordered or edges != [list(e) for e in ordered]:
            raise ValueError("empty or unsorted graph")
        digest = graph_digest(n, ordered)
        if digest != proof["graph"]["sha256"]:
            raise ValueError("graph checksum mismatch")
        if graph is not None:
            from .graph import as_graph
            g = as_graph(graph)
            supplied = sorted((int(i), int(j)) for i, j in zip(g.u, g.v))
            if g.n != n or graph_digest(g.n, supplied) != digest:
                raise ValueError("certificate does not match the supplied normalized graph")
        tau = 2 * len(ordered)
        if proof["kind"] == "universal":
            q = Fraction(1)
        elif proof["kind"] == "triangle":
            scale = int(proof["scale"])
            if scale <= 0:
                raise ValueError("scale must be positive")
            residual, dual_sum = {}, 0
            for i, j, k, encoded in proof["triangles"]:
                if any(type(v) is not int or not 0 <= v < n for v in (i, j, k)) or len({i, j, k}) != 3:
                    raise ValueError("invalid triangle")
                y = int(encoded)
                if y < 0:
                    raise ValueError("negative dual multiplier")
                dual_sum += y
                for a, b, sign in ((i, j, 1), (j, k, 1), (i, k, -1)):
                    key = (min(a, b), max(a, b))
                    residual[key] = residual.get(key, 0) + sign * y
            numerator = -sum(d * d for d in degrees) * scale + dual_sum
            for i in range(n):
                for j in range(i + 1, n):
                    w = 2 * (tau * ((i, j) in seen) - degrees[i] * degrees[j])
                    numerator += max(0, w * scale - residual.get((i, j), 0))
            q = min(Fraction(1), Fraction(numerator, tau * tau * scale))
        else:
            raise ValueError("unknown proof kind")
        if q < 0 or Fraction(proof["upper_exact"]) != q:
            raise ValueError("stored upper bound disagrees with exact replay")
        report = dict(passed=True, UB=upper_float(q), upper_exact=str(q), graph_sha256=digest)
        if "partition" in proof:
            labels = proof["partition"]["membership"]
            if len(labels) != n or any(type(c) is not int for c in labels):
                raise ValueError("invalid partition")
            volumes = {}
            for c, d in zip(labels, degrees):
                volumes[c] = volumes.get(c, 0) + d
            inside = sum(labels[i] == labels[j] for i, j in ordered)
            lower = Fraction(2 * inside, tau) - Fraction(sum(v * v for v in volumes.values()), tau * tau)
            if str(lower) != proof["partition"]["modularity_exact"] or lower > q:
                raise ValueError("invalid partition score or upper bound below partition")
            report.update(lower_exact=str(lower), optimal=(lower == q))
        return report
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError, ZeroDivisionError) as err:
        return dict(passed=False, fail=str(err))
