"""Optional small-instance triangle LP with exact, independently replayable bounds.

Adapted from the November 18, 2026 experiment. For same-community variables
0 <= x <= 1, triangle rows Gx <= 1, and any y >= 0,
max w.x <= sum(y) + sum(max(0, w - G.T y)). The objective numerator has
w_ij = 2*(2m*A_ij - d_i*d_j) and constant -sum(d_i**2), divided by (2m)**2.
Solver duals only suggest y. Integer arithmetic establishes the saved bound.
"""
from __future__ import annotations

import heapq
import json
import math
import os
import tempfile
import time
from fractions import Fraction

import numpy as np

from .graph import as_graph
from .modularity import as_membership, modularity
from .triangle_certificate import FORMAT, graph_digest, upper_float, verify_triangle_certificate


SCALE = 1 << 24
BYTES_PER_PAIR = 200
BYTES_PER_ROW = 4096


def pair_lp_memory_gb(n):
    """Conservative allocation estimate in GiB, not an OS-enforced RSS limit."""
    return (BYTES_PER_PAIR * n * (n - 1) // 2 + 32 * n * n) / 2 ** 30


def _check_options(time_limit, memory_gb, target_percent):
    if not math.isfinite(time_limit) or time_limit < 0:
        raise ValueError("tightening time_limit must be finite and nonnegative")
    if not math.isfinite(memory_gb) or memory_gb < 0:
        raise ValueError("tightening memory_gb must be finite and nonnegative")
    if target_percent is not None and (not math.isfinite(target_percent) or not 0 <= target_percent <= 100):
        raise ValueError("target_percent must be between 0 and 100")


def _target_met(lower, upper, target_percent):
    return (lower is not None and lower > 0 and target_percent is not None and
            100 * (upper - lower) <= Fraction(str(target_percent)) * upper)


def _separate(n, iu, ju, x, active, limit, deadline):
    """Keep at most limit violated rows, even on dense inputs."""
    X = np.zeros((n, n))
    X[iu, ju] = x
    X += X.T
    heap = []
    for j in range(n):
        S = np.flatnonzero(X[j] > 1e-7)
        for a, i in enumerate(S[:-1]):
            if time.monotonic() >= deadline:
                return heap, False
            K = S[a + 1:]
            violations = X[i, j] + X[j, K] - X[i, K] - 1.0
            for pos in np.flatnonzero(violations > 1e-7):
                k = int(K[pos])
                key = (int(i), j, k)
                if key not in active:
                    item = (float(violations[pos]), *key)
                    if len(heap) < limit:
                        heapq.heappush(heap, item)
                    elif item > heap[0]:
                        heapq.heapreplace(heap, item)
        if time.monotonic() >= deadline:
            return heap, False
    return sorted(heap, reverse=True), True


def _dual_bound(w, const, D, rows, dual, pid):
    # The LP objective is normalized by D. Convert its multipliers back to the
    # unnormalized integer objective before rounding onto an exact rational grid.
    Y = [max(0, round(float(y) * D * SCALE)) for y in dual]
    residual = {}
    for y, (i, j, k) in zip(Y, rows):
        for p, sign in ((pid(i, j), 1), (pid(j, k), 1), (pid(i, k), -1)):
            residual[p] = residual.get(p, 0) + sign * y
    numerator = const * SCALE + sum(Y)
    numerator += sum(max(0, int(v) * SCALE - residual.get(p, 0)) for p, v in enumerate(w))
    return min(Fraction(1), Fraction(numerator, D * SCALE)), Y


def triangle_bound(graph, *, partition=None, time_limit=600.0, memory_gb=8.0,
                   target_percent=None, workdir=None):
    """Return an independently verified UpperBound from the full-pair triangle LP.

    The optional ``highspy`` dependency is needed only when optimization runs.
    A zero budget or a failed allocation estimate returns the universal U=1.
    Budgets govern search, not exact certificate construction/replay. A saved
    partition is evaluated exactly; known optima are never inputs to this routine.
    """
    from .api import UpperBound
    _check_options(time_limit, memory_gb, target_percent)
    start = time.monotonic()
    deadline = start + time_limit
    g = as_graph(graph)
    n, m = g.n, g.m
    if n < 2 or m < 1:
        raise ValueError("triangle bound requires a graph with at least one edge")
    root = os.path.abspath(workdir) if workdir else tempfile.mkdtemp(prefix="lagbound_triangle_")
    os.makedirs(root, exist_ok=True)
    edges = sorted((int(i), int(j)) for i, j in zip(g.u, g.v))
    proof = dict(format=FORMAT, kind="universal", graph=dict(nodes=n, edges=edges,
                 sha256=graph_digest(n, edges)), upper_exact="1")
    lower = None
    if partition is not None:
        labels = np.unique(as_membership(g, partition), return_inverse=True)[1]
        lower = modularity(g, labels, exact=True)
        proof["partition"] = dict(membership=labels.tolist(), modularity_exact=str(lower))
    history, best, reason = [], Fraction(1), "time budget"
    initial_memory = pair_lp_memory_gb(n)
    if _target_met(lower, best, target_percent):
        reason = "target met"
    elif initial_memory > memory_gb:
        reason = "memory estimate"
    elif time.monotonic() < deadline:
        try:
            import highspy
        except ImportError as err:
            raise ImportError('Triangle tightening requires highspy; install ".[tighten]" from the repository clone') from err
        D = (2 * m) ** 2
        if 2 * D > np.iinfo(np.int64).max:
            raise ValueError("graph is too large for triangle-LP integer coefficients")
        iu, ju = np.triu_indices(n, 1)
        degrees = g.degrees
        A = np.zeros((n, n), dtype=np.int8)
        A[g.u, g.v] = 1
        w = 2 * (2 * m * A[iu, ju].astype(np.int64) - degrees[iu] * degrees[ju])
        del A
        const = -sum(int(d) ** 2 for d in degrees)
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        h.addCols(len(w), w / D, np.zeros(len(w)), np.ones(len(w)), 0,
                  np.array([], np.int32), np.array([], np.int32), np.array([], float))
        h.changeObjectiveSense(highspy.ObjSense.kMaximize)
        h.changeObjectiveOffset(const / D)
        rows, active = [], set()

        def pid(i, j):
            if i > j:
                i, j = j, i
            return i * n - i * (i + 1) // 2 + j - i - 1

        for rnd in range(1000):
            left = deadline - time.monotonic()
            if left <= 0:
                reason = "time budget"
                break
            # HiGHS retains cumulative solver time across run() calls.
            h.setOptionValue("time_limit", h.getRunTime() + left)
            h.run()
            solution = h.getSolution()
            status = h.modelStatusToString(h.getModelStatus())
            if solution.dual_valid and np.all(np.isfinite(solution.row_dual)):
                q, Y = _dual_bound(w, const, D, rows, solution.row_dual, pid)
                if q < best:
                    best = q
                    proof.update(kind="triangle", scale=str(SCALE), upper_exact=str(best),
                                 triangles=[[i, j, k, str(y)] for (i, j, k), y in zip(rows, Y) if y])
            history.append(dict(round=rnd, rows=len(rows), status=status, upper_exact=str(best)))
            if lower is not None and best < lower:
                raise ArithmeticError("triangle upper bound is below the exact partition score")
            if lower is not None and best == lower:
                reason = "exact equality"
                break
            if _target_met(lower, best, target_percent):
                reason = "target met"
                break
            if h.getModelStatus() != highspy.HighsModelStatus.kOptimal or not solution.value_valid:
                reason = "LP status " + status
                break
            room = int((memory_gb - initial_memory) * 2 ** 30 // BYTES_PER_ROW) - len(rows)
            if room <= 0:
                reason = "memory estimate"
                break
            cuts, complete = _separate(n, iu, ju, np.asarray(solution.col_value), active,
                                       min(room, max(200, 20 * n)), deadline)
            if not complete:
                reason = "time budget"
                break
            if not cuts:
                reason = "converged within numerical tolerances"
                break
            starts, indices, values = [], [], []
            for _, i, j, k in cuts:
                rows.append((i, j, k))
                active.add((i, j, k))
                starts.append(len(indices))
                indices.extend((pid(i, j), pid(j, k), pid(i, k)))
                values.extend((1.0, 1.0, -1.0))
            h.addRows(len(cuts), np.full(len(cuts), -highspy.kHighsInf), np.ones(len(cuts)), len(indices),
                      np.asarray(starts, np.int32), np.asarray(indices, np.int32), np.asarray(values))
        else:
            reason = "round limit"
    proof["search"] = dict(stop=reason, history=history, time_limit=time_limit,
                           memory_gb=memory_gb, seconds=time.monotonic() - start)
    path = os.path.join(root, "triangle_certificate.json")
    with open(path, "w") as fh:
        json.dump(proof, fh)
        fh.write("\n")
    report = verify_triangle_certificate(path, g)
    details = dict(method="triangle_lp", exact=str(best), verification=report, **proof["search"])
    return UpperBound(value=upper_float(best), status="verified" if report["passed"] else "rejected",
                      estimate=float(best), certificate=path, details=details,
                      seconds=time.monotonic() - start, workdir=root)
