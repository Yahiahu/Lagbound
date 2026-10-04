"""Public API: :func:`lower_bound`, :func:`upper_bound`, and :func:`bound` (both)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from .graph import as_graph
from .modularity import as_membership, modularity
from .pipeline import TAG, Workspace


@dataclass
class LowerBound:
    """Best partition found. ``membership[i]`` is the cluster of node ``i`` (non-isolated nodes, in the order of
    ``as_graph(graph).labels``); ``partition`` maps every original node identifier to a cluster."""

    value: float
    exact: Fraction
    membership: np.ndarray
    partition: dict
    seconds: float
    workdir: str | None


@dataclass
class UpperBound:
    """Upper bound on the maximum modularity.

    ``status`` is ``"verified"`` (certified and re-checked by the independent verifier), ``"certified"``, or
    ``"estimate"``; ``value`` is the bound with that status. ``estimate`` is the solver's estimate before
    certification, and ``certificate`` is the path of the stored proof object.
    """

    value: float
    status: str
    estimate: float
    certificate: str | None
    details: dict
    seconds: float
    workdir: str
    stages: list = field(default_factory=list)

    def verify(self, method: str = "auto", max_gb: float | None = None) -> dict:
        """Re-check the certificate with the independent verifier; ``method`` is ``ldl``, ``sn``, ``eig``, or
        ``auto`` (``ldl`` up to about 10^9 factor nonzeros, ``sn`` beyond). ``max_gb`` caps the supernodal factor."""
        if self.certificate is None:
            raise ValueError("no certificate to verify")
        ws = Workspace(None, self.workdir)
        return ws.verify(self.certificate, _verify_method(method, self.details), self.details.get("labels"),
                         max_gb or self.details.get("verify_max_gb", 120.0))


@dataclass
class BoundResult:
    lower: LowerBound | None
    upper: UpperBound | None

    @property
    def absolute_gap(self) -> float | None:
        """``U - L``: no partition has modularity more than this above the lower bound's partition."""
        if self.lower is None or self.upper is None:
            return None
        return self.upper.value - self.lower.value

    @property
    def gap(self) -> float | None:
        """``(U - L) / L``, the relative version of :attr:`absolute_gap`; ``None`` unless ``L > 0``."""
        if self.absolute_gap is None or self.lower.value <= 0:
            return None
        return self.absolute_gap / self.lower.value

    @property
    def partition(self) -> dict | None:
        return None if self.lower is None else self.lower.partition

    def summary(self) -> str:
        rows = []
        if self.lower is not None:
            rows.append(f"lower bound  {self.lower.value:.6f}  (best partition found)")
        if self.upper is not None:
            rows.append(f"upper bound  {self.upper.value:.6f}  ({self.upper.status})")
        if self.absolute_gap is not None:
            rel = f"  ({100 * self.gap:.2f}%)" if self.gap is not None else ""
            rows.append(f"gap          {self.absolute_gap:.6f}{rel}")
        return "\n".join(rows)


# Leiden seeds of the production label files (leiden_r16_s17, leiden_r32_s21, leiden_r64_s27, leiden_r128_s31)
_LEVEL_SEEDS = {16.0: 17, 32.0: 21, 64.0: 27, 128.0: 31}


def _verify_method(method: str, details: dict) -> str:
    if method != "auto":
        return method
    return "ldl" if details.get("fill", 0) <= 1e9 else "sn"


def lower_bound(graph, seeds: int = 10, refine_time: float = 60.0, workdir: str | None = None,
                verbose: bool = False) -> LowerBound:
    """Best partition found by Leiden (``seeds`` converged runs) followed by ensemble refinement for
    ``refine_time`` seconds. The value is the exact modularity of the returned partition. Without ``workdir`` the
    temporary files are deleted afterwards."""
    t0 = time.time()
    g = as_graph(graph)
    ws = Workspace(g, workdir, verbose=verbose)
    best_file, best_q = None, -1.0
    for s in range(seeds):
        f, q = ws.leiden(s, 1.0, -1)
        if q > best_q:
            best_file, best_q = f, q
    ws.say(f"best of {seeds} Leiden runs: {best_q:.6f}")
    labels_file = ws.improve(best_file, refine_time)[0] if refine_time > 0 else best_file
    membership = np.unique(np.load(labels_file), return_inverse=True)[1]
    exact = modularity(g, membership, exact=True)
    ws.say(f"lower bound {float(exact):.6f}")
    if workdir is None:
        ws.cleanup()
    return LowerBound(value=float(exact), exact=exact, membership=membership, partition=g.partition_dict(membership),
                      seconds=time.time() - t0, workdir=workdir and ws.root)


def upper_bound(graph, time_limit: float = 3600.0, partition=None, lower: float | None = None,
                levels: tuple = (16, 64), vc_level: float = 64, max_clusters: int = 6000,
                proximal_fraction: float = 0.4, fill_filter: bool = False, certify: bool = True,
                margin: float = 2.0, bisect: int = 0, bisect_lo: float = 0.0, backend: str = "auto",
                verify: bool = False, verify_method: str = "auto", verify_max_gb: float = 120.0,
                threads: int | None = None, max_iterations: int = 3000, workdir: str | None = None,
                verbose: bool = False) -> UpperBound:
    """Upper bound on the maximum modularity of ``graph`` by the pipeline of the paper.

    1. Leiden clusterings at the resolutions in ``levels`` (cluster-pair rows, at most ``max_clusters`` clusters per
       level) and ``vc_level`` (node-cluster rows). Resolution 1 means the starting partition: ``partition`` if
       given, otherwise a Leiden run at resolution 1.
    2. Exact reductions (leaves, true twins).
    3. Augmented Lagrangian solver, main phase, for ``(1 - proximal_fraction) * time_limit`` seconds.
    4. Optional filter of fill-incompatible triangles (``fill_filter``).
    5. Proximal phase from the main phase's last checkpoint for the remaining time.
    6. Bound estimates (with rank escape) of both final checkpoints; the better one is certified.
    7. Certificate: mixing repair, estimate, repair, sparse lifted certificate starting at ``margin`` times the
       estimated shift, with ``bisect`` extra factorizations searching down to ``bisect_lo`` times that shift.
    8. Optional independent verification; ``verify_max_gb`` caps the verifier's supernodal factor.

    The defaults are the settings of the coPapersDBLP run; the README lists those of the other reported graphs. ``lower`` is only used to report
    gaps in the logs.
    """
    t0 = time.time()
    g = as_graph(graph)
    ws = Workspace(g, workdir, threads=threads, verbose=verbose)
    stages = []

    if partition is not None:
        init = ws.path("data", TAG, "partition.npy")
        np.save(init, np.unique(as_membership(g, partition), return_inverse=True)[1].astype(np.int64))
    else:
        init, _ = ws.leiden(11, 1.0, 2)
    files = {1.0: init}
    for r in sorted(set(map(float, levels)) | {float(vc_level)}):
        if r not in files:
            files[r], _ = ws.leiden(_LEVEL_SEEDS.get(r, 7), r, 2)
    ws.reduce(list(files.values()))
    init_r = ws.reduced_labels(init)
    pairs_r = [ws.reduced_labels(files[float(r)]) for r in levels]
    vc_r = ws.reduced_labels(files[float(vc_level)])
    with open(ws.path("data", TAG + "__rx", "graph.json")) as fh:
        n_red = json.load(fh)["n"]
    lb_arg = lower if lower is not None else float("nan")
    nbatch = 512 if n_red >= 512 * 64 else max(1, n_red // 64)
    ws.say(f"reduced instance: {n_red} nodes; solver batches {nbatch}")

    solver = dict(lb=lb_arg, outer=max_iterations, nbatch=nbatch, max_clusters=max_clusters)
    t_main = (1.0 - proximal_fraction) * time_limit
    main_ckpt, weight = ws.solve("main", pairs_r, vc_r, time_limit=t_main, init=init_r, **solver)
    stages.append(("main phase", time.time() - t0))
    if fill_filter:
        main_ckpt = ws.filter_triangles(main_ckpt, ws.path("runs", "main_filtered.npz"))
    candidates = [("main", main_ckpt)]
    t_prox = max(0.0, time_limit - (time.time() - t0))
    if proximal_fraction > 0 and t_prox > 0:
        prox_ckpt, _ = ws.solve("proximal", pairs_r, vc_r, time_limit=t_prox, resume=main_ckpt, pair_weight=weight,
                                proximal=True, **solver)
        candidates.append(("proximal", prox_ckpt))
        stages.append(("proximal phase", time.time() - t0))
    estimates = {name: ws.estimate(f"est_{name}", ckpt, lb_arg) for name, ckpt in candidates}
    best_name = min(estimates, key=lambda k: estimates[k]["ub"])
    best_ckpt = dict(candidates)[best_name]
    estimate = float(estimates[best_name]["ub"])
    ws.say("estimates: " + ", ".join(f"{k} {v['ub']:.6f}" for k, v in estimates.items()))
    stages.append(("estimates", time.time() - t0))

    value, status, cert_file = estimate, "estimate", None
    details = dict(labels=init, estimates=estimates, verify_max_gb=verify_max_gb)
    if certify:
        fix_v = ws.repair(best_ckpt, ws.path("runs", "cert_fixV.npz"), 10)
        est = ws.estimate("cert_est", best_ckpt, lb_arg, init_v=fix_v, save_w=ws.path("runs", "cert_W.npz"),
                          polish_time=1500, eig_time=1200)
        w2 = ws.repair(best_ckpt, ws.path("runs", "cert_W2.npz"), 3, init_w=ws.path("runs", "cert_W.npz"))
        cert = ws.certify("certificate", best_ckpt, w2, est["json"], margin=margin, bisect=bisect,
                          bisect_lo=bisect_lo, backend=backend, order_notri=fill_filter)
        value, status, cert_file = float(cert["UB"]), "certified", cert["file"]
        details.update(fill=cert.get("fill"), certificate=cert)
        ws.say(f"certified upper bound {value:.6f}")
        stages.append(("certificate", time.time() - t0))
        if verify:
            res = ws.verify(cert_file, _verify_method(verify_method, details), init, verify_max_gb)
            details["verification"] = res
            if res.get("passed"):
                value, status = float(res["UB"]), "verified"
                ws.say(f"verified upper bound {value:.6f}")
            else:
                ws.say(f"verification failed: {res.get('fail', 'see verify.log')}")
            stages.append(("verification", time.time() - t0))
    return UpperBound(value=value, status=status, estimate=estimate, certificate=cert_file, details=details,
                      seconds=time.time() - t0, workdir=ws.root, stages=stages)


def bound(graph, lower: bool = True, upper: bool = True, time_limit: float = 3600.0, seeds: int = 10,
          refine_time: float = 60.0, certify: bool = True, verify: bool = False, workdir: str | None = None,
          verbose: bool = False, **upper_options) -> BoundResult:
    """Lower bound, upper bound, or both (default). With both, the lower bound's partition initializes the solver."""
    g = as_graph(graph)
    lb = lower_bound(g, seeds=seeds, refine_time=refine_time, workdir=workdir, verbose=verbose) if lower else None
    ub = None
    if upper:
        ub = upper_bound(g, time_limit=time_limit, partition=None if lb is None else lb.membership,
                         lower=None if lb is None else lb.value, certify=certify, verify=verify, workdir=workdir,
                         verbose=verbose, **upper_options)
    return BoundResult(lower=lb, upper=ub)
