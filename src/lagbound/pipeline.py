"""Runs the research pipeline (``lagbound/_core``) on a working directory.

The scripts in ``_core`` are the ones that produced the certified bounds in the paper. They read graphs from
``DATA_DIR/<tag>/{graph.json, keys.npy}`` and write checkpoints, JSON summaries, and logs. This module writes the
input in that format, calls each script with the flags used in the production runs, and reads the results back.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from .graph import Graph

CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_core")
TAG = "g"


class PipelineError(RuntimeError):
    """A pipeline stage failed; the message ends with the tail of its log."""


class Workspace:
    """Working directory holding the graph, label files, checkpoints, certificates, and logs."""

    def __init__(self, graph: Graph | None, workdir: str | None = None, threads: int | None = None,
                 verbose: bool = False):
        """Create the working directory and write ``graph`` into it; ``graph=None`` reopens an existing one."""
        self.root = os.path.abspath(workdir) if workdir else tempfile.mkdtemp(prefix="lagbound_")
        self.data = os.path.join(self.root, "data")
        self.runs = os.path.join(self.root, "runs")
        self.threads = threads or os.cpu_count() or 1
        self.verbose = verbose
        os.makedirs(os.path.join(self.data, TAG), exist_ok=True)
        os.makedirs(self.runs, exist_ok=True)
        if graph is None:
            return
        n = graph.n
        keys = np.unique(graph.u.astype(np.int64) * n + graph.v)
        np.save(os.path.join(self.data, TAG, "keys.npy"), keys)
        with open(os.path.join(self.data, TAG, "graph.json"), "w") as fh:
            json.dump({"n": int(n), "m": int(len(keys))}, fh)

    # ------------------------------------------------------------------ plumbing
    def path(self, *parts) -> str:
        return os.path.join(self.root, *parts)

    def say(self, msg: str):
        if self.verbose:
            print(f"[lagbound] {msg}", flush=True)

    def run(self, script: str, args: list, log: str, threads: int | None = None, blas_threads: int | None = None,
            extra_env: dict | None = None) -> str:
        """Run ``_core/<script>`` with ``args`` from the working directory; returns the log path."""
        th = str(threads or self.threads)
        env = dict(os.environ)
        env.update(DATA_DIR=self.data + os.sep, NUMBA_NUM_THREADS=th, OMP_NUM_THREADS=th,
                   OPENBLAS_NUM_THREADS=str(blas_threads or th), MKL_NUM_THREADS=str(blas_threads or th))
        env.setdefault("NUMBA_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "lagbound", "numba"))
        if extra_env:
            env.update(extra_env)
        log_path = os.path.join(self.runs, log)
        with open(log_path, "w") as fh:
            proc = subprocess.run([sys.executable, os.path.join(CORE, script)] + [str(a) for a in args], cwd=self.root,
                                  env=env, stdout=fh, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            with open(log_path) as fh:
                tail = "".join(fh.readlines()[-15:])
            raise PipelineError(f"{script} failed (exit {proc.returncode}); log {log_path}:\n{tail}")
        return log_path

    # ------------------------------------------------------------------ stages
    def leiden(self, seed: int, resolution: float, iterations: int, tag: str = TAG):
        """One Leiden run (``leiden_one.py``); returns the label file and the modularity at resolution 1."""
        self.run("leiden_one.py", ["data", tag, seed, resolution, iterations], f"leiden_r{resolution:g}_s{seed}.log", threads=1)
        stem = os.path.join(self.data, tag, f"leiden_r{resolution:g}_s{seed}")
        with open(stem + ".json") as fh:
            return stem + ".npy", float(json.load(fh)["Q"])

    def improve(self, start_labels: str, budget: float):
        """Ensemble refinement of the best partition (``improve2.py``) for ``budget`` seconds."""
        out = self.path("lb")
        os.makedirs(out, exist_ok=True)
        self.run("improve2.py", [TAG, budget, "--start", start_labels], "improve.log", threads=1,
                 extra_env=dict(LAGBOUND_OUT=out))
        with open(os.path.join(out, f"best_labels_{TAG}.json")) as fh:
            info = json.load(fh)
        return os.path.join(out, f"best_labels_{TAG}.npy"), info["Q_exact"]

    def reduce(self, label_files: list) -> str:
        """Exact reductions (``reduce_exact.py``); label files are restricted to the reduced instance."""
        self.run("reduce_exact.py", [TAG] + label_files, "reduce.log", threads=1)
        return TAG + "__rx"

    def reduced_labels(self, label_file: str) -> str:
        """Path of ``label_file`` restricted to the reduced instance."""
        return os.path.join(self.data, TAG + "__rx", os.path.basename(label_file))

    def solve(self, name: str, pairs: list, vc: str, lb: float, time_limit: float, outer: int, nbatch: int,
              init: str | None = None, resume: str | None = None, pair_weight: float | None = None,
              proximal: bool = False, max_clusters: int = 6000, ckpt_every: int | None = None):
        """Augmented Lagrangian solver (``alm_vcs.py``).

        Main phase: the flags of the production launch script. Proximal phase: resume from a checkpoint with no new
        triangles, four sweeps per multiplier update, and the pair weight of the main phase (production phase-2 runs).
        Node-cluster separation stays on in both phases, as in production. ``ckpt_every`` defaults to 50 outer
        iterations in the main phase and 20 in the proximal phase. Returns the last checkpoint and the pair weight
        chosen by the scaling rule.
        """
        args = ["--tag", TAG + "__rx", "--pair_labels"] + pairs + [
            "--max_clusters", max_clusters, "--k", 32, "--nbatch", nbatch, "--inner", 1, "--est_every", 0,
            "--tri_max", 16000000, "--mom", 0, "--lb", lb, "--shift_p", 0, "--time_limit", time_limit, "--seed", 0,
            "--vcs_labels", vc, "--vcs_max", 150000000, "--vcs_add", 10000000, "--outer", outer,
            "--save", os.path.join("runs", name + ".npz")]
        if proximal:
            args += ["--resume", resume, "--tri_add", 0, "--tri1_add", 0, "--tri_every", 1000000,
                     "--pairlvl_weight", pair_weight, "--phase2_at_it", 1, "--phase2_w", pair_weight,
                     "--phase2_inner", 4, "--phase2_tri_every", 1000000, "--phase2_est_every", 0,
                     "--ckpt_every", ckpt_every or 20]
        else:
            args += ["--tri_add", 600000, "--tri1_add", 300000, "--tri_every", 9, "--init_partition", init,
                     "--pw_rule", "default", "--ckpt_every", ckpt_every or 50]
        log = self.run("alm_vcs.py", args, name + ".log", blas_threads=1)
        weight = pair_weight
        with open(log) as fh:
            for line in fh:
                hit = re.search(r"-> pairlvl_weight ([0-9.eE+-]+)", line)
                if hit:
                    weight = float(hit.group(1))
        return os.path.join(self.runs, name + "_last.npz"), weight

    def filter_triangles(self, ckpt: str, out: str) -> str:
        """Keep only fill-compatible triangles (the production restricted runs; dropping multipliers is valid)."""
        code = ("import sys, numpy as np; sys.path.insert(0, {core!r}); import certify_lifted_vc as CV\n"
                "P = CV.load_problem({tag!r}, {ck!r}, {data!r})\n"
                "Q = CV.tri_fill_filter(P, dict(mu=1e4))\n"
                "z = dict(np.load({ck!r})); z['TI'] = Q['TI']; z['lam'] = Q['lam']; np.savez({out!r}, **z)\n"
                "print('kept', len(Q['lam']), 'of', len(P['lam']))\n").format(core=CORE, tag=TAG + "__rx", ck=ckpt,
                                                                             data=self.data + os.sep, out=out)
        script = os.path.join(self.runs, "filter_triangles.py")
        with open(script, "w") as fh:
            fh.write(code)
        env = dict(os.environ, DATA_DIR=self.data + os.sep)
        with open(os.path.join(self.runs, "filter.log"), "w") as fh:
            proc = subprocess.run([sys.executable, script], cwd=self.root, env=env, stdout=fh, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise PipelineError(f"triangle filter failed; see {os.path.join(self.runs, 'filter.log')}")
        return out

    def estimate(self, name: str, ckpt: str, lb: float, init_v: str | None = None, save_w: str | None = None,
                 polish_time: float = 900, eig_time: float = 900) -> dict:
        """Bound estimate with rank escape (``est_dual_vcs.py``)."""
        out = os.path.join(self.runs, name + ".json")
        args = ["--tag", TAG + "__rx", "--ckpt", ckpt, "--lb", lb, "--polish", "rgd", "--polish_time", polish_time,
                "--scale", "vol", "--eig_time", eig_time, "--out", out]
        if init_v:
            args += ["--initV", init_v]
        if save_w:
            args += ["--saveW", save_w]
        self.run("est_dual_vcs.py", args, name + ".log")
        with open(out) as fh:
            res = json.load(fh)
        res["json"] = out
        return res

    def repair(self, ckpt: str, out: str, iterations: int, init_w: str | None = None) -> str:
        """Mixing repair of rows stuck at saddle points (``hw_fix.py``)."""
        args = [TAG + "__rx", ckpt, out, iterations] + ([init_w] if init_w else [])
        self.run("hw_fix.py", args, os.path.basename(out).replace(".npz", ".log"))
        return out

    def certify(self, name: str, ckpt: str, init_v: str, est_json: str, margin: float = 2.0, bisect: int = 0,
                bisect_lo: float = 0.0, backend: str = "auto", order_notri: bool = False) -> dict:
        """Sparse lifted certificate (``certify_lifted_vc.py``); also writes the proof object for the verifier.

        The script starts from ``margin`` times the estimated shift and multiplies it by 1.25 until the matrix
        factors, for at most 40 steps; ``bisect`` extra factorizations then search down towards ``bisect_lo`` times
        the starting shift. When the estimate is far too small (it happens on tiny graphs), the search is
        continued from the last shift tried, up to three more times.
        """
        out = os.path.join(self.runs, name + ".json")
        dump = os.path.join(self.runs, name + ".npz")
        args = [TAG + "__rx", ckpt, "--initV", init_v, "--bisect", bisect, "--bisect_lo", bisect_lo,
                "--backend", backend, "--max_fill", 3e11,
                "--dump_cert", dump, "--out", out] + (["--order_notri", 1] if order_notri else [])
        start = ["--shift_est", est_json, "--margin", margin]
        for attempt in range(4):
            try:
                self.run("certify_lifted_vc.py", args + start, name + ".log")
                break
            except PipelineError as err:
                with open(os.path.join(self.runs, name + ".log")) as fh:
                    tried = re.findall(r"shift ([0-9.eE+-]+): factor ok False", fh.read())
                if attempt == 3 or "could not factor" not in str(err) or not tried:
                    raise
                start = ["--shift", float(tried[-1]) * 1.25]
                self.say(f"certificate: continuing the shift search from {start[1]:.3e}")
        with open(out) as fh:
            res = json.load(fh)
        res["file"] = dump
        return res

    def verify(self, cert_file: str, method: str, labels: str | None = None, max_gb: float = 120.0) -> dict:
        """Independent verification (``verify_cert.py``), including the exact reduction check. ``max_gb`` caps the
        supernodal factor of method ``sn`` (Hollywood-2009 needed 142 GB and was run with 400)."""
        out = os.path.join(self.runs, "verify.json")
        args = [os.path.join(self.data, TAG + "__rx"), cert_file, "--method", method, "--sn_max_GB", max_gb,
                "--orig", os.path.join(self.data, TAG), "--out", out]
        if labels:
            args += ["--labels", labels]
        if os.path.exists(out):
            os.remove(out)                          # never read a result left over from an earlier run
        error = None
        try:
            self.run("verify_cert.py", args, "verify.log")
        except PipelineError as err:
            error = str(err)                        # a rejected certificate also exits nonzero
        if not os.path.exists(out):
            return dict(passed=False, fail=error or "the verifier wrote no result")
        with open(out) as fh:
            res = json.load(fh)
        res["passed"] = error is None and bool(res.get("passed"))
        return res

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)
