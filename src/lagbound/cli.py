"""Command line interface: ``lagbound GRAPH [options]``."""
from __future__ import annotations

import argparse
import json
import sys

from .api import bound


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="lagbound", description="Lower and certified upper bounds on the maximum "
                                "modularity of a graph.")
    p.add_argument("graph", help="edge-list file: one 'u v' pair per line, '#' or '%%' comments, .gz allowed")
    which = p.add_mutually_exclusive_group()
    which.add_argument("--lower-only", action="store_true", help="only compute the lower bound (best partition)")
    which.add_argument("--upper-only", action="store_true", help="only compute the upper bound")
    p.add_argument("--time-limit", type=float, default=3600, help="solver time for the upper bound, seconds (3600)")
    p.add_argument("--seeds", type=int, default=10, help="Leiden runs for the lower bound (10)")
    p.add_argument("--refine-time", type=float, default=60, help="refinement time for the lower bound, seconds (60)")
    p.add_argument("--levels", type=float, nargs="+", default=[16, 64],
                   help="Leiden resolutions of the cluster-pair rows (16 64); 1 means the starting partition")
    p.add_argument("--vc-level", type=float, default=64, help="Leiden resolution of the node-cluster rows (64)")
    p.add_argument("--max-clusters", type=int, default=6000, help="cluster cap per cluster-pair level (6000)")
    p.add_argument("--fill-filter", action="store_true", help="drop fill-incompatible triangles before the proximal "
                   "phase (for graphs whose certificate factor would not fit in memory)")
    p.add_argument("--no-certify", action="store_true", help="stop at the bound estimate, skip the certificate")
    p.add_argument("--margin", type=float, default=2.0, help="starting certificate shift / estimated shift (2)")
    p.add_argument("--bisect", type=int, default=0, help="extra factorizations searching the shift down (0)")
    p.add_argument("--bisect-lo", type=float, default=0.0, help="lowest shift searched, fraction of the start (0)")
    p.add_argument("--backend", default="auto", choices=["auto", "cholmod", "superlu", "cholcert"],
                   help="factorization backend of the certificate (auto: cholmod if installed, else superlu)")
    p.add_argument("--verify", action="store_true", help="re-check the certificate with the independent verifier")
    p.add_argument("--verify-max-gb", type=float, default=120, help="memory cap of the verifier's factor, GB (120)")
    p.add_argument("--threads", type=int, default=None, help="solver threads (all cores)")
    p.add_argument("--workdir", default=None, help="keep checkpoints, certificate, and logs here")
    p.add_argument("--partition-out", default=None, help="write the best partition here ('node cluster' per line)")
    p.add_argument("--json", action="store_true", help="print the result as JSON")
    p.add_argument("-q", "--quiet", action="store_true", help="no progress messages")
    a = p.parse_args(argv)

    res = bound(a.graph, lower=not a.upper_only, upper=not a.lower_only, time_limit=a.time_limit, seeds=a.seeds,
                refine_time=a.refine_time, certify=not a.no_certify, verify=a.verify, workdir=a.workdir,
                verbose=not a.quiet, threads=a.threads, levels=tuple(a.levels), vc_level=a.vc_level,
                max_clusters=a.max_clusters, fill_filter=a.fill_filter, margin=a.margin, bisect=a.bisect,
                bisect_lo=a.bisect_lo, backend=a.backend, verify_max_gb=a.verify_max_gb)

    if a.partition_out and res.partition is not None:
        with open(a.partition_out, "w") as fh:
            for node, cluster in res.partition.items():
                fh.write(f"{node} {cluster}\n")
    if a.json:
        out = {"gap": res.gap}
        if res.lower is not None:
            out["lower"] = {"value": res.lower.value, "exact": str(res.lower.exact),
                            "clusters": int(res.lower.membership.max()) + 1, "seconds": res.lower.seconds}
        if res.upper is not None:
            out["upper"] = {"value": res.upper.value, "status": res.upper.status, "estimate": res.upper.estimate,
                            "certificate": res.upper.certificate, "workdir": res.upper.workdir,
                            "seconds": res.upper.seconds}
        json.dump(out, sys.stdout, indent=1)
        print()
    else:
        print(res.summary())
        if res.upper is not None and res.upper.certificate:
            print(f"certificate  {res.upper.certificate}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
