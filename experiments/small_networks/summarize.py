"""Summarize recorded scalars; this does not replay the historical certificates.

Run with Python 3.10+: python experiments/small_networks/summarize.py
All inequalities and zero/equality checks use rational arithmetic. Missing or
unaccepted bounds are errors, never evidence of a successful soundness check.
"""
import csv
from fractions import Fraction
from pathlib import Path
import statistics
import sys


def summarize(path):
    rows = list(csv.DictReader(Path(path).open()))
    base, final = [], []
    matched, consistent, zero = 0, 0, 0
    for row in rows:
        if row["atris_status"] != "verified":
            raise ValueError(f"{row['name']}: no recorded verified SDP bound")
        L = Fraction(row["L_exact"])
        qstar = Fraction(row["qstar_exact_posthoc"])
        Ua = Fraction(row["U_atris_binary_exact"])
        Ut = Fraction(row["U_triangle_exact"])
        U = min(Ua, Ut)
        if L > U or L < 0:
            raise ValueError(f"{row['name']}: inconsistent bounds")
        matched += L == qstar
        consistent += Ua >= qstar and Ut >= qstar
        if L == 0:
            zero += 1
            if U != 0 or qstar != 0:
                raise ValueError(f"{row['name']}: zero score does not establish zero optimum")
            continue
        base.append(100 * (Ua - L) / Ua)
        final.append(100 * (U - L) / U)
    print(f"Recorded networks: {len(rows)}; positive-score: {len(base)}; zero-optimum: {zero}")
    print(f"Exact scores matching reference optima: {matched}; both recorded bounds >= reference: {consistent}")
    print("The scalar checks above do not replace certificate replay.")
    for label, values in [("Atris alone", base), ("With triangle LP", final)]:
        print(f"{label}: median {float(statistics.median(values)):.2f}%, "
              f"mean {float(statistics.mean(values)):.2f}%, worst {float(max(values)):.2f}% (nearest rounding)")
        print("  " + ", ".join(f"within {t}%: {sum(v <= Fraction(t) for v in values)}/{len(values)}"
                                for t in ("1", "1.85", "4.97")))
    print("Separately recorded component runtimes; not measured end-to-end times:")
    for label, field in [("VieClus", "vieclus_seconds"), ("SDP", "atris_seconds"),
                         ("Triangle LP", "triangle_seconds")]:
        values = [float(row[field]) for row in rows]
        print(f"{label}: median {statistics.median(values):.2f}s, maximum {max(values):.2f}s")
    return rows


if __name__ == "__main__":
    summarize(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("recorded_155.csv"))
