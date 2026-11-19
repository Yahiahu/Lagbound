# Changes

## Documentation — November 19, 2026

- Align the algorithm overview and results documentation with the revised manuscript's optional tightening step.
- Include all three small-network component runtimes and their recorded values in the reproducible summary.
  These are separate measurements, not end-to-end timings or a comparison on common hardware.
- No algorithm, default settings, or reported bound values changed in this documentation update.

## 0.1.2 — November 18, 2026

- Add opt-in triangle-LP tightening through `tighten=True` / `--tighten`, with separate time and memory estimates,
  an optional target percentage, and retention of the better accepted bound. The default SDP path is unchanged.
- Save the normalized graph, exact dual multipliers, triangle rows, and supplied partition in replayable JSON
  certificates. Verify them with an independent integer-arithmetic implementation that does not import HiGHS.
- Correct rational-to-float outward rounding, bound the cut-selection buffer, and account for cumulative HiGHS
  solver time. A numerical convergence status is separate from certificate validity.
- Accept an incumbent in the combined `bound()` workflow and the CLI. Evaluate it exactly and retain the best
  candidate against the heuristic and the all-in-one partition. Add exact rational upper-bound and percentage fields.
- Keep native solver output off the CLI JSON stream. Preserve accepted SDP results if optional tightening fails.
- Document the 155-network research records and the two largest remaining gaps. Remove tolerance-based optimality
  claims from the published summary and distinguish historical scalar consistency checks from proof replay.

This code update did not change the manuscript or the eight large-network result values. It does not claim a rerun of
all 155 historical experiments or introduce an exact-solver/branching fallback.

Validation on macOS with Python 3.12 and HiGHS 1.16.0 passed all 75 tests, with 11 expected graph-normalization
warnings. The wheel build, CLI JSON output, and relative documentation links passed their checks. A full
SDP-plus-tightening run on karate club passed independent replay. Fresh triangle-LP runs on cattle and
high_tech_company reproduced the reported weaker LP bounds to the displayed precision and passed exact replay.
