# FRE-GY-9743-Assignments-2026fall
Assignments for NYU, FRE-GY 9743, Fall 2026.

## Assignment 7 submission: Raj Pawar (`Raj_Pawar_Branch`)

| Deliverable | Location |
|---|---|
| The interpolator: `Interpolator1DMonotoneConvex`, a shape-preserving quadratic spline through the IFR points with a continuous derivative except for a kink where a collinear run meets data it cannot bend into. Node slopes come from the neighbouring chords (weighted harmonic mean, zero at a local extremum, capped at twice the smaller chord; parabolic end slopes limited the same way; a collinear run and its end nodes take the run's chord, so collinear points reproduce as their line, and nearly equal chords blend toward that rule so the curve depends continuously on the data); each node interval is one quadratic or two quadratics joined with a continuous derivative at a knot placed so that one piece is linear when the end slopes straddle the chord. `interpolate` and `integrate` are implemented; the two optional gradient stubs stay as supplied | `fixedincomelib/utilities/numerics.py` |
| Professor's notebook with the TODO cell filled (IFR plot of both curves under both interpolations and a grid check of monotonicity and convexity on every node interval), one explanation cell added after it, all six product cells executed in order; PCP values unchanged from the professor's | `hw7_monotone_convex_interpolator.ipynb` |
| Tests: one-node and two-node curves, interpolation through the points, continuity and C1, node slope rules and caps, monotone/convex pieces for convex, concave, hump and bowl data, the weighted-knot case, collinear runs (exact, tied only up to rounding, two runs meeting in a kink, a run ending at a maximum, a capped run end), continuity of the curve in the data around a tie, the blended slope for nearly equal chords, a finite bump-and-reprice limit on round quotes, random no-overshoot data, integrals against Simpson quadrature, additivity, flat extrapolation, negative rates, a three-node hand calculation, model discount factors, the locality of a bump, duplicate node times, and the six notebook PVs; plus a plain-Python re-derivation of the six PVs from a second implementation of the recipe and hand-written schedules | `tests/test_monotone_convex_interpolator.py`, `tests/test_hand_present_values.py` |
| Write-up: the requirements, the method with its monotonicity and convexity argument, how incompatible data are handled, the relation to Hagan and West, and the PV differences explained | `writeup/Assignment7_Writeup_RajPawar.pdf` (source `.tex` alongside) |
| Project page, numbers and figure generated from the library | `docs/index.html` (built by `docs/build_page.py`) |

### How to run

```bash
pip install -r requirements.txt                       # Python 3.11, QuantLib 1.41, matplotlib for the plot cell
jupyter notebook hw7_monotone_convex_interpolator.ipynb   # Run All
python tests/test_monotone_convex_interpolator.py     # or, with pytest installed: python -m pytest tests -q
python tests/test_hand_present_values.py
```

Within `fixedincomelib/` only `utilities/numerics.py` changed (the `Interpolator1DMonotoneConvex` class). `requirements.txt` gained matplotlib for the plot cell. Everything else on the branch (notebook outputs, tests, write-up, docs page) is listed in the table above.
