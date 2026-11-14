# FRE-GY-9743-Assignments-2026fall
Assignments for NYU, FRE-GY 9743, Fall 2026.

## Assignment 7 submission: Raj Pawar (`Raj_Pawar_Branch`)

| Deliverable | Location |
|---|---|
| The interpolator: `Interpolator1DMonotoneConvex`, a shape-preserving C1 quadratic spline through the IFR points. Node slopes come from the neighbouring chords (weighted harmonic mean, zero at a local extremum, capped at twice the smaller chord; parabolic end slopes limited the same way); each node interval is one quadratic or two quadratics joined with a continuous derivative at a knot placed so that one piece is linear. `interpolate` and `integrate` are implemented; the two optional gradient stubs stay as supplied | `fixedincomelib/utilities/numerics.py` |
| Professor's notebook with the TODO cell filled (IFR plot of both curves under both interpolations and a grid check of monotonicity and convexity on every node interval), one explanation cell added after it, all six product cells executed in order; PCP values unchanged from the professor's | `hw7_monotone_convex_interpolator.ipynb` |
| Tests: one-node and two-node curves, interpolation through the points, continuity and C1, node slope rules, monotone/convex pieces for convex, concave, hump and bowl data, the midpoint-knot case, random no-overshoot data, integrals against Simpson quadrature, additivity, flat extrapolation, a three-node hand calculation, model discount factors, bumping, and the six notebook PVs | `tests/test_monotone_convex_interpolator.py` |
| Write-up: the requirements, the method with its monotonicity and convexity argument, how incompatible data are handled, the relation to Hagan and West, and the PV differences explained | `writeup/Assignment7_Writeup_RajPawar.pdf` (source `.tex` alongside) |
| Project page, numbers and figure generated from the library | `docs/index.html` (built by `docs/build_page.py`) |

### How to run

```bash
pip install -r requirements.txt                       # Python 3.11, QuantLib 1.41, matplotlib for the plot cell
jupyter notebook hw7_monotone_convex_interpolator.ipynb   # Run All
python tests/test_monotone_convex_interpolator.py     # or, with pytest installed: python -m pytest tests -q
```

Within `fixedincomelib/` only `utilities/numerics.py` changed (the `Interpolator1DMonotoneConvex` class). `requirements.txt` gained matplotlib for the plot cell. Everything else on the branch (notebook outputs, tests, write-up, docs page) is listed in the table above.
