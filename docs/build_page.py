"""Generate docs/index.html: the Assignment 7 project page (the two IFR curves under both interpolations, the method, the PVs).

Run from the repository root with the QuantLib interpreter:  python docs/build_page.py
"""
import html, io, os, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests")); os.chdir(ROOT)
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from test_monotone_convex_interpolator import FF, NODE_DATES, SOFR, TIMES, notebook_model  # noqa: E402
from fixedincomelib import qfDiscountFactor  # noqa: E402
from fixedincomelib.date import Date  # noqa: E402
from fixedincomelib.utilities import Interpolator1DMonotoneConvex  # noqa: E402

P, M = notebook_model("PIECEWISE_CONSTANT_LEFT_CONTINUOUS"), notebook_model("MONOTONE_CONVEX")
def act360(a, b): return (Date(b).serialNumber() - Date(a).serialNumber()) / 360
def comp(model, idx, a, b): return (qfDiscountFactor(model, idx, a) / qfDiscountFactor(model, idx, b) - 1) / act360(a, b)
def dfund(model, d): return qfDiscountFactor(model, "SOFR-1B-FLAT", d)
N = 1e6
FRA = ("2026-05-05", "2026-11-05")
SWAP = [("2026-04-02", "2027-04-02", "2027-04-06"), ("2027-04-02", "2028-04-03", "2028-04-05"), ("2028-04-03", "2029-04-02", "2029-04-04")]
QUARTERS = [("2026-10-05", "2027-01-04", "2027-01-06"), ("2027-01-04", "2027-04-05", "2027-04-07"), ("2027-04-05", "2027-07-06", "2027-07-08"),
            ("2027-07-06", "2027-10-04", "2027-10-06"), ("2027-10-04", "2028-01-03", "2028-01-05"), ("2028-01-03", "2028-04-03", "2028-04-05"),
            ("2028-04-03", "2028-07-03", "2028-07-06")]
LEG1 = [("2026-07-02", "2026-07-06", "2026-07-08"), ("2026-07-06", "2026-10-05", "2026-10-07")] + QUARTERS
LEG2 = [("2026-07-02", "2026-07-03", "2026-07-07"), ("2026-07-03", "2026-10-05", "2026-10-07")] + QUARTERS
def pvs(model):
    fra = N * act360(*FRA) * (0.032 - comp(model, "SOFR-1B", *FRA)) * dfund(model, FRA[1])
    fixed = sum(N * 0.034 * act360(a, b) * dfund(model, p) for a, b, p in SWAP)
    flt = sum(N * act360(a, b) * comp(model, "SOFR-1B", a, b) * dfund(model, p) for a, b, p in SWAP)
    leg1 = sum(N * act360(a, b) * (comp(model, "SOFR-1B", a, b) + 0.001) * dfund(model, p) for a, b, p in LEG1)
    leg2 = sum(N * act360(a, b) * comp(model, "FF-1B", a, b) * dfund(model, p) for a, b, p in LEG2)
    return fra, fixed, flt, fixed - flt, leg1, leg2, leg1 - leg2
pp, pm = pvs(P), pvs(M)
import subprocess  # noqa: E402
n_tests, n_hand = (int([ln for ln in subprocess.run([sys.executable, f"tests/{name}.py"], capture_output=True, text=True, check=True)
                        .stdout.splitlines() if ln.endswith("tests passed")][0].split()[0])
                   for name in ("test_monotone_convex_interpolator", "test_hand_present_values"))

grid = np.linspace(0.0, 5.5, 2201)
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
for ax, index in zip(axes, ("SOFR-1B", "FF-1B")):
    for model, style, label in ((P, "C0--", "piecewise constant (left-continuous)"), (M, "C3-", "monotone convex")):
        interp = model.retrieve_model_component(index).state_data_interpolator
        ax.plot(grid, [100 * interp.interpolate(t) for t in grid], style, lw=1.4, label=label)
    nodes = M.retrieve_model_component(index).state_data_interpolator
    ax.plot(nodes.axis1, 100 * nodes.values, "ko", ms=4, label="supplied IFR points")
    ax.set_title(f"{index} instantaneous forward rate", fontsize=10); ax.set_xlabel("years from 2026-01-02", fontsize=9)
    ax.tick_params(labelsize=8); ax.grid(alpha=0.3)
axes[0].set_ylabel("IFR (%)", fontsize=9); axes[0].legend(loc="upper left", fontsize=8, frameon=False)
fig.tight_layout()
buf = io.StringIO(); fig.savefig(buf, format="svg", metadata={"Date": None}); plt.close(fig)
svg = buf.getvalue()[buf.getvalue().index("<svg"):]
svg = re.sub(r'\s(width|height)="[^"]*pt"', "", svg, count=2).replace("<svg ", '<svg style="width:100%;height:auto" role="img" aria-label="SOFR and Fed Funds IFR curves under both interpolations" ', 1)

def table(headers, rows):
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
f2 = lambda v: f"{v:,.2f}"
sofr_it = M.retrieve_model_component("SOFR-1B").state_data_interpolator
d = Interpolator1DMonotoneConvex._node_slopes(np.array(TIMES), np.array(SOFR))
node_rows = [(dt, f"{t:.4f}", f"{100*s:.2f}%", f"{100*fv:.2f}%", f"{100*ds:.4f}%/yr") for dt, t, s, fv, ds in zip(NODE_DATES, TIMES, SOFR, FF, d)]
shape_rows = []
for a, b in zip(TIMES[:-1], TIMES[1:]):
    row = [f"[{a:.3f}, {b:.3f}]"]
    for idx in ("SOFR-1B", "FF-1B"):
        it = M.retrieve_model_component(idx).state_data_interpolator
        v = np.array([it.interpolate(t) for t in np.linspace(a, b, 2001)])
        row += ["yes" if (np.diff(v) >= -1e-15).all() else "no", "yes" if (np.diff(v, 2) >= -1e-13).all() else "no"]
    shape_rows.append(row)
rate_rows = [(a, b, p, f"{100*comp(P, 'SOFR-1B', a, b):.4f}%", f"{100*comp(M, 'SOFR-1B', a, b):.4f}%", f"{dfund(P, p):.6f}", f"{dfund(M, p):.6f}") for a, b, p in SWAP]
gap_rows = [(a, b, f"{1e4*(comp(P, 'SOFR-1B', a, b) - comp(P, 'FF-1B', a, b)):.1f}", f"{1e4*(comp(M, 'SOFR-1B', a, b) - comp(M, 'FF-1B', a, b)):.1f}") for a, b, _ in QUARTERS]
pv_rows = [("FRA 2026-05-05 to 2026-11-05, receive 3.2%", f2(pp[0]), f2(pm[0]), f2(pm[0] - pp[0])),
           ("RFR swap 2026-04-02 to 2029-04-02, receive 3.4% annual", f2(pp[3]), f2(pm[3]), f2(pm[3] - pp[3])),
           ("Overnight basis swap 2026-07-02 to 2028-07-03, receive SOFR + 10 bp, pay FF", f2(pp[6]), f2(pm[6]), f2(pm[6] - pp[6]))]
leg_rows = [("Swap fixed leg", f2(pp[1]), f2(pm[1])), ("Swap floating leg", f2(pp[2]), f2(pm[2])),
            ("Basis leg 1: SOFR + 10 bp", f2(pp[4]), f2(pm[4])), ("Basis leg 2: Fed Funds", f2(pp[5]), f2(pm[5]))]

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Assignment 7: a monotone convex IFR interpolator (Raj Pawar)</title>
<style>
body{{font-family:Georgia,'Times New Roman',serif;max-width:900px;margin:32px auto;padding:0 16px;color:#222;line-height:1.45;background:#fff}}
h1{{font-size:1.6em;margin-bottom:0}} h2{{font-size:1.2em;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:3px}}
.meta{{color:#555;margin-top:4px}} code{{font-family:Menlo,Consolas,monospace;font-size:0.92em;background:#f4f4f4;padding:1px 4px;border-radius:3px}}
table{{border-collapse:collapse;font-size:0.85em;margin:10px 0;width:100%;overflow-x:auto;display:block}} th,td{{border:1px solid #ddd;padding:4px 6px;text-align:right;white-space:nowrap}}
th{{background:#f7f7f7}} td:first-child,th:first-child{{text-align:left}} .formula{{background:#f8f8ff;border-left:3px solid #1f4e79;padding:8px 12px;margin:8px 0;font-size:0.95em}}
.note{{color:#444;font-size:0.95em}} sup,sub{{line-height:0}} figure{{margin:12px 0}} figcaption{{color:#555;font-size:0.9em}}
</style></head><body>
<h1>Assignment 7: A Monotone Convex Interpolator for the IFR Curve</h1>
<p class="meta">Raj Pawar, FRE-GY 9743 (Spec Topics in Risk Management), NYU Tandon, November 2026.
Branch <a href="https://github.com/pawarraj8888/FRE-GY-9743-Assignments-7/tree/Raj_Pawar_Branch"><code>Raj_Pawar_Branch</code></a>,
write-up <a href="https://github.com/pawarraj8888/FRE-GY-9743-Assignments-7/blob/Raj_Pawar_Branch/writeup/Assignment7_Writeup_RajPawar.pdf">PDF</a>,
code in <code>fixedincomelib/utilities/numerics.py</code> (class <code>Interpolator1DMonotoneConvex</code>).</p>

<h2>The curves</h2>
<p class="note">Value date 2026-01-02. The library's piecewise constant curve holds each interval at the value of its right-hand node; the new curve passes
through the points, has a continuous derivative, and is monotone and convex on every node interval. The one-node funding curve (5 bp over SOFR) is a constant under both.</p>
<figure>{svg}<figcaption>Both index curves under the two interpolations, with flat extrapolation on either side of the nodes.</figcaption></figure>

<h2>The method in one paragraph</h2>
<p>The inputs are IFR <i>values</i> at the nodes, so the free quantities are the node slopes. An interior node gets the weighted harmonic mean of the two neighbouring
chord slopes (zero at a local extremum or next to a flat interval, never more than twice the smaller chord); the two end nodes get the slope of the parabola through the
three outermost points, limited the same way; and where two consecutive chords are equal (within a relative tolerance) the node between them and the two end nodes of that
collinear run take the common chord, so collinear points are reproduced by their straight line, and chords within a tenth of each other blend toward that rule so the curve
depends continuously on the data. On each interval the derivative is piecewise linear through the left slope,
a knot slope and the right slope, and the curve is two quadratics joined with a continuous derivative. When the end slopes straddle the chord (the data are strictly convex
there) the knot is placed where one piece becomes a straight line, so the derivative runs monotonically from one end slope to the other: the piece is convex (or concave for
concave data) and, since every slope has the chord's sign, monotone. When both slopes sit on the same side of the chord no convex C<sup>1</sup> piece exists; the knot moves
toward the end whose slope is closer to the chord and the slope cap keeps the piece monotone. Where a collinear run meets data it cannot bend into smoothly (an extremum, a
second run, a capped neighbour) the run stays straight and the curve takes a kink at that node. This is Hagan and West's recipe (local node estimates, piecewise quadratics,
a flat segment inserted to protect monotonicity) moved from interval averages to point values.</p>
<div class="formula">interior slope: d<sub>i</sub> = (w<sub>1</sub> + w<sub>2</sub>) / (w<sub>1</sub>/s<sub>i</sub> + w<sub>2</sub>/s<sub>i+1</sub>),
&nbsp; w<sub>1</sub> = 2h<sub>i+1</sub> + h<sub>i</sub>, w<sub>2</sub> = h<sub>i+1</sub> + 2h<sub>i</sub>, &nbsp; |d<sub>i</sub>| &le; 2 min(|s<sub>i</sub>|, |s<sub>i+1</sub>|);
&nbsp; knot slope m = 2s<sub>i</sub> &minus; (&alpha;d<sub>i&minus;1</sub> + &beta;d<sub>i</sub>)/h<sub>i</sub>.</div>
{table(["Node date", "t (years)", "SOFR IFR", "FF IFR", "SOFR node slope"], node_rows)}
{table(["Node interval (years)", "SOFR monotone", "SOFR convex", "FF monotone", "FF convex"], shape_rows)}
<p class="note">The shape table evaluates the curve on a 2001-point grid inside each interval and checks the first and second differences.</p>

<h2>Present values</h2>
{table(["Product (notional 1,000,000)", "PCP PV", "Monotone convex PV", "Difference"], pv_rows)}
{table(["Leg", "PCP", "Monotone convex"], leg_rows)}
<p>On rising data the piecewise constant step sits above the point curve everywhere, so the new curve gives smaller floating legs and slightly larger discount factors:
both receivers of fixed gain. The swap moves most through its last period, where the step jumps to the 5Y value of 4.7% right after the 3Y node while the new curve is
still near 3.6%. The basis swap receives the SOFR&ndash;FF gap, which the step takes from the right-hand node of each interval; the new curves pass through the point gaps,
so the gap, and the PV, shrink.</p>
{table(["Swap period start", "end", "payment", "compounded SOFR, PCP", "compounded SOFR, monotone convex", "funding DF, PCP", "funding DF, monotone convex"], rate_rows)}
{table(["Basis quarter start", "end", "SOFR minus FF (bp), PCP", "SOFR minus FF (bp), monotone convex"], gap_rows)}

<h2>Checks</h2>
<p class="note">{n_tests} tests in <code>tests/test_monotone_convex_interpolator.py</code>: one-node and two-node curves, interpolation through the points, continuity and a continuous
derivative, the slope rules and caps, shape on rising, falling, concave, hump and bowl data, the weighted-knot case, collinear runs (exact and tied only up to rounding, two runs
meeting in a kink, a run ending at a maximum, a capped run end), continuity of the curve in the data around a tie and the blended slope for nearly equal chords, a finite
bump-and-reprice limit on round quotes, random no-overshoot data, integrals against Simpson quadrature, flat
extrapolation, negative rates, a three-node hand calculation, model discount factors, the locality of a bump, and the six notebook PVs (the PCP values equal the professor's
printed numbers). <code>tests/test_hand_present_values.py</code> ({n_hand} tests) re-derives the six PVs in plain Python from a second implementation of the recipe with
hand-written schedules and checks them against the notebook and the library.</p>
<p class="note">Source for this page: <code>docs/build_page.py</code> (the figure and every number above come from the library).</p>
</body></html>
"""
(ROOT / "docs" / "index.html").write_text(page)
print("wrote docs/index.html", len(page), "bytes")
