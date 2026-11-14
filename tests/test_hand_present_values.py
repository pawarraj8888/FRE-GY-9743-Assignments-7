"""The six notebook present values re-derived in plain Python, without the library.

A second implementation of the interpolation recipe (node slopes, knots, a piecewise-linear derivative
integrated with the trapezoid rule, the curve integrated with Simpson per piece), the left-continuous
step curve for PCP, ACT/ACT ISDA node times, ACT/360 accruals, compounded coupons as (D(Ts)/D(Te) - 1)/alpha
and hand-written schedules.  The tests compare these values with the library's and with the notebook.

    python tests/test_hand_present_values.py          # or: python -m pytest tests -q
"""
import os
import sys
from datetime import date
from math import exp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VALUE_DATE = "2026-01-02"
NODES = ["2026-04-02", "2026-10-02", "2027-07-02", "2028-01-03", "2029-01-02", "2031-01-02"]
SOFR = [0.025, 0.026, 0.028, 0.030, 0.035, 0.047]
FF = [0.023, 0.0237, 0.0252, 0.0265, 0.0305, 0.0405]
FUNDING_SPREAD = 0.0005                      # one node, so a constant under either interpolation
NOTIONAL, FRA_RATE, SWAP_RATE, BASIS_SPREAD = 1_000_000, 0.032, 0.034, 0.001
NOTEBOOK = {"PCP": (2908.2866392379638, 6222.517513070154, 8351.348908181993),
            "MC": (3261.8437894226518, 12874.702872320777, 7631.9437072256405)}

# schedules by hand: annual periods roll 2028-04-02 (Sunday) to Monday; payments two US business days later;
# the quarterly basis schedule runs backward from 2028-07-03 and leaves a short first stub whose end differs
# between the SOFR calendar (July 3, 2026 is a holiday there) and the Fed Funds calendar.
FRA = ("2026-05-05", "2026-11-05", "2026-11-05")
SWAP = [("2026-04-02", "2027-04-02", "2027-04-06"), ("2027-04-02", "2028-04-03", "2028-04-05"), ("2028-04-03", "2029-04-02", "2029-04-04")]
QUARTERS = [("2026-10-05", "2027-01-04", "2027-01-06"), ("2027-01-04", "2027-04-05", "2027-04-07"), ("2027-04-05", "2027-07-06", "2027-07-08"),
            ("2027-07-06", "2027-10-04", "2027-10-06"), ("2027-10-04", "2028-01-03", "2028-01-05"), ("2028-01-03", "2028-04-03", "2028-04-05"),
            ("2028-04-03", "2028-07-03", "2028-07-06")]
LEG1 = [("2026-07-02", "2026-07-06", "2026-07-08"), ("2026-07-06", "2026-10-05", "2026-10-07")] + QUARTERS
LEG2 = [("2026-07-02", "2026-07-03", "2026-07-07"), ("2026-07-03", "2026-10-05", "2026-10-07")] + QUARTERS


def days(a, b):
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def _leap(y):
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


def isda_years(a, b):
    """ACT/ACT (ISDA): each calendar year's slice over that year's length."""
    d0, d1 = date.fromisoformat(a), date.fromisoformat(b)
    if d0.year == d1.year:
        return (d1 - d0).days / (366 if _leap(d0.year) else 365)
    first = (date(d0.year + 1, 1, 1) - d0).days / (366 if _leap(d0.year) else 365)
    last = (d1 - date(d1.year, 1, 1)).days / (366 if _leap(d1.year) else 365)
    return first + (d1.year - d0.year - 1) + last


def years(d):
    return isda_years(VALUE_DATE, d)


def act360(a, b):
    return days(a, b) / 360


def pcp_integral(x, y, T):
    """Left-continuous steps: node i's value on (x[i-1], x[i]], the first value before x[0], the last after x[-1]."""
    total, lower = 0.0, 0.0
    for i, (xi, yi) in enumerate(zip(x, y)):
        upper = float("inf") if i == len(x) - 1 else xi
        total += yi * max(0.0, min(T, upper) - lower)
        lower = upper
    return total


TOL = 1e-10
BLEND = 0.1


def _limit(d, s1, s2):
    if d * s1 <= 0 or d * s2 <= 0:
        return 0.0
    return (1 if d > 0 else -1) * min(abs(d), 2 * min(abs(s1), abs(s2)))


def node_slopes(x, y):
    n = len(x) - 1
    h = [x[i + 1] - x[i] for i in range(n)]
    s = [(y[i + 1] - y[i]) / h[i] for i in range(n)]
    if n == 1:
        return s, [s[0], s[0]]
    d = [0.0] * (n + 1)
    for i in range(1, n):
        if s[i - 1] * s[i] > 0:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            d[i] = _limit((w1 + w2) / (w1 / s[i - 1] + w2 / s[i]), s[i - 1], s[i])
    d[0] = _limit(((2 * h[0] + h[1]) * s[0] - h[0] * s[1]) / (h[0] + h[1]), s[0], s[0])
    d[n] = _limit(((2 * h[n - 1] + h[n - 2]) * s[n - 1] - h[n - 1] * s[n - 2]) / (h[n - 1] + h[n - 2]), s[n - 1], s[n - 1])
    proposals = {}
    for i in range(n - 1):
        gap = abs(s[i] - s[i + 1]) / max(abs(s[i]), abs(s[i + 1]), 1e-300)
        weight = 1.0 if gap <= TOL else max(0.0, 1.0 - gap / BLEND)
        if weight > 0.0:
            for node, chord in ((i, s[i]), (i + 1, (s[i] + s[i + 1]) / 2), (i + 2, s[i + 1])):
                proposals.setdefault(node, []).append((weight, chord))
    for node, items in proposals.items():
        weight, chord = max(items, key=lambda item: item[0])
        d[node] = _limit((1 - weight) * d[node] + weight * chord, s[max(node - 1, 0)], s[min(node, n - 1)])
    return s, d


def derivative_breakpoints(x, y):
    """Per interval, the (time, slope) points of the piecewise-linear derivative."""
    s, d = node_slopes(x, y)
    pieces = []
    for i in range(len(x) - 1):
        h, x0, x1 = x[i + 1] - x[i], x[i], x[i + 1]
        a, b = s[i] - d[i], d[i + 1] - s[i]
        scale = TOL * (abs(s[i]) + abs(d[i]) + abs(d[i + 1]))
        a = 0.0 if abs(a) <= scale else a
        b = 0.0 if abs(b) <= scale else b
        if a == 0.0 or b == 0.0:
            seg = [(x0, s[i]), (x1, s[i])]
        elif a * b > 0:
            if abs(abs(a) - abs(b)) <= scale:
                seg = [(x0, d[i]), (x1, 2 * s[i] - d[i])]
            elif abs(b) < abs(a):
                knot = x0 + 2 * h * abs(b) / (abs(a) + abs(b))
                seg = [(x0, d[i]), (knot, d[i + 1]), (x1, d[i + 1])]
            else:
                knot = x0 + h * (abs(b) - abs(a)) / (abs(a) + abs(b))
                seg = [(x0, d[i]), (knot, d[i]), (x1, d[i + 1])]
        else:
            w = abs(b) / (abs(a) + abs(b))
            seg = [(x0, d[i]), (x0 + w * h, 2 * s[i] - w * d[i] - (1 - w) * d[i + 1]), (x1, d[i + 1])]
        pieces.append(seg)
    return pieces


def mc_curve(x, y):
    """value(t) and integral(T) of the monotone convex curve, computed from the derivative breakpoints."""
    if len(x) == 1:
        return (lambda t: y[0]), (lambda T: y[0] * T)
    pieces = derivative_breakpoints(x, y)

    def value(t):
        if t <= x[0]:
            return y[0]
        if t >= x[-1]:
            return y[-1]
        i = max(k for k in range(len(x) - 1) if x[k] <= t)
        total, seg = y[i], pieces[i]
        for (t0, g0), (t1, g1) in zip(seg[:-1], seg[1:]):
            if t <= t0:
                break
            hi = min(t, t1)
            g_hi = g0 if t1 == t0 else g0 + (g1 - g0) * (hi - t0) / (t1 - t0)
            total += (g0 + g_hi) / 2 * (hi - t0)                  # trapezoid is exact for a linear derivative
        return total

    def integral(T):
        if T <= x[0]:
            return y[0] * T
        total = y[0] * x[0]
        for seg in pieces:
            for (t0, _), (t1, _) in zip(seg[:-1], seg[1:]):
                hi = min(t1, T)
                if hi <= t0:
                    continue
                total += (hi - t0) / 6 * (value(t0) + 4 * value((t0 + hi) / 2) + value(hi))   # Simpson: exact for quadratics
        if T > x[-1]:
            total += y[-1] * (T - x[-1])
        return total

    for xi, yi in zip(x, y):
        assert abs(value(xi) - yi) < 1e-12
    return value, integral


def hand_pvs():
    x = [years(d) for d in NODES]
    curves = {"PCP": {"SOFR": lambda T: pcp_integral(x, SOFR, T), "FF": lambda T: pcp_integral(x, FF, T)},
              "MC": {"SOFR": mc_curve(x, SOFR)[1], "FF": mc_curve(x, FF)[1]}}

    def df(m, idx, d):
        return exp(-curves[m][idx](years(d)))

    def df_fund(m, d):
        return df(m, "SOFR", d) * exp(-FUNDING_SPREAD * years(d))

    def compounded(m, idx, a, b):
        return (df(m, idx, a) / df(m, idx, b) - 1.0) / act360(a, b)

    out = {}
    for m in ("PCP", "MC"):
        a, b, p = FRA
        fra = NOTIONAL * act360(a, b) * (FRA_RATE - compounded(m, "SOFR", a, b)) * df_fund(m, p)
        fixed = sum(NOTIONAL * SWAP_RATE * act360(a, b) * df_fund(m, p) for a, b, p in SWAP)
        floating = sum(NOTIONAL * act360(a, b) * compounded(m, "SOFR", a, b) * df_fund(m, p) for a, b, p in SWAP)
        leg1 = sum(NOTIONAL * act360(a, b) * (compounded(m, "SOFR", a, b) + BASIS_SPREAD) * df_fund(m, p) for a, b, p in LEG1)
        leg2 = sum(NOTIONAL * act360(a, b) * compounded(m, "FF", a, b) * df_fund(m, p) for a, b, p in LEG2)
        out[m] = (fra, fixed - floating, leg1 - leg2)
    return out


def test_hand_values_match_the_notebook():
    got = hand_pvs()
    for m in ("PCP", "MC"):
        for g, want in zip(got[m], NOTEBOOK[m]):
            assert abs(g - want) < 1e-6, (m, g, want)


def test_hand_curve_matches_the_library_curve():
    from fixedincomelib.utilities import ExtrapMethod, InterpMethod, InterpolatorFactory
    x = [years(d) for d in NODES]
    for y in (SOFR, FF, [0.0, 1.0, 2.0, 4.0, 6.0, 5.0], [0.01, 0.0125, 0.015, 0.0175, 0.0215, 0.03]):
        lib = InterpolatorFactory.create_1d_interpolator(x, y, InterpMethod.MONOTONE_CONVEX, ExtrapMethod.FLAT)
        value, integral = mc_curve(x, y)
        for k in range(0, 56):
            t = 0.1 * k
            assert abs(value(t) - lib.interpolate(t)) < 1e-12, (y, t)
            assert abs(integral(t) - lib.integrate(0.0, t)) < 1e-12, (y, t)


def test_hand_schedules_and_day_counts_match_the_library():
    from fixedincomelib.date import Date, accrued
    for d in NODES + [p for _, _, p in SWAP + LEG1 + LEG2] + [FRA[0], FRA[1]]:
        assert abs(years(d) - accrued(Date(VALUE_DATE), Date(d))) < 1e-14, d
    from fixedincomelib import qfCreateProductOvernightIndexBasisSwap, qfCreateProductRFRSwap
    swap = qfCreateProductRFRSwap(effective_date="2026-04-02", term_or_termination_date="2029-04-02", on_index="SOFR-1B",
                                  fixed_rate=SWAP_RATE, notional=NOTIONAL, pay_or_rec="receive", accrual_period="1Y",
                                  floating_leg_accrual_period="1Y", accrual_basis="ACT/360", compounding_method="compound",
                                  spread=0.0, payment_off_set="2D", pay_business_day_convention="F", pay_holiday_convention="USGS")
    basis = qfCreateProductOvernightIndexBasisSwap(effective_date="2026-07-02", term_or_termination_date="2028-07-03",
                                                   on_index_leg_1="SOFR-1B", on_index_leg_2="FF-1B", spread_over_leg_1=BASIS_SPREAD,
                                                   notional=NOTIONAL, pay_or_rec_leg_1="receive", accrual_period_1="3M",
                                                   accrual_period_2="3M", accrual_basis="ACT/360", compounding_method="compound",
                                                   payment_off_set="2D", pay_business_day_convention="F", pay_holiday_convention="USGS")

    def dates(stream):
        return [(cf[0].effective_date_.ISO(), cf[0].termination_date_.ISO(), cf[0].payment_date_.ISO()) for cf in stream.elements_]
    assert dates(swap.floating_leg) == SWAP
    assert dates(basis.floating_leg_1) == LEG1 and dates(basis.floating_leg_2) == LEG2


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")
