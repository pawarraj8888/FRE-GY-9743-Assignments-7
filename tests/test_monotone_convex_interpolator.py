"""Tests for the HW7 monotone convex IFR interpolator (fixedincomelib/utilities/numerics.py).

Run with the course environment (Python 3.11, QuantLib 1.41):
    python tests/test_monotone_convex_interpolator.py          # plain runner
    python -m pytest tests -q                                  # with pytest installed
"""
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixedincomelib import (  # noqa: E402
    qfCreateBuildMethod, qfCreateData1D, qfCreateDataCollection, qfCreateModel,
    qfCreateModelBuildMethodCollection, qfCreateProductFRA, qfCreateProductOvernightIndexBasisSwap,
    qfCreateProductRFRSwap, qfCreateValuationParameters, qfCreateValuationParametersCollection,
    qfCreateValueReport, qfDiscountFactor,
)
from fixedincomelib.date import Date, accrued  # noqa: E402
from fixedincomelib.utilities import (  # noqa: E402
    ExtrapMethod, InterpMethod, Interpolator1DMonotoneConvex, InterpolatorFactory,
)

# The notebook's curves: node times from the 2026-01-02 value date (ACT/ACT ISDA) and IFR values.
NODE_DATES = ["2026-04-02", "2026-10-02", "2027-07-02", "2028-01-03", "2029-01-02", "2031-01-02"]
TIMES = [accrued(Date("2026-01-02"), Date(d)) for d in NODE_DATES]
SOFR = [0.025, 0.026, 0.028, 0.030, 0.035, 0.047]
FF = [0.023, 0.0237, 0.0252, 0.0265, 0.0305, 0.0405]


def curve(x, y):
    return InterpolatorFactory.create_1d_interpolator(x, y, InterpMethod.MONOTONE_CONVEX, ExtrapMethod.FLAT)


def sample(it, a, b, n=4001):
    grid = np.linspace(a, b, n)
    return grid, np.array([it.interpolate(t) for t in grid])


def simpson(it, a, b, n=20000):
    """Composite Simpson on interpolate() only: a check of integrate() that does not use it."""
    grid = np.linspace(a, b, n + 1)
    f = np.array([it.interpolate(t) for t in grid])
    h = (b - a) / n
    return h / 3.0 * (f[0] + f[-1] + 4.0 * f[1:-1:2].sum() + 2.0 * f[2:-1:2].sum())


def assert_close(a, b, tol=1e-12):
    assert abs(a - b) <= tol, (a, b, a - b)


def derivative_jump(it, p, eps=1e-6):
    """f'(p+) - f'(p-) from second-order one-sided differences (bias O(eps^2), so ~1e-12 on these curves)."""
    f = it.interpolate
    left = (3.0 * f(p) - 4.0 * f(p - eps) + f(p - 2.0 * eps)) / (2.0 * eps)
    right = (-3.0 * f(p) + 4.0 * f(p + eps) - f(p + 2.0 * eps)) / (2.0 * eps)
    return right - left


# ---- basic contract ----------------------------------------------------------------------------
def test_one_node_curve_is_constant_everywhere():
    it = curve([1.0054794521], [0.0005])
    for t in (0.0, 0.5, 1.0054794521, 3.0, 10.0):
        assert_close(it.interpolate(t), 0.0005)
    assert_close(it.integrate(0.0, 3.0), 0.0015)
    assert_close(it.integrate(2.0, 0.5), -0.00075)


def test_two_nodes_give_the_straight_line():
    it = curve([1.0, 3.0], [0.02, 0.03])
    assert_close(it.interpolate(1.5), 0.0225)
    assert_close(it.interpolate(2.0), 0.025)
    assert_close(it.integrate(1.0, 3.0), 0.05)


def test_curve_passes_through_the_supplied_points():
    for values in (SOFR, FF):
        it = curve(TIMES, values)
        for t, v in zip(TIMES, values):
            assert_close(it.interpolate(t), v, 1e-15)


def test_factory_returns_the_monotone_convex_class_and_requires_flat_extrapolation():
    it = curve(TIMES, SOFR)
    assert isinstance(it, Interpolator1DMonotoneConvex)
    assert it.interp_method == "MONOTONE_CONVEX" and it.extrap_method == "FLAT"
    assert it.length == 6
    try:
        InterpolatorFactory.create_1d_interpolator(TIMES, SOFR, InterpMethod.MONOTONE_CONVEX, ExtrapMethod.LINEAR)
    except AssertionError:
        return
    raise AssertionError("LINEAR extrapolation should be rejected")


def test_optional_gradient_stubs_are_left_as_supplied():
    it = curve(TIMES, SOFR)
    for call in (lambda: it.gradient_wrt_ordinate(1.0),
                 lambda: it.gradient_of_integrated_value_wrt_ordinate(0.0, 1.0)):
        try:
            call()
        except NotImplementedError:
            continue
        raise AssertionError("gradient stub should still raise NotImplementedError")


# ---- continuity and smoothness -----------------------------------------------------------------
def test_curve_is_continuous_and_has_a_continuous_derivative():
    for values in (SOFR, FF):
        it = curve(TIMES, values)
        eps = 1e-7
        for p in list(it.breaks_) + [TIMES[0] - 0.3, TIMES[-1] + 0.3]:
            assert abs(it.interpolate(p + eps) - it.interpolate(p - eps)) < 1e-8
        for p in it.breaks_[1:-1]:                      # interior breakpoints: nodes and knots
            assert abs(derivative_jump(it, p)) < 1e-8, p


def test_node_slopes_follow_the_chords_and_stay_capped():
    x = np.array([0.0, 0.1, 2.1, 2.2, 3.0, 3.5, 5.0])
    y = np.array([0.0, 0.01, 0.012, 0.022, 0.02, 0.02, 0.01])
    d = Interpolator1DMonotoneConvex._node_slopes(x, y)
    s = np.diff(y) / np.diff(x)
    assert d[3] == 0.0 and d[4] == 0.0 and d[5] == 0.0   # local maximum, then a flat interval on either side
    for i in range(1, len(x) - 1):
        if d[i] != 0.0:
            assert np.sign(d[i]) == np.sign(s[i - 1]) == np.sign(s[i])
            assert abs(d[i]) <= 2.0 * min(abs(s[i - 1]), abs(s[i])) + 1e-15
            assert min(s[i - 1], s[i]) <= d[i] <= max(s[i - 1], s[i])
    assert d[0] * s[0] > 0 and abs(d[0]) <= 2.0 * abs(s[0])
    assert d[-1] * s[-1] > 0 and abs(d[-1]) <= 2.0 * abs(s[-1])
    # a short steep interval next to a long flat one: the harmonic mean (0.00281) exceeds twice the flat chord
    w1, w2 = 2.0 * 2.0 + 0.1, 2.0 + 2.0 * 0.1
    assert (w1 + w2) / (w1 / s[0] + w2 / s[1]) > 2.0 * s[1]
    assert d[1] == 2.0 * s[1] and d[2] == 2.0 * s[1]


def test_end_slopes_follow_the_sign_rule_and_the_cap():
    slopes = Interpolator1DMonotoneConvex._node_slopes
    # parabola through (0,0), (1,0.01), (2,0.05) has slope -0.005 at 0: wrong sign against the chord 0.01 -> zero
    assert slopes(np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.01, 0.05]))[0] == 0.0
    # parabola through (0,0), (1,0.01), (2,-0.05) has slope 0.045 at 0: capped at twice the end chord
    assert slopes(np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.01, -0.05]))[0] == 0.02
    # mirror images at the last node
    assert slopes(np.array([0.0, 1.0, 2.0]), np.array([0.05, 0.01, 0.0]))[-1] == 0.0
    assert slopes(np.array([0.0, 1.0, 2.0]), np.array([-0.05, 0.01, 0.0]))[-1] == -0.02


# ---- shape within node intervals ---------------------------------------------------------------
def shape_per_interval(it, x):
    out = []
    for a, b in zip(x[:-1], x[1:]):
        _, v = sample(it, a, b)
        first, second = np.diff(v), np.diff(np.diff(v))
        out.append((bool((first >= -1e-15).all()), bool((first <= 1e-15).all()),
                    bool((second >= -1e-13).all()), bool((second <= 1e-13).all())))
    return out      # (increasing, decreasing, convex, concave) per interval


def test_increasing_convex_data_give_increasing_convex_pieces():
    for values in (SOFR, FF):
        for inc, dec, cvx, ccv in shape_per_interval(curve(TIMES, values), TIMES):
            assert inc and cvx


def test_decreasing_convex_data_give_decreasing_convex_pieces():
    x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    y = [0.05, 0.04, 0.033, 0.028, 0.025, 0.024]         # chord slopes rise from -0.010 to -0.001
    for inc, dec, cvx, ccv in shape_per_interval(curve(x, y), x):
        assert dec and cvx


def test_concave_data_give_monotone_concave_pieces():
    x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    y = [0.01, 0.02, 0.027, 0.032, 0.035, 0.036]         # chord slopes fall from 0.010 to 0.001
    for inc, dec, cvx, ccv in shape_per_interval(curve(x, y), x):
        assert inc and ccv


def test_local_maximum_and_minimum_stay_at_the_nodes_without_overshoot():
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    hump = curve(x, [0.01, 0.03, 0.04, 0.03, 0.01])
    grid, v = sample(hump, 0.0, 4.0)
    assert v.max() == 0.04 and grid[v.argmax()] == 2.0 and v.min() == 0.01
    shapes = shape_per_interval(hump, x)
    assert shapes[0][0] and shapes[1][0] and shapes[2][1] and shapes[3][1]
    assert shapes[1][3] and shapes[2][3]                  # the data approach the peak concavely: concave pieces
    bowl = curve(x, [0.04, 0.02, 0.01, 0.02, 0.04])
    grid, v = sample(bowl, 0.0, 4.0)
    assert v.min() == 0.01 and grid[v.argmin()] == 2.0 and v.max() == 0.04
    shapes = shape_per_interval(bowl, x)
    assert shapes[1][2] and shapes[2][2]                  # convex approach: convex pieces
    assert abs(derivative_jump(hump, 2.0)) < 1e-8          # smooth over the peak, with zero slope there
    assert abs((hump.interpolate(2.0 + 1e-6) - hump.interpolate(2.0 - 1e-6)) / 2e-6) < 1e-8


def test_flat_interval_is_constant_and_neighbours_land_flat():
    it = curve([0.0, 1.0, 2.0, 3.0], [0.02, 0.03, 0.03, 0.05])
    _, v = sample(it, 1.0, 2.0)
    assert (v == 0.03).all()
    eps = 1e-7
    assert abs(it.interpolate(1.0 - eps) - 0.03) < 1e-12 and abs(it.interpolate(2.0 + eps) - 0.03) < 1e-12


def test_slopes_on_the_same_side_of_the_chord_use_the_weighted_knot_and_stay_monotone():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 1.9, 3.9])                    # chords 1, 0.9, 2: a slight concave bend, then a convex one
    it = curve(x, y)
    d = Interpolator1DMonotoneConvex._node_slopes(x, y)
    s = np.diff(y) / np.diff(x)
    a, b = s[1] - d[1], d[2] - s[1]
    assert a < 0 < b                                      # both end slopes above the middle chord
    weight = abs(b) / (abs(a) + abs(b))
    assert any(abs(brk - (1.0 + weight)) < 1e-12 for brk in it.breaks_)
    knot_slope = 2.0 * s[1] - weight * d[1] - (1.0 - weight) * d[2]
    eps = 1e-7
    assert abs((it.interpolate(1.0 + weight + eps) - it.interpolate(1.0 + weight - eps)) / (2 * eps) - knot_slope) < 1e-7
    inc, dec, cvx, ccv = shape_per_interval(it, x)[1]
    assert inc and not cvx and not ccv                    # monotone with one inflection
    assert_close(it.interpolate(2.0), 1.9, 1e-15)


def test_collinear_run_is_reproduced_by_its_line_and_the_bend_stays_convex():
    x, y = np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0, 4.0])
    it = curve(x, y)
    assert np.allclose(Interpolator1DMonotoneConvex._node_slopes(x, y), [1.0, 1.0, 1.0, 2.5])
    for t in (0.25, 0.5, 1.5, 1.9):
        assert_close(it.interpolate(t), t, 1e-15)        # the run is the straight line
    assert not [brk for brk in it.breaks_ if 0.0 < brk < 2.0 and brk != 1.0]   # no knot inside the run
    assert shape_per_interval(it, x) == [(True, False, True, True), (True, False, True, True), (True, False, True, False)]
    for node in (1.0, 2.0):                               # smooth at both nodes, including the end of the run
        assert abs(derivative_jump(it, node)) < 1e-8


def test_tie_broken_only_by_rounding_counts_as_a_tie():
    x, y = np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([0.01, 0.0125, 0.015, 0.0175, 0.0215])
    chords = np.diff(y) / np.diff(x)
    assert chords[0] != chords[1] or chords[1] != chords[2]   # equal in decimal, not in binary
    it = curve(x, y)
    assert np.allclose(Interpolator1DMonotoneConvex._node_slopes(x, y)[:4], 0.0025)
    for t in (1.5, 2.5, 3.5):
        assert_close(it.interpolate(t), 0.01 + 0.0025 * (t - 1.0), 1e-15)
    for node in (2.0, 3.0, 4.0):
        assert abs(derivative_jump(it, node)) < 1e-8
    assert shape_per_interval(it, x)[3][2]                # the bend after the run is convex


def test_curve_depends_continuously_on_the_data_around_a_tie():
    base = curve([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 4.0])
    grid = np.linspace(0.0, 3.0, 301)                     # the run and the interval beyond it
    for eps in (1e-9, -1e-9, 1e-6, -1e-6, 1e-4, -1e-4, 1e-2, -1e-2):
        moved = curve([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0 + eps, 4.0])
        assert max(abs(moved.interpolate(t) - base.interpolate(t)) for t in grid) <= 2.0 * abs(eps)


def test_nearly_equal_chords_blend_the_run_end_slope():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    plain = Interpolator1DMonotoneConvex._node_slopes(x, np.array([0.0, 1.0, 2.3, 4.3]))   # chords 1, 1.3, 2: no blend
    assert abs(plain[2] - 6.0 / (3.0 / 1.3 + 3.0 / 2.0)) < 1e-15                              # plain harmonic mean (equal spacing)
    half = Interpolator1DMonotoneConvex._node_slopes(x, np.array([0.0, 1.0, 2.05, 4.05]))    # chords 1, 1.05, 2: gap 0.048
    weight = 1.0 - (0.05 / 1.05) / Interpolator1DMonotoneConvex.BLEND_TOLERANCE
    harmonic = 6.0 / (3.0 / 1.05 + 3.0 / 2.0)
    assert abs(half[2] - ((1.0 - weight) * harmonic + weight * 1.05)) < 1e-12
    assert 1.05 < half[2] < harmonic                        # between the run's chord and the default


def test_bump_sensitivity_has_a_finite_limit_on_round_quote_data():
    # the first three chords are exactly 0.2% per year, so a bump of node 1 makes and breaks a tie
    sofr = [0.025, 0.026003, 0.027499, 0.028512, 0.033, 0.047]
    base = notebook_model("MONOTONE_CONVEX", sofr=sofr)
    df0 = qfDiscountFactor(base, "SOFR-1B", "2028-07-03")
    per_unit = []
    for bump in (1e-6, 1e-7, 1e-8):
        bumped = notebook_model("MONOTONE_CONVEX", sofr=sofr)
        bumped.perturb_model_parameter("SOFR-1B", 1, bump)
        per_unit.append((qfDiscountFactor(bumped, "SOFR-1B", "2028-07-03") - df0) / bump)
    assert max(per_unit) / min(per_unit) < 1.01 and all(v < 0 for v in per_unit)


def test_two_collinear_runs_meet_in_a_single_kink():
    x, y = np.array([0.0, 1.0, 2.0, 3.0, 4.0]), np.array([0.0, 1.0, 2.0, 4.0, 6.0])
    it = curve(x, y)                                      # the only convex interpolant is piecewise linear
    for t, v in ((0.5, 0.5), (1.5, 1.5), (2.5, 3.0), (3.5, 5.0)):
        assert_close(it.interpolate(t), v, 1e-15)
    jumps = [derivative_jump(it, n) for n in (1.0, 2.0, 3.0)]
    assert abs(jumps[0]) < 1e-8 and abs(jumps[2]) < 1e-8 and abs(jumps[1] - 1.0) < 1e-8
    assert all(cvx for _, _, cvx, _ in shape_per_interval(it, x))


def test_run_ending_at_a_maximum_stays_straight_up_to_the_peak():
    x, y = np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 0.01, 0.02, 0.015])
    it = curve(x, y)
    assert Interpolator1DMonotoneConvex._node_slopes(x, y)[2] == 0.0
    assert_close(it.interpolate(1.5), 0.015, 1e-15)      # the run is the line; the peak is at the node
    shapes = shape_per_interval(it, x)
    assert shapes[0][2] and shapes[1][2] and shapes[2][1] and shapes[2][3]
    assert abs(derivative_jump(it, 2.0) - (0.0 - 0.01)) < 1e-8   # the kink at the peak: slope 0.01 then 0
    _, v = sample(it, 0.0, 3.0)
    assert v.max() <= 0.02 and it.interpolate(2.0) == 0.02


def test_steep_run_after_a_gentle_interval_keeps_the_cap_and_kinks_once():
    x, y = np.array([0.0, 1.0, 2.0, 3.0, 4.0]), np.array([0.0, 0.01, 0.011, 0.021, 0.031])
    d = Interpolator1DMonotoneConvex._node_slopes(x, y)
    s = np.diff(y) / np.diff(x)
    assert d[2] == 2.0 * s[1] and abs(d[3] - s[2]) < 1e-15 and abs(d[4] - s[3]) < 1e-15   # the cap holds the run's end slope down
    it = curve(x, y)
    for t in (2.5, 3.5):
        assert_close(it.interpolate(t), 0.011 + 0.01 * (t - 2.0), 1e-15)
    assert abs(derivative_jump(it, 2.0) - (s[2] - d[2])) < 1e-8
    assert all(inc for inc, _, _, _ in shape_per_interval(it, x))


def test_negative_rates_shift_the_curve_exactly():
    base, shifted = curve(TIMES, SOFR), curve(TIMES, [v - 0.05 for v in SOFR])
    for t in np.linspace(0.0, 5.5, 111):
        assert_close(shifted.interpolate(t), base.interpolate(t) - 0.05, 1e-15)
    assert_close(shifted.integrate(0.0, 4.0), base.integrate(0.0, 4.0) - 0.2, 1e-14)


def test_decreasing_concave_data_give_decreasing_concave_pieces():
    x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    y = [0.036, 0.035, 0.032, 0.027, 0.020, 0.010]       # chord slopes fall from -0.001 to -0.010
    for inc, dec, cvx, ccv in shape_per_interval(curve(x, y), x):
        assert dec and ccv


def test_duplicate_node_times_are_rejected():
    try:
        curve([0.0, 1.0, 1.0, 2.0], [0.01, 0.02, 0.02, 0.03])
    except AssertionError:
        return
    raise AssertionError("duplicate node times should be rejected")


def test_random_data_pieces_never_overshoot_and_integrate_consistently():
    rng = np.random.default_rng(7)
    for _ in range(50):
        n = int(rng.integers(3, 9))
        x = np.cumsum(rng.uniform(0.1, 2.0, size=n))
        y = np.cumsum(rng.uniform(-0.01, 0.01, size=n)) + 0.03
        it = curve(x, y)
        for a, b, ya, yb in zip(x[:-1], x[1:], y[:-1], y[1:]):
            _, v = sample(it, a, b, 401)
            assert v.min() >= min(ya, yb) - 1e-12 and v.max() <= max(ya, yb) + 1e-12
            first = np.diff(v)
            assert (first >= -1e-12).all() or (first <= 1e-12).all()
        assert_close(it.integrate(x[0], x[-1]), simpson(it, x[0], x[-1], 4000), 1e-9)


# ---- integration and extrapolation -------------------------------------------------------------
def test_integral_matches_simpson_quadrature_of_interpolate():
    for values in (SOFR, FF):
        it = curve(TIMES, values)
        for a, b in ((0.0, 3.2465753425), (0.3369863014, 0.8410958904), (1.0, 4.0), (0.0, 7.0), (4.5, 6.5)):
            assert_close(it.integrate(a, b), simpson(it, a, b), 1e-10)


def test_integral_is_additive_and_antisymmetric():
    it = curve(TIMES, SOFR)
    a, b, c = 0.1, 1.7, 5.3
    assert_close(it.integrate(a, c), it.integrate(a, b) + it.integrate(b, c), 1e-15)
    assert_close(it.integrate(c, a), -it.integrate(a, c), 1e-15)
    assert_close(it.integrate(b, b), 0.0, 1e-18)


def test_flat_extrapolation_on_both_sides():
    it = curve(TIMES, SOFR)
    assert it.interpolate(0.0) == 0.025 and it.interpolate(0.2) == 0.025
    assert it.interpolate(5.0) == 0.047 and it.interpolate(9.0) == 0.047
    assert_close(it.integrate(0.0, TIMES[0]), 0.025 * TIMES[0], 1e-15)
    assert_close(it.integrate(5.0, 7.0), 0.047 * 2.0, 1e-15)


def test_hand_calculation_of_a_three_node_curve():
    # x = [0, 1, 3], y = [0, 1, 2]: chords s = [1, 1/2], h = [1, 2]
    # interior slope: harmonic mean with w1 = 2*2 + 1 = 5, w2 = 2 + 2*1 = 4  ->  9 / (5/1 + 4/0.5) = 9/13
    # ends: parabola through three points: d0 = (4*1 - 1*0.5) / 3 = 7/6,  d2 = (5*0.5 - 2*1) / 3 = 1/6
    d0, d1, d2 = 7.0 / 6.0, 9.0 / 13.0, 1.0 / 6.0
    it = curve([0.0, 1.0, 3.0], [0.0, 1.0, 2.0])
    assert np.allclose(it._node_slopes(np.array([0.0, 1.0, 3.0]), np.array([0.0, 1.0, 2.0])), [d0, d1, d2])
    # interval 1: a = 1 - 7/6 < 0, b = 9/13 - 1 < 0 (concave), |b| > |a| -> linear piece of length
    # alpha = h (|b| - |a|) / (|a| + |b|) with slope d0, then a quadratic running from d0 to d1
    a, b = 1.0 - d0, d1 - 1.0
    alpha = 1.0 * (abs(b) - abs(a)) / (abs(a) + abs(b))
    assert_close(it.breaks_[1], alpha)
    assert_close(it.interpolate(alpha / 2.0), d0 * alpha / 2.0)
    L = 1.0 - alpha
    assert_close(it.interpolate(alpha + L / 2.0), d0 * alpha + d0 * L / 2.0 + (d1 - d0) / (2.0 * L) * (L / 2.0) ** 2)
    integral_1 = d0 * alpha ** 2 / 2.0 + (d0 * alpha) * L + d0 * L ** 2 / 2.0 + (d1 - d0) / (2.0 * L) * L ** 3 / 3.0
    assert_close(it.integrate(0.0, 1.0), integral_1)
    # interval 2: a = 1/2 - 9/13 < 0, b = 1/6 - 1/2 < 0, |b| > |a| -> linear with slope d1 for
    # alpha2 = 2 (|b| - |a|) / (|a| + |b|), then a quadratic from d1 to d2
    a2, b2 = 0.5 - d1, d2 - 0.5
    alpha2 = 2.0 * (abs(b2) - abs(a2)) / (abs(a2) + abs(b2))
    assert_close(it.breaks_[3], 1.0 + alpha2)
    L2 = 2.0 - alpha2
    assert_close(it.interpolate(1.0 + alpha2 / 2.0), 1.0 + d1 * alpha2 / 2.0)                       # linear part
    assert_close(it.interpolate(1.0 + alpha2 + L2 / 2.0), 1.0 + d1 * alpha2 + d1 * L2 / 2.0 + (d2 - d1) / (2.0 * L2) * (L2 / 2.0) ** 2)
    assert_close(1.0 + d1 * alpha2 + d1 * L2 + (d2 - d1) / (2.0 * L2) * L2 ** 2, 2.0, 1e-14)          # the hand formula lands on f_2
    integral_2 = (1.0 * alpha2 + d1 * alpha2 ** 2 / 2.0 + (1.0 + d1 * alpha2) * L2 + d1 * L2 ** 2 / 2.0
                  + (d2 - d1) / (2.0 * L2) * L2 ** 3 / 3.0)
    assert_close(it.integrate(1.0, 3.0), integral_2)
    assert_close(it.integrate(-1.0, 4.0), integral_1 + integral_2 + 0.0 * 1.0 + 2.0 * 1.0)


# ---- the curve inside the library -------------------------------------------------------------
def notebook_model(method, sofr=SOFR):
    node_dates = NODE_DATES
    sofr_quotes = pd.DataFrame({"values": sofr}, index=node_dates)
    ff_quotes = pd.DataFrame({"values": FF}, index=node_dates)
    funding_quotes = pd.DataFrame({"values": [0.0005]}, index=["2027-01-04"])
    data_collection = qfCreateDataCollection([
        qfCreateData1D("INSTANTANEOUS FORWARD RATE", "SOFR-1B-IFR", sofr_quotes),
        qfCreateData1D("INSTANTANEOUS FORWARD RATE", "FF-1B-IFR", ff_quotes),
        qfCreateData1D("INSTANTANEOUS FORWARD RATE", "SOFR-1B-FUNDING-IFR", funding_quotes),
    ])
    curve_inputs = [
        ("YIELD_CURVE_INDEX", {"TARGET": "SOFR-1B", "INSTANTANEOUS FORWARD RATE": "SOFR-1B-IFR"}),
        ("YIELD_CURVE_INDEX", {"TARGET": "FF-1B", "INSTANTANEOUS FORWARD RATE": "FF-1B-IFR"}),
        ("YIELD_CURVE_FUNDING", {"TARGET": "SOFR-1B-FLAT", "REFERENCE INDEX": "SOFR-1B",
                                "INSTANTANEOUS FORWARD RATE": "SOFR-1B-FUNDING-IFR"}),
    ]
    build_methods = qfCreateModelBuildMethodCollection([
        qfCreateBuildMethod(kind, {**inputs, "INTERPOLATION METHOD": method, "EXTRAPOLATION METHOD": "FLAT"})
        for kind, inputs in curve_inputs
    ])
    return qfCreateModel("2026-01-02", "YIELD_CURVE", data_collection, build_methods)


def test_model_discount_factors_come_from_the_curve_integral():
    model = notebook_model("MONOTONE_CONVEX")
    sofr = curve(TIMES, SOFR)
    for d in ("2026-05-05", "2026-11-05", "2027-04-06", "2029-04-04", "2032-01-02"):
        T = accrued(Date("2026-01-02"), Date(d))
        assert_close(qfDiscountFactor(model, "SOFR-1B", d), math.exp(-sofr.integrate(0.0, T)), 1e-14)
        # the one-node funding curve is a constant 5 bp over SOFR
        assert_close(qfDiscountFactor(model, "SOFR-1B-FLAT", d) / qfDiscountFactor(model, "SOFR-1B", d),
                     math.exp(-0.0005 * T), 1e-14)
    assert isinstance(model.retrieve_model_component("SOFR-1B").state_data_interpolator, Interpolator1DMonotoneConvex)


def test_bumping_a_node_rebuilds_the_curve_locally():
    model = notebook_model("MONOTONE_CONVEX")
    before = model.retrieve_model_component("SOFR-1B").state_data_interpolator
    far = np.linspace(TIMES[4], TIMES[5], 50)              # the last interval is two nodes away from the bump
    near = np.linspace(TIMES[1], TIMES[3], 50)
    far_values = [before.interpolate(t) for t in far]
    near_values = [before.interpolate(t) for t in near]
    df_before = qfDiscountFactor(model, "SOFR-1B", "2027-01-04")
    model.perturb_model_parameter("SOFR-1B", 2, 0.0001)                  # bump the 2027-07-02 node by 1 bp
    after = model.retrieve_model_component("SOFR-1B").state_data_interpolator
    assert after is not before and after.interpolate(TIMES[2]) == 0.0281
    assert max(abs(after.interpolate(t) - v) for t, v in zip(far, far_values)) < 1e-15
    assert max(abs(after.interpolate(t) - v) for t, v in zip(near, near_values)) > 1e-5
    assert qfDiscountFactor(model, "SOFR-1B", "2027-01-04") < df_before


def value(model, product, label="FUNDING INDEX PARAMETER"):
    params = qfCreateValuationParametersCollection([qfCreateValuationParameters(label, {"FUNDING INDEX": "SOFR-1B-FLAT"})])
    return qfCreateValueReport(model, product, params, "PV")[0][1]


def test_notebook_present_values_under_both_interpolations():
    fra = qfCreateProductFRA(effective_date="2026-05-05", term_or_termination_date="2026-11-05",
                             projection_index="SOFR-1B", fixed_rate=0.032, notional=1000000, pay_or_rec="receive",
                             accrual_basis="ACT/360", business_day_convention="F", holiday_convention="USGS")
    swap = qfCreateProductRFRSwap(effective_date="2026-04-02", term_or_termination_date="2029-04-02", on_index="SOFR-1B",
                                  fixed_rate=0.034, notional=1000000, pay_or_rec="receive", accrual_period="1Y",
                                  floating_leg_accrual_period="1Y", accrual_basis="ACT/360", compounding_method="compound",
                                  spread=0.0, payment_off_set="2D", pay_business_day_convention="F",
                                  pay_holiday_convention="USGS")
    basis = qfCreateProductOvernightIndexBasisSwap(effective_date="2026-07-02", term_or_termination_date="2028-07-03",
                                                   on_index_leg_1="SOFR-1B", on_index_leg_2="FF-1B", spread_over_leg_1=0.001,
                                                   notional=1000000, pay_or_rec_leg_1="receive", accrual_period_1="3M",
                                                   accrual_period_2="3M", accrual_basis="ACT/360",
                                                   compounding_method="compound", payment_off_set="2D",
                                                   pay_business_day_convention="F", pay_holiday_convention="USGS")
    pcp, mc = notebook_model("PIECEWISE_CONSTANT_LEFT_CONTINUOUS"), notebook_model("MONOTONE_CONVEX")
    # the professor's printed PCP values, and the monotone convex values the notebook now prints
    assert_close(value(pcp, fra), 2908.2866392379638, 1e-6)
    assert_close(value(pcp, swap), 6222.517513070154, 1e-6)
    assert_close(value(pcp, basis), 8351.348908181993, 1e-6)
    assert_close(value(mc, fra), 3261.8437894226518, 1e-6)
    assert_close(value(mc, swap), 12874.702872320777, 1e-6)
    assert_close(value(mc, basis), 7631.9437072256405, 1e-6)


def test_pv_differences_have_the_sign_the_curves_imply():
    # PCP holds each interval at the value of its right-hand node, so it sits above a rising point-interpolant.
    pcp_curve = InterpolatorFactory.create_1d_interpolator(TIMES, SOFR, InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                                                           ExtrapMethod.FLAT)
    mc_curve = curve(TIMES, SOFR)
    grid = np.linspace(TIMES[0] + 1e-9, TIMES[-1], 2001)
    assert all(mc_curve.interpolate(t) <= pcp_curve.interpolate(t) + 1e-15 for t in grid)
    # lower forwards mean a smaller floating leg, so both receivers of fixed gain
    fra = qfCreateProductFRA(effective_date="2026-05-05", term_or_termination_date="2026-11-05",
                             projection_index="SOFR-1B", fixed_rate=0.032, notional=1000000, pay_or_rec="receive",
                             accrual_basis="ACT/360", business_day_convention="F", holiday_convention="USGS")
    swap = qfCreateProductRFRSwap(effective_date="2026-04-02", term_or_termination_date="2029-04-02", on_index="SOFR-1B",
                                  fixed_rate=0.034, notional=1000000, pay_or_rec="receive", accrual_period="1Y",
                                  floating_leg_accrual_period="1Y", accrual_basis="ACT/360", compounding_method="compound",
                                  spread=0.0, payment_off_set="2D", pay_business_day_convention="F",
                                  pay_holiday_convention="USGS")
    pcp, mc = notebook_model("PIECEWISE_CONSTANT_LEFT_CONTINUOUS"), notebook_model("MONOTONE_CONVEX")
    assert value(mc, fra) > value(pcp, fra)
    assert value(mc, swap) > value(pcp, swap)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")
