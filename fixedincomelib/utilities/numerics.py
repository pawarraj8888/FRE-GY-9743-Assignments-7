import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'
    MONOTONE_CONVEX = 'MONOTONE_CONVEX'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        return float(self.values_[idx])

    def integrate(self, start_x: float, end_x: float) -> float:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        overlaps = np.maximum(hi - lo, 0.0)
        return float(np.dot(np.asarray(self.values_, dtype=float), overlaps))

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        grad = np.zeros(self.length_)
        grad[idx] = 1.0
        return grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        return np.maximum(hi - lo, 0.0)



class Interpolator1DMonotoneConvex(Interpolator1D):
    """HW7: a shape-preserving C1 quadratic spline through the IFR points.

    The curve passes through every (axis1, value) pair, is continuous with a
    continuous first derivative, and is monotone and convex on every node
    interval whose data allow it.  Construction, in the spirit of Hagan and
    West (2006):

    1. Node slopes.  Each interior node gets the weighted harmonic mean of the
       two neighbouring chord slopes (Fritsch and Butland), zero when the node
       is a local extremum or next to a flat interval, and never more than
       twice the smaller neighbouring chord in magnitude.  End nodes take the
       slope of the parabola through the three outermost points, limited the
       same way.  So a node slope always has the sign of the chords around it.
    2. Interval pieces.  On each interval the derivative is piecewise linear
       through (left slope, knot slope, right slope).  When the two end slopes
       straddle the chord slope the knot is placed so that one piece is linear
       (Schumaker, 1983; the analogue of Hagan-West's flat segment) and the
       derivative runs monotonically from one end slope to the other: the
       piece is monotone and convex (slopes below-then-above the chord) or
       monotone and concave (above-then-below).  When both end slopes sit on
       the same side of the chord no C1 convex piece exists; the knot goes to
       the midpoint and the slope cap keeps the piece monotone.
    3. Extrapolation is flat on both sides; one node gives a constant curve.
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values, InterpMethod.MONOTONE_CONVEX,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT
        x = np.asarray(self.axis1_, dtype=float)
        y = np.asarray(self.values_, dtype=float)
        if self.length_ == 1:
            self.breaks_ = np.array([x[0], x[0]])
            self.coefs_ = np.zeros((1, 3))
            self.coefs_[0, 0] = y[0]
        else:
            self.breaks_, self.coefs_ = self._build_pieces(x, y, self._node_slopes(x, y))
        lengths = np.diff(self.breaks_)
        piece_areas = (self.coefs_[:, 0] * lengths + self.coefs_[:, 1] * lengths ** 2 / 2.0
                       + self.coefs_[:, 2] * lengths ** 3 / 3.0)
        self.cumulative_ = np.concatenate(([0.0], np.cumsum(piece_areas)))

    # -- construction ---------------------------------------------------------------------------
    @staticmethod
    def _node_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        h = np.diff(x)
        s = np.diff(y) / h
        n = len(h)
        d = np.zeros(n + 1)
        if n == 1:
            d[:] = s[0]
            return d
        for i in range(1, n):
            if s[i - 1] * s[i] <= 0.0:
                continue
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            d[i] = Interpolator1DMonotoneConvex._limit(
                (w1 + w2) / (w1 / s[i - 1] + w2 / s[i]), s[i - 1], s[i])
        d[0] = Interpolator1DMonotoneConvex._end_slope(h[0], h[1], s[0], s[1])
        d[n] = Interpolator1DMonotoneConvex._end_slope(h[n - 1], h[n - 2], s[n - 1], s[n - 2])
        return d

    @staticmethod
    def _end_slope(h_end: float, h_next: float, s_end: float, s_next: float) -> float:
        parabola = ((2.0 * h_end + h_next) * s_end - h_end * s_next) / (h_end + h_next)
        return Interpolator1DMonotoneConvex._limit(parabola, s_end, s_end)

    @staticmethod
    def _limit(slope: float, s_left: float, s_right: float) -> float:
        """Zero unless the slope has the sign of both chords; magnitude at most twice the smaller chord."""
        if slope * s_left <= 0.0 or slope * s_right <= 0.0:
            return 0.0
        return float(np.sign(slope) * min(abs(slope), 2.0 * min(abs(s_left), abs(s_right))))

    @staticmethod
    def _interval_segments(h: float, s: float, d0: float, d1: float):
        """(length, left slope, right slope) of the one or two pieces covering an interval."""
        a, b = s - d0, d1 - s                       # end slopes relative to the chord
        if a == 0.0 and b == 0.0:
            return [(h, d0, d1)]
        if a * b > 0.0:                             # slopes straddle the chord: one linear piece
            if abs(a) == abs(b):
                return [(h, d0, d1)]
            if abs(b) < abs(a):
                alpha = 2.0 * h * abs(b) / (abs(a) + abs(b))
                return [(alpha, d0, d1), (h - alpha, d1, d1)]
            alpha = h * (abs(b) - abs(a)) / (abs(a) + abs(b))
            return [(alpha, d0, d0), (h - alpha, d0, d1)]
        knot_slope = 2.0 * s - 0.5 * (d0 + d1)      # same side of the chord: midpoint knot
        return [(0.5 * h, d0, knot_slope), (0.5 * h, knot_slope, d1)]

    @staticmethod
    def _build_pieces(x: np.ndarray, y: np.ndarray, d: np.ndarray):
        breaks, coefs = [x[0]], []
        for i in range(len(x) - 1):
            h = x[i + 1] - x[i]
            level, start = y[i], x[i]
            for length, left, right in Interpolator1DMonotoneConvex._interval_segments(
                    h, (y[i + 1] - y[i]) / h, d[i], d[i + 1]):
                c2 = 0.0 if length == 0.0 else (right - left) / (2.0 * length)
                coefs.append([level, left, c2])
                level = level + length * (left + c2 * length)
                start = start + length
                breaks.append(start)
            breaks[-1] = x[i + 1]                   # land exactly on the node
        return np.array(breaks), np.array(coefs)

    # -- evaluation -----------------------------------------------------------------------------
    def _piece(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.breaks_, t, side='right') - 1, 0, len(self.coefs_) - 1))

    def interpolate(self, x: float) -> float:
        t = float(x)
        if t <= self.breaks_[0]:
            return float(self.values_[0])
        if t >= self.breaks_[-1]:
            return float(self.values_[-1])
        k = self._piece(t)
        u = t - self.breaks_[k]
        c0, c1, c2 = self.coefs_[k]
        return float(c0 + u * (c1 + c2 * u))

    def _antiderivative(self, t: float) -> float:
        """Integral of the curve from the first node to t, flat beyond the nodes."""
        if t <= self.breaks_[0]:
            return float(self.values_[0]) * (t - float(self.breaks_[0]))
        if t >= self.breaks_[-1]:
            return float(self.cumulative_[-1]) + float(self.values_[-1]) * (t - float(self.breaks_[-1]))
        k = self._piece(t)
        u = t - self.breaks_[k]
        c0, c1, c2 = self.coefs_[k]
        return float(self.cumulative_[k] + u * (c0 + u * (c1 / 2.0 + c2 * u / 3.0)))

    def integrate(self, start_x: float, end_x: float) -> float:
        return self._antiderivative(float(end_x)) - self._antiderivative(float(start_x))

    # Optional risk interface
    # The test cells for risk are not prepared but you can do it by your own.
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        raise NotImplementedError("Optional risk: ordinate gradient is not implemented")

    def gradient_of_integrated_value_wrt_ordinate(
        self, start_x: float, end_x: float,
    ) -> np.ndarray:
        raise NotImplementedError("Optional risk: integrated ordinate gradient is not implemented")


class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        elif interpolation_method == InterpMethod.MONOTONE_CONVEX:
            return Interpolator1DMonotoneConvex(axis1_, values_, extrapolation_method)
        else:
            raise ValueError(f'Unsupported interpolation method: {interpolation_method}')
