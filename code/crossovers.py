"""Crossovers of circular repeat ground tracks: exact counts and locations.

Notation (as in the paper).  A k:m repeat track (k nodal revolutions in m
nodal days, gcd(k,m)=1) at inclination i, c = cos i.  A crossover is an
unordered pair of track parameters theta_1 != theta_2 (mod 2 pi k) with the
same sub-satellite point.  They are the pairs

    theta_{1,2} = pi/2 + pi s -/+ k w,   s in Z_{2k},  w in (0,pi),  E(w,c)=0,

modulo (s,w) ~ (s+k, pi-w), where

    E(w,c) = sin(kw) cos(mw) - c cos(kw) sin(mw).

L(i) = number of zeros of E(.,c) in (0,pi);  N_tot = k L.

Everything here uses only closed-form breakpoints and bracketed scalar
root-finding on intervals where the function is strictly monotone.
"""
from math import gcd, pi, cos, sin, acos, floor
import numpy as np
from scipy.optimize import brentq

TOL_C = 1e-12  # |c - c*| below this is treated as "exactly critical"


def E(w, k, m, c):
    return np.sin(k * w) * np.cos(m * w) - c * np.cos(k * w) * np.sin(m * w)


def gamma(w, k, m):
    return np.tan(k * w) / np.tan(m * w)


def G(w, k, m):
    return k * np.sin(2 * m * w) - m * np.sin(2 * k * w)


# --------------------------------------------------------------------------
# critical points of gamma in (0, pi/2): zeros of G
# --------------------------------------------------------------------------
def G_zeros(k, m):
    """All zeros of G in the open interval (0, pi/2), certified.

    G' = 4 k m sin((k+m)w) sin((k-m)w), so G is strictly monotone between
    consecutive points of {p pi/(k+m)} U {p pi/|k-m|}.  G(0)=G(pi/2)=0, so
    the first and last pieces contain no interior zero.
    """
    bp = {0.0, pi / 2}
    for p in range(1, (k + m) // 2 + 1):
        x = p * pi / (k + m)
        if 0 < x < pi / 2:
            bp.add(x)
    d = abs(k - m)
    for p in range(1, d // 2 + 1):
        x = p * pi / d
        if 0 < x < pi / 2:
            bp.add(x)
    bp = sorted(bp)
    # merge numerically coincident breakpoints
    pts = [bp[0]]
    for x in bp[1:]:
        if x - pts[-1] > 1e-14:
            pts.append(x)
    zeros = []
    for a, b in zip(pts[1:-2], pts[2:-1]):
        ga, gb = G(a, k, m), G(b, k, m)
        if ga * gb < 0:
            zeros.append(brentq(G, a, b, args=(k, m), xtol=1e-15, rtol=1e-15))
    return zeros


def critical_values(k, m):
    """Critical values of cos i in [0,1) and their type.

    Returns list of (c_star, w_star, kind), kind in
      'fold'      interior local maximum of gamma in (0,pi/2), value in (0,1)
      'equator'   pitchfork at w=pi/2 (k>m both odd), c = m/k
      'cusp'      pitchfork at w=0 (k<m), c = k/m
      'polar'     k even: local maximum of gamma at w=pi/2 with value 0
    """
    out = []
    for w in G_zeros(k, m):
        g = float(gamma(w, k, m))
        if 0 < g < 1:
            out.append((g, w, 'fold'))
    if k > m and k % 2 == 1 and m % 2 == 1:
        out.append((m / k, pi / 2, 'equator'))
    if k < m:
        out.append((k / m, 0.0, 'cusp'))
    if k % 2 == 0:
        out.append((0.0, pi / 2, 'polar'))
    out.sort(key=lambda t: -t[0])
    return out


def predicted_fold_count(k, m):
    """V of the counting theorem."""
    return m // 2 if k > m else (k - 1) // 2


def L_theory(k, m, c, crit=None):
    """Number of zeros of E(.,c) in (0,pi), from the counting theorem."""
    if k == m:                      # 1:1, the only coprime pair with k = m
        return 1
    if crit is None:
        crit = critical_values(k, m)
    if c < -TOL_C:
        return k + m - 1
    if abs(c) <= TOL_C:
        return k + m - 1 - (1 if k % 2 == 0 else 0)
    L = abs(k - m) - 1
    for cs, _, kind in crit:
        if kind == 'polar':
            continue
        if kind == 'fold':
            if c < cs - TOL_C:
                L += 4
            elif abs(c - cs) <= TOL_C:
                L += 2
        else:  # pitchforks: no change at the critical value itself
            if c < cs - TOL_C:
                L += 2
    return L


# --------------------------------------------------------------------------
# all zeros of E(.,c) in (0, pi), certified by monotone bracketing
# --------------------------------------------------------------------------
def roots(k, m, c, crit=None):
    """Sorted zeros of E(.,c) in (0,pi) (double zeros listed once).

    The poles and the stationary points of gamma divide (0, pi/2) into laps
    on which gamma is strictly monotone.  E(.,c) has exactly one zero in a
    lap if c lies strictly between the two end limits of gamma, and none
    otherwise.  The end limits are known: +-infinity at a pole, k/m at w = 0,
    the stationary value at a stationary point, and at w = pi/2 the value m/k
    (k, m odd), 0 (k even) or a pole (m even).  The zero is then refined by a
    bracketed solver applied to E divided by its known simple factors at the
    ends (sin w at 0, cos w at pi/2 when k and m are odd), so that the
    function has opposite signs at the two ends of the bracket.

    A value of c within TOL_C of a finite end limit is treated as equal to
    it (exactly critical inclination): the lap then has no zero, and if the
    limit is an interior stationary value the stationary point itself is
    returned once, as a double zero.
    """
    if k == m:                      # only 1:1 is coprime: E = (1-c) sin w cos w
        return [pi / 2]
    both_odd = (k % 2 == 1 and m % 2 == 1)
    if abs(c) <= TOL_C:
        # E = sin(kw) cos(mw): closed form
        zs = {p * pi / k for p in range(1, k)}
        zs |= {(2 * b + 1) * pi / (2 * m) for b in range(m)}
        zs = sorted(zs)
        out = [zs[0]]
        for z in zs[1:]:
            if z - out[-1] > 1e-13:
                out.append(z)
        return out
    INF = float('inf')
    half_pi = pi / 2
    # breakpoints in [0, pi/2]: (w, kind, finite limit of gamma or None)
    pts = [(0.0, 'end', k / m)]
    poles = {(2 * q + 1) * pi / (2 * k) for q in range(k)} | {p * pi / m for p in range(1, m)}
    pts += [(w, 'pole', None) for w in poles if w < half_pi - 1e-14]
    pts += [(w, 'stat', float(gamma(w, k, m))) for w in G_zeros(k, m)]
    if both_odd:
        pts.append((half_pi, 'mid', m / k))      # removable singularity, value m/k
    elif k % 2 == 0:
        pts.append((half_pi, 'mid', 0.0))        # maximum with value zero
    else:
        pts.append((half_pi, 'pole', None))      # m even
    pts.sort(key=lambda t: t[0])
    skm = sin(k * half_pi) * sin(m * half_pi)

    def make_f(at_zero, at_half):
        # E / [sin w]^at_zero / [cos w]^at_half, with its limits at the ends
        def f(w):
            if at_zero and w == 0.0:
                return k - c * m
            if at_half:
                z = half_pi - w
                if z == 0.0:
                    return skm * (m - c * k)
                e = skm * (cos(k * z) * sin(m * z) - c * sin(k * z) * cos(m * z)) / sin(z)
            else:
                e = sin(k * w) * cos(m * w) - c * cos(k * w) * sin(m * w)
            return e / sin(w) if at_zero else e
        return f

    half = []
    for (a, ka, va), (b, kb, vb) in zip(pts[:-1], pts[1:]):
        if b - a < 1e-14:
            continue
        rising = G(0.5 * (a + b), k, m) > 0
        lo = va if ka != 'pole' else (-INF if rising else INF)
        hi = vb if kb != 'pole' else (INF if rising else -INF)
        if not (min(lo, hi) + TOL_C < c < max(lo, hi) - TOL_C):
            continue
        f = make_f(a == 0.0, b == half_pi and both_odd)
        fa, fb = f(a), f(b)
        if fa * fb >= 0:
            raise ArithmeticError(f'bracket lost for {k}:{m}, c={c!r}, lap ({a}, {b})')
        half.append(brentq(f, a, b, xtol=1e-15, rtol=1e-15))
    # exact tangency: the fold point is a double zero, listed once
    half += [w for (w, kind, v) in pts if kind == 'stat' and abs(c - v) <= TOL_C]
    half.sort()
    mid = [half_pi] if both_odd else []
    return half + mid + [pi - w for w in reversed(half)]


def crossovers(k, m, inc_deg, lam0=0.0):
    """All crossovers: arrays (lat, lon, w, s) in degrees; one row per event.

    Events are (s, w), s in Z_{2k}, w a zero of E in (0,pi), modulo
    (s,w) ~ (s+k, pi-w); representatives: w in (0,pi/2) with all s, and
    w = pi/2 with s in 0..k-1.
    """
    inc = np.radians(inc_deg)
    c, si = cos(inc), sin(inc)
    ws = np.array(roots(k, m, c))
    wl = ws[ws < pi / 2 - 1e-12]
    wm = ws[np.abs(ws - pi / 2) <= 1e-12]
    W = np.concatenate([np.repeat(wl, 2 * k), np.repeat(wm, k)])
    S = np.concatenate([np.tile(np.arange(2 * k), len(wl)),
                        np.tile(np.arange(k), len(wm))])
    th1 = pi / 2 + pi * S - k * W
    lat = np.degrees(np.arcsin(np.clip(si * np.sin(th1), -1, 1)))
    lon = lam0 + np.arctan2(c * np.sin(th1), np.cos(th1)) - (m / k) * th1
    lon = np.degrees((lon + pi) % (2 * pi) - pi)
    return lat, lon, W, S


def track(k, m, inc_deg, n_per_rev=720, lam0=0.0):
    inc = np.radians(inc_deg)
    th = np.linspace(0, 2 * pi * k, n_per_rev * k + 1)
    lat = np.degrees(np.arcsin(sin(inc) * np.sin(th)))
    lon = lam0 + np.arctan2(cos(inc) * np.sin(th), np.cos(th)) - (m / k) * th
    lon = np.degrees((lon + pi) % (2 * pi) - pi)
    return th, lat, lon


def critical_inclinations_deg(k, m):
    return [(np.degrees(acos(cs)), kind) for cs, w, kind in critical_values(k, m)]


# --------------------------------------------------------------------------
# eigenvalue (colleague-matrix) method: all roots in one call
# --------------------------------------------------------------------------
def _colleague_roots(a):
    """Real roots in (-1,1) of sum_j a[j] U_j(x), U_j Chebyshev polynomials of
    the second kind, as eigenvalues of the colleague matrix."""
    n = len(a) - 1
    A = np.zeros((n, n))
    j = np.arange(n - 1)
    A[j, j + 1] = 0.5
    A[j + 1, j] = 0.5
    A[n - 1, :] -= np.asarray(a[:n]) / (2.0 * a[n])
    ev = np.linalg.eigvals(A)
    return ev


def roots_eig(k, m, c, merge_tol=1e-6):
    """Zeros of E(.,c) in (0,pi) from one eigenvalue problem of size k+m-1.

    E(w,c)/sin w = (1-c)/2 U_{k+m-1}(x) + sgn(k-m)(1+c)/2 U_{|k-m|-1}(x),
    x = cos w.  At an exactly critical inclination the polynomial has a
    multiple root, which an eigenvalue solver resolves only to ~eps**(1/2)
    (double root, fold) or ~eps**(1/3) (triple root at x = 0, equatorial
    pitchfork); roots closer than merge_tol are merged.  Very near a critical
    inclination use roots(), whose count is certified.
    """
    if k == m:
        return np.array([pi / 2])
    n = k + m - 1
    a = np.zeros(n + 1)
    a[n] = (1 - c) / 2
    d = abs(k - m)
    a[d - 1] += np.sign(k - m) * (1 + c) / 2
    ev = _colleague_roots(a)
    x = ev[(np.abs(ev.imag) < merge_tol) & (np.abs(ev.real) < 1)].real
    w = np.sort(np.arccos(x))
    # at a cusp pitchfork (cos i = k/m) the polynomial has roots at x = +-1,
    # i.e. w = 0 and pi, which are not crossovers
    w = w[(w > merge_tol) & (w < pi - merge_tol)]
    if len(w) == 0:
        return w
    keep = [w[0]]
    for v in w[1:]:
        if v - keep[-1] > merge_tol:
            keep.append(v)
    return np.array(keep)


def critical_values_eig(k, m):
    """Fold values from the zeros of G via one eigenvalue problem.

    G(w)/sin w = k U_{2m-1}(x) - m U_{2k-1}(x)."""
    n = 2 * max(k, m) - 1
    a = np.zeros(n + 1)
    a[2 * m - 1] += k
    a[2 * k - 1] -= m
    ev = _colleague_roots(a)
    x = ev[(np.abs(ev.imag) < 1e-9) & (np.abs(ev.real) < 1)].real
    w = np.arccos(x)
    # The polynomial has simple roots at x = +-1 (w = 0, pi) and a root at
    # x = 0 (w = pi/2) that is triple when k and m are both odd; they are not
    # stationary points of gamma in (0, pi/2) and are discarded.  G is
    # monotone between consecutive zeros of G', so every genuine zero lies in
    # [pi/(k+m), pi/2 - pi/(2(k+m))].
    w = np.sort(w[(w > 0.999 * pi / (k + m)) & (w < pi / 2 - 0.499 * pi / (k + m))])
    g = gamma(w, k, m)
    return np.sort(g[(g > 0) & (g < 1)])[::-1]
