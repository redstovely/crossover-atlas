"""Fast checks (a few seconds).  The full suite is code/verify.py."""
import os
import sys
from math import acos, cos, degrees, pi, radians

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))
import crossovers as cx  # noqa: E402


def test_published_counts():
    for k, m, inc, n in [(7, 3, 70, 35), (24, 7, 55, 384), (13, 4, 83, 156),
                         (14, 5, 110, 252), (19, 3, 81, 323),
                         (127, 10, 66.0, 14732), (385, 27, 98.65, 158235)]:
        assert k * cx.L_theory(k, m, cos(radians(inc))) == n
        assert len(cx.crossovers(k, m, inc)[0]) == n


def test_number_of_fold_inclinations():
    for k in range(2, 30):
        for m in range(1, 30):
            if k == m or np.gcd(k, m) != 1:
                continue
            folds = [c for c, _, kind in cx.critical_values(k, m) if kind == 'fold']
            assert len(folds) == (m // 2 if k > m else (k - 1) // 2)
            assert all(c < min(m / k, k / m) for c in folds)


def test_eigenvalue_method_agrees():
    for k, m, inc in [(7, 3, 70), (8, 3, 95), (3, 8, 75), (24, 7, 55), (127, 10, 66.0)]:
        c = cos(radians(inc))
        a, b = np.array(cx.roots(k, m, c)), cx.roots_eig(k, m, c)
        assert len(a) == len(b) and np.abs(a - b).max() < 1e-10


def test_pitchfork_19_3():
    ip = degrees(acos(3 / 19))
    assert 19 * cx.L_theory(19, 3, 3 / 19) == 285
    assert len(cx.roots(19, 3, 3 / 19)) == 15
    assert 19 * cx.L_theory(19, 3, cos(radians(ip + 0.01))) == 323


def test_exactly_critical_inclinations():
    # equatorial pitchfork, cusp and fold: no spurious zeros at the critical value
    assert len(cx.roots(7, 3, 3 / 7)) == cx.L_theory(7, 3, 3 / 7) == 3
    assert len(cx.roots(2, 3, 2 / 3)) == cx.L_theory(2, 3, 2 / 3) == 0
    assert len(cx.roots_eig(2, 3, 2 / 3)) == 0
    assert len(cx.roots(3, 2, 1 / 9)) == cx.L_theory(3, 2, 1 / 9) == 2


def test_one_to_one_and_polar():
    for c in (0.8, 0.0, -0.5):
        assert cx.roots(1, 1, c) == [pi / 2] and cx.L_theory(1, 1, c) == 1
    lat = cx.crossovers(8, 3, 90.0)[0]
    assert len(lat) == 72 and int(np.sum(np.abs(lat) > 90 - 1e-6)) == 56


def test_crossover_meridians():
    # crossovers lie on 2k meridians pi/k apart, at every inclination (King, 1975)
    for k, m, inc in [(7, 3, 70.0), (8, 3, 95.0), (3, 8, 75.0), (24, 7, 55.0)]:
        lon = cx.crossovers(k, m, inc)[1]
        r = lon * 2 * k / 180.0
        assert np.abs(r - np.round(r)).max() < 1e-9
        assert np.all(np.mod(np.round(r), 2) == (k - m) % 2)
