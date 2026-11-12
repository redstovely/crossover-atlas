"""Catalogue of circular repeat sun-synchronous orbits (mean secular J2 model).

For each coprime (k,m) with m <= M_MAX solve simultaneously
    (omega_dot + M_dot) / (omega_E - Omega_dot) = k/m      (repeat condition)
    Omega_dot = Omega_dot_sun                              (sun-synchronism)
for (a, i); keep altitudes in [H_MIN, H_MAX].  Writes ../data/sso_catalogue.csv
"""
from math import gcd, pi, sqrt, acos, degrees, ceil, floor
import csv
from scipy.optimize import brentq

MU = 398600.4418        # km^3/s^2
RE = 6378.137           # km
J2 = 1.08262668e-3
W_E = 2 * pi / 86164.0905           # rad/s, sidereal rotation
W_SUN = 2 * pi / (365.2421897 * 86400.0)
H_MIN, H_MAX, M_MAX = 300.0, 2000.0, 15


def cos_i_ss(a):
    n = sqrt(MU / a**3)
    return -W_SUN / (1.5 * J2 * n * (RE / a)**2)


def xi(a, c):
    n = sqrt(MU / a**3)
    return n * (1 + 0.75 * J2 * (RE / a)**2 * (8 * c * c - 2))


def revs_per_nodal_day(a):
    c = cos_i_ss(a)
    return xi(a, c) / (W_E - W_SUN)


def solve(k, m):
    f = lambda a: revs_per_nodal_day(a) - k / m
    lo, hi = RE + H_MIN, RE + H_MAX
    if f(lo) * f(hi) > 0:
        return None
    a = brentq(f, lo, hi, xtol=1e-9)
    return a, degrees(acos(cos_i_ss(a)))


def catalogue():
    qmax, qmin = revs_per_nodal_day(RE + H_MIN), revs_per_nodal_day(RE + H_MAX)
    rows = []
    for m in range(1, M_MAX + 1):
        for k in range(floor(qmin * m), ceil(qmax * m) + 1):
            if gcd(k, m) != 1:
                continue
            r = solve(k, m)
            if r is None:
                continue
            a, inc = r
            L = k + m - 1
            rows.append((k, m, a, a - RE, inc, L, k * L))
    return rows, qmin, qmax


if __name__ == '__main__':
    rows, qmin, qmax = catalogue()
    print(f'revs/day range for {H_MAX:.0f}..{H_MIN:.0f} km: {qmin:.4f} .. {qmax:.4f}')
    print('number of repeat SSOs:', len(rows))
    with open('../data/sso_catalogue.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['k', 'm', 'a_km', 'altitude_km', 'inclination_deg', 'L', 'N_tot'])
        for r in rows:
            w.writerow([r[0], r[1], f'{r[2]:.3f}', f'{r[3]:.3f}', f'{r[4]:.4f}', r[5], r[6]])
    for km in [(143, 10), (233, 16), (29, 2), (385, 27)]:
        for r in rows:
            if (r[0], r[1]) == km:
                print(r)
