#!/usr/bin/env python3
"""B6: SSO + repeat-track catalogue numbers (verified polynomials E7.2/E7.9).

deg-7  (J2-compatible a, given i):  c1 x^7 + c2 x^4 + c3 = 0, x = sqrt(a)
deg-14 (compatible + sun-synchronous): c1 x^14 + c2 x^7 + c3 x^4 + c4 = 0,
then cos i = -K_ss * (R/a)^{7/2} ... i from (7.21).

Constants: J2 = 1.08263e-3, R = 6378.137 km, mu = 398600.4418 km^3/s^2,
T_sid = 86164.0905 s, K_ss such that cos i (R/a)^{7/2} = -0.0989 (E7.8,
re-derived: -2*AdotRM*R^{3/2}/(3 J2 sqrt(mu)) with AdotRM = 360/365.25
deg/day). We recompute the constant from first principles rather than
hardcoding 0.0989.
"""
import numpy as np

J2 = 1.08263e-3
R = 6378.137          # km
MU = 398600.4418      # km^3/s^2
T_SID = 86164.0905    # s
ADOT = np.radians(360.0 / 365.25) / 86400.0   # rad/s, sun mean motion

# E7.8 constant: Omega_dot = -(3/2) J2 n (R/p)^2 cos i = ADOT  (e=0, p=a)
# n = sqrt(MU) a^{-3/2}  =>  cos i * a^{-7/2} = -2 ADOT /(3 J2 sqrt(MU) R^2)
KSS = 2.0 * ADOT / (3.0 * J2 * np.sqrt(MU)) * R ** 1.5  # cos i (R/a)^3.5 = -KSS
print(f"SSO constant (E7.8): cos i (R/a)^(7/2) = -{KSS:.5f}  (thesis: 0.0989)")


def sso_repeat(k, m):
    """Solve the degree-14 polynomial for x = sqrt(a) (km^0.5), return a, i."""
    c1 = -6.0 * J2 * m * KSS**2 / (R**5 * k)
    c2 = 2.0 * np.pi / (np.sqrt(MU) * T_SID) - 1.5 * J2 * KSS / R**1.5
    c3 = -m / k
    c4 = 1.5 * R**2 * J2 * m / k
    coeffs = np.zeros(15)
    coeffs[0] = c1; coeffs[7] = c2; coeffs[10] = c3; coeffs[14] = c4
    roots = np.roots(coeffs)
    a_kepler = (MU * (m * T_SID / (k * 2 * np.pi)) ** 2) ** (1.0 / 3.0)
    best = None
    for r in roots:
        if abs(r.imag) < 1e-9 and r.real > 0:
            a = r.real ** 2
            if best is None or abs(a - a_kepler) < abs(best - a_kepler):
                best = a
    a = best
    cosi = -KSS * (a / R) ** 3.5   # E7.8: cos i (R/a)^{7/2} = -KSS
    return a, np.degrees(np.arccos(cosi))


def basic_roots(k, m, i_rad):
    c = np.cos(i_rad)
    odd = (k % 2 == 1) and (m % 2 == 1)
    s = -1.0 if odd else 1.0
    F = lambda z: (1-c)*np.sin((k+m)*z) + s*(1+c)*np.sin((k-m)*z)
    n = 4000 * k
    eps = 1e-9
    zs = np.linspace(eps, np.pi/2 - eps, n)
    f = F(zs)
    sign = np.sign(f)
    idx = np.nonzero(sign[:-1]*sign[1:] < 0)[0]
    roots = []
    for j in idx:
        a_, b_ = zs[j], zs[j+1]
        for _ in range(60):
            mid = 0.5*(a_+b_)
            if np.sign(F(a_)) != np.sign(F(mid)): b_ = mid
            else: a_ = mid
        roots.append(0.5*(a_+b_))
    if odd:
        roots = [0.0] + roots
    return roots, odd


def crossover_summary(k, m, i_deg):
    i = np.radians(i_deg)
    roots, odd = basic_roots(k, m, i)
    N = len(roots)
    total = k*(2*N-1) if odd else 2*k*N
    lats = []
    for z in roots:
        x = k*z
        r = int(np.floor(x/np.pi + 1e-12))
        u2 = x - r*np.pi
        if not odd:
            u2 = k*z - np.pi/2 - r*np.pi  # placeholder, recompute below
    # latitudes directly: u2 from z (odd) or w (mixed)
    lats = []
    for z in roots:
        if odd:
            u2 = k*z - np.pi*int(np.floor(k*z/np.pi + 1e-12))
        else:
            x = k*z - np.pi/2
            r = int(np.floor(x/np.pi + 1e-12))
            u2 = x - r*np.pi
            if u2 < 0: u2 += np.pi
        lats.append(np.degrees(np.arcsin(np.sin(u2)*np.sin(i))))
    lats = sorted(abs(l) for l in lats)
    return N, total, (lats[0] if lats else None), (lats[-1] if lats else None)


if __name__ == "__main__":
    print("\n== zero-distortion closed forms (E7.7) ==")
    print(f"  arccos((1+sqrt(17))/8) = {np.degrees(np.arccos((1+np.sqrt(17))/8)):.3f} deg")
    print(f"  arccos((1-sqrt(17))/8) = {np.degrees(np.arccos((1-np.sqrt(17))/8)):.3f} deg")

    print("\n== worked SSO examples (deg-14 polynomial + E7.8) ==")
    for (k, m, name) in [(143, 10, "Sentinel-2-like"), (233, 16, "Landsat-8-like"),
                         (211, 15, ""), (29, 2, "")]:
        a, ideg = sso_repeat(k, m)
        h = a - R
        N, total, lmin, lmax = crossover_summary(k, m, ideg)
        print(f"  k={k:4d} m={m:2d} {name:16s}: a={a:9.3f} km h={h:7.1f} km "
              f"i={ideg:8.4f} deg | N_basic={N:3d} total={total:5d} "
              f"|phi| in [{lmin:.2f}, {lmax:.2f}] deg")

    print("\n== catalogue size check (h in [300,2000] km, m <= 15) ==")
    count = 0
    from math import gcd
    for m in range(1, 16):
        for k in range(1, 260):
            if gcd(k, m) != 1:
                continue
            a, ideg = sso_repeat(k, m)
            h = a - R
            if 300.0 <= h <= 2000.0 and 90.0 < ideg < 113.0:
                count += 1
    print(f"  admissible repeat SSOs: {count}")
