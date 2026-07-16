#!/usr/bin/env python3
"""Exhaustive scan (task A2): critical-inclination counts for all coprime
(k, m), k <= KMAX, both parity families.

Checks, for each pair:
  1. G(z) = k sin(2mz) - m sin(2kz) has exactly m-1 zeros in (0, pi/2)
     (Prop. Gcount, both parities).
  2. Valid criticals (critical value of c in (0,1)):
       odd-odd  (m, k odd):        c(z) = tan(mz)/tan(kz), predict (m-1)/2
       mixed,   m odd  (k even):   c(w) = tan(kw)/tan(mw), predict (m-1)/2
       mixed,   m even (k odd):    c(w) = tan(kw)/tan(mw), predict m/2
  3. Staircase cross-check: for a few sample inclinations, the number of
     master-equation roots matches the census prediction
       odd-odd:  N = (k-m)/2 + [i>i1*] + 2*#{tangency criticals < i}
       mixed:    N = (k-m-1)/2 + 2*#{criticals < i}
  4. Distinctness: minimum gap between critical inclinations (reported).
  5. Alternation of valid/invalid along ordered G-roots (empirical).

Root finding: sign-change bracketing on a fine grid + bisection (brentq).
Grid density scales with k so no roots are missed (G has at most ~2k
oscillations on the interval).
"""
import math
from fractions import Fraction
from math import gcd, pi, sin, cos, tan, acos

from scipy.optimize import brentq

KMAX = 40
GRID_PER_UNIT = 4000  # grid points per unit of k (>= 100 oscillation samples)


def g_roots(k, m):
    """Zeros of G(z) = k sin(2mz) - m sin(2kz) in the OPEN interval (0, pi/2).
    Excludes the endpoint z=0 (triple zero) and z=pi/2 (zero iff k, m odd or
    boundary cases); uses a small guard band eps and checks none are missed by
    counting sign changes of G away from its zero endpoints."""
    G = lambda z: k * math.sin(2 * m * z) - m * math.sin(2 * k * z)
    n = GRID_PER_UNIT * k
    # guard band: near 0, G ~ (4/3) k m (k^2-m^2) z^3 > 0, root-free by lemma
    eps = 1e-4 / k
    zs = [eps + (pi / 2 - 2 * eps) * j / n for j in range(n + 1)]
    roots = []
    for a, b in zip(zs[:-1], zs[1:]):
        ga, gb = G(a), G(b)
        if ga == 0.0:
            roots.append(a)
        elif ga * gb < 0:
            roots.append(brentq(G, a, b, xtol=1e-14))
    # deduplicate (tangential double counting cannot occur: interior zeros of
    # G at x=m integer are simple except cubic crossings which are still
    # sign changes)
    out = []
    for r in roots:
        if not out or abs(r - out[-1]) > 1e-9:
            out.append(r)
    return out


def c_ratio(z, num_mult, den_mult):
    """tan(num_mult*z)/tan(den_mult*z), computed stably via sines/cosines."""
    sn, cn = math.sin(num_mult * z), math.cos(num_mult * z)
    sd, cd = math.sin(den_mult * z), math.cos(den_mult * z)
    denom = cn * sd
    if abs(denom) < 1e-13:
        return math.inf
    return (sn * cd) / denom


def master_root_count(k, m, i_rad, odd_case):
    """Count solutions of the master equation in [0, pi/2) (odd case,
    tan(mz)=tan(kz)cos i, z=0 counts) or (0, pi/2] (mixed parity,
    tan(kw)=tan(mw)cos i) at inclination i, via the polynomial form F."""
    c = math.cos(i_rad)
    if odd_case:
        F = lambda z: (1 - c) * math.sin((k + m) * z) - (1 + c) * math.sin((k - m) * z)
        lo, hi, include_zero = 0.0, pi / 2, True
        spurious = [pi / 2]  # z=pi/2 spurious always (odd case)
    else:
        F = lambda w: (1 - c) * math.sin((k + m) * w) + (1 + c) * math.sin((k - m) * w)
        lo, hi, include_zero = 0.0, pi / 2, False
        # mixed parity: w=0 spurious (F root, not master root); w=pi/2 genuine
        # except it solves F only when m even (never) or k even at i=pi/2.
        spurious = [0.0]
    n = GRID_PER_UNIT * k
    eps = 1e-7
    xs = [lo + eps + (hi - lo - 2 * eps) * j / n for j in range(n + 1)]
    count = 0
    roots = []
    for a, b in zip(xs[:-1], xs[1:]):
        fa, fb = F(a), F(b)
        if fa * fb < 0:
            roots.append(brentq(F, a, b, xtol=1e-14))
    if include_zero:
        roots.append(0.0)
    # count pi/2 endpoint for mixed parity if it is a genuine root (only at
    # i=pi/2 exactly for k even; we never sample that in the staircase check)
    return len(roots)


def analyze_pair(k, m):
    odd_case = (m % 2 == 1) and (k % 2 == 1)
    roots = g_roots(k, m)
    n_roots = len(roots)

    if odd_case:
        cvals = [c_ratio(z, m, k) for z in roots]  # c = tan(mz)/tan(kz)
        predict_valid = (m - 1) // 2
    else:
        cvals = [c_ratio(w, k, m) for w in roots]  # c = tan(kw)/tan(mw)
        predict_valid = (m - 1) // 2 if m % 2 == 1 else m // 2

    valid = [(z, cv) for z, cv in zip(roots, cvals) if 0.0 < cv < 1.0]
    n_valid = len(valid)

    # criticals in degrees, sorted
    crit_deg = sorted(math.degrees(acos(cv)) for _, cv in valid)
    if odd_case:
        crit_deg = sorted(crit_deg + [math.degrees(acos(m / k))])  # pitchfork

    # distinctness
    gaps = [b - a for a, b in zip(crit_deg[:-1], crit_deg[1:])]
    min_gap = min(gaps) if gaps else None

    # alternation pattern along ordered roots
    pattern = "".join("V" if 0 < cv < 1 else "x" for cv in cvals)

    # staircase spot-check at 3 inclinations (avoid criticals by 0.05 deg)
    stair_ok = True
    all_crit = crit_deg
    for i_deg in (20.0, 55.0, 87.0):
        if any(abs(i_deg - cd) < 0.05 for cd in all_crit):
            i_deg += 0.11
        n_below = sum(1 for cd in all_crit if cd < i_deg)
        if odd_case:
            i1 = math.degrees(acos(m / k))
            n_tan_below = sum(1 for cd in all_crit if cd < i_deg and abs(cd - i1) > 1e-9)
            pred = (k - m) // 2 + (1 if i_deg > i1 else 0) + 2 * n_tan_below
        else:
            pred = (k - m - 1) // 2 + 2 * n_below
        got = master_root_count(k, m, math.radians(i_deg), odd_case)
        if got != pred:
            stair_ok = False
    return dict(k=k, m=m, odd=odd_case, n_groots=n_roots, n_valid=n_valid,
                predict_valid=predict_valid,
                groots_ok=(n_roots == m - 1),
                valid_ok=(n_valid == predict_valid),
                stair_ok=stair_ok, min_gap=min_gap, pattern=pattern,
                crit_deg=crit_deg)


def main():
    fails = []
    n_pairs = 0
    global_min_gap = (1e9, None)
    for k in range(2, KMAX + 1):
        for m in range(1, k):
            if gcd(k, m) != 1:
                continue
            n_pairs += 1
            r = analyze_pair(k, m)
            if not (r["groots_ok"] and r["valid_ok"] and r["stair_ok"]):
                fails.append(r)
            if r["min_gap"] is not None and r["min_gap"] < global_min_gap[0]:
                global_min_gap = (r["min_gap"], (k, m))
            # alternation check: pattern should alternate strictly
            pat = r["pattern"]
            alt = all(a != b for a, b in zip(pat[:-1], pat[1:]))
            if pat and not alt:
                print(f"  NOTE non-alternating pattern (k,m)=({k},{m}): {pat}")
    print(f"scanned {n_pairs} coprime pairs, k <= {KMAX}")
    if fails:
        print(f"FAILURES: {len(fails)}")
        for r in fails[:20]:
            print("  ", {kk: r[kk] for kk in
                         ("k", "m", "odd", "n_groots", "n_valid",
                          "predict_valid", "groots_ok", "valid_ok", "stair_ok")})
    else:
        print("ALL CHECKS PASS: G-root count m-1; valid criticals (m-1)/2 "
              "(odd-odd and mixed m odd) or m/2 (mixed m even); staircase "
              "census confirmed at sample inclinations.")
    print(f"minimum critical-inclination gap: {global_min_gap[0]:.6f} deg "
          f"at (k,m)={global_min_gap[1]} -> all criticals pairwise distinct")
    # headline numbers for the paper (k=7,m=3 second critical etc.)
    r = analyze_pair(7, 3)
    print("(k,m)=(7,3) criticals [deg]:", [f"{x:.4f}" for x in r["crit_deg"]])


if __name__ == "__main__":
    main()
