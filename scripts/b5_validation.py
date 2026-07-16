#!/usr/bin/env python3
"""B5: regenerate every validation number of Sec. 7.
7.1 Kim M*L law: our L = 2N-[odd] vs Kim's bounds L in [k-m-1, k+m-1].
7.2 All six Dilectis-Mortari figure counts (incl. Fig. 10 correction).
7.3 Arnas-Linares Thm 2: first fold (k=m+1) == max tan(kw)/tan(mw).
7.5 Real-mission table: counts, extreme |latitudes|, timing.
"""
import time
import numpy as np
from bench_b3 import basic_roots_companion, count_map_points

R2D = 180/np.pi


def latitudes(k, m, i_rad):
    z, odd = basic_roots_companion(k, m, i_rad)
    lats = []
    for zz in z:
        if odd:
            x = k*zz
        else:
            x = k*zz - np.pi/2
        r = int(np.floor(x/np.pi + 1e-12))
        u2 = x - r*np.pi
        if u2 < 0:
            u2 += np.pi
        lats.append(abs(np.degrees(np.arcsin(np.sin(u2)*np.sin(i_rad)))))
    return sorted(lats)


def main():
    print("== 7.1 Kim M*L: staircase L vs Kim's bounds, sample pairs ==")
    for (k, m) in [(7, 3), (8, 3), (24, 7), (19, 3)]:
        Ls = []
        for ideg in np.arange(1, 180, 1.0):
            tot, N = count_map_points(k, m, np.radians(ideg))
            odd = (k % 2 == 1) and (m % 2 == 1)
            L = 2*N - 1 if odd else 2*N
            Ls.append(L)
        lo, hi = min(Ls), max(Ls)
        print(f"  ({k},{m}): L range observed [{lo},{hi}]  "
              f"Kim bounds [{k-m-1},{k+m-1}]  "
              f"{'OK' if lo >= k-m-1 and hi <= k+m-1 and lo == k-m-1 and hi == k+m-1 else 'MISMATCH'}")

    print("== 7.2 Dilectis-Mortari figure counts ==")
    for (k, m, ideg, claimed) in [(24, 7, 55, 384), (18, 5, 65, 216),
                                  (13, 4, 83, 156), (21, 8, 80, 336),
                                  (14, 5, 110, 252), (19, 3, 81, 285)]:
        tot, N = count_map_points(k, m, np.radians(ideg))
        tag = "MATCH" if tot == claimed else f"OURS={tot} vs CLAIMED={claimed}"
        print(f"  ({k}:{m}) i={ideg}: {tag}")

    print("== 7.3 Arnas-Linares Thm 2 (k = m+1): threshold equality ==")
    from scipy.optimize import brentq, minimize_scalar
    for m in (2, 4, 7, 9):
        k = m + 1
        # our route: G-roots -> max valid c-tilde
        G = lambda w: k*np.sin(2*m*w) - m*np.sin(2*k*w)
        ws = np.linspace(1e-6, np.pi/2 - 1e-6, 200000)
        g = G(ws)
        roots = []
        for j in np.nonzero(np.sign(g[:-1]) * np.sign(g[1:]) < 0)[0]:
            roots.append(brentq(G, ws[j], ws[j+1], xtol=1e-15))
        ct = lambda w: np.tan(k*w)/np.tan(m*w)
        ours = max(ct(w) for w in roots if 0 < ct(w) < 1)
        # their route: max over tau in (1/(k+m), 1.5/(k+m)] of tan(pi k t)/tan(pi m t)
        f = lambda t: -np.tan(np.pi*k*t)/np.tan(np.pi*m*t)
        res = minimize_scalar(f, bounds=(1/(k+m)+1e-9, 1.5/(k+m)), method="bounded",
                              options={"xatol": 1e-14})
        theirs = -res.fun
        print(f"  k={k},m={m}: ours={ours:.12f} theirs={theirs:.12f} "
              f"diff={abs(ours-theirs):.1e}  i* = {np.degrees(np.arccos(ours)):.4f} deg")

    print("== 7.5 real missions ==")
    for (k, m, ideg, name) in [(127, 10, 66.04, "TOPEX/Jason"),
                               (385, 27, 98.65, "Sentinel-3"),
                               (1387, 91, 92.00, "ICESat-2")]:
        t0 = time.perf_counter()
        tot, N = count_map_points(k, m, np.radians(ideg))
        lats = latitudes(k, m, np.radians(ideg))
        dt = time.perf_counter() - t0
        print(f"  {name} ({k}:{m}, i={ideg}): basic={N} total={tot} "
              f"|phi| in [{lats[0]:.3f}, {lats[-1]:.3f}] deg  t={dt:.2f} s")


if __name__ == "__main__":
    main()
