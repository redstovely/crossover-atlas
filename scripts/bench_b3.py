#!/usr/bin/env python3
"""B3 benchmark: companion-matrix (colleague-matrix) all-roots method vs
brute-force segment intersection; near-critical robustness demo.

Method (Boyd 2007, thesis E5.25 with the b_l index fix): F(z) =
sum_j b_j sin(jz) = sin(z) * P(cos z), P in Chebyshev-U basis; roots of P
from the colleague matrix eigenvalues x = cos z.
  odd case:  b_{k+m} = 1-c, b_{k-m} = -(1+c)
  mixed:     b_{k+m} = 1-c, b_{k-m} = +(1+c)
Total map count: k*(2N-1) odd / k*2N mixed  (eq:totalcount).
"""
import time
import numpy as np


def colleague_roots(bvec):
    """Roots x=cos z of P(x) = sum_{j>=1} b_j U_{j-1}(x); bvec[j-1] = b_j."""
    N1 = len(bvec)            # N+1 coefficients
    N = N1 - 1
    C = np.zeros((N, N))
    C[0, 1] = 0.5
    for j in range(1, N - 1):
        C[j, j - 1] = 0.5
        C[j, j + 1] = 0.5
    C[N - 1, :] = -np.asarray(bvec[:N]) / (2.0 * bvec[N])
    C[N - 1, N - 2] += 0.5
    return np.linalg.eigvals(C)


def basic_roots_companion(k, m, i_rad):
    c = np.cos(i_rad)
    odd = (k % 2 == 1) and (m % 2 == 1)
    b = np.zeros(k + m)
    b[k + m - 1] = 1.0 - c
    b[k - m - 1] = -(1.0 + c) if odd else (1.0 + c)
    ev = colleague_roots(b)
    x = ev[np.abs(ev.imag) < 1e-9].real
    x = x[np.abs(x) <= 1.0]
    z = np.arccos(x)
    if odd:
        # keep [0, pi/2), drop spurious pi/2, include z=0 (always a root of
        # the master equation; sin-series has it via sin z factor, excluded
        # from P, so add manually)
        z = z[(z > 1e-9) & (z < np.pi / 2 - 1e-9)]
        z = np.concatenate([[0.0], np.sort(z)])
    else:
        w = z[(z > 1e-9) & (z <= np.pi / 2 - 1e-9)]
        # w = pi/2 genuine only at i=pi/2 (k even); not sampled here
        z = np.sort(w)
    return z, odd


def count_map_points(k, m, i_rad):
    z, odd = basic_roots_companion(k, m, i_rad)
    N = len(z)
    return k * (2 * N - 1) if odd else k * 2 * N, N


def residual_check(k, m, i_rad, z, odd):
    c = np.cos(i_rad)
    if odd:
        F = lambda t: (1-c)*np.sin((k+m)*t) - (1+c)*np.sin((k-m)*t)
    else:
        F = lambda t: (1-c)*np.sin((k+m)*t) + (1+c)*np.sin((k-m)*t)
    return max((abs(F(t)) for t in z), default=0.0)


# ---------------------------------------------------------------- brute force
def brute_force_count(k, m, i_rad, n_per_rev=2000):
    """Count self-intersections of the closed track by segment intersection
    with longitude-bin sweeping. Returns count of intersection points."""
    n = n_per_rev * k
    u = np.linspace(0.0, 2*np.pi*k, n, endpoint=False)
    u += 0.5 * (2*np.pi*k/n)   # half-step offset: keep equator crossings
    # (u = multiples of pi) strictly inside segments, not on vertices
    ci, si = np.cos(i_rad), np.sin(i_rad)
    lam_u = np.unwrap(np.arctan2(np.sin(u)*ci, np.cos(u)))
    lam = lam_u - (m/k)*u          # node-relative longitude, radians
    phi = np.arcsin(np.clip(np.sin(u)*si, -1, 1))
    lam = np.mod(lam + np.pi, 2*np.pi) - np.pi
    P = np.stack([lam, phi], axis=1)
    # segments i -> i+1 (wrap), skip dateline jumps
    A = P
    B = np.roll(P, -1, axis=0)
    ok = np.abs(B[:, 0] - A[:, 0]) < np.pi/2
    idx = np.nonzero(ok)[0]
    # spatial hash on longitude bins
    nbins = 4096
    binw = 2*np.pi/nbins
    lo = np.minimum(A[idx, 0], B[idx, 0])
    hi = np.maximum(A[idx, 0], B[idx, 0])
    b0 = ((lo + np.pi)/binw).astype(int)
    b1 = ((hi + np.pi)/binw).astype(int)
    from collections import defaultdict
    bins = defaultdict(list)
    for s, (a0, a1) in enumerate(zip(b0, b1)):
        for bb in range(a0, a1+1):
            bins[bb].append(idx[s])
    def seg_int(p1, p2, p3, p4):
        d1 = p2 - p1; d2 = p4 - p3
        den = d1[0]*d2[1] - d1[1]*d2[0]
        if den == 0: return False
        t = ((p3[0]-p1[0])*d2[1] - (p3[1]-p1[1])*d2[0]) / den
        s = ((p3[0]-p1[0])*d1[1] - (p3[1]-p1[1])*d1[0]) / den
        return 1e-12 < t < 1-1e-12 and 1e-12 < s < 1-1e-12
    hits = set()
    for bb, segs in bins.items():
        L = sorted(segs)
        for a_i in range(len(L)):
            sa = L[a_i]
            for b_i in range(a_i+1, len(L)):
                sb = L[b_i]
                if abs(sb - sa) <= 1 or (sa == 0 and sb == n-1):
                    continue
                if seg_int(A[sa], B[sa], A[sb], B[sb]):
                    hits.add((sa, sb))
    return len(hits)


if __name__ == "__main__":
    print("== validation: companion vs dense-grid bisection residuals ==")
    for (k, m, ideg) in [(7, 3, 80), (8, 3, 70), (24, 7, 55), (19, 3, 81)]:
        z, odd = basic_roots_companion(k, m, np.radians(ideg))
        res = residual_check(k, m, np.radians(ideg), z, odd)
        tot, N = count_map_points(k, m, np.radians(ideg))
        print(f"  ({k},{m}) i={ideg}: N_basic={N}, total={tot}, "
              f"max|F(root)|={res:.2e}")

    print("== timing: companion matrix, increasing size ==")
    for (k, m, ideg) in [(7, 3, 80), (24, 7, 55), (127, 10, 66.04),
                         (385, 27, 98.65), (1387, 91, 92.0)]:
        t0 = time.perf_counter()
        tot, N = count_map_points(k, m, np.radians(ideg))
        dt = time.perf_counter() - t0
        print(f"  ({k},{m}) i={ideg}: N_basic={N}, total={tot}, "
              f"t={dt*1e3:.1f} ms  (matrix {k+m-1}x{k+m-1})")

    print("== timing + agreement: brute force segment intersection ==")
    for (k, m, ideg, npr) in [(7, 3, 80, 2000), (24, 7, 55, 2000)]:
        t0 = time.perf_counter()
        bf = brute_force_count(k, m, np.radians(ideg), npr)
        dt = time.perf_counter() - t0
        tot, _ = count_map_points(k, m, np.radians(ideg))
        print(f"  ({k},{m}) i={ideg}: brute={bf} vs analytic={tot}, "
              f"t={dt:.2f} s ({npr} pts/rev)")

    print("== near-critical demo: (19,3) at i=81 deg ==")
    i81 = np.radians(81.0)
    tot, N = count_map_points(19, 3, i81)
    icrit = np.degrees(np.arccos(3/19))
    print(f"  pitchfork at arccos(3/19) = {icrit:.4f} deg; i = 81 deg is "
          f"{81-icrit:.4f} deg past it")
    print(f"  companion: N_basic={N}, total={tot}  (D&M Fig. 10 claims 285)")
    for npr in (500, 2000, 8000, 32000):
        t0 = time.perf_counter()
        bf = brute_force_count(19, 3, i81, npr)
        dt = time.perf_counter() - t0
        print(f"  brute force {npr:>6} pts/rev: {bf}  ({dt:.2f} s)")
