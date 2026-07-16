#!/usr/bin/env python3
"""Smoke test: fast end-to-end sanity check of the whole toolchain
(~10 s). Plain asserts, no test framework needed:

    python3 tests/test_smoke.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

from scan_counts import analyze_pair
from bench_b3 import basic_roots_companion, count_map_points, residual_check
from b5_validation import latitudes
from proto_recovery import basic_crossovers, dist_to_track
import b6_sso

passed = 0


def ok(label, cond):
    global passed
    assert cond, f"SMOKE FAIL: {label}"
    passed += 1
    print(f"  ok - {label}")


# counting: known (N_basic, total) at sample configurations
t, N = count_map_points(7, 3, np.radians(80))
ok("(7,3) i=80: 3 basic, 35 total", N == 3 and t == 35)
t, N = count_map_points(24, 7, np.radians(55))
ok("(24,7) i=55: 8 basic, 384 total", N == 8 and t == 384)
t, _ = count_map_points(19, 3, np.radians(81))
ok("(19,3) i=81: 323 total (near-critical)", t == 323)

# solver: companion-matrix roots satisfy the master equation
z, odd = basic_roots_companion(7, 3, np.radians(80))
ok("(7,3) residuals < 1e-9",
   residual_check(7, 3, np.radians(80), z, odd) < 1e-9)

# critical inclinations
r = analyze_pair(7, 3)
ok("(7,3) criticals = [64.6231, 88.2693]",
   len(r["crit_deg"]) == 2
   and abs(r["crit_deg"][0] - 64.6231) < 5e-4
   and abs(r["crit_deg"][1] - 88.2693) < 5e-4)

# Arnas-Linares threshold, exact rational value
r32 = analyze_pair(3, 2)
ok("(3,2) threshold cos(i*) = 1/9",
   abs(math.cos(math.radians(r32["crit_deg"][0])) - 1 / 9) < 1e-9)

# latitudes
lat = latitudes(127, 10, np.radians(66.04))
ok("TOPEX latitudes span [1.977, 66.033]",
   abs(lat[0] - 1.977) < 2e-3 and abs(lat[-1] - 66.033) < 2e-3)

# SSO design
a, ideg = b6_sso.sso_repeat(143, 10)
ok("Sentinel-2-like 143:10 SSO: a=7164.26 km, i=98.544 deg",
   abs(a - 7164.257) < 0.01 and abs(ideg - 98.544) < 1e-3)
N29, tot29, _, _ = b6_sso.crossover_summary(29, 2,
                                            b6_sso.sso_repeat(29, 2)[1])
ok("29:2 SSO: 15 basic, 870 total", N29 == 15 and tot29 == 870)

# geometric recovery: computed crossovers lie on the ground track
pts, case = basic_crossovers(7, 3, np.radians(80))
worst = max(dist_to_track(p[0], p[1], 7, 3, np.radians(80)) for p in pts)
ok("(7,3) crossovers lie on track (< 0.01 deg)", worst < 0.01)

print(f"\nSMOKE OK: {passed} checks passed")
