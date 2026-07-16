#!/usr/bin/env python3
"""Numeric audit: recompute every headline number of the paper from scratch
and compare against the published values (hardcoded below) and the shipped
data/ files. Exit code 0 iff every check passes.

This is the standalone version of the audit: it verifies the *science*
(recomputation vs published values). The companion manuscript audit, which
additionally greps the LaTeX sources for each value, lives with the
manuscript and is not part of this repository.

Runtime ~15 s (dominated by the ICESat-2 1387:91 companion matrix and
the (39,38) critical-inclination scan).
"""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_counts import analyze_pair, g_roots, c_ratio
from bench_b3 import count_map_points
from b5_validation import latitudes
import b6_sso

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

results = []


def check(cid, ok, detail=""):
    results.append((cid, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {cid} {detail if not ok else ''}")


print("== critical inclinations ==")
r73 = analyze_pair(7, 3)
check("(7,3) first critical = 64.6231 (pitchfork arccos(3/7))",
      abs(r73["crit_deg"][0] - 64.6231) < 5e-4)
check("(7,3) second critical = 88.2693",
      abs(r73["crit_deg"][1] - 88.2693) < 5e-4)
# the tangency root behind the second critical: z* and c(z*)
z73 = [z for z in g_roots(7, 3) if 0.0 < c_ratio(z, 3, 7) < 1.0]
check("(7,3) exactly one valid G-root", len(z73) == 1)
zstar = math.degrees(z73[0])
cstar = c_ratio(z73[0], 3, 7)
check("z* = 62.1001 deg", abs(zstar - 62.1001) < 5e-4,
      f"got {zstar:.4f}")
check("c(z*) = 0.030202", abs(cstar - 0.030202) < 5e-6,
      f"got {cstar:.6f}")

# min gap claim (recompute over all pairs k<=40 is slow; recompute (39,38))
r3938 = analyze_pair(39, 38)
gaps = [b - a for a, b in zip(r3938["crit_deg"][:-1], r3938["crit_deg"][1:])]
check("min gap 0.191316 deg at (39,38)",
      abs(min(gaps) - 0.191316) < 2e-3)

print("== Sec. 5 table (counts and totals) ==")
for (k, m, ideg, nb, tot) in [(7, 3, 80, 3, 35), (24, 7, 55, 8, 384),
                              (127, 10, 66.04, 58, 14732),
                              (385, 27, 98.65, 206, 158235),
                              (1387, 91, 92.0, 739, 2048599)]:
    t, N = count_map_points(k, m, np.radians(ideg))
    check(f"({k},{m}) i={ideg}: N_basic={nb}, total={tot}",
          N == nb and t == tot, f"got N={N}, total={t}")

print("== near-critical demo ==")
pf = math.degrees(math.acos(3 / 19))
check("pitchfork (19,3) = 80.9153 deg", abs(pf - 80.9153) < 5e-4)
check("offset at i=81 deg is 0.0847 deg", abs(81 - pf - 0.0847) < 5e-4)
t193, N193 = count_map_points(19, 3, np.radians(81))
check("(19,3) at 81 deg: total = 323 (vs 285 in Dilectis-Mortari Fig. 10)",
      t193 == 323, f"got {t193}")

print("== Dilectis-Mortari counts ==")
for (k, m, ideg, c) in [(18, 5, 65, 216), (13, 4, 83, 156),
                        (21, 8, 80, 336), (14, 5, 110, 252)]:
    t, _ = count_map_points(k, m, np.radians(ideg))
    check(f"D&M ({k}:{m},{ideg}) = {c}", t == c, f"got {t}")

print("== Arnas-Linares threshold ==")
r32 = analyze_pair(3, 2)
c32 = math.cos(math.radians(r32["crit_deg"][0]))
check("(3,2) threshold = 1/9 exactly", abs(c32 - 1 / 9) < 1e-9,
      f"got {c32:.12f}")

print("== mission latitudes ==")
lat_t = latitudes(127, 10, np.radians(66.04))
check("TOPEX lat range [1.977, 66.033] deg",
      abs(lat_t[0] - 1.977) < 2e-3 and abs(lat_t[-1] - 66.033) < 2e-3,
      f"got [{lat_t[0]:.3f}, {lat_t[-1]:.3f}]")
lat_i = latitudes(1387, 91, np.radians(92.0))
check("ICESat-2 max lat = 88.00 deg", abs(lat_i[-1] - 88.0) < 5e-3,
      f"got {lat_i[-1]:.3f}")
lat_s = latitudes(385, 27, np.radians(98.65))
check("Sentinel-3 max lat = 81.35 deg", abs(lat_s[-1] - 81.35) < 5e-3,
      f"got {lat_s[-1]:.3f}")

print("== SSO section ==")
check("K_ss = 0.09892", abs(b6_sso.KSS - 0.09892) < 5e-6,
      f"got {b6_sso.KSS:.6f}")
zd1 = math.degrees(math.acos((1 + math.sqrt(17)) / 8))
zd2 = math.degrees(math.acos((1 - math.sqrt(17)) / 8))
check("zero-distortion inclinations 50.18 / 112.98 deg",
      abs(zd1 - 50.179) < 1e-3 and abs(zd2 - 112.98) < 5e-3,
      f"got {zd1:.3f} / {zd2:.3f}")

from math import gcd
count_sso = 0
for m in range(1, 16):
    for k in range(1, 260):
        if gcd(k, m) != 1:
            continue
        a, ideg = b6_sso.sso_repeat(k, m)
        h = a - b6_sso.R
        if 300.0 <= h <= 2000.0 and 90.0 < ideg < 113.0:
            count_sso += 1
check("329 admissible repeat SSOs", count_sso == 329, f"got {count_sso}")

# anchors computed directly (Landsat 233:16 has m=16, outside the m<=15
# catalogue loop -- it is a worked example, not a catalogue member)
anchor = {}
for (k, m) in [(143, 10), (233, 16), (29, 2)]:
    a_, i_ = b6_sso.sso_repeat(k, m)
    anchor[(k, m)] = (a_, a_ - b6_sso.R, i_)
a, h, ideg = anchor[(143, 10)]
check("Sentinel-2 anchor: a=7164.26 km, h=786.1 km, i=98.54 deg",
      abs(a - 7164.257) < 0.01 and abs(h - 786.1) < 0.1
      and abs(ideg - 98.544) < 1e-3, f"got {a:.3f}/{h:.1f}/{ideg:.4f}")
a, h, ideg = anchor[(233, 16)]
check("Landsat anchor: a=7077.72 km, i=98.19 deg",
      abs(a - 7077.721) < 0.01 and abs(ideg - 98.186) < 1e-3,
      f"got {a:.3f}/{ideg:.4f}")
N29, tot29, _, _ = b6_sso.crossover_summary(29, 2, anchor[(29, 2)][2])
check("29:2 -> 15 basic, 870 total", N29 == 15 and tot29 == 870,
      f"got {N29}, {tot29}")

print("== data/critical_inclinations.csv vs fresh recomputation (k <= 9) ==")
crit_csv = {}
csv_ok = True
path = os.path.join(DATA, "critical_inclinations.csv")
if not os.path.exists(path):
    check("critical_inclinations.csv present (run make_data.py)", False)
else:
    with open(path) as fh:
        for row in csv.DictReader(fh):
            key = (int(row["k"]), int(row["m"]))
            crit_csv.setdefault(key, []).append(float(row["i_crit_deg"]))
    for k in range(2, 10):
        for m in range(1, k):
            if gcd(k, m) != 1:
                continue
            fresh = analyze_pair(k, m)["crit_deg"]
            shipped = crit_csv.get((k, m), [])
            if len(fresh) != len(shipped) or any(
                    abs(a - b) > 5e-6 for a, b in zip(sorted(fresh),
                                                      sorted(shipped))):
                csv_ok = False
                print(f"    mismatch ({k},{m}): fresh={fresh} csv={shipped}")
    check("all k<=9 criticals match the shipped CSV", csv_ok)

print("== data/sso_catalogue.csv vs fresh recomputation ==")
path = os.path.join(DATA, "sso_catalogue.csv")
if not os.path.exists(path):
    check("sso_catalogue.csv present (run make_data.py)", False)
else:
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    check("catalogue has 329 rows", len(rows) == 329, f"got {len(rows)}")
    cat_ok = True
    for row in rows:
        k, m = int(row["k_revs"]), int(row["m_days"])
        a_, i_ = b6_sso.sso_repeat(k, m)
        N, total, lmin, lmax = b6_sso.crossover_summary(k, m, i_)
        if (abs(a_ - float(row["a_km"])) > 5e-3
                or abs(i_ - float(row["i_deg"])) > 5e-4
                or N != int(row["n_basic_crossovers"])
                or total != int(row["total_crossovers"])):
            cat_ok = False
            print(f"    mismatch {k}:{m}")
    check("all catalogue rows match fresh recomputation", cat_ok)

print("== A6 sweep headline numbers (shipped results file) ==")
path = os.path.join(HERE, "A6_RESULTS.md")
if not os.path.exists(path):
    check("A6_RESULTS.md present", False)
else:
    a6 = open(path).read()
    for s in ["configurations: **5588**",
              "within validity bound: **4895**",
              "(100.00 %), deviations: **0**",
              "count == m in 482",
              "criticals above 90 deg encountered: **0**"]:
        check(f"A6: '{s}'", s in a6)

fails = [c for c, ok, _ in results if not ok]
print(f"\n{len(results)} checks, {len(fails)} failures")
if fails:
    print("FAILED:", fails)
sys.exit(1 if fails else 0)
