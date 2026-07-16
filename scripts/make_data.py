#!/usr/bin/env python3
"""Generate the data/ files shipped with the repository:

  data/critical_inclinations.csv  all critical inclinations, coprime (k,m),
                                  2 <= k <= KMAX (default 40)
  data/sso_catalogue.csv          the 329 admissible repeat sun-synchronous
                                  orbits (h in [300, 2000] km, m <= 15) with
                                  crossover counts and latitude ranges
  data/constants.csv              physical constants used throughout

Usage: python3 make_data.py [--kmax N]   (run from anywhere; paths are
relative to this file). Full k <= 40 scan takes a few minutes.
"""
import argparse
import csv
import math
import os
import sys
from math import gcd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_counts import analyze_pair
import b6_sso

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def write_criticals(kmax):
    """One row per critical inclination (interior tangencies; for odd-odd
    pairs the equatorial pitchfork arccos(m/k) is included and labelled).
    The polar critical at i = 90 deg (k even, present at exactly i = pi/2)
    is a boundary case and is not listed."""
    path = os.path.join(DATA, "critical_inclinations.csv")
    n_rows = 0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["k", "m", "case", "n_criticals", "crit_index",
                    "i_crit_deg", "kind"])
        for k in range(2, kmax + 1):
            for m in range(1, k):
                if gcd(k, m) != 1:
                    continue
                r = analyze_pair(k, m)
                case = "odd-odd" if r["odd"] else "mixed"
                pitch = math.degrees(math.acos(m / k)) if r["odd"] else None
                for j, cd in enumerate(r["crit_deg"], start=1):
                    kind = ("pitchfork" if pitch is not None
                            and abs(cd - pitch) < 1e-9 else "tangency")
                    w.writerow([k, m, case, len(r["crit_deg"]), j,
                                f"{cd:.6f}", kind])
                    n_rows += 1
    print(f"  wrote {path} ({n_rows} criticals, k <= {kmax})")


def write_sso_catalogue():
    path = os.path.join(DATA, "sso_catalogue.csv")
    rows = []
    for m in range(1, 16):
        for k in range(1, 260):
            if gcd(k, m) != 1:
                continue
            a, ideg = b6_sso.sso_repeat(k, m)
            h = a - b6_sso.R
            if 300.0 <= h <= 2000.0 and 90.0 < ideg < 113.0:
                N, total, lmin, lmax = b6_sso.crossover_summary(k, m, ideg)
                rows.append([k, m, f"{a:.3f}", f"{h:.1f}", f"{ideg:.4f}",
                             N, total, f"{lmin:.2f}", f"{lmax:.2f}"])
    rows.sort(key=lambda r: (int(r[1]), int(r[0])))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["k_revs", "m_days", "a_km", "h_km", "i_deg",
                    "n_basic_crossovers", "total_crossovers",
                    "lat_min_deg", "lat_max_deg"])
        w.writerows(rows)
    print(f"  wrote {path} ({len(rows)} orbits)")


def write_constants():
    path = os.path.join(DATA, "constants.csv")
    consts = [
        ("J2", b6_sso.J2, "-", "Earth oblateness coefficient"),
        ("R", b6_sso.R, "km", "Earth equatorial radius"),
        ("MU", b6_sso.MU, "km^3/s^2", "Earth gravitational parameter"),
        ("KSS", b6_sso.KSS, "-",
         "sun-synchronicity constant: cos(i) (R/a)^3.5 = -KSS"),
    ]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "value", "unit", "description"])
        for name, val, unit, desc in consts:
            w.writerow([name, repr(val), unit, desc])
    print(f"  wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmax", type=int, default=40,
                    help="max k for the criticals scan (default 40)")
    args = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    print("== generating data/ ==")
    write_constants()
    write_sso_catalogue()
    write_criticals(args.kmax)
