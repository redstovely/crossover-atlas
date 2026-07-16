#!/usr/bin/env python3
"""Run the verification suite. Usage:

    python3 run_all.py smoke     # fast sanity check              (~10 s)
    python3 run_all.py audit     # full numeric audit vs paper    (~2 min)
    python3 run_all.py figures   # regenerate all paper figures -> figures/
    python3 run_all.py data      # regenerate data/ CSVs          (~minutes)
    python3 run_all.py full      # everything, in dependency order

Each step is an independent script; run_all just sequences them, times
them, and stops at the first failure. Paths are __file__-relative, so the
working directory does not matter.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

STEPS = {
    "smoke":   [("tests/test_smoke.py", [])],
    "audit":   [("scripts/audit_numbers.py", [])],
    "figures": [("scripts/make_figures.py", []),
                ("scripts/figs_c1b.py", [])],
    "data":    [("scripts/make_data.py", [])],
    "full":    [("tests/test_smoke.py", []),
                ("scripts/scan_counts.py", []),
                ("scripts/bench_b3.py", []),
                ("scripts/b5_validation.py", []),
                ("scripts/b6_sso.py", []),
                ("scripts/proto_recovery.py", []),
                ("scripts/bench_a5.py", []),
                ("scripts/sweep_a6.py", []),
                ("scripts/make_data.py", []),
                ("scripts/audit_numbers.py", []),
                ("scripts/make_figures.py", []),
                ("scripts/figs_c1b.py", [])],
}


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if target not in STEPS:
        print(f"unknown target '{target}'; pick one of: {', '.join(STEPS)}")
        return 2
    t00 = time.perf_counter()
    for script, extra in STEPS[target]:
        path = os.path.join(HERE, script)
        print(f"\n=== {script} ===")
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, path] + extra)
        dt = time.perf_counter() - t0
        if r.returncode != 0:
            print(f"\nFAILED: {script} (exit {r.returncode}) after {dt:.1f} s")
            return 1
        print(f"--- {script} passed in {dt:.1f} s")
    print(f"\nALL '{target}' STEPS PASSED in "
          f"{time.perf_counter() - t00:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
