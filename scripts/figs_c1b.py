#!/usr/bin/env python3
"""C1b: the two optional Sec. 6 figures.
  fig_foldcurve  — critical-eccentricity fold curves e_fold(i) for the
                   pair born at the circular critical ((7,5) and (7,3)),
                   from exact-system continuation (bench_a5 machinery).
  fig_driftloops — crossover drift loops on the map as omega sweeps
                   0..360 deg at fixed e ((7,3), i=50 deg, O(e^2) d-E law).
Same style as make_figures.py.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import radians, degrees, pi

from bench_a5 import (circular_roots, continue_branch, scan_roots,
                      position_deg, M_exact_factory, M_dE_factory)
from make_figures import BLUE, VERM, GRAY, INK, save  # rcParams applied

E_SCAN = np.linspace(0.0, 0.25, 126)


def first_fold_e(k, m, ideg, om_deg=60.0):
    """Smallest fold eccentricity over all branches at inclination ideg."""
    inc, om = radians(ideg), radians(om_deg)
    folds = []
    for (u0, r, j) in circular_roots(k, m, inc):
        _, lost = continue_branch(k, m, r, j, inc, om, M_exact_factory,
                                  E_SCAN, u0)
        if lost is not None:
            folds.append(lost["e_last"])
    return min(folds) if folds else None


def fig_foldcurve():
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for (k, m, istar, imax, color, mk) in [
            (7, 5, 44.4153, 58.0, BLUE, "o"),
            (7, 3, 64.6231, 78.0, VERM, "s")]:
        igrid = np.concatenate([
            istar + np.array([0.15, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0]),
            np.arange(np.ceil(istar) + 4, imax + 0.1, 2.0)])
        ef = []
        for ideg in igrid:
            ef.append(first_fold_e(k, m, float(ideg)))
        ok = [(i_, e_) for i_, e_ in zip(igrid, ef) if e_ is not None]
        xs = [istar] + [p[0] for p in ok]
        ys = [0.0] + [p[1] for p in ok]
        ax.plot(xs, ys, color=color, lw=1.0, marker=mk, ms=2.6,
                label=f"$k={k},\\ m={m}$")
        ax.axvline(istar, color=GRAY, lw=0.5, ls=(0, (2, 2)), zorder=0)
    ax.set_xlabel("inclination $i$ [deg]")
    ax.set_ylabel("fold eccentricity $e_{\\mathrm{fold}}$")
    ax.set_ylim(0, 0.26)
    ax.legend(loc="upper left")
    fig.tight_layout(pad=0.3)
    save(fig, "fig_foldcurve")


def fig_driftloops():
    k, m, ideg, e = 7, 3, 50.0, 0.05
    inc = radians(ideg)
    mk = m / k
    oms = np.arange(0.0, 360.1, 3.0)
    # circular reference positions
    M0, _ = M_dE_factory(0.0)
    refs = []
    for (u0, r, j) in circular_roots(k, m, inc):
        # drop the equatorial family (u2 near 0 or pi): its drift is
        # predominantly longitudinal and the (u,r) matching jumps branches
        if abs(np.sin(u0)) < 0.15:
            continue
        if abs(u0) > pi / 2:
            continue  # pass-swap duplicate (u and pi-u = same crossover)
        refs.append((u0, r))
    Mf = M_dE_factory
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    tracks = {ir: ([], []) for ir in range(len(refs))}
    for om_deg in oms:
        om = radians(om_deg)
        Me, _ = Mf(e)
        roots = scan_roots(k, m, inc, om, e, Mf)
        for iref, (u0, r0) in enumerate(refs):
            # drift relative to the CIRCULAR solution at the SAME omega:
            # isolates the O(e) displacement (the perigee-epoch convention
            # slides absolute longitudes by (m/k)*omega even at e = 0)
            p0, l0 = position_deg(u0, r0, inc, om, M0, mk)
            cands = [(abs(u - u0), u, r) for (u, r, j) in roots if r == r0]
            if not cands:
                continue
            _, u, r = min(cands)
            phi, lam = position_deg(u, r, inc, om, Me, mk)
            dlam = (lam - l0 + 180) % 360 - 180
            tracks[iref][0].append(dlam)
            tracks[iref][1].append(phi - p0)
    plotted = 0
    for iref, (xs, ys) in tracks.items():
        if len(xs) < len(oms) - 2 or plotted >= 3:
            continue
        u0 = refs[iref][0]
        col = [BLUE, VERM, INK][plotted]
        hemi = "northern" if np.sin(u0) * np.sin(inc) > 0 else "southern"
        lab = f"{hemi} crossover ($u_2^{{(0)}} = {degrees(u0):.1f}^\\circ$)"
        ax.plot(xs + xs[:1], ys + ys[:1], color=col, lw=0.9, label=lab)
        ax.plot([xs[0]], [ys[0]], marker="o", ms=3, color=col)
        plotted += 1
    ax.axhline(0, color=GRAY, lw=0.4, zorder=0)
    ax.axvline(0, color=GRAY, lw=0.4, zorder=0)
    ax.set_xlabel("longitude drift $\\Delta\\lambda$ [deg]")
    ax.set_ylabel("latitude drift $\\Delta\\varphi$ [deg]")
    ax.legend(fontsize=6.5, loc="best", handlelength=1.4)
    fig.tight_layout(pad=0.3)
    save(fig, "fig_driftloops")


if __name__ == "__main__":
    fig_driftloops()
    fig_foldcurve()
