#!/usr/bin/env python3
"""C1: regenerate the paper figure set with one consistent style.
Outputs vector PDFs to ../figures/ (+ PNG previews for inspection).

Style: AIAA single column (3.5 in), serif/STIX, thin black linework,
two validated accents (blue #0072B2, vermillion #D55E00 - CVD dE=92,
contrast 5.2/3.9 on white, distinct grayscale luminance), identity never
color-alone (linestyle/marker + direct labels).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scan_counts import analyze_pair

BLUE, VERM, GRAY, INK = "#0072B2", "#D55E00", "#8a8a8a", "#000000"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": False, "ytick.right": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 0.9, "legend.frameon": False,
    "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=220)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- helpers
def master_roots(k, m, i_rad):
    """Roots of the master equation: z in [0, pi/2) (odd) / w in (0, pi/2]
    (mixed), via the polynomial form + bisection."""
    c = np.cos(i_rad)
    odd = (k % 2 == 1) and (m % 2 == 1)
    sgn = -1.0 if odd else 1.0
    F = lambda t: (1 - c) * np.sin((k + m) * t) + sgn * (1 + c) * np.sin((k - m) * t)
    n = 3000 * k
    ts = np.linspace(1e-9, np.pi / 2 - 1e-9, n)
    f = F(ts)
    roots = []
    idx = np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
    for j in idx:
        a, b = ts[j], ts[j + 1]
        for _ in range(55):
            mid = 0.5 * (a + b)
            if np.sign(F(a)) != np.sign(F(mid)):
                b = mid
            else:
                a = mid
        roots.append(0.5 * (a + b))
    if odd:
        roots = [0.0] + roots
    return roots, odd


def track_lonlat(k, m, i_rad, n=12000):
    u = np.linspace(0, 2 * np.pi * k, n)
    lam_u = np.unwrap(np.arctan2(np.sin(u) * np.cos(i_rad), np.cos(u)))
    lam = np.degrees(lam_u - (m / k) * u)
    lam = (lam + 180) % 360 - 180
    phi = np.degrees(np.arcsin(np.clip(np.sin(u) * np.sin(i_rad), -1, 1)))
    return lam, phi


def crossover_points(k, m, i_rad):
    """All map crossover points (node-relative lon, lat) via copies+mirror."""
    roots, odd = master_roots(k, m, i_rad)
    pts = []
    for t in roots:
        x = k * t if odd else k * t - np.pi / 2
        r = int(np.floor(x / np.pi + 1e-12))
        u1 = x - r * np.pi
        if u1 < 0:
            u1 += np.pi
            r -= 1
        phi = np.degrees(np.arcsin(np.sin(u1) * np.sin(i_rad)))
        lam_u = np.arctan2(np.sin(u1) * np.cos(i_rad), np.cos(u1))
        lam = np.degrees(lam_u - (m / k) * u1)
        for j in range(k):
            lj = (lam + j * 360.0 / k + 180) % 360 - 180
            pts.append((lj, phi))
            if abs(phi) > 1e-9:
                lm = (-lam + j * 360.0 / k + 180) % 360 - 180
                pts.append((lm, -phi))
    return np.array(pts)


# ------------------------------------------------- fig 1: track + crossovers
def fig_track():
    k, m, ideg = 7, 3, 70.0
    lam, phi = track_lonlat(k, m, np.radians(ideg))
    pts = crossover_points(k, m, np.radians(ideg))
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    # break the polyline at dateline jumps
    jump = np.abs(np.diff(lam)) > 90
    lam_plot = lam.copy(); phi_plot = phi.copy()
    lam_plot[1:][jump] = np.nan
    ax.plot(lam_plot, phi_plot, color=GRAY, lw=0.5, zorder=1)
    ax.scatter(pts[:, 0], pts[:, 1], s=9, facecolor=BLUE, edgecolor="white",
               linewidth=0.4, zorder=2)
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-60, -30, 0, 30, 60])
    ax.set_xlabel("node-relative longitude [deg]")
    ax.set_ylabel("latitude [deg]")
    fig.tight_layout(pad=0.3)
    save(fig, "fig_track_7_3_i70")


# ----------------------------------------------- fig 2: staircases N(i)
def staircase_data(k, m):
    r = analyze_pair(k, m)
    crit = r["crit_deg"]
    odd = r["odd"]
    if odd:
        base = (k - m) // 2
        i1 = np.degrees(np.arccos(m / k))
        steps = [(0.0, base)]
        n = base
        for cd in crit:
            n += 1 if abs(cd - i1) < 1e-9 else 2
            steps.append((cd, n))
        return steps, (m + 1) // 2, odd  # pole-collapse value
    else:
        base = (k - m - 1) // 2
        steps = [(0.0, base)]
        n = base
        for cd in crit:
            n += 2
            steps.append((cd, n))
        if k % 2 == 0:
            steps.append((90.0, n + 1))
        return steps, None, odd


def fig_staircase():
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.3), sharex=True)
    for ax, (k, m) in zip(axes, [(7, 3), (8, 3)]):
        steps, polecount, odd = staircase_data(k, m)
        xs, ys = [0.0], [steps[0][1]]
        for cd, n in steps[1:]:
            xs += [cd, cd]; ys += [ys[-1], n]
        xs.append(180.0); ys.append(ys[-1])
        ax.plot(xs, ys, color=INK, lw=1.0)
        prev_cd = -1e9
        for cd, n in steps[1:]:
            ax.axvline(cd, color=GRAY, lw=0.5, ls=(0, (2, 2)), zorder=0)
            # stagger: if the previous critical is close, label on the right
            right = (cd - prev_cd) < 8.0
            ax.annotate(f"{cd:.2f}$^\\circ$", (cd, ax.get_ylim()[0]),
                        xytext=(cd + 1.5 if right else cd - 1.5,
                                steps[0][1] + 0.15), rotation=90,
                        fontsize=6.5, color="#555555",
                        ha="left" if right else "right", va="bottom")
            prev_cd = cd
        if odd and polecount is not None:
            ax.plot([90], [polecount], marker="o", ms=3.5, mfc=VERM,
                    mec=VERM, ls="none")
            ax.annotate("pole collapse", (90, polecount),
                        xytext=(97, polecount - 0.1), fontsize=6.5,
                        color=VERM, va="center")
        fam = "odd–odd" if odd else "mixed parity"
        ax.text(0.02, 0.93, f"$k={k},\\ m={m}$ ({fam})",
                transform=ax.transAxes, fontsize=8, va="top")
        ax.set_ylabel("basic solutions $N$")
        ax.set_ylim(steps[0][1] - 0.6, steps[-1][1] + 0.7)
        ax.set_yticks(range(steps[0][1], steps[-1][1] + 1))
    axes[1].set_xlabel("inclination $i$ [deg]")
    axes[1].set_xlim(0, 180)
    axes[1].set_xticks([0, 30, 60, 90, 120, 150, 180])
    fig.tight_layout(pad=0.3, h_pad=0.8)
    save(fig, "fig_staircase")


# ------------------------------------------- fig 3: bifurcation diagrams
def fig_bifurcation():
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.6), sharex=True)
    for ax, (k, m) in zip(axes, [(7, 3), (8, 3)]):
        r = analyze_pair(k, m)
        crit = r["crit_deg"]
        odd = (k % 2 == 1) and (m % 2 == 1)
        igrid = np.linspace(0.2, 179.8, 1400)
        I, Z = [], []
        for ideg in igrid:
            roots, _ = master_roots(k, m, np.radians(ideg))
            for t in roots:
                I.append(ideg); Z.append(np.degrees(t))
        ax.plot(I, Z, ls="none", marker=".", ms=0.7, color=INK,
                rasterized=True)
        if not odd and k % 2 == 0:
            crit = crit + [90.0]  # polar critical: branch enters at w=90
            ax.plot([90], [90], marker="o", ms=4, mfc="white", mec=VERM,
                    mew=0.9, ls="none", zorder=3, clip_on=False)
        for cd in crit:
            ax.axvline(cd, color=GRAY, lw=0.5, ls=(0, (2, 2)), zorder=0)
        # mark critical points on the branches
        for cd in crit:
            roots_after, _ = master_roots(k, m, np.radians(cd + 0.4))
            roots_before, _ = master_roots(k, m, np.radians(cd - 0.4))
            # newborn roots = those in 'after' far from every 'before'
            for t in roots_after:
                if all(abs(t - tb) > 0.01 for tb in roots_before):
                    ax.plot([cd], [np.degrees(t)], marker="o", ms=4,
                            mfc="white", mec=VERM, mew=0.9, ls="none",
                            zorder=3)
        var = "z" if odd else "w"
        fam = "odd–odd" if odd else "mixed parity"
        ax.text(0.02, 0.95, f"$k={k},\\ m={m}$ ({fam})",
                transform=ax.transAxes, fontsize=8, va="top")
        ax.set_ylabel(f"root ${var}$ [deg]")
        ax.set_ylim(-3, 93)
    axes[1].set_xlabel("inclination $i$ [deg]")
    axes[1].set_xlim(0, 180)
    axes[1].set_xticks([0, 30, 60, 90, 120, 150, 180])
    fig.tight_layout(pad=0.3, h_pad=0.8)
    save(fig, "fig_bifurcation")


# ------------------------------------------------- fig 4: SSO catalogue
def fig_ssocat():
    from math import gcd
    from b6_sso import sso_repeat, crossover_summary, R
    rows = []
    for m in range(1, 16):
        for k in range(1, 260):
            if gcd(k, m) != 1:
                continue
            a, ideg = sso_repeat(k, m)
            h = a - R
            if 300.0 <= h <= 2000.0 and 90.0 < ideg < 113.0:
                N, total, lmin, lmax = crossover_summary(k, m, ideg)
                rows.append((h, total, lmin, lmax, m))
    rows = np.array(rows)
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.6), sharex=True)
    ax = axes[0]
    ax.scatter(rows[:, 0], rows[:, 1], s=6, facecolor="none",
               edgecolor=INK, linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("crossover points per cycle")
    ax = axes[1]
    ax.vlines(rows[:, 0], rows[:, 2], rows[:, 3], color=GRAY, lw=0.5)
    ax.scatter(rows[:, 0], rows[:, 3], s=4, color=BLUE, zorder=3,
               label="highest $|\\varphi|$")
    ax.scatter(rows[:, 0], rows[:, 2], s=4, color=VERM, zorder=3,
               marker="s", label="lowest $|\\varphi|$")
    ax.legend(loc="center right", handletextpad=0.3)
    ax.set_ylabel("crossover latitude $|\\varphi|$ [deg]")
    ax.set_xlabel("altitude [km]")
    fig.tight_layout(pad=0.3, h_pad=0.8)
    save(fig, "fig_ssocat")
    print(f"  catalogue rows: {len(rows)}")


# ------------------------------------------------- table: criticals k<=9
def table_criticals():
    from math import gcd
    lines = []
    for k in range(2, 10):
        for m in range(1, k):
            if gcd(k, m) != 1:
                continue
            r = analyze_pair(k, m)
            fam = "odd" if r["odd"] else ("mixed, $k$ even" if k % 2 == 0
                                          else "mixed, $k$ odd")
            crit = ", ".join(f"{c:.3f}" for c in r["crit_deg"])
            polar = " (+ polar)" if (not r["odd"] and k % 2 == 0) else ""
            lines.append(f"{k} & {m} & {fam} & {crit}{polar}\\\\")
    out = os.path.join(OUT, "table_criticals.tex")
    with open(out, "w") as f:
        f.write("\\begin{tabular}{ccll}\n\\toprule\n"
                "$k$ & $m$ & family & critical inclinations [deg]\\\\\n"
                "\\midrule\n")
        f.write("\n".join(lines) + "\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("wrote table_criticals.tex,", len(lines), "rows")


if __name__ == "__main__":
    fig_track()
    fig_staircase()
    fig_bifurcation()
    fig_ssocat()
    table_criticals()


# ============== C1c (2026-07-07): prettier / additional figures ==============

def _Fgrid(k, m, odd, i_deg, t_deg):
    """F on a (i, t) grid, vectorized; t = z (odd) or w (mixed), degrees."""
    I, T = np.meshgrid(np.radians(i_deg), np.radians(t_deg), indexing="ij")
    c = np.cos(I)
    sgn = -1.0 if odd else 1.0
    return (1 - c) * np.sin((k + m) * T) + sgn * (1 + c) * np.sin((k - m) * T)


def fig_bifurcation_smooth():
    """Solution branches as zero-contours of F: smooth vector curves.
    Spurious lines (z=90 odd; w=0 mixed) drawn separately in gray."""
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.9), sharex=True)
    for ax, (k, m) in zip(axes, [(7, 3), (8, 3)]):
        odd = (k % 2 == 1) and (m % 2 == 1)
        ideg = np.linspace(0.05, 179.95, 1600)
        tdeg = np.linspace(0.06, 92.0, 1900)  # start off 0: F=0 on the
        # t=0 grid row makes marching-squares degenerate there
        F = _Fgrid(k, m, odd, ideg, tdeg)
        cs = ax.contour(ideg, tdeg, F.T, levels=[0.0],
                        colors=INK, linewidths=0.9)
        # remove the spurious contour path (z=90 odd / w=0 mixed) and
        # redraw it as a labeled gray dashed line
        spur = 90.0 if odd else 0.0
        for coll in cs.collections:
            paths_keep = []
            for p in coll.get_paths():
                v = p.vertices
                if np.max(np.abs(v[:, 1] - spur)) < 0.2:
                    continue
                paths_keep.append(p)
            coll.set_paths(paths_keep)
        if odd:
            # genuine equatorial branch z=0 (exists for every i)
            ax.axhline(0.0, color=INK, lw=0.9, zorder=2)
        ax.axhline(spur, color=GRAY, lw=0.6, ls=(0, (3, 2)))
        lab = ("spurious root $z=90^\\circ$" if odd
               else "spurious root $w=0$")
        ax.annotate(lab, (135, spur), xytext=(135, spur + (2.5 if not odd else -6.5)),
                    fontsize=6.3, color="#777777")
        # criticals: dashed verticals + birth markers
        r = analyze_pair(k, m)
        crit = list(r["crit_deg"])
        for cd in crit:
            ax.axvline(cd, color=GRAY, lw=0.5, ls=(0, (2, 2)), zorder=0)
            roots_a, _ = master_roots(k, m, np.radians(cd + 0.4))
            roots_b, _ = master_roots(k, m, np.radians(cd - 0.4))
            for t in roots_a:
                if all(abs(t - tb) > 0.01 for tb in roots_b):
                    ax.plot([cd], [np.degrees(t)], marker="o", ms=4.2,
                            mfc="white", mec=VERM, mew=1.0, ls="none",
                            zorder=5)
        if not odd and k % 2 == 0:
            ax.axvline(90, color=GRAY, lw=0.5, ls=(0, (2, 2)), zorder=0)
            ax.plot([90], [90], marker="o", ms=4.2, mfc="white", mec=VERM,
                    mew=1.0, ls="none", zorder=5, clip_on=False)
        var = "z" if odd else "w"
        fam = "odd–odd" if odd else "mixed parity"
        ax.text(0.02, 0.965, f"$k={k},\\ m={m}$ ({fam})",
                transform=ax.transAxes, fontsize=8, va="top")
        ax.set_ylabel(f"root ${var}$ [deg]")
        ax.set_ylim(-4, 94)
        ax.set_yticks([0, 30, 60, 90])
    axes[1].set_xlabel("inclination $i$ [deg]")
    axes[1].set_xlim(0, 180)
    axes[1].set_xticks([0, 30, 60, 90, 120, 150, 180])
    fig.tight_layout(pad=0.3, h_pad=0.7)
    save(fig, "fig_bifurcation")


def _stereo(lam_deg, phi_deg, cap_deg):
    """North-polar stereographic (conformal); radius normalized to cap."""
    colat = np.radians(90.0 - np.asarray(phi_deg))
    rho = np.tan(colat / 2.0) / np.tan(np.radians(90.0 - cap_deg) / 2.0)
    lam = np.radians(np.asarray(lam_deg))
    return rho * np.sin(lam), rho * np.cos(lam)


def fig_polar():
    """Conformal polar view: crossovers accumulating toward the pole as
    i -> 90 deg ((7,3), two inclinations); crossing angles faithful."""
    cap = 70.0
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.0))
    for ax, ideg in zip(axes, [88.6, 89.7]):
        i = np.radians(ideg)
        lam, phi = track_lonlat(7, 3, i, n=400000)
        mask = phi > cap
        xs, ys = _stereo(lam[mask], phi[mask], cap)
        # break segments at gaps (mask discontinuities)
        idx = np.nonzero(mask)[0]
        brk = np.nonzero(np.diff(idx) > 1)[0]
        xs2, ys2 = xs.copy(), ys.copy()
        xs2[brk + 1] = np.nan
        ax.plot(xs2, ys2, color=GRAY, lw=0.45, zorder=1)
        pts = crossover_points(7, 3, i)
        sel = pts[:, 1] > cap
        px, py = _stereo(pts[sel, 0], pts[sel, 1], cap)
        ax.scatter(px, py, s=8, facecolor=BLUE, edgecolor="white",
                   linewidth=0.35, zorder=3)
        # graticule
        th = np.linspace(0, 2 * np.pi, 361)
        for gp in (75, 80, 85):
            gx, gy = _stereo(np.degrees(th), np.full_like(th, gp), cap)
            ax.plot(gx, gy, color="#cccccc", lw=0.35, zorder=0)
            ax.annotate(f"{gp}$^\\circ$", (0, _stereo(0, gp, cap)[1]),
                        fontsize=5.2, color="#999999", ha="center",
                        va="bottom", xytext=(0, _stereo(0, gp, cap)[1] + 0.015))
        bx, by = _stereo(np.degrees(th), np.full_like(th, cap), cap)
        ax.plot(bx, by, color="#aaaaaa", lw=0.5, zorder=0)
        ax.plot([0], [0], marker="+", ms=5, color=INK, mew=0.7)
        ax.set_title(f"$i={ideg}^\\circ$", fontsize=8, pad=2)
        ax.set_aspect("equal")
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
        ax.axis("off")
    fig.tight_layout(pad=0.2, w_pad=0.6)
    save(fig, "fig_polar")


def fig_gallery():
    """Track + crossovers at four inclinations ((7,3)): the staircase in
    pictures. Two-column-ish width."""
    cases = [50.0, 70.0, 88.0, 89.3]
    fig, axes = plt.subplots(2, 2, figsize=(5.9, 3.4),
                             sharex=True, sharey=True)
    for ax, ideg in zip(axes.flat, cases):
        i = np.radians(ideg)
        lam, phi = track_lonlat(7, 3, i, n=30000)
        jump = np.abs(np.diff(lam)) > 90
        lam_p = lam.copy(); lam_p[1:][jump] = np.nan
        ax.plot(lam_p, phi, color=GRAY, lw=0.4, zorder=1)
        for gl in (-60, -30, 0, 30, 60):
            ax.axhline(gl, color="#e3e3e3", lw=0.3, zorder=0)
        pts = crossover_points(7, 3, i)
        ax.scatter(pts[:, 0], pts[:, 1], s=6.5, facecolor=BLUE,
                   edgecolor="white", linewidth=0.3, zorder=3)
        ax.text(0.02, 0.96, f"$i={ideg}^\\circ$: {len(pts)} crossovers",
                transform=ax.transAxes, fontsize=7.2, va="top",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_yticks([-60, 0, 60])
    for ax in axes[1]:
        ax.set_xlabel("node-rel. longitude [deg]", fontsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("latitude [deg]", fontsize=7.5)
    fig.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.5)
    save(fig, "fig_gallery")


def fig_track_pretty():
    """fig_track with a light graticule and emphasized equator."""
    k, m, ideg = 7, 3, 70.0
    lam, phi = track_lonlat(k, m, np.radians(ideg))
    pts = crossover_points(k, m, np.radians(ideg))
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    jump = np.abs(np.diff(lam)) > 90
    lam_plot = lam.copy(); lam_plot[1:][jump] = np.nan
    for gl in (-60, -30, 30, 60):
        ax.axhline(gl, color="#e6e6e6", lw=0.35, zorder=0)
    for gl in (-120, -60, 0, 60, 120):
        ax.axvline(gl, color="#e6e6e6", lw=0.35, zorder=0)
    ax.axhline(0, color="#c8c8c8", lw=0.5, zorder=0)
    ax.plot(lam_plot, phi, color=GRAY, lw=0.5, zorder=1)
    ax.scatter(pts[:, 0], pts[:, 1], s=10, facecolor=BLUE,
               edgecolor="white", linewidth=0.4, zorder=2)
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-60, -30, 0, 30, 60])
    ax.set_xlabel("node-relative longitude [deg]")
    ax.set_ylabel("latitude [deg]")
    fig.tight_layout(pad=0.3)
    save(fig, "fig_track_7_3_i70")


# ============== C1d (2026-07-07): fan, mixed polar, Sentinel, families =======
# 4-class family palette (Okabe-Ito; validated: worst CVD dE 17.9, tritan
# 17.0, contrast >= 3.06 on white):
C_EQ, C_PF, C_ORIG, C_FOLD = "#D55E00", "#009E73", "#0072B2", "#CC79A7"


def _all_lats(k, m, i_rad):
    """All basic-family crossover |latitudes| [deg]."""
    roots, odd = master_roots(k, m, i_rad)
    lats = []
    for t in roots:
        x = k * t if odd else k * t - np.pi / 2
        r = int(np.floor(x / np.pi + 1e-12))
        u2 = x - r * np.pi
        if u2 < 0:
            u2 += np.pi
        lats.append(abs(np.degrees(np.arcsin(np.sin(u2) * np.sin(i_rad)))))
    return lats


def fig_ssocat_fan():
    """SSO catalogue with the FULL latitude fan (thesis Fig. 7-6 style):
    every crossover latitude of every catalogued orbit, colored by m."""
    from math import gcd
    from b6_sso import sso_repeat, R
    from matplotlib import cm
    from matplotlib.colors import Normalize
    rows = []       # (h, total, m)
    fan = []        # (h, |phi|, m)
    for m in range(1, 16):
        for k in range(1, 260):
            if gcd(k, m) != 1:
                continue
            a, ideg = sso_repeat(k, m)
            h = a - R
            if not (300.0 <= h <= 2000.0 and 90.0 < ideg < 113.0):
                continue
            i = np.radians(ideg)
            lats = _all_lats(k, m, i)
            odd = (k % 2 == 1) and (m % 2 == 1)
            N = len(lats)
            total = k * (2 * N - 1) if odd else 2 * k * N
            rows.append((h, total, m))
            fan += [(h, la, m) for la in lats]
    rows = np.array(rows); fan = np.array(fan)
    norm = Normalize(vmin=1, vmax=15)
    cmap = cm.Blues
    cvals = lambda mm: cmap(0.35 + 0.65 * (norm(mm)))
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.2), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.6]})
    ax = axes[0]
    ax.scatter(rows[:, 0], rows[:, 1], s=7, c=[cvals(mm) for mm in rows[:, 2]],
               edgecolor="none")
    ax.set_yscale("log")
    ax.set_ylabel("crossover points per cycle")
    ax = axes[1]
    ax.scatter(fan[:, 0], fan[:, 1], s=1.1, c=[cvals(mm) for mm in fan[:, 2]],
               edgecolor="none", rasterized=True)
    ax.axhline(66.7433, color="#999999", lw=0.5, ls=(0, (4, 2)))
    ax.annotate("arctic circle", (330, 66.7433), xytext=(330, 68.3),
                fontsize=6.2, color="#666666",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.4))
    ax.set_ylim(0, 90)
    ax.set_ylabel("crossover latitudes $|\\varphi|$ [deg]")
    ax.set_xlabel("altitude [km]")
    sm = cm.ScalarMappable(norm=Normalize(1, 15), cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, fraction=0.035, pad=0.02, aspect=38)
    cb.set_label("repeat cycle $m$ [nodal days]", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    save(fig, "fig_ssocat")
    print(f"  fan points: {len(fan)}, orbits: {len(rows)}")


def _points_by_root(k, m, i_rad):
    """Map points per basic root: list of (class_z_deg, pts array)."""
    roots, odd = master_roots(k, m, i_rad)
    out = []
    for t in roots:
        x = k * t if odd else k * t - np.pi / 2
        r = int(np.floor(x / np.pi + 1e-12))
        u1 = x - r * np.pi
        if u1 < 0:
            u1 += np.pi; r -= 1
        phi = np.degrees(np.arcsin(np.sin(u1) * np.sin(i_rad)))
        lam_u = np.arctan2(np.sin(u1) * np.cos(i_rad), np.cos(u1))
        lam = np.degrees(lam_u - (m / k) * u1)
        pts = []
        for j in range(k):
            pts.append(((lam + j * 360 / k + 180) % 360 - 180, phi))
            if abs(phi) > 1e-9:
                pts.append(((-lam + j * 360 / k + 180) % 360 - 180, -phi))
        out.append((np.degrees(t), np.array(pts)))
    return out, odd


def fig_gallery_families():
    """Gallery colored by family provenance ((7,3)): equatorial /
    original / pitchfork-born / fold-born."""
    cases = [50.0, 70.0, 88.0, 89.3]
    fig, axes = plt.subplots(2, 2, figsize=(5.9, 3.55),
                             sharex=True, sharey=True)
    handles = {}
    for ax, ideg in zip(axes.flat, cases):
        i = np.radians(ideg)
        lam, phi = track_lonlat(7, 3, i, n=30000)
        jump = np.abs(np.diff(lam)) > 90
        lam_p = lam.copy(); lam_p[1:][jump] = np.nan
        ax.plot(lam_p, phi, color=GRAY, lw=0.4, zorder=1)
        for gl in (-60, -30, 0, 30, 60):
            ax.axhline(gl, color="#e6e6e6", lw=0.3, zorder=0)
        groups, _ = _points_by_root(7, 3, i)
        n = len(groups)
        # (7,3): sorted z: [eq, (pf), orig, (fold_lo, fold_hi)]
        if n == 2:
            classes = [("equatorial", C_EQ), ("original", C_ORIG)]
        elif n == 3:
            classes = [("equatorial", C_EQ), ("pitchfork-born", C_PF),
                       ("original", C_ORIG)]
        else:
            classes = [("equatorial", C_EQ), ("pitchfork-born", C_PF),
                       ("original", C_ORIG), ("fold-born", C_FOLD),
                       ("fold-born", C_FOLD)]
        total = 0
        for (zdeg, pts), (name, col) in zip(groups, classes):
            sc = ax.scatter(pts[:, 0], pts[:, 1], s=7.5, facecolor=col,
                            edgecolor="white", linewidth=0.3, zorder=3)
            handles.setdefault(name, sc)
            total += len(pts)
        ax.text(0.02, 0.96, f"$i={ideg}^\\circ$: {total} crossovers",
                transform=ax.transAxes, fontsize=7.2, va="top",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_yticks([-60, 0, 60])
    for ax in axes[1]:
        ax.set_xlabel("node-rel. longitude [deg]", fontsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("latitude [deg]", fontsize=7.5)
    order = ["equatorial", "original", "pitchfork-born", "fold-born"]
    fig.legend([handles[o] for o in order], order, ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, 1.005),
               fontsize=7, frameon=False, columnspacing=1.2,
               handletextpad=0.2)
    fig.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.5, rect=(0, 0, 1, 0.955))
    save(fig, "fig_gallery")


def fig_polar_mixed():
    """Mixed-parity conformal polar view ((8,3)): the newborn polar
    crossover of thm:mixedcount(ii) appears at exactly i=90."""
    cap = 70.0
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.0))
    for ax, ideg in zip(axes, [89.2, 90.6]):
        i = np.radians(ideg)
        lam, phi = track_lonlat(8, 3, i, n=400000)
        mask = phi > cap
        xs, ys = _stereo(lam[mask], phi[mask], cap)
        idx = np.nonzero(mask)[0]
        brk = np.nonzero(np.diff(idx) > 1)[0]
        xs2 = xs.copy(); xs2[brk + 1] = np.nan
        ax.plot(xs2, ys, color=GRAY, lw=0.45, zorder=1)
        groups, _ = _points_by_root(8, 3, i)
        for (wdeg, pts) in groups:
            sel = pts[:, 1] > cap
            if not sel.any():
                continue
            px, py = _stereo(pts[sel, 0], pts[sel, 1], cap)
            col = VERM if wdeg > 80.0 else BLUE   # newborn polar family
            ax.scatter(px, py, s=8, facecolor=col, edgecolor="white",
                       linewidth=0.35, zorder=3)
        th = np.linspace(0, 2 * np.pi, 361)
        for gp in (75, 80, 85):
            gx, gy = _stereo(np.degrees(th), np.full_like(th, gp), cap)
            ax.plot(gx, gy, color="#cccccc", lw=0.35, zorder=0)
        bx, by = _stereo(np.degrees(th), np.full_like(th, cap), cap)
        ax.plot(bx, by, color="#aaaaaa", lw=0.5, zorder=0)
        ax.plot([0], [0], marker="+", ms=5, color=INK, mew=0.7)
        ax.set_title(f"$i={ideg}^\\circ$", fontsize=8, pad=2)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    fig.tight_layout(pad=0.2, w_pad=0.6)
    save(fig, "fig_polar_mixed")


def fig_sentinel():
    """The full crossover lattice of Sentinel-3 (385:27, i=98.65):
    158,235 points, computed in ~0.1 s, plotted raw."""
    k, m, ideg = 385, 27, 98.65
    pts = crossover_points(k, m, np.radians(ideg))
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    ax.scatter(pts[:, 0], pts[:, 1], s=0.22, facecolor=BLUE,
               edgecolor="none", rasterized=True)
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_yticks([-60, -30, 0, 30, 60])
    ax.set_xlabel("node-relative longitude [deg]")
    ax.set_ylabel("latitude [deg]")
    fig.tight_layout(pad=0.3)
    save(fig, "fig_sentinel")
    print(f"  points plotted: {len(pts)}")


# ====== C1e: polar panels with per-panel zoom caps (user feedback) ==========
def _polar_panel(ax, k, m, ideg, cap, grat, newborn_rule=False):
    i = np.radians(ideg)
    lam, phi = track_lonlat(k, m, i, n=600000)
    mask = phi > cap
    xs, ys = _stereo(lam[mask], phi[mask], cap)
    idx = np.nonzero(mask)[0]
    brk = np.nonzero(np.diff(idx) > 1)[0]
    xs2 = xs.copy(); xs2[brk + 1] = np.nan
    ax.plot(xs2, ys, color=GRAY, lw=0.4, zorder=1)
    groups, _ = _points_by_root(k, m, i)
    for (tdeg, pts) in groups:
        sel = pts[:, 1] > cap
        if not sel.any():
            continue
        px, py = _stereo(pts[sel, 0], pts[sel, 1], cap)
        col = VERM if (newborn_rule and tdeg > 80.0 and ideg > 90.0) else BLUE
        ax.scatter(px, py, s=9, facecolor=col, edgecolor="white",
                   linewidth=0.35, zorder=3)
    th = np.linspace(0, 2 * np.pi, 361)
    for gp in grat:
        gx, gy = _stereo(np.degrees(th), np.full_like(th, gp), cap)
        ax.plot(gx, gy, color="#cccccc", lw=0.35, zorder=0)
        ax.annotate(f"{gp}$^\\circ$", (0, _stereo(0, gp, cap)[1]),
                    fontsize=5.2, color="#999999", ha="center", va="bottom",
                    xytext=(0, _stereo(0, gp, cap)[1] + 0.015))
    bx, by = _stereo(np.degrees(th), np.full_like(th, cap), cap)
    ax.plot(bx, by, color="#aaaaaa", lw=0.5, zorder=0)
    ax.plot([0], [0], marker="+", ms=5, color=INK, mew=0.7)
    ax.set_title(f"$i={ideg}^\\circ$ (cap ${cap}^\\circ$)", fontsize=7.5,
                 pad=2)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)


def fig_polar_v2():
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.05))
    _polar_panel(axes[0], 7, 3, 88.6, 70.0, (75, 80, 85))
    _polar_panel(axes[1], 7, 3, 89.7, 85.0, (87, 88, 89))
    fig.tight_layout(pad=0.2, w_pad=0.6)
    save(fig, "fig_polar")


def fig_polar_mixed_v2():
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.05))
    _polar_panel(axes[0], 8, 3, 89.2, 85.0, (87, 88, 89))
    _polar_panel(axes[1], 8, 3, 90.6, 75.0, (80, 84, 88),
                 newborn_rule=True)
    fig.tight_layout(pad=0.2, w_pad=0.6)
    save(fig, "fig_polar_mixed")


def fig_sentinel_v2():
    """Sentinel-3 crossover lattice: conformal polar cap (the full map is
    solid ink at 158k points; near the 81.35 deg turning latitude the
    ring structure is the story)."""
    k, m, ideg = 385, 27, 98.65
    cap = 65.0
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    i = np.radians(ideg)
    pts = crossover_points(k, m, i)
    sel = pts[:, 1] > cap
    px, py = _stereo(pts[sel, 0], pts[sel, 1], cap)
    ax.scatter(px, py, s=0.4, facecolor=BLUE, edgecolor="none",
               rasterized=True, zorder=3)
    th = np.linspace(0, 2 * np.pi, 361)
    for gp in (70, 75, 80):
        gx, gy = _stereo(np.degrees(th), np.full_like(th, gp), cap)
        ax.plot(gx, gy, color="#cccccc", lw=0.35, zorder=0)
        ax.annotate(f"{gp}$^\\circ$", (0, _stereo(0, gp, cap)[1]),
                    fontsize=5.5, color="#777777", ha="center", va="bottom",
                    xytext=(0, _stereo(0, gp, cap)[1] + 0.012), zorder=6,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3))
    bx, by = _stereo(np.degrees(th), np.full_like(th, cap), cap)
    ax.plot(bx, by, color="#aaaaaa", lw=0.5, zorder=0)
    ax.plot([0], [0], marker="+", ms=5, color=INK, mew=0.7)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.03, 1.03); ax.set_ylim(-1.03, 1.03)
    fig.tight_layout(pad=0.15)
    save(fig, "fig_sentinel")
    print(f"  cap points: {sel.sum()} of {len(pts)}")
