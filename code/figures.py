"""Regenerates every figure and table of the paper into ../figures.

Figures follow the AIAA journal guidelines: no background or frame around the
artwork, axis labels of the form "Quantity, unit", lettering of at least 8 pt
at the size at which the figure is included, and part labels a), b), ...
below and to the left of each part.  Two widths are used: W_COL for artwork
that fits one journal column and W_PAGE for artwork that spans the page.
"""
from math import gcd, pi, cos, sin, tan, acos, atan, sqrt, radians, degrees
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import crossovers as cx
import sso

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'STIXGeneral'],
    'mathtext.fontset': 'stix', 'font.size': 9, 'axes.linewidth': 0.6,
    'axes.labelsize': 9, 'legend.fontsize': 8, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'lines.linewidth': 0.9, 'pdf.fonttype': 42,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})
BLUE, ORANGE, GREEN, GRAY, INK = '#0072B2', '#D55E00', '#009E73', '#8a8a8a', '#222222'
W_PAGE = 6.5   # manuscript text width, in.
W_COL = 3.4    # journal column width, in.
W = W_PAGE


def plot_track(ax, k, m, inc, lw=0.5, ms=2.6, color=BLUE):
    th, lat, lon = cx.track(k, m, inc, 1440)
    brk = np.where(np.abs(np.diff(lon)) > 180)[0] + 1
    for seg_lon, seg_lat in zip(np.split(lon, brk), np.split(lat, brk)):
        ax.plot(seg_lon, seg_lat, color=color, lw=lw)
    la, lo, _, _ = cx.crossovers(k, m, inc)
    ax.plot(lo, la, 'o', ms=ms, mfc=ORANGE, mec='none')
    ax.set_xlim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 60))
    return len(la)


# --------------------------------------------------------------- Fig 1
def fig_track():
    fig, ax = plt.subplots(figsize=(W, 2.6))
    n = plot_track(ax, 7, 3, 70, lw=0.7, ms=3.6)
    assert n == 35
    ax.set_ylim(-80, 80)
    ax.set_yticks(np.arange(-60, 61, 30))
    ax.set_xlabel('Longitude, deg')
    ax.set_ylabel('Latitude, deg')
    fig.savefig('../figures/fig_track.pdf')
    plt.close(fig)


# --------------------------------------------------------------- Fig 2
def fig_gamma():
    cases = [(7, 3), (8, 3), (3, 8)]
    fig, axes = plt.subplots(3, 1, figsize=(W_PAGE, 7.0))
    for ax, (k, m), part in zip(axes, cases, 'abc'):
        poles = sorted({(2 * q + 1) * pi / (2 * k) for q in range(k)} |
                       {p * pi / m for p in range(1, m)})
        both_odd = k % 2 == 1 and m % 2 == 1
        if both_odd:
            poles = [p for p in poles if abs(p - pi / 2) > 1e-9]
        edges = [0.0] + poles + [pi]
        ax.axhspan(-1, 1, color='#f1f1f1', zorder=0)
        for a, b in zip(edges[:-1], edges[1:]):
            w = np.linspace(a, b, 1202)[1:-1]
            if both_odd:
                w = w[np.abs(w - pi / 2) > 1e-6]
            g = cx.gamma(w, k, m)
            g[np.abs(g) > 6] = np.nan
            ax.plot(np.degrees(w), g, color=BLUE)
        for p in poles:
            ax.axvline(np.degrees(p), color=GRAY, lw=0.35, ls=':')
        if both_odd:
            ax.axvline(90, color=GREEN, lw=1.0)
        crit = cx.critical_values(k, m)
        for cs, w, kind in crit:
            if kind == 'fold':
                ax.plot(np.degrees([w, pi - w]), [cs, cs], 'o', ms=4, mfc=ORANGE, mec='none')
            elif kind == 'equator':
                ax.plot([90], [cs], 's', ms=4.5, mfc=GREEN, mec='none')
            elif kind == 'cusp':
                ax.plot([0, 180], [cs, cs], 'D', ms=4.2, mfc=GREEN, mec='none', clip_on=False, zorder=5)
            elif kind == 'polar':
                ax.plot([90], [0], '^', ms=5, mfc=ORANGE, mec='none')
        for w in cx.G_zeros(k, m):
            g = float(cx.gamma(w, k, m))
            if g > 1 and g < 2.4:
                ax.plot(np.degrees([w, pi - w]), [g, g], 'o', ms=3.5, mfc='white', mec=GRAY, mew=0.7)
        ax.set_ylim(-1.6, 2.4)
        ax.set_xlim(0, 180)
        ax.set_ylabel(r'$\gamma(w)$')
        ax.set_xlabel(r'$w$, deg', labelpad=1)
        ax.set_xticks(np.arange(0, 181, 30))
        ax.text(0.0, -0.2, rf'{part}) ${k}{{:}}{m}$', transform=ax.transAxes, va='top', ha='left')
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ticks = [0, 60, 90, 120, 180]
        ax2.set_yticks([cos(radians(t)) for t in ticks])
        ax2.set_yticklabels([str(t) for t in ticks])
        ax2.set_ylabel(r'$i$, deg')
    fig.subplots_adjust(hspace=0.42)
    fig.savefig('../figures/fig_gamma.pdf')
    plt.close(fig)


# --------------------------------------------------------------- Fig 3
def fig_staircase():
    cases = [(7, 3, BLUE, '-'), (8, 3, ORANGE, '--'), (3, 8, GREEN, '-.')]
    fig, ax = plt.subplots(figsize=(W_COL, 2.4))
    for k, m, col, ls in cases:
        crit = cx.critical_values(k, m)
        inc = np.linspace(0.05, 179.95, 7200)
        L = [cx.L_theory(k, m, cos(radians(x)), crit) for x in inc]
        ax.step(inc, L, where='mid', color=col, ls=ls, label=rf'${k}{{:}}{m}$', lw=1.1)
        for cs, _, kind in crit:      # value at the critical inclination itself
            ax.plot([degrees(acos(cs))], [cx.L_theory(k, m, cs, crit)], 'o', ms=3.4, mfc=col, mec='white',
                    mew=0.5, zorder=5, clip_on=False)
    ax.set_xlabel(r'Inclination $i$, deg')
    ax.set_ylabel(r'$L$')
    ax.set_xlim(0, 180)
    ax.set_xticks(np.arange(0, 181, 30))
    ax.set_yticks(range(0, 11, 2))
    ax.legend(frameon=False, loc='lower right', handlelength=2.6)
    fig.savefig('../figures/fig_staircase.pdf')
    plt.close(fig)


# --------------------------------------------------------------- Fig 4
def fig_gallery():
    fig, axes = plt.subplots(2, 2, figsize=(W_PAGE, 5.6), sharey=True)
    counts = {}
    for ax, inc, part in zip(axes.flat, [50, 70, 88, 89.3], 'abcd'):
        n = plot_track(ax, 7, 3, inc, lw=0.5, ms=2.8)
        counts[inc] = n
        ax.set_ylim(-90, 90)
        ax.set_yticks(np.arange(-90, 91, 30))
        ax.set_xticks(np.arange(-120, 121, 60))
        ax.set_xlabel('Longitude, deg', labelpad=1)
        ax.text(0.0, -0.2, rf'{part}) $i={inc}$ deg, {n} crossovers', transform=ax.transAxes,
                va='top', ha='left')
    assert counts == {50: 21, 70: 35, 88: 35, 89.3: 63}, counts
    for ax in axes[:, 0]:
        ax.set_ylabel('Latitude, deg')
    fig.subplots_adjust(hspace=0.42, wspace=0.06)
    fig.savefig('../figures/fig_gallery.pdf')
    plt.close(fig)


# --------------------------------------------------------------- Fig 5
def fig_pitchfork():
    """Longitude gap between the descending and the ascending pass that meet
    at an equatorial crossover of the 19:3 track, versus latitude."""
    k, m = 19, 3
    ic = degrees(acos(m / k))
    fig, ax = plt.subplots(figsize=(W_COL, 3.0))
    u = np.linspace(-0.36, 0.36, 4001)
    lam = lambda th, c: np.arctan2(c * np.sin(th), np.cos(th)) - (m / k) * th
    styles = [(80.7, GRAY, ':'), (ic, INK, '-'), (80.95, BLUE, '--'), (81.0, ORANGE, '-')]
    for inc, col, ls in styles:
        c, si = cos(radians(inc)), sin(radians(inc))
        tha = pi / 2 + k * pi / 2 + (u if (k % 4 == 3) else -u)   # theta_2 branch
        thd = pi / 2 - k * pi / 2 + (-u if (k % 4 == 3) else u)   # theta_1 branch
        lat = np.degrees(np.arcsin(si * np.sin(tha)))
        assert np.allclose(lat, np.degrees(np.arcsin(si * np.sin(thd))), atol=1e-9)
        d = lam(thd, c) - lam(tha, c)
        d = np.degrees((d + pi) % (2 * pi) - pi)
        lab = rf'{inc:.3f} deg ($i_p$)' if abs(inc - ic) < 1e-9 else rf'{inc:.3f} deg'
        ax.plot(d, lat, color=col, ls=ls, label=lab, lw=1.1)
        nz = int(np.sum(d[:-1] * d[1:] < 0) + np.sum(d == 0))
        print('  pitchfork fig: i=%.3f zero crossings of the gap: %d' % (inc, nz))
    ax.axvline(0, color=INK, lw=0.4)
    ax.set_ylim(-18, 18)
    ax.set_xlabel('Longitude difference, deg')
    ax.set_ylabel('Latitude, deg')
    ax.legend(frameon=False, loc='upper right', handlelength=2.6, borderaxespad=0.2,
              title=r'$i$', title_fontsize=8)
    ax.set_xlim(-0.05, 0.085)
    ax.set_xticks([-0.04, 0, 0.04, 0.08])
    fig.savefig('../figures/fig_pitchfork.pdf')
    plt.close(fig)


# --------------------------------------------------------------- Fig 6
def fig_critmap():
    fig, ax = plt.subplots(figsize=(W_PAGE, 3.6))
    xs, ys, xp, yp = [], [], [], []
    for k in range(2, 41):
        for m in range(1, k):
            if gcd(k, m) != 1:
                continue
            for cs, w, kind in cx.critical_values(k, m):
                if kind == 'fold':
                    xs.append(m / k); ys.append(degrees(acos(cs)))
                elif kind == 'equator':
                    xp.append(m / k); yp.append(degrees(acos(cs)))
    ax.plot(xs, ys, '.', ms=2.0, color=BLUE, label='Fold', rasterized=True)
    ax.plot(xp, yp, 's', ms=2.4, mfc=ORANGE, mec='none', label='Equatorial pitchfork', rasterized=True)
    q = np.linspace(0, 1, 400)
    ax.plot(q, np.degrees(np.arccos(q)), color=INK, lw=0.6, label=r'$\arccos(m/k)$')
    ax.set_xlabel(r'$m/k$')
    ax.set_ylabel('Critical inclination, deg')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 90)
    ax.set_yticks(np.arange(0, 91, 15))
    ax.legend(frameon=False, loc='lower left', markerscale=2.5)
    fig.savefig('../figures/fig_critmap.pdf', dpi=600)
    plt.close(fig)


# --------------------------------------------------------------- Fig 7
def fig_sso():
    rows, _, _ = sso.catalogue()
    fig, ax = plt.subplots(figsize=(W_PAGE * 0.8, 3.1))
    h = [r[3] for r in rows]
    n = [r[6] for r in rows]
    mm = [r[1] for r in rows]
    sc = ax.scatter(h, n, c=mm, s=9, cmap='viridis', lw=0)
    ax.set_yscale('log')
    ax.set_xlabel('Altitude, km')
    ax.set_ylabel('Crossovers per repeat cycle')
    ax.set_ylim(1e2, 2.5e5)
    cb = fig.colorbar(sc, ax=ax, pad=0.015)
    cb.set_label(r'$m$, nodal days')
    for (k, m, name, xy) in [(143, 10, '143:10', (1000, 1.5e5)), (15, 1, '15:1', (640, 1.3e2)),
                             (12, 1, '12:1', (1850, 2.6e2))]:
        for r in rows:
            if (r[0], r[1]) == (k, m):
                ax.annotate(name, (r[3], r[6]), xytext=xy, textcoords='data', fontsize=8,
                            va='center', arrowprops=dict(arrowstyle='-', lw=0.5, shrinkB=2.5))
    fig.savefig('../figures/fig_sso.pdf')
    plt.close(fig)
    return len(rows)



# ------------------------------------------------- geometry of the problem
PURPLE = '#AA4499'


def _view(az, el):
    az, el = radians(az), radians(el)
    v = np.array([cos(el) * cos(az), cos(el) * sin(az), sin(el)])
    r = np.array([-sin(az), cos(az), 0.0])
    return v, r, np.cross(v, r)


def _proj(P, view):
    v, r, up = view
    P = np.atleast_2d(np.asarray(P, float))
    return P @ r, P @ up, P @ v


def _curve(ax, P, view, back=True, **kw):
    """Curve on the unit sphere: solid where visible, thin and dashed behind."""
    X, Y, D = _proj(P, view)
    vis = D >= -1e-9
    ax.plot(np.where(vis, X, np.nan), np.where(vis, Y, np.nan), solid_capstyle='butt', **kw)
    if back:
        kb = dict(kw, lw=0.45, ls=(0, (3, 2)), alpha=0.75)
        kb.pop('zorder', None)
        ax.plot(np.where(~vis, X, np.nan), np.where(~vis, Y, np.nan), zorder=1, **kb)


def _head(ax, P, view, color, size=7):
    """Arrowhead at the end of the curve P."""
    X, Y, _ = _proj(P[-2:], view)
    ax.annotate('', xy=(X[1], Y[1]), xytext=(X[0], Y[0]),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=0, mutation_scale=size,
                                shrinkA=0, shrinkB=0), zorder=6)


def _label(ax, p, view, s, dx=0.0, dy=0.0, **kw):
    X, Y, _ = _proj(p, view)
    kw.setdefault('ha', 'center')
    kw.setdefault('va', 'center')
    ax.text(X[0] + dx, Y[0] + dy, s, **kw)


def fig_geometry():
    """a) orbit, subsatellite point and angles in the frame of the node;
    b) the two passes of a crossover, mirror images in the meridian plane of
    the turning point."""
    inc = radians(55.0)
    c, s = cos(inc), sin(inc)
    orb = lambda u: np.stack([np.cos(u), c * np.sin(u), s * np.sin(u)], -1)
    sph = lambda la, lo: np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo),
                                   np.sin(la) + 0 * lo], -1)
    Lam = lambda u: np.arctan2(c * np.sin(u), np.cos(u))
    full = np.linspace(0, 2 * pi, 721)
    n = 80
    z = np.zeros(n)
    fig, axes = plt.subplots(1, 2, figsize=(W_PAGE, 3.25))

    def base(ax, view, view_az):
        t = np.linspace(0, 2 * pi, 361)
        ax.plot(np.cos(t), np.sin(t), color=INK, lw=0.7)
        _curve(ax, sph(0 * full, full), view, color=GRAY, lw=0.7)
        _curve(ax, orb(full), view, color=BLUE, lw=1.0)
        # rotation axis and direction of the node
        for p, lab, off in [((0, 0, 1.0), '$z$', (0.06, 0.03)), ((1.0, 0, 0), '$x$', (-0.03, -0.07))]:
            p = np.array(p)
            X, Y, _ = _proj([0 * p, p, 1.3 * p], view)
            ax.plot(X[:2], Y[:2], color=INK, lw=0.5, ls=':')
            ax.annotate('', xy=(X[2], Y[2]), xytext=(X[1], Y[1]),
                        arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.6, mutation_scale=7,
                                        shrinkA=0, shrinkB=0))
            ax.text(X[2] + off[0], Y[2] + off[1], lab, ha='center', va='center')
        # rotation of the Earth relative to the node
        lo = np.radians(np.linspace(view_az - 150, view_az + 75, 80))
        ring = sph(np.radians(80.0) + 0 * lo, lo) + np.array([0, 0, 0.17])
        X, Y, _ = _proj(ring, view)
        ax.plot(X, Y, color=INK, lw=0.6)
        _head(ax, ring, view, INK)
        ax.text(X[-1] + 0.07, Y[-1], r'$\nu$', ha='left', va='center')
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_xlim(-1.22, 1.22)
        ax.set_ylim(-1.1, 1.42)

    # ---- a) variables
    ax = axes[0]
    view = _view(52.0, 20.0)
    base(ax, view, 52.0)
    u0 = radians(52.0)
    L0 = float(Lam(u0))
    ph0 = float(np.arcsin(s * sin(u0)))
    _curve(ax, orb(np.linspace(0, u0, n)), view, back=False, color=ORANGE, lw=2.0, zorder=4)
    _curve(ax, sph(z, np.linspace(0, L0, n)), view, back=False, color=GREEN, lw=2.0, zorder=4)
    _curve(ax, sph(np.linspace(0, ph0, n), L0 + z), view, back=False, color=PURPLE, lw=2.0, zorder=4)
    _curve(ax, sph(np.linspace(ph0, pi / 2, n), L0 + z), view, back=False, color=GRAY, lw=0.45,
           zorder=2)
    # inclination mark at the node
    be = np.linspace(0, inc, 30)
    d = 0.26
    arc = np.stack([cos(d) + 0 * be, sin(d) * np.cos(be), sin(d) * np.sin(be)], -1)
    _curve(ax, arc, view, back=False, color=INK, lw=0.6, zorder=4)
    _label(ax, arc[15], view, '$i$', dx=0.07, dy=0.0)
    _head(ax, orb(np.linspace(radians(96), radians(100), 2)), view, BLUE, size=9)
    X, Y, _ = _proj([orb(u0), orb(0.0)], view)
    ax.plot(X[0], Y[0], 'o', ms=4.5, mfc=INK, mec='none', zorder=7)
    ax.plot(X[1], Y[1], 'o', ms=3.2, mfc='white', mec=INK, mew=0.7, zorder=7)
    ax.text(X[0] + 0.05, Y[0] + 0.07, '$P$', ha='left', va='center')
    _label(ax, orb(u0 * 0.52), view, '$u$', dx=-0.085, dy=0.045, color=ORANGE)
    _label(ax, sph(0.0, np.array(L0 * 0.5)), view, r'$\Lambda$', dy=-0.085, color=GREEN)
    _label(ax, sph(np.array(ph0 * 0.5), L0), view, r'$\varphi$', dx=0.075, color=PURPLE)
    ax.text(X[1] - 0.05, Y[1] + 0.075, 'node', ha='right', va='center', fontsize=8)
    ax.text(0.0, -0.02, 'a)', transform=ax.transAxes, va='top', ha='left')

    # ---- b) the two passes of a crossover
    ax = axes[1]
    view = _view(66.0, 20.0)
    base(ax, view, 66.0)
    a = radians(33.0)
    u1, u2 = pi / 2 - a, pi / 2 + a
    L1, L2 = float(Lam(u1)), float(Lam(u2))
    ph = float(np.arcsin(s * sin(u1)))
    _curve(ax, sph(ph + 0 * full, full), view, color=GRAY, lw=0.5)
    _curve(ax, sph(np.linspace(0, pi / 2, n), pi / 2 + z), view, back=False, color=INK, lw=0.5,
           ls=(0, (1, 1.5)), zorder=2)
    _curve(ax, orb(np.linspace(u1, u2, n)), view, back=False, color=BLUE, lw=2.2, zorder=4)
    par = sph(ph + z, np.linspace(L1, L2, n))
    _curve(ax, par, view, back=False, color=ORANGE, lw=2.0, zorder=4)
    _head(ax, par[n // 4 - 1:n // 4 + 1], view, ORANGE, size=9)
    _head(ax, par[3 * n // 4 - 1:3 * n // 4 + 1], view, ORANGE, size=9)
    _head(ax, orb(np.linspace(radians(25), radians(29), 2)), view, BLUE, size=9)
    _head(ax, orb(np.linspace(radians(151), radians(155), 2)), view, BLUE, size=9)
    X, Y, _ = _proj([orb(u1), orb(u2), orb(pi / 2), orb(0.0)], view)
    ax.plot(X[:2], Y[:2], 'o', ms=4.5, mfc=INK, mec='none', zorder=7)
    ax.plot(X[2], Y[2], 'o', ms=3.4, mfc='white', mec=INK, mew=0.7, zorder=7)
    ax.plot(X[3], Y[3], 'o', ms=3.2, mfc='white', mec=INK, mew=0.7, zorder=7)
    ax.text(X[0] - 0.05, Y[0] - 0.02, r'$P(\theta_1)$', ha='right', va='center')
    ax.text(X[1] + 0.05, Y[1] - 0.02, r'$P(\theta_2)$', ha='left', va='center')
    ax.text(X[3] - 0.04, Y[3] + 0.08, 'node', ha='right', va='center', fontsize=8)
    ax.annotate('turning point', xy=(X[2], Y[2] + 0.03), xytext=(X[2] - 0.12, Y[2] + 0.27),
                ha='center', va='center', fontsize=8,
                arrowprops=dict(arrowstyle='-', color=INK, lw=0.4, shrinkA=1, shrinkB=1))
    _label(ax, orb(pi / 2 - 0.6 * a), view, '$kw$', dx=-0.08, dy=0.075, color=BLUE)
    _label(ax, orb(pi / 2 + 0.6 * a), view, '$kw$', dx=0.09, dy=0.06, color=BLUE)
    _label(ax, sph(np.array(ph), 0.5 * (L1 + pi / 2)), view, '$mw$', dy=-0.085, color=ORANGE)
    _label(ax, sph(np.array(ph), 0.5 * (L2 + pi / 2)), view, '$mw$', dy=-0.085, color=ORANGE)
    ax.text(0.0, -0.02, 'b)', transform=ax.transAxes, va='top', ha='left')
    fig.subplots_adjust(wspace=0.02)
    fig.savefig('../figures/fig_geometry.pdf')
    plt.close(fig)


def fig_symmetry():
    """From the fundamental crossovers to all of them (7:3 track at 70 deg):
    a) images under the symmetries of the track, b) pairs of passes as chords
    of the repeat cycle."""
    k, m, inc = 7, 3, 70.0
    la, lo, Wz, S = cx.crossovers(k, m, inc)
    assert len(la) == 35
    fam_lat = np.array(sorted(set(np.round(np.abs(la), 6)), reverse=True))   # 67.07, 43.42, 0
    assert len(fam_lat) == 3
    fam = np.array([int(np.argmin(np.abs(fam_lat - abs(x)))) for x in la])
    cols, marks = [ORANGE, GREEN, PURPLE], ['o', 's', '^']
    # fundamental sector: 360/k deg of longitude, northern hemisphere and equator
    sec = 360.0 / k
    north = la > -1e-9
    off = np.mod(lo[north], sec)
    cand = np.linspace(0, sec, 721)[:-1]
    dist = lambda x: np.min(np.minimum(np.mod(off - x, sec), np.mod(x - off, sec)))
    x0 = cand[int(np.argmax([dist(x) for x in cand]))]
    lam_a = x0 - sec * np.ceil((x0 + sec / 2) / sec)          # sector closest to longitude 0
    inside = north & (np.mod(lo - lam_a, 360.0) < sec)
    assert inside.sum() == 3 and sorted(fam[inside]) == [0, 1, 2]

    fig = plt.figure(figsize=(W_PAGE, 2.95))
    ax = fig.add_axes([0.075, 0.2, 0.56, 0.77])
    ax.fill_between([lam_a, lam_a + sec], 0, 90, color='#dddddd', lw=0, zorder=0)
    th, lat, lon = cx.track(k, m, inc, 1440)
    brk = np.where(np.abs(np.diff(lon)) > 180)[0] + 1
    for seg_lon, seg_lat in zip(np.split(lon, brk), np.split(lat, brk)):
        ax.plot(seg_lon, seg_lat, color=BLUE, lw=0.45, zorder=1)
    for f in range(3):
        sel = (fam == f) & ~inside
        ax.plot(lo[sel], la[sel], marks[f], ms=3.6, mfc='white', mec=cols[f], mew=0.9, zorder=3)
        sel = (fam == f) & inside
        ax.plot(lo[sel], la[sel], marks[f], ms=5.2, mfc=cols[f], mec=INK, mew=0.6, zorder=4)
    ax.annotate('', xy=(lam_a + sec, 80), xytext=(lam_a, 80),
                arrowprops=dict(arrowstyle='<|-|>', color=INK, lw=0.6, mutation_scale=6,
                                shrinkA=0, shrinkB=0))
    ax.text(lam_a + sec + 4, 80, r'$360/k$ deg', ha='left', va='center', fontsize=8,
            bbox=dict(fc='white', ec='none', pad=0.5))
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.set_xlabel('Longitude, deg', labelpad=1)
    ax.set_ylabel('Latitude, deg', labelpad=1)
    ax.text(0.0, -0.2, 'a)', transform=ax.transAxes, va='top', ha='left')

    # ---- b) chord diagram of the repeat cycle: psi = theta/k, clockwise from the top
    bx = fig.add_axes([0.665, 0.2, 0.335, 0.77])
    pt = lambda psi, r=1.0: (r * np.sin(psi), r * np.cos(psi))
    t = np.linspace(0, 2 * pi, 361)
    bx.plot(np.cos(t), np.sin(t), color=INK, lw=0.7, zorder=2)
    for order in (False, True):                 # fundamental chords on top
        for j in range(len(la)):
            if inside[j] != order:
                continue
            pc = (pi / 2 + pi * S[j]) / k
            (x1, y1), (x2, y2) = pt(pc - Wz[j]), pt(pc + Wz[j])
            bx.plot([x1, x2], [y1, y2], color=cols[fam[j]], lw=1.9 if order else 0.55,
                    alpha=1.0 if order else 0.8, zorder=4 if order else 3,
                    solid_capstyle='round')
    for j in range(k):                          # ascending nodes and turning points
        (x1, y1), (x2, y2) = pt(2 * pi * j / k, 1.0), pt(2 * pi * j / k, 1.07)
        bx.plot([x1, x2], [y1, y2], color=INK, lw=0.7)
        x, y = pt(2 * pi * j / k, 1.17)
        bx.text(x, y, str(j), ha='center', va='center', fontsize=8)
        x, y = pt((pi / 2 + 2 * pi * j) / k)
        bx.plot(x, y, 'o', ms=2.6, mfc=INK, mec=INK, mew=0.5, zorder=5)
        x, y = pt((3 * pi / 2 + 2 * pi * j) / k)
        bx.plot(x, y, 'o', ms=2.6, mfc='white', mec=INK, mew=0.5, zorder=5)
    # angle subtended by the fundamental chord of the first family
    j = int(np.where(inside & (fam == 0))[0][0])
    pc = (pi / 2 + pi * S[j]) / k
    arc = np.linspace(pc - Wz[j], pc + Wz[j], 60)
    x, y = pt(arc, 1.36)
    bx.plot(x, y, color=cols[0], lw=0.8)
    for e, d in ((0, 1), (-1, -1)):
        bx.annotate('', xy=(x[e], y[e]), xytext=(x[e + d], y[e + d]),
                    arrowprops=dict(arrowstyle='-|>', color=cols[0], lw=0, mutation_scale=7,
                                    shrinkA=0, shrinkB=0))
    x, y = pt(pc, 1.5)
    bx.text(x, y, '$2w$', ha='center', va='center', color=cols[0])
    bx.set_aspect('equal')
    bx.axis('off')
    bx.set_xlim(-1.6, 1.6)
    bx.set_ylim(-1.6, 1.6)
    bx.text(0.0, -0.2, 'b)', transform=bx.transAxes, va='top', ha='left')
    fig.savefig('../figures/fig_symmetry.pdf')
    plt.close(fig)
    print('  symmetry fig: sector [%.2f, %.2f] deg, fundamental (lat, lon, w, s):'
          % (lam_a, lam_a + sec),
          [(round(float(la[j]), 2), round(float(lo[j]), 2), round(degrees(Wz[j]), 2), int(S[j]))
           for j in np.where(inside)[0]])


# ------------------------------------------------- polar stereographic views
def polar_cap(ax, k, m, inc, lat_rim, parallels=(), label_parallels=False, npr=6000,
              ms=3.2, lw=0.7, c_exact=None):
    """North polar cap in polar stereographic projection (conformal): the pole
    is at the center, the rim is the parallel lat_rim, longitude 0 is at the
    top and east runs counterclockwise.  Returns the number of crossovers
    inside the cap and the total number."""
    R = tan(radians(90 - lat_rim) / 2)
    rho = lambda la: np.tan(np.radians(90 - la) / 2) / R
    xy = lambda la, lo: (-rho(la) * np.sin(np.radians(lo)), rho(la) * np.cos(np.radians(lo)))
    t = np.linspace(0, 2 * pi, 361)
    for lo in np.arange(0, 360, 30):
        ax.plot([0, -sin(radians(lo))], [0, cos(radians(lo))], color=GRAY, lw=0.3, ls=':')
    for p in parallels:
        r = float(rho(p))
        ax.plot(r * np.cos(t), r * np.sin(t), color=GRAY, lw=0.3, ls=':')
    th, lat, lon = cx.track(k, m, inc, npr)
    x, y = xy(lat, lon)
    inside = lat >= lat_rim
    ax.plot(np.where(inside, x, np.nan), np.where(inside, y, np.nan), color=BLUE, lw=lw)
    la, lo, _, _ = cx.crossovers(k, m, inc)
    sel = la >= lat_rim
    px, py = xy(la[sel], lo[sel])
    ax.plot(px, py, 'o', ms=ms, mfc=ORANGE, mec='none', zorder=4)
    ax.plot(np.cos(t), np.sin(t), color=INK, lw=0.7)
    if label_parallels:
        a = radians(-112.5)   # between two meridians
        for p in tuple(parallels) + (lat_rim,):
            r = float(rho(p)) - (0.075 if p == lat_rim else 0.0)
            ax.text(r * cos(a), r * sin(a), f'{p:g}', fontsize=8, color='#555555', ha='center',
                    va='center', bbox=dict(fc='white', ec='none', pad=0.6), zorder=5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    return int(sel.sum()), len(la)


def _cap_rows(rows, fname, height):
    fig, axes = plt.subplots(len(rows), 3, figsize=(W_PAGE, height))
    parts = iter('abcdefghi')
    for r, (k, m, incs, rim, par, expect, cex) in enumerate(rows):
        for c, inc in enumerate(incs):
            ax = axes[r, c]
            n, ntot = polar_cap(ax, k, m, inc, rim, par, label_parallels=(c == 0),
                                c_exact=(cex if c == 1 else None))
            assert (n, ntot) == expect[c], (k, m, inc, n, ntot)
            print(f'  cap {k}:{m} i={inc:.3f}: {n} of {ntot} crossovers north of {rim} deg')
            ax.text(0.0, -0.04, rf'{next(parts)}) ${k}{{:}}{m}$, $i={inc:.2f}$ deg', transform=ax.transAxes,
                    va='top', ha='left')
    fig.subplots_adjust(hspace=0.16, wspace=0.05)
    fig.savefig(fname)
    plt.close(fig)


def fig_conformal():
    """Birth of crossovers at a fold (3:2) and at a cusp pitchfork (2:3)."""
    i32, i23 = degrees(acos(1 / 9)), degrees(acos(2 / 3))
    _cap_rows([(3, 2, [82.0, i32, 85.0], 40, (60, 80), [(0, 0), (3, 6), (6, 12)], None),
               (2, 3, [42.0, i23, 54.0], 20, (40, 60, 80), [(0, 0), (0, 0), (2, 4)], 2 / 3)],
              '../figures/fig_conformal.pdf', 4.75)


def fig_polar():
    """Polar limit: k even (8:3) and k odd (7:3) on both sides of 90 deg."""
    _cap_rows([(8, 3, [88.6, 90.0, 91.4], 74, (80, 85), [(24, 64), (28, 72), (32, 80)], None),
               (7, 3, [88.6, 90.0, 91.4], 74, (80, 85), [(21, 63), (21, 63), (21, 63)], None)],
              '../figures/fig_polar.pdf', 4.75)


# ------------------------------------------------------------- dense track
def fig_dense():
    """a) Conformal view of the TOPEX/Poseidon track near its turning latitude;
    b) distribution of the crossover latitudes of three missions."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W_PAGE, 3.3), gridspec_kw=dict(width_ratios=[1, 1.02]))
    # ---- a) polar stereographic zoom, north up, east to the right
    k, m, inc = 127, 10, 66.0
    lat0, lat1, dlon = 52.0, 66.0, 11.0
    rho = lambda la: np.tan(np.radians(90 - np.asarray(la, float)) / 2)
    xy = lambda la, lo: (rho(la) * np.sin(np.radians(lo)), -rho(la) * np.cos(np.radians(lo)))
    x1 = float(xy(lat0, dlon)[0])
    y0, y1 = float(xy(lat0, 0)[1]), float(xy(lat1, 0)[1])
    y1 += 0.05 * (y1 - y0)
    th, lat, lon = cx.track(k, m, inc, 6000)
    near = (lat >= lat0 - 3) & (np.abs(lon) <= 3 * dlon)
    x, y = xy(lat, lon)
    ax.plot(np.where(near, x, np.nan), np.where(near, y, np.nan), color=BLUE, lw=0.55)
    la, lo, _, _ = cx.crossovers(k, m, inc)
    assert len(la) == 14732
    sel = (la >= lat0 - 3) & (np.abs(lo) <= 3 * dlon)
    px, py = xy(la[sel], lo[sel])
    ax.plot(px, py, 'o', ms=2.4, mfc=ORANGE, mec='none', zorder=4)
    ll = np.linspace(-3 * dlon, 3 * dlon, 200)
    for p in (55, 60, 65):
        gx, gy = xy(np.full_like(ll, p), ll)
        ax.plot(gx, gy, color=GRAY, lw=0.35, ls=':')
        lam = np.arcsin(x1 / rho(p))
        ax.text(x1 * 1.03, -rho(p) * np.cos(lam), f'{p}', fontsize=8, va='center', ha='left')
    for q in (-10, -5, 0, 5, 10):
        gx, gy = xy(np.array([lat0 - 3, 80.0]), np.array([q, q]))
        ax.plot(gx, gy, color=GRAY, lw=0.35, ls=':')
        ax.text(-y0 * tan(radians(q)), y0 - 0.03 * (y1 - y0), f'{q}', fontsize=8, va='top', ha='center')
    ax.set_xlim(-x1, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, -0.1, 'Longitude, deg', transform=ax.transAxes, ha='center', va='top')
    ax.text(1.115, 0.5, 'Latitude, deg', transform=ax.transAxes, ha='left', va='center', rotation=90)
    # ---- b) distribution of crossover latitudes
    stats = []
    for (name, k, m, inc, col) in [('TOPEX/Poseidon', 127, 10, 66.0, BLUE),
                                   ('Sentinel-3', 385, 27, 98.65, ORANGE),
                                   ('ICESat-2', 1387, 91, 92.0, GREEN)]:
        i_ = radians(inc)
        c, si, ti = cos(i_), sin(i_), abs(tan(i_))
        sg = 1 if c > 0 else -1
        turn = min(i_, pi - i_)
        w = np.array(cx.roots(k, m, c))
        lat = np.sort(np.arctan(ti * np.abs(np.cos(m * w))))       # Eq. (10)
        F = (np.arange(len(lat)) + 0.5) / len(lat)
        law = lambda ph: 2 * (k * np.arcsin(np.clip(np.tan(ph) / ti, 0, 1))
                              - sg * m * np.arcsin(np.clip(np.sin(ph) / si, 0, 1))) / (pi * (k - sg * m))
        bx.plot(lat / turn, F, color=col, lw=1.3, label=name)
        pm = turn * (1 - np.linspace(0, 1, 17)[1:-1]**2)             # markers, denser near the turn
        bx.plot(pm / turn, law(pm), 'o', ms=3.2, mfc='white', mec=col, mew=0.8, zorder=4)
        dev = float(np.abs(F - law(lat)).max())
        stats.append((name, dev, 1 / len(lat)))
        print(f'  dense: {name}: max |F - law| = {dev:.4f}, 2/L = {2/len(lat):.4f}')
    assert all(d <= 2 * l for _, d, l in stats)
    bx.plot([], [], 'o', ms=3.2, mfc='white', mec=INK, mew=0.8, label='Closed form')
    bx.set_xlim(0, 1)
    bx.set_ylim(0, 1)
    bx.set_xlabel(r'$\varphi\,/\,\varphi_{\max}$', labelpad=1)
    bx.set_ylabel('Fraction of crossovers below $|\\varphi|$')
    bx.legend(frameon=False, loc='upper left', handlelength=2.0)
    fig.subplots_adjust(wspace=0.42)
    fig.canvas.draw()
    for a_, lab in ((ax, 'a) Track and crossovers near the turning latitude'), (bx, 'b) Distribution in latitude')):
        fig.text(a_.get_position().x0, -0.06, lab, ha='left', va='top')
    fig.savefig('../figures/fig_dense.pdf')
    plt.close(fig)
    return stats

# --------------------------------------------------------------- Table
def table_criticals(kmax=9):
    lines = []
    for k in range(2, kmax + 1):
        for m in range(1, k):
            if gcd(k, m) != 1:
                continue
            crit = cx.critical_values(k, m)
            ent = []
            for cs, w, kind in crit:
                v = degrees(acos(cs))
                if kind == 'equator':
                    ent.append(rf'{v:.3f}$^{{\rm a}}$')
                elif kind == 'fold':
                    ent.append(f'{v:.3f}')
                elif kind == 'polar':
                    ent.append(r'90$^{\rm b}$')
            Lmin, Lmax = k - m - 1, k + m - 1
            lines.append(rf'{k}:{m} & {Lmin} & {Lmax} & ' + ', '.join(ent) + r' \\')
    half = (len(lines) + 1) // 2
    left, right = lines[:half], lines[half:]
    out = []
    for j, l in enumerate(left):
        r = right[j] if j < len(right) else r' & & & \\'
        out.append(l[:-3].rstrip() + ' & ' + r)
    open('../figures/table_merged.tex', 'w').write('\n'.join(out) + '\n')


def table_missions():
    """Rows of the performance table, from data/missions.json (written by verify.py)."""
    import json
    rows = {r['name']: r for r in json.load(open('../data/missions.json'))}
    sig = lambda x: f'{x:.2f}' if x < 0.995 else (f'{x:.1f}' if x < 9.95 else f'{round(x, -1 if x >= 100 else 0):.0f}')
    out = []
    for label, key in [('Example', '7:3'), ('Example', '24:7'), ('TOPEX/Poseidon, Jason', 'TOPEX/Jason'),
                       ('Sentinel-3', 'Sentinel-3'), ('ICESat-2', 'ICESat-2')]:
        r = rows[key]
        out.append(rf"{label} & {r['k']}:{r['m']} & {r['inc']:.2f} & {r['L']} & {r['N']:,} & "
                   rf"{sig(1e3 * r['t_eig'])} & {sig(1e3 * r['t_interval'])} \\")
    open('../figures/table_missions.tex', 'w').write('\n'.join(out) + '\n')
    print('  ICESat-2, all crossovers with coordinates: %.2f s' % rows['ICESat-2']['t_all'])
    # numbers quoted in the running text, so that they always match the last run
    open('../figures/numbers.tex', 'w').write(
        '\\newcommand{\\ICESatAllTime}{%.2f}\n' % rows['ICESat-2']['t_all'])


if __name__ == '__main__':
    fig_gamma(); fig_staircase(); fig_gallery(); fig_pitchfork()
    fig_critmap(); print('SSO:', fig_sso()); table_criticals()
    fig_conformal(); fig_polar(); fig_dense(); table_missions()
    fig_geometry(); fig_symmetry()
    print('done')
