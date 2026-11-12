"""Independent verification of every statement and number used in the paper.

Run:  python verify.py            (full, a few minutes)
      python verify.py quick      (reduced grids)
Writes results to ../data/verification_log.txt
"""
import sys, time, json
from math import gcd, pi, cos, sin, tan, acos, asin, atan, sqrt, floor, radians, degrees
import numpy as np
from scipy.spatial import cKDTree
import crossovers as cx

QUICK = len(sys.argv) > 1 and sys.argv[1] == 'quick'
LOG = []
FAIL = 0


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def check(name, ok, detail=''):
    global FAIL
    if not ok:
        FAIL += 1
    log(('PASS' if ok else 'FAIL'), name, detail)


# ---------------------------------------------------------------- helpers
def L_grid(k, m, c, n_per=400):
    """Count zeros of E(.,c) in (0,pi) by sign changes on a fine grid
    (independent of the theory; valid away from critical inclinations)."""
    n = n_per * (k + m)
    w = (np.arange(n) + 0.5) * pi / n
    e = cx.E(w, k, m, c)
    return int(np.sum(e[:-1] * e[1:] < 0)) + int(np.sum(e == 0))


def brute_force_3d(k, m, inc_deg, n_per_rev=2000):
    """Count self-intersections of the closed ground track on the unit
    sphere, treating it as a polyline of great-circle arcs.  Uses nothing
    but the definition of the track."""
    inc = radians(inc_deg)
    c, s = cos(inc), sin(inc)
    n = n_per_rev * k
    th = (np.arange(n) + 0.3183098861) * 2 * pi * k / n
    x, y, z = np.cos(th), c * np.sin(th), s * np.sin(th)
    a = -(m / k) * th
    P = np.stack([x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a), z], axis=1)
    Q = np.roll(P, -1, axis=0)
    mid = (P + Q) / 2
    seg = np.linalg.norm(Q - P, axis=1).max()
    tree = cKDTree(mid)
    pairs = tree.query_pairs(r=1.05 * seg, output_type='ndarray')
    i, j = pairs[:, 0], pairs[:, 1]
    d = np.abs(i - j)
    keep = (d > 1) & (d < n - 1)
    i, j = i[keep], j[keep]
    n1 = np.cross(P[i], Q[i])
    n2 = np.cross(P[j], Q[j])
    v = np.cross(n1, n2)
    nv = np.linalg.norm(v, axis=1)
    ok = nv > 0
    v = v[ok] / nv[ok, None]
    i, j, n1, n2 = i[ok], j[ok], n1[ok], n2[ok]

    def inside(v):
        a1 = np.einsum('ij,ij->i', np.cross(P[i], v), n1) >= 0
        a2 = np.einsum('ij,ij->i', np.cross(v, Q[i]), n1) > 0
        b1 = np.einsum('ij,ij->i', np.cross(P[j], v), n2) >= 0
        b2 = np.einsum('ij,ij->i', np.cross(v, Q[j]), n2) > 0
        return a1 & a2 & b1 & b2
    return int(np.sum(inside(v)) + np.sum(inside(-v)))


def coprime_pairs(kmax):
    return [(k, m) for k in range(1, kmax + 1) for m in range(1, kmax + 1)
            if k != m and gcd(k, m) == 1]


t00 = time.time()

# ------------------------------------------------------------------ T1
log('\n== T1: counting theorem vs grid count of zeros of E vs bracketing solver')
KM = 12 if QUICK else 20
incs = np.arange(2.37, 180, 3.91 if QUICK else 1.93)
ncase = nbad = 0
for (k, m) in coprime_pairs(KM):
    crit = cx.critical_values(k, m)
    for inc in incs:
        c = cos(radians(inc))
        if min(abs(c - cs) for cs, _, _ in crit) < 2e-5 if crit else False:
            continue
        Lt = cx.L_theory(k, m, c, crit)
        Lg = L_grid(k, m, c)
        Lr = len(cx.roots(k, m, c, crit))
        ncase += 1
        if not (Lt == Lg == Lr):
            nbad += 1
            if nbad < 10:
                log('  mismatch', k, m, inc, Lt, Lg, Lr)
check('T1 L_theory == L_grid == #roots', nbad == 0, f'{ncase} cases ({len(coprime_pairs(KM))} pairs x {len(incs)} inclinations), k,m<={KM}, {nbad} mismatches')

# ------------------------------------------------------------------ T2
log('\n== T2: critical points of gamma (zeros of G), all coprime pairs')
KM2 = 25 if QUICK else 60
nb = 0
npairs = 0
min_gap = (9e9, None)
min_gap_polar = (9e9, None)
first_is_pitch = True
kim_range = True
type_ok = True
for (k, m) in coprime_pairs(KM2):
    npairs += 1
    z = cx.G_zeros(k, m)
    if len(z) != min(k, m) - 1:
        nb += 1
        log('  G-zero count', k, m, len(z))
    g = [float(cx.gamma(w, k, m)) for w in z]
    V = sum(1 for x in g if 0 < x < 1)
    if V != cx.predicted_fold_count(k, m):
        nb += 1
        log('  V mismatch', k, m, V)
    q = min(k / m, m / k)
    for w, x in zip(z, g):
        if not (0 < x < q or x > 1 / q):
            kim_range = False
        # type: second derivative sign of gamma = sign(S1*S2)
        s12 = sin((k + m) * w) * sin((k - m) * w)
        if (x < 1 and not s12 < 0) or (x > 1 and not s12 > 0):
            type_ok = False
    cv = sorted([cs for cs, _, kind in cx.critical_values(k, m) if kind != 'polar'])
    for a, b in zip(cv[:-1], cv[1:]):
        gap = degrees(acos(a)) - degrees(acos(b))
        if gap < min_gap[0]:
            min_gap = (gap, (k, m))
    if k % 2 == 0 and cv:
        gap = 90.0 - degrees(acos(cv[0]))
        if gap < min_gap_polar[0]:
            min_gap_polar = (gap, (k, m))
check('T2a #zeros of G in (0,pi/2) = min(k,m)-1 and V as predicted', nb == 0, f'{npairs} pairs, k,m<={KM2}')
check('T2b critical values in (0,min(q,1/q)) U (max(q,1/q),inf)', kim_range)
check('T2c value<1 => local max, value>1 => local min', type_ok)
log('  minimum gap between fold/pitchfork inclinations of one pair [deg]:', min_gap)
log('  minimum gap between the last fold and the polar transition (k even) [deg]:', min_gap_polar)

# ------------------------------------------------------------------ T3
log('\n== T3: 3-D brute-force self-intersection count vs k*L')
cases = [(7, 3, 50), (7, 3, 70), (7, 3, 89.3), (7, 3, 110), (8, 3, 60), (8, 3, 86),
         (8, 3, 95), (3, 2, 60), (3, 2, 85), (3, 2, 120), (2, 3, 30), (2, 3, 60),
         (2, 3, 100), (3, 7, 50), (3, 7, 70), (3, 7, 85), (3, 7, 130), (1, 2, 40),
         (1, 2, 75), (5, 2, 45), (5, 2, 88), (9, 5, 40), (9, 5, 62), (9, 5, 80),
         (9, 5, 89), (13, 4, 83), (14, 5, 110), (4, 5, 30), (4, 5, 45), (4, 5, 80),
         (5, 9, 50), (5, 9, 70), (5, 9, 88)]
if not QUICK:
    cases += [(24, 7, 55), (18, 5, 65), (21, 8, 80), (19, 3, 70), (15, 1, 97), (16, 1, 60)]
nb = 0
for (k, m, inc) in cases:
    Lt = cx.L_theory(k, m, cos(radians(inc)))
    nbf = brute_force_3d(k, m, inc, 4000 if k < 10 else 3000)
    if nbf != k * Lt:
        nb += 1
    log(f'  {k}:{m} i={inc}: theory {k*Lt}, brute force {nbf}')
check('T3 brute force == k L', nb == 0, f'{len(cases)} cases')

# ------------------------------------------------------------------ T4
log('\n== T4: every reported event is a genuine double point')
worst = 0
nev = 0
for (k, m, inc) in [(7, 3, 70), (8, 3, 95), (3, 7, 60), (24, 7, 55), (127, 10, 66.0), (5, 9, 88)]:
    lat, lon, W, S = cx.crossovers(k, m, inc)
    i_ = radians(inc)
    for sign in (-1, 1):
        th = pi / 2 + pi * S + sign * k * W
        x, y, z = np.cos(th), cos(i_) * np.sin(th), sin(i_) * np.sin(th)
        a = -(m / k) * th
        Pp = np.stack([x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a), z], 1)
        if sign == -1:
            P1 = Pp
        else:
            worst = max(worst, np.abs(P1 - Pp).max())
    nev += len(lat)
    # distinct events <-> distinct points (i != 90)
    pts = np.round(np.stack([lat, lon], 1), 6)
    assert len(np.unique(pts, axis=0)) == len(lat), (k, m, inc)
    assert len(lat) == k * cx.L_theory(k, m, cos(i_))
check('T4 coincidence of the two passes', worst < 1e-9, f'{nev} events, max |r1-r2| = {worst:.2e}')

# ------------------------------------------------------------------ T5
log('\n== T5: 19:3 near its first critical inclination')
ic = degrees(acos(3 / 19))
for inc in (80.5, ic, 81.0):
    c = cos(radians(inc))
    Lt = cx.L_theory(19, 3, c)
    log(f'  i={inc:.4f}: L={Lt}, N_tot={19*Lt}, #roots={len(cx.roots(19,3,c))}')
check('T5a count at arccos(3/19) is 285', 19 * cx.L_theory(19, 3, 3 / 19) == 285)
check('T5b count at 81.000 deg is 323', 19 * cx.L_theory(19, 3, cos(radians(81))) == 323)
for npr in (2000, 8000, 32000):
    log(f'  3-D brute force at 81 deg, {npr} samples/rev:', brute_force_3d(19, 3, 81.0, npr))
bf = brute_force_3d(19, 3, 81.0, 32000)
check('T5c brute force at 81 deg == 323', bf == 323, str(bf))

# ------------------------------------------------------------------ T6
log('\n== T6: polar inclination, i = 90 deg exactly')
nb = 0
for (k, m) in coprime_pairs(12):
    r = cx.roots(k, m, 0.0)
    if len(r) != k + m - 1 - (1 if k % 2 == 0 else 0):
        nb += 1
check('T6 L(90) = k+m-1-[k even]', nb == 0)

# ------------------------------------------------------------------ T6b
log('\n== T6b: duality L_{k,m} = L_{m,k} for k,m both odd')
nb = 0
for (k, m) in coprime_pairs(25):
    if k % 2 and m % 2 and k > m:
        for inc in np.arange(1.7, 180, 2.3):
            c = cos(radians(inc))
            if cx.L_theory(k, m, c) != cx.L_theory(m, k, c) or L_grid(k, m, c) != L_grid(m, k, c):
                nb += 1
check('T6b odd-odd duality', nb == 0)

# ------------------------------------------------------------------ T7
log('\n== T7: zero-crossover thresholds (|k-m|=1)')
ok = True
for m in range(1, 40):
    k = m + 1
    cv = [cs for cs, _, kind in cx.critical_values(k, m)]
    cmax = max(cv)
    tau = np.linspace(1 / (k + m), 1.5 / (k + m), 200001)[1:-1]
    al = max(0.0, np.max(np.tan(pi * k * tau) / np.tan(pi * m * tau)))
    if abs(al - cmax) > 1e-9:
        ok = False
    if m in (1, 2, 4, 7):
        log(f'  k={k}, m={m}: cos i* = {cmax:.15f}  (AL max over tau window: {al:.12f})')
    k2, m2 = m, m + 1
    cv2 = [cs for cs, _, kind in cx.critical_values(k2, m2) if kind != 'polar']
    if abs(max(cv2) - k2 / m2) > 1e-15:
        ok = False
check('T7 thresholds agree with Arnas-Linares Theorem 2', ok)
ok = True
for k in range(1, 30):
    m = k + 1
    c = k / m
    if cx.L_theory(k, m, c) != 0 or L_grid(k, m, c) != 0 or L_grid(k, m, c - 1e-3) != 2:
        ok = False
check('T7b k=m-1: no crossovers at cos i = k/m itself, two crossover latitudes just beyond', ok)
log('  (3,2): cos i* =', max(cs for cs, _, kd in cx.critical_values(3, 2) if kd != 'polar'), ' 1/9 =', 1 / 9)

# ------------------------------------------------------------------ T9
log('\n== T9: eigenvalue method vs bracketing method')
nb = ncase = 0
worst = 0.0
for (k, m) in coprime_pairs(12 if QUICK else 20):
    crit = cx.critical_values(k, m)
    fe = cx.critical_values_eig(k, m)
    fb = np.array(sorted([cs for cs, _, kd in crit if kd == 'fold'], reverse=True))
    if len(fe) != len(fb) or (len(fb) and np.abs(fe - fb).max() > 1e-9):
        nb += 1
    for inc in np.arange(2.37, 180, 5.77):
        c = cos(radians(inc))
        if crit and min(abs(c - cs) for cs, _, _ in crit) < 2e-5:
            continue
        rb = np.array(cx.roots(k, m, c, crit))
        re_ = cx.roots_eig(k, m, c)
        ncase += 1
        if len(rb) != len(re_):
            nb += 1
        elif len(rb):
            worst = max(worst, np.abs(rb - re_).max())
check('T9 eigenvalue roots == bracketing roots', nb == 0, f'{ncase} cases, max |dw| = {worst:.1e}')

# ------------------------------------------------------------------ T10
log('\n== T10: maximum of the de Dilectis-Mortari function f at the fold inclinations')
worst = 0.0
nfold = npairs10 = 0
count_ok = True
for k in range(2, 26):
    for m in range(1, k):
        if gcd(k, m) != 1:
            continue
        npairs10 += 1
        folds = [cs for cs, _, kd in cx.critical_values(k, m) if kd == 'fold']
        vals = []
        for c in folds:
            # interior stationary point of f: tan^2 u = (m - k c) / (c (k - m c))
            u = atan(sqrt((m - k * c) / (c * (k - m * c))))
            fm = (k - m) / 2 + (m * u - k * atan(c * tan(u))) / pi
            worst = max(worst, abs(fm - round(fm)))
            vals.append(round(fm))
            nfold += 1
        # the integers strictly between (k-m)/2 and k/2, each attained once
        expect = [j for j in range(0, k + 1) if (k - m) / 2 < j < k / 2]
        if sorted(vals) != expect:
            count_ok = False
check('T10a max of f is an integer at every fold inclination', worst < 1e-12,
      f'{npairs10} pairs with m<k<=25, {nfold} folds, max deviation {worst:.1e}')
check('T10b those integers are exactly the ones strictly between (k-m)/2 and k/2', count_ok)

# ------------------------------------------------------------------ T11
log('\n== T11: L is the number of distinct crossover latitudes (i != 90 deg)')
nb = ncase = 0
for (k, m) in coprime_pairs(10 if QUICK else 15):
    for inc in np.arange(3.1, 180, 7.3):
        c = cos(radians(inc))
        r = np.array(cx.roots(k, m, c))
        lat = sin(radians(inc)) * np.cos(k * r)
        nlat = len(set(np.round(np.concatenate([lat, -lat]), 8)))
        ncase += 1
        if nlat != len(r):
            nb += 1
check('T11 #distinct latitudes == L', nb == 0, f'{ncase} cases')

# ------------------------------------------------------------------ T12
log('\n== T12: latitude formula, crossing angle, tangency latitude at the folds')
def _dir(k, m, c, si, th, h=1e-6):
    """east and north components of d(subsatellite point)/d(theta), by differences"""
    lam = lambda t: np.arctan2(c * np.sin(t), np.cos(t)) - (m / k) * t
    phi = lambda t: np.arcsin(si * np.sin(t))
    dl = (lam(th + h) - lam(th - h) + pi) % (2 * pi) - pi
    return np.cos(phi(th)) * dl / (2 * h), (phi(th + h) - phi(th - h)) / (2 * h)
w_lat = w_ang = w_sym = 0.0
ncr = 0
for (k, m, inc) in [(7, 3, 70), (7, 3, 89.3), (8, 3, 95), (3, 8, 75), (24, 7, 55), (5, 9, 88), (127, 10, 66.0)]:
    i_ = radians(inc)
    c, si, q = cos(i_), sin(i_), m / k
    lat, lon, W, S = cx.crossovers(k, m, inc)
    ph = np.radians(lat)
    # |tan(lat)| = |tan i| |cos(m w)|
    w_lat = max(w_lat, np.abs(np.abs(np.tan(ph)) - abs(tan(i_)) * np.abs(np.cos(m * W))).max())
    e1, n1 = _dir(k, m, c, si, pi / 2 + pi * S - k * W)
    e2, n2 = _dir(k, m, c, si, pi / 2 + pi * S + k * W)
    ok = np.abs(np.cos(ph)**2 - c * c) > 1e-6      # away from the turning latitude
    w_sym = max(w_sym, np.abs(e1 - e2)[ok].max(), np.abs(n1 + n2)[ok].max())
    alpha = np.arctan2(np.abs(c - q * np.cos(ph)**2), np.sqrt(np.clip(np.cos(ph)**2 - c * c, 0, None)))
    w_ang = max(w_ang, np.abs(np.arctan2(np.abs(e1), np.abs(n1)) - alpha)[ok].max())
    ncr += len(lat)
check('T12a |tan lat| = |tan i| |cos mw| at every crossover', w_lat < 1e-8, f'{ncr} crossovers, max error {w_lat:.1e}')
check('T12b the two passes have equal east and opposite north components', w_sym < 1e-6, f'max error {w_sym:.1e}')
check('T12c angle to the meridian: tan(alpha) = |c - q cos^2(lat)| / sqrt(cos^2(lat) - c^2)', w_ang < 1e-6, f'max error {w_ang:.1e} rad')
worst = 0.0
nf = 0
for (k, m) in coprime_pairs(12 if QUICK else 30):
    for cs, w, kd in cx.critical_values(k, m):
        if kd == 'fold':
            sin2 = (1 - cs * cs) * cos(k * w)**2          # sin^2(lat) of the tangency
            worst = max(worst, abs((1 - sin2) - cs * k / m))
            nf += 1
check('T12d tangency latitude at a fold: cos^2(lat) = (k/m) cos i_j', worst < 1e-10, f'{nf} folds, max error {worst:.1e}')

# ------------------------------------------------------------------ T13
log('\n== T13: number of crossover latitudes below a latitude (k > m, regimes of Corollary 1)')
def lat_count_error(k, m, inc):
    """max |n(phi) - closed form| over the crossover latitudes, where n(phi) is the number of
    zeros with |latitude| <= phi and the closed form is 1 + 2 floor(X) (k, m odd) or
    2 floor(X + 1/2), X = (k Lam - s m u)/pi, s = sign(cos i)."""
    i_ = radians(inc)
    c, si, ti = cos(i_), sin(i_), abs(tan(i_))
    sg = 1 if c > 0 else -1
    w = np.array(cx.roots(k, m, c))
    lat = np.sort(np.arctan(ti * np.abs(np.cos(m * w))))
    bad = 0
    # evaluate between consecutive distinct latitudes
    if len(lat) < 2:
        return 0, 0
    u_ = [lat[0]]
    for v in lat[1:]:                 # mirror zeros share a latitude: merge them
        if v - u_[-1] > 1e-9:
            u_.append(v)
    u_ = np.array(u_)
    mids = (u_[:-1] + u_[1:]) / 2
    for ph in mids:
        n = int(np.sum(lat <= ph))
        X = (k * asin(min(1.0, tan(ph) / ti)) - sg * m * asin(min(1.0, sin(ph) / si))) / pi
        pred = 1 + 2 * floor(X) if (k % 2 and m % 2) else 2 * floor(X + 0.5)
        bad += (n != pred)
    return bad, len(mids)
nbad = ntest = 0
for (k, m) in coprime_pairs(12 if QUICK else 25):
    if k < m:
        continue
    for inc in list(np.arange(2.5, 180, 8.3)):
        c = cos(radians(inc))
        if 0 <= c < m / k + 1e-9:        # outside the two regimes
            continue
        b_, n_ = lat_count_error(k, m, inc)
        nbad += b_
        ntest += n_
for (k, m, inc) in [(127, 10, 66.0), (385, 27, 98.65), (1387, 91, 92.0)]:
    b_, n_ = lat_count_error(k, m, inc)
    nbad += b_
    ntest += n_
check('T13a count of crossover latitudes below a latitude equals the closed form', nbad == 0,
      f'{ntest} latitude intervals, {nbad} mismatches')
ok = True
for name, k, m, inc in [('TOPEX/Jason', 127, 10, 66.0), ('Sentinel-3', 385, 27, 98.65), ('ICESat-2', 1387, 91, 92.0)]:
    i_ = radians(inc)
    c, si, ti = cos(i_), sin(i_), abs(tan(i_))
    sg = 1 if c > 0 else -1
    w = np.array(cx.roots(k, m, c))
    lat = np.sort(np.arctan(ti * np.abs(np.cos(m * w))))
    F = (np.arange(len(lat)) + 0.5) / len(lat)
    Flaw = 2 * (k * np.arcsin(np.clip(np.tan(lat) / ti, 0, 1)) - sg * m * np.arcsin(np.clip(np.sin(lat) / si, 0, 1))) / (pi * (k - sg * m))
    Fasin = 2 / pi * np.arcsin(np.clip(np.tan(lat) / ti, 0, 1))
    from scipy.optimize import brentq as _bq
    g = lambda ph: 2 * (k * asin(min(1.0, tan(ph) / ti)) - sg * m * asin(min(1.0, sin(ph) / si))) / (pi * (k - sg * m)) - 0.5
    med_law = degrees(_bq(g, 0.0, min(i_, pi - i_) - 1e-12))
    q = m / k
    h = lambda ph: abs(c - q * cos(ph)**2) - sqrt(max(cos(ph)**2 - c * c, 0.0))
    lat45 = degrees(_bq(h, 0.0, min(i_, pi - i_) - 1e-9))
    log(f'  {name}: max |F - law| = {np.abs(F - Flaw).max():.4f} (1/L = {1/len(lat):.4f}); '
        f'without the q-term {np.abs(F - Fasin).max():.3f}; median latitude {degrees(np.median(lat)):.2f} deg, '
        f'law {med_law:.2f} deg; passes perpendicular at latitude {lat45:.2f} deg; '
        f'alpha at the equator {degrees(atan(abs(c - q) / si)):.2f} deg')
    ok = ok and np.abs(F - Flaw).max() <= 2.0 / len(lat)
check('T13b fraction of crossovers below a latitude within 2/L of the law', ok)

# ------------------------------------------------------------------ T14
log('\n== T14: exactly critical inclinations and their two sides, all types')
nbad = ncase = 0
def _consistent(k, m, c, grid=True):
    """roots, crossover events and the counting law agree; roots distinct"""
    r = cx.roots(k, m, c)
    Lt = cx.L_theory(k, m, c)
    okc = (len(r) == Lt) and all(b - a > 1e-9 for a, b in zip(r[:-1], r[1:]))
    if grid:
        okc = okc and (L_grid(k, m, c) == Lt)
    if abs(c) < 1:
        lat, lon, W, S = cx.crossovers(k, m, degrees(acos(c)))
        okc = okc and (len(lat) == k * Lt)
    return okc
for (k, m) in coprime_pairs(9 if QUICK else 14):
    for cs, w, kind in cx.critical_values(k, m):
        # at the critical value itself (no grid count: multiple or boundary roots)
        ncase += 1
        nbad += not _consistent(k, m, cs, grid=False)
        for d in (1e-3, 1e-6, 1e-9):
            for cc in (cs + d, cs - d):
                if -1 < cc < 1:
                    ncase += 1
                    nbad += not _consistent(k, m, cc, grid=(d >= 1e-3))   # a grid cannot resolve closer roots
check('T14a roots == events/k == counting law at and around every critical inclination', nbad == 0,
      f'{ncase} cases, k,m<={9 if QUICK else 14}, {nbad} mismatches')
# the cases reported as failing before the solver was made limit-based
okc = (len(cx.roots(7, 3, 3 / 7)) == 3 and len(cx.roots(3, 7, 3 / 7)) == 3
       and len(cx.roots_eig(2, 3, 2 / 3)) == 0 and len(cx.crossovers(7, 3, degrees(acos(3 / 7)))[0]) == 21
       and len(cx.crossovers(3, 7, degrees(acos(3 / 7)))[0]) == 9)
check('T14b pitchfork and cusp inclinations: no spurious roots', okc)
okc = (cx.L_theory(1, 1, 0.3) == 1 and cx.roots(1, 1, -0.4) == [pi / 2] and brute_force_3d(1, 1, 50.0) == 1
       and brute_force_3d(1, 1, 120.0) == 1)
check('T14c 1:1 track: one crossover at every inclination (theory, solver, brute force)', okc)
# 8:3 at exactly 90 deg: pairs versus distinct geographical points
lat, lon, W, S = cx.crossovers(8, 3, 90.0)
polar = np.abs(np.abs(lat) - 90) < 1e-9
npts = 2 + len(np.unique(np.round(np.stack([lat[~polar], lon[~polar]], 1), 6), axis=0))
log(f'  8:3 at 90 deg: {len(lat)} crossovers (pairs of passes), {int(polar.sum())} of them at the poles; '
    f'{npts} distinct points')
check('T14d polar orbit: 72 pairs, 56 at the poles, 18 distinct points for 8:3', (len(lat), int(polar.sum()), npts) == (72, 56, 18))

# ------------------------------------------------------------------ T15
log('\n== T15: symmetries of the track and fundamental crossovers')
def _wrap(x):
    return (x + 180.0) % 360.0 - 180.0
def _same_set(lat1, lon1, lat2, lon2, tol=1e-7):
    """Two sets of points on the sphere coincide (as sets)."""
    xyz = lambda la, lo: np.stack([np.cos(np.radians(la)) * np.cos(np.radians(lo)),
                                   np.cos(np.radians(la)) * np.sin(np.radians(lo)),
                                   np.sin(np.radians(la))], 1)
    a, b = xyz(lat1, lon1), xyz(lat2, lon2)
    return len(a) == len(b) and cKDTree(a).query(b)[0].max() < tol and cKDTree(b).query(a)[0].max() < tol
nsym = nbad = nfund = nfbad = 0
worst = 0.0
for (k, m) in coprime_pairs(12):
    for inc in (20.0, 47.0, 63.4, 75.0, 86.0, 97.0, 120.0, 160.0):
        c, si = cos(radians(inc)), sin(radians(inc))
        # (a) pointwise: theta+2pi rotates by -2pi m/k; theta+pi reflects and rotates by pi(1-m/k)
        th = np.linspace(0.0, 2 * pi * k, 997)
        P = lambda t: (np.degrees(np.arcsin(si * np.sin(t))),
                       np.degrees(np.arctan2(c * np.sin(t), np.cos(t)) - (m / k) * t))
        la0, lo0 = P(th); la1, lo1 = P(th + 2 * pi); la2, lo2 = P(th + pi)
        e1 = max(np.abs(la1 - la0).max(), np.abs(_wrap(lo1 - lo0 + 360.0 * m / k)).max())
        e2 = max(np.abs(la2 + la0).max(), np.abs(_wrap(lo2 - lo0 - 180.0 * (1 - m / k))).max())
        worst = max(worst, e1, e2)
        # (b) the crossover set is invariant under the rotation of 2pi/k and under the
        #     reflection in the equator, combined with a rotation of pi/k unless k, m are odd
        lat, lon, W, S = cx.crossovers(k, m, inc)
        if len(lat) == 0:
            continue
        shift = 0.0 if (k % 2 and m % 2) else 180.0 / k
        ok = _same_set(lat, lon, lat, lon + 360.0 / k) and _same_set(lat, lon, -lat, lon + shift)
        nsym += 1; nbad += (not ok)
        # (c) a sector of 2pi/k of the northern hemisphere, equator included, holds exactly
        #     one crossover per zero of E in (0, pi/2]
        nz = sum(1 for w in cx.roots(k, m, c) if w <= pi / 2 + 1e-12)
        for lam_a in (-171.3, -41.234, 3.7, 111.1):   # no crossover on a sector boundary
            ins = (lat > -1e-9) & (np.mod(lon - lam_a, 360.0) < 360.0 / k)
            dlat = np.abs(lat[ins][:, None] - lat[ins][None, :]) + np.eye(int(ins.sum()))
            good = (int(ins.sum()) == nz) and (ins.sum() < 2 or dlat.min() > 1e-7)
            nfund += 1; nfbad += (not good)
check('T15a P(theta+2pi) and P(theta+pi) are the stated rotation and rotoreflection', worst < 1e-9, f'max error {worst:.1e} deg')
check('T15b crossover set invariant under the 2k symmetries of the track', nbad == 0, f'{nsym} cases')
check('T15c one crossover per zero of E in (0,pi/2] in every fundamental sector', nfbad == 0, f'{nfund} sectors')
# (d) crossover meridians (King 1975): longitudes are multiples of pi/(2k) from the reference
#     node, odd multiples if k - m is odd and even ones if k and m are odd; and King's Eq. (7):
#     on the ascending pass, atan(c tan th') - (m/k) th' is a multiple of pi/(2k)
nmer = 0; emer = eking = 0.0; badpar = 0
for (k, m) in coprime_pairs(12):
    for inc in (20.0, 47.0, 63.4, 75.0, 86.0, 97.0, 120.0, 160.0):
        lat, lon, W, S = cx.crossovers(k, m, inc)
        if len(lat) == 0:
            continue
        r = lon * 2 * k / 180.0
        ri = np.round(r)
        emer = max(emer, float(np.abs(r - ri).max())); nmer += len(lat)
        badpar += int(np.sum(np.mod(ri, 2) != (k - m) % 2))
        badpar += int(len(set(np.mod(ri, 4 * k).astype(int))) > 2 * k)
        c = cos(radians(inc))
        th1 = pi / 2 + pi * S - k * W; th2 = pi / 2 + pi * S + k * W
        thp = (np.where(np.cos(th1) > 0, th1, th2) + pi) % (2 * pi) - pi
        x = (np.arctan(c * np.tan(thp)) - (m / k) * thp) * 2 * k / pi
        eking = max(eking, float(np.abs(x - np.round(x)).max()))
check('T15d crossovers lie on 2k meridians pi/k apart; King (1975) Eq. (7) holds', emer < 1e-9 and eking < 1e-9 and badpar == 0,
      f'{nmer} crossovers, max error {max(emer, eking):.1e} (units of pi/2k)')

# ------------------------------------------------------------------ T8
log('\n== T8: missions and timings')
def _t(f):
    a = time.perf_counter(); f(); return time.perf_counter() - a
_t(lambda: cx.roots_eig(7, 3, 0.3))
rows = []
for name, k, m, inc in [('TOPEX/Jason', 127, 10, 66.0), ('Sentinel-3', 385, 27, 98.65),
                        ('ICESat-2', 1387, 91, 92.0), ('7:3', 7, 3, 70.0), ('24:7', 24, 7, 55.0), ('Sentinel-2', 143, 10, 98.57),
                        ('Landsat-8/9', 233, 16, 98.2)]:
    NREP = 5   # every time below is the best of NREP repetitions
    crit = cx.critical_values(k, m)
    r = cx.roots(k, m, cos(radians(inc)), crit)
    lat, lon, W, S = cx.crossovers(k, m, inc)
    tc = min(_t(lambda: cx.critical_values(k, m)) for _ in range(NREP))
    tr = min(_t(lambda: cx.roots(k, m, cos(radians(inc)), crit)) for _ in range(NREP))
    ta = min(_t(lambda: cx.crossovers(k, m, inc)) for _ in range(NREP))
    te = min(_t(lambda: cx.roots_eig(k, m, cos(radians(inc)))) for _ in range(NREP))
    Lt = cx.L_theory(k, m, cos(radians(inc)), crit)
    assert len(r) == Lt and len(lat) == k * Lt
    nfold = sum(1 for cs, _, kd in crit if kd == 'fold')
    first = max((cs for cs, _, kd in crit if kd != 'polar'), default=None)
    rows.append(dict(name=name, k=k, m=m, inc=inc, L=Lt, N=k * Lt, nfold=nfold,
                     first_crit_deg=degrees(acos(first)),
                     t_crit=tc, t_roots=tr, t_interval=tc + tr, t_all=ta, t_eig=te,
                     latmin=float(np.abs(lat).min()), latmax=float(np.abs(lat).max())))
    log('  ', rows[-1])
json.dump(rows, open('../data/missions.json', 'w'), indent=1)

log(f'\nTOTAL: {FAIL} failed checks; elapsed {time.time()-t00:.0f} s')
open('../data/verification_log.txt', 'w').write('\n'.join(LOG) + '\n')
sys.exit(1 if FAIL else 0)
