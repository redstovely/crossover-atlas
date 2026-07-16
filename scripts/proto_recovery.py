import numpy as np

def master_residual(u1, r, k, m, i):
    lhs = np.tan(np.pi/2 + (u1 - np.pi/2)*m/k - r*np.pi/k)
    rhs = np.tan(u1)*np.cos(i)
    return abs(np.arctan(lhs) - np.arctan(rhs))  # compare via atan to tame branches

def basic_crossovers(k, m, i):
    """Return list of (phi, lam, u1, r) for basic crossovers (phi>=0)."""
    ci = np.cos(i)
    odd = (k % 2 == 1) and (m % 2 == 1)
    out = []
    if odd:
        F = lambda z: (1-ci)*np.sin((k+m)*z) - (1+ci)*np.sin((k-m)*z)
        # z=0: equatorial crossover
        out.append(0.0)
        zs = np.linspace(1e-10, np.pi/2 - 1e-10, 40001)
        f = F(zs)
        roots = []
        for j in range(len(zs)-1):
            if np.sign(f[j]) != np.sign(f[j+1]):
                a, b = zs[j], zs[j+1]
                for _ in range(60):
                    c = 0.5*(a+b)
                    if np.sign(F(a)) != np.sign(F(c)): b = c
                    else: a = c
                roots.append(0.5*(a+b))
        out += roots
        pts = []
        for z in out:
            x = k*z
            r = int(np.floor(x/np.pi + 1e-12))
            u1 = x - r*np.pi
            phi = np.arcsin(np.sin(u1)*np.sin(i))
            lam_u = np.arctan(np.tan(u1)*ci) if abs(np.cos(u1))>1e-12 else np.sign(np.tan(u1)*ci)*np.pi/2
            # lam_u must be in same quadrant as u1 (prograde) / opposite (retrograde): use atan2 form
            lam_u = np.arctan2(np.sin(u1)*ci, np.cos(u1))
            lam = lam_u - (m/k)*u1
            res = master_residual(u1, r, k, m, i)
            pts.append((np.degrees(phi), np.degrees(lam), u1, r, res))
        return pts, 'odd'
    else:
        F = lambda w: (1-ci)*np.sin((k+m)*w) + (1+ci)*np.sin((k-m)*w)
        ws = np.linspace(1e-10, np.pi/2 - 1e-10, 40001)
        f = F(ws)
        roots = []
        for j in range(len(ws)-1):
            if np.sign(f[j]) != np.sign(f[j+1]):
                a, b = ws[j], ws[j+1]
                for _ in range(60):
                    c = 0.5*(a+b)
                    if np.sign(F(a)) != np.sign(F(c)): b = c
                    else: a = c
                roots.append(0.5*(a+b))
        pts = []
        for w in roots:
            x = k*w - np.pi/2   # u1 + r*pi = k*w - pi/2
            r = int(np.floor(x/np.pi + 1e-12))
            u1 = x - r*np.pi
            if u1 < 0: u1 += np.pi; r -= 1
            phi = np.arcsin(np.sin(u1)*np.sin(i))
            lam_u = np.arctan2(np.sin(u1)*ci, np.cos(u1))
            lam = lam_u - (m/k)*u1
            res = master_residual(u1, r, k, m, i)
            pts.append((np.degrees(phi), np.degrees(lam), u1, r, res))
        return pts, 'even'

def track_points(k, m, i, N=20000):
    u = np.linspace(0, 2*np.pi*k, N)
    phi = np.degrees(np.arcsin(np.clip(np.sin(u)*np.sin(i),-1,1)))
    lam_u = np.unwrap(np.arctan2(np.sin(u)*np.cos(i), np.cos(u)))
    lam = np.degrees(lam_u - (m/k)*u)
    lam = (lam + 180) % 360 - 180
    return lam, phi

def dist_to_track(phi0, lam0, k, m, i):
    lam, phi = track_points(k, m, i, 400000)
    dl = np.abs((lam - lam0 + 180) % 360 - 180)
    dp = np.abs(phi - phi0)
    return np.min(np.hypot(dl, dp))

# verify: crossover points must lie ON the track (distance ~ 0) and residual small
def main():
    for (k, m, ideg) in [(7,3,80),(7,3,44.5),(8,3,70),(15,1,98),(13,4,83),(24,7,55),(5,3,120)]:
        i = np.radians(ideg)
        pts, case = basic_crossovers(k, m, i)
        worst = 0
        for (phi, lam, u1, r, res) in pts:
            d = dist_to_track(phi, lam, k, m, i)
            worst = max(worst, d)
        print(f"k={k} m={m} i={ideg} case={case}: {len(pts)} basic, worst dist-to-track={worst:.5f} deg, max residual={max(p[4] for p in pts):.2e}")
        # also verify each copy: (phi, lam + j*360/k) and mirror (-phi, -lam)
        phi0, lam0 = pts[-1][0], pts[-1][1]
        dcopy = dist_to_track(phi0, (lam0 + 360.0/k + 180)%360-180, k, m, i)
        dmir = dist_to_track(-phi0, (-lam0 + 180)%360-180, k, m, i)
        print(f"   copy shift 360/k: dist={dcopy:.5f}  mirror(-phi,-lam): dist={dmir:.5f}")

if __name__ == "__main__":
    main()
