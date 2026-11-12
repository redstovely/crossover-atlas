"""Checks the JavaScript core of the interactive tool against crossovers.py.

The math core of the tool (../index.html, or ../tool/crossover-atlas.html; root finder, counting law and
crossover coordinates) is extracted and run with Node.js on every coprime
pair with k, m <= KMAX at eight generic inclinations, at every critical
inclination, and at 1e-3 and 1e-6 deg on both sides of each.  The count from
the law, the number of crossovers and their positions on the sphere must
agree with the Python implementation.  Requires `node` in the PATH.

Usage:  python check_tool.py [path-to-html]
"""
import json
import subprocess
import sys
import tempfile
from math import acos, cos, degrees, gcd, radians
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import crossovers as cx

KMAX = 14
HARNESS = r"""
const cases = JSON.parse(require('fs').readFileSync(process.argv[2]));
process.stdout.write(JSON.stringify(cases.map(([k, m, inc]) => {
  const r = allCrossovers(k, m, inc);
  return {L: countLaw(k, m, inc), pts: r.pts.map(p => [p.phi, p.lam]),
          crit: criticalInclinations(k, m).map(c => c.ideg)};
})));
"""


def js_core(html):
    lines = Path(html).read_text(encoding='utf-8').splitlines()
    a = next(i for i, s in enumerate(lines) if s.startswith('function gcd(a,b)'))
    b = next(i for i, s in enumerate(lines) if s.startswith('/* ground track polylines'))
    return 'const D2R = Math.PI/180, R2D = 180/Math.PI;\n' + '\n'.join(lines[a:b]) + HARNESS


def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)


def main(html):
    cases = [(1, 1, inc, 'generic') for inc in (30.0, 90.0, 140.0)]
    for k in range(1, KMAX + 1):
        for m in range(1, KMAX + 1):
            if k == m or gcd(k, m) != 1:
                continue
            cases += [(k, m, inc, 'generic') for inc in (20.0, 47.0, 63.4, 75.0, 86.0, 97.0, 120.0, 160.0)]
            for cs, _, kind in cx.critical_values(k, m):
                ic = degrees(acos(cs))
                cases.append((k, m, ic, 'critical'))
                cases += [(k, m, ic + s * d, 'near-critical') for d in (1e-3, 1e-6) for s in (-1, 1)]
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / 'core.js').write_text(js_core(html))
        (Path(tmp) / 'cases.json').write_text(json.dumps([c[:3] for c in cases]))
        res = json.loads(subprocess.run(['node', str(Path(tmp) / 'core.js'), str(Path(tmp) / 'cases.json')],
                                        capture_output=True, text=True, check=True).stdout)
    bad, total = [], {}
    for (k, m, inc, tag), r in zip(cases, res):
        total[tag] = total.get(tag, 0) + 1
        lat, lon, _, _ = cx.crossovers(k, m, inc)
        ok = r['L'] == cx.L_theory(k, m, cos(radians(inc))) and len(r['pts']) == len(lat)
        if ok and len(lat):
            P = np.array(r['pts'])
            a, b = xyz(lat, lon), xyz(P[:, 0], P[:, 1])
            ok = cKDTree(a).query(b)[0].max() < 1e-6 and cKDTree(b).query(a)[0].max() < 1e-6
        if ok and tag == 'generic' and k != m:
            crit = sorted(degrees(acos(cs)) for cs, _, _ in cx.critical_values(k, m))
            ok = len(crit) == len(r['crit']) and all(abs(x - y) < 1e-9 for x, y in zip(crit, sorted(r['crit'])))
        if not ok:
            bad.append((k, m, inc, tag, r['L'], len(r['pts']), len(lat)))
    print('tool vs crossovers.py:', ', '.join(f'{n} {t}' for t, n in total.items()), '->',
          'PASS' if not bad else f'FAIL ({len(bad)})')
    for row in bad[:20]:
        print('  ', row)
    return 1 if bad else 0


if __name__ == '__main__':
    root = Path(__file__).resolve().parent.parent
    candidates = [root / 'index.html', root / 'tool' / 'crossover-atlas.html']
    default = next((c for c in candidates if c.exists()), candidates[0])
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else default))
