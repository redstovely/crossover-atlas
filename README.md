# Crossover Atlas

[![verify](https://github.com/redstovely/crossover-atlas/actions/workflows/ci.yml/badge.svg)](https://github.com/redstovely/crossover-atlas/actions/workflows/ci.yml)

Exact crossover counts, critical inclinations and crossover locations of
circular repeat ground tracks. Companion code, data and interactive tool for

> R. Vázquez, E. Tesón and A. Montero Miñán, *Exact Crossover Counts and
> Critical Inclinations of Repeat Ground Tracks* (submitted).

A satellite whose ground track repeats after `k` revolutions and `m` nodal
days crosses its own track at `N = k·L` points. This package gives `L` in
closed form for every coprime `(k, m)` and every inclination, and computes
every crossover.

**Live tool:** <https://redstovely.github.io/crossover-atlas/>

## Interactive tool

[`index.html`](index.html) is a self-contained browser application: no
installation, no server, no network access (coastlines derived from
[Natural Earth](https://www.naturalearthdata.com/), public domain). For any
`(k, m, i)` it draws the track and its crossovers on the map and on the polar
caps (polar stereographic projection), shows the count given by the counting
law and the critical inclinations, sweeps the inclination, and exports the
crossover catalogue as CSV.

## Python

```bash
pip install -r requirements.txt
```

```python
import crossovers as cx                      # in code/
lat, lon, w, s = cx.crossovers(127, 10, 66.0)    # TOPEX/Poseidon: 14 732 crossovers
cx.critical_inclinations_deg(7, 3)               # equatorial pitchfork at 64.62 deg, fold at 88.27 deg
cx.L_theory(1387, 91, -0.0349)                   # counting law: 1477 (ICESat-2)
cx.roots_eig(7, 3, 0.342)                        # all roots from one eigenvalue problem
```

| File | Purpose |
|---|---|
| `code/crossovers.py` | counting law, critical inclinations, eigenvalue and interval root finders, crossover coordinates |
| `code/verify.py` | independent checks: 3-D brute-force self-intersections, grid counts, structure of the crossover locus, symmetries, latitude and crossing-angle formulas |
| `code/check_tool.py` | checks the JavaScript core of `index.html` against `crossovers.py` (needs Node.js) |
| `code/figures.py` | regenerates every figure and table of the paper into `figures/` |
| `code/sso.py` | catalogue of repeat sun-synchronous orbits (`data/sso_catalogue.csv`) |
| `tests/` | fast tests (`pytest`) |

## Reproducing the paper

Run inside `code/`:

```bash
python sso.py && python verify.py && python figures.py && python check_tool.py
```

`verify.py` takes about half a minute and writes `data/verification_log.txt`;
every check must report `PASS`. `figures.py` writes the figures and tables
into `figures/` (not committed). The timings in `data/missions.json` depend
on the machine.

## Earlier version

The package published in July 2026, which accompanied an earlier draft and
included exploratory scripts for eccentric orbits, is preserved at the tag
[`july-2026`](https://github.com/redstovely/crossover-atlas/tree/july-2026).
Eccentric orbits are outside the scope of the paper and of this version.

## License

MIT. See `LICENSE` and `CITATION.cff`.
