# Crossover Atlas — reproducibility package

[![verify](https://github.com/redstovely/crossover-atlas/actions/workflows/ci.yml/badge.svg)](https://github.com/redstovely/crossover-atlas/actions/workflows/ci.yml)

Companion code, data, and interactive tool for:

> R. Vázquez, E. Tesón Muñoz, A. Montero Miñán, "Crossover Points of
> Satellite Repeat Ground Tracks: Complete Counts, Critical Inclinations,
> and the Eccentric Case," in preparation for the *Journal of Guidance,
> Control, and Dynamics*.

**Live tool:** <https://redstovely.github.io/crossover-atlas/>

## Interactive tool

[`index.html`](index.html) is a fully self-contained browser application —
no server, no external requests, coastline embedded (derived from
[Natural Earth](https://www.naturalearthdata.com/), public domain). Open it
locally in any browser or use the GitHub Pages link above. For any coprime
repeat pair (k revolutions, m nodal days) it computes: all crossover points
on the map, the counting staircase, critical inclinations, the bifurcation
diagram, and a design catalogue with CSV export. Circular case (v1); the
eccentric mode following the paper's Section 6 formulation is planned as v2.

## Quick start

Tested on macOS and Linux with Python 3.12 (any Python ≥ 3.10 with the
pinned packages should work). All commands run from the repository root;
scripts locate their inputs/outputs relative to their own file, so the
working directory does not matter.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python run_all.py smoke     # fast sanity check          (~30 s)
.venv/bin/python run_all.py audit     # every paper number, fresh  (~15 s)
.venv/bin/python run_all.py figures   # all paper figures -> figures/
.venv/bin/python run_all.py full      # everything                 (~35 min)
```

Expected output: every step prints `PASS`/`ok` lines and `run_all.py`
finishes with `ALL '<target>' STEPS PASSED`. A non-zero exit code means a
check failed. The `figures` target writes PDF + PNG files into `figures/`
(not committed); `full` additionally regenerates `data/`,
`scripts/A5_RESULTS.md`, `scripts/A6_RESULTS.md`, and the two detail CSVs
next to them (not committed).

## Scripts (`scripts/`)

All figures, tables, and numeric claims of the paper regenerate from these
(Python 3, numpy/scipy/matplotlib):

| script | regenerates |
|---|---|
| `scan_counts.py` | exhaustive critical-inclination scan, all 489 coprime (k,m), k ≤ 40, both parity families (Theorems 2–3 verification) |
| `bench_b3.py` | colleague-matrix all-roots solver + timing table + brute-force cross-check + the (19,3) near-critical demo (Sec. 5) |
| `b5_validation.py` | Kim M·L bounds, Dilectis–Mortari counts, Arnas–Linares threshold equality, mission table (Sec. 7) |
| `b6_sso.py` | degree-7/degree-14 sun-synchronous repeat compatibility, worked examples (Sec. 8) |
| `bench_a5.py` | eccentric-formulation benchmark vs the exact Keplerian system (Sec. 6 error tables; writes `A5_RESULTS.md`) |
| `sweep_a6.py` | eccentric counting-conjecture sweep, 5588 configurations (Sec. 6; writes `A6_RESULTS.md`) |
| `make_figures.py`, `figs_c1b.py` | every figure in the paper (→ `figures/`) |
| `make_data.py` | the `data/` CSVs below |
| `audit_numbers.py` | numeric audit: recomputes every headline number and compares against the published values and `data/` (exit 0 iff all pass; run in CI) |
| `proto_recovery.py` | coordinate recovery + on-track verification prototype |

## Data (`data/`)

| file | contents |
|---|---|
| `critical_inclinations.csv` | every critical inclination for all coprime (k,m), 2 ≤ k ≤ 40: pair, parity case, ordered critical inclinations in degrees (6 decimals), kind (`tangency` or the odd-odd equatorial `pitchfork` at arccos(m/k)). The polar boundary critical at i = 90° (k even) is not listed. |
| `sso_catalogue.csv` | the 329 admissible repeat sun-synchronous orbits (altitude 300–2000 km, m ≤ 15 nodal days): k, m, semi-major axis, altitude, inclination, basic/total crossover counts, crossover latitude range. |
| `constants.csv` | physical constants used throughout (J2, Earth radius, gravitational parameter, sun-synchronicity constant). |

`SHA256SUMS` holds checksums of the committed `data/` files and
`index.html`; verify with `shasum -a 256 -c SHA256SUMS`. Regenerated CSVs
can differ in the last digit across platforms/BLAS builds; the audit
compares values with tolerances, not byte-exactly.

## Continuous integration

Every push runs `smoke` + `audit` on Ubuntu / Python 3.12 with the pinned
`requirements.txt` (see `.github/workflows/ci.yml`) — the full numeric
content of the paper is re-derived from scratch on each commit.

## Citing

See [`CITATION.cff`](CITATION.cff). A journal article is in preparation;
until it appears, please cite the software archive (Zenodo DOI to be added
on first release).

## License

[MIT](LICENSE). The embedded coastline derives from Natural Earth (public
domain). The SSO catalogue and all other data files are released under the
same MIT terms.
