# Fast Cortical Wiring
Fast Python tools for intrinsic cortical wiring metrics from Ecker et al. (2013), using geodesic distances on cortical meshes.

## What this repo contains
- `fastcw_pp3d.py`: primary implementation (potpourri3d heat-method backend).
- `fastcw.py`: legacy implementation (pycortex-based backend).
- `validate_fastcw_pp3d.py`: synthetic-surface generator + validation harness for `fastcw_pp3d.py`.

## Metrics computed
For each cortical vertex:
- `MSD` (Mean Separation Distance): area-weighted mean geodesic distance to other cortical vertices.
- `Radius function`: geodesic radius needed to enclose a fixed target area fraction.
- `Perimeter function`: perimeter of that geodesic disc.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` includes:
- Runtime: `numpy`, `nibabel`, `pandas`, `matplotlib`, `tqdm`, `numba`, `potpourri3d`
- Legacy backend support: `pycortex` (needed for `fastcw.py`)

## Script guide

### 1) `fastcw_pp3d.py` (recommended)
Uses `potpourri3d.MeshHeatMethodDistanceSolver(..., use_robust=True)` on the cortex-only submesh.

Run:
```bash
python fastcw_pp3d.py <subjects_dir> <subject_id> [options]
```

Common options:
- `--hemispheres lh rh`
- `--surf-type pial`
- `--custom-label cortex6`
- `--scale 0.05`
- `--area-tol 0.01`
- `--no-compute-msd`
- `--overwrite`
- `--visualize`

Outputs:
- CSV with per-vertex metrics
- FreeSurfer overlays (`.mgh`) for MSD/radius/perimeter

### 2) `fastcw.py` (legacy)
Older pycortex-based path, kept for compatibility/comparison.

Run:
```bash
python fastcw.py <subjects_dir> <subject_id> [options]
```

### 3) `validate_fastcw_pp3d.py`
Generates synthetic validation surfaces and validates area inversion/perimeter consistency for `fastcw_pp3d.py`.

Current validation matrix includes fs-like resolutions (`fs5`, `fs6`, `fs7`) for:
- `sphere_R100_*`
- `sphere_R30_*`
- `plane_large_*`
- `cylinder_R50_*`

It evaluates area fractions `{0.01, 0.03, 0.05, 0.10}` (explicit 5% reporting), applies mesh-size/radius-scale gates, and writes:
- `validation_results_detailed.csv`
- `validation_results_tiers_f05.csv`

Run:
```bash
python validate_fastcw_pp3d.py --base-dir test_surfaces_fs_like --regenerate
```

Optional integration check (slower):
```bash
python validate_fastcw_pp3d.py --base-dir test_surfaces_fs_like --run-integration
```

## Notes
- Input data are expected in FreeSurfer subject layout (`surf/`, `label/`).
- For custom meshes/surfaces, provide matching cortex labels (`--custom-label`) where appropriate.
- `fastcw_pp3d.py` is the maintained backend going forward.
