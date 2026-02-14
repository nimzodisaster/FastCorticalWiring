---

# Fast Cortical Wiring

High-performance Python tools for computing intrinsic cortical wiring metrics from surface meshes, following the framework of Ecker et al. (2013). This implementation is optimized for modern cortical mesh resolutions (fsaverage5–fsaverage and higher) and uses state-of-the-art geodesic solvers to achieve practical runtimes on large surfaces.

---

## Overview

The cerebral cortex is a folded 2-dimensional sheet embedded in 3D. Many biologically meaningful distance measures—such as connection length or wiring cost—must be computed along the surface itself, not through Euclidean space.

This package computes three intrinsic wiring metrics at each cortical vertex:

* **Mean Separation Distance (MSD)**
  Area-weighted mean geodesic distance to all other cortical vertices. A global measure of intrinsic wiring cost or spatial isolation.

* **Radius Function (Local Wiring Scale)**
  The geodesic radius required to enclose a fixed fraction of cortical surface area (typically 5%).

* **Perimeter Function (Boundary Wiring Cost)**
  The perimeter of the geodesic disc defined by the radius function. Related to the cost of connecting to neighboring cortical territories.

These metrics provide quantitative descriptions of cortical spatial organization and intrinsic wiring constraints.

---

## Key Features

### High-performance geodesic computation

Uses the **heat method** (Crane et al., 2013) via:

* `potpourri3d` (primary backend, recommended)
* `pycortex` (legacy backend)

The heat method solves geodesic distance to all vertices simultaneously using sparse linear algebra. This is dramatically faster than graph-based methods like Dijkstra’s algorithm for the one-to-all distance computations required here.

Typical speedup over traditional approaches: **10×–100×**

---

### Optimized geometric computations

Performance-critical operations are optimized using several techniques:

* **Numba JIT compilation**
  Accelerates geometric clipping, area integration, and perimeter extraction loops.

* **Precomputed mesh geometry**

  * Face areas
  * Vertex areas
  * Edge lengths
  * Median mesh length scale (L)

* **BFS-ordered vertex processing**

  * Improves locality
  * Enables warm-started radius bracketing
  * Reduces iterations needed for radius inversion

* **Robust radius inversion**

  * Uses bracketed root-finding for stability
  * Avoids Newton-method instability on discretized meshes

---

### Designed for real cortical mesh scales

Fully validated at realistic resolutions:

| Mesh       | Vertices / hemi |
| ---------- | --------------- |
| fsaverage5 | ~10k            |
| fsaverage6 | ~41k            |
| fsaverage  | ~164k           |

Validation uses analytic surfaces with known ground truth and area-fraction-based radius targets.

---

### Cortex-only submesh support

Automatically restricts computation to cortex label:

* excludes medial wall
* reduces computational load
* matches typical neuroimaging workflows

---

### Standard neuroimaging outputs

Produces both analysis-ready and visualization-ready outputs:

* CSV tables with per-vertex metrics
* FreeSurfer `.mgh` overlays
* compatible with FreeSurfer, nilearn, PySurfer, etc.

---

## Repository contents

```
fastcw_pp3d.py
    Primary implementation using potpourri3d heat method

fastcw.py
    Legacy implementation using pycortex backend

validate_fastcw_pp3d.py
    Synthetic validation harness with fsaverage-scale meshes

requirements.txt
```

---

## Installation

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Core dependencies:

Runtime:

```
numpy
scipy
nibabel
potpourri3d
numba
pandas
matplotlib
tqdm
```

Optional legacy support:

```
pycortex
```

For best performance, build potpourri3d locally with SuiteSparse.

---

## Usage

Primary tool:

```
python fastcw_pp3d.py <subjects_dir> <subject_id>
```

Common options:

```
--hemispheres lh rh
--surf-type pial
--custom-label cortex
--scale 0.05
--area-tol 0.01
--no-compute-msd
--overwrite
--visualize
```

Example:

```
python fastcw_pp3d.py $SUBJECTS_DIR fsaverage \
  --hemispheres lh rh \
  --scale 0.05
```

---

## Validation

Synthetic validation using analytic surfaces:

```
python validate_fastcw_pp3d.py \
  --base-dir test_surfaces_fs_like \
  --regenerate
```

Optional full integration validation:

```
python validate_fastcw_pp3d.py \
  --base-dir test_surfaces_fs_like \
  --run-integration
```

Validation includes:

* spheres
* planes
* cylinders
* fsaverage-like resolutions (fs5/fs6/fs7)
* area fractions `{0.01, 0.03, 0.05, 0.10}` with explicit 5% reporting
* mesh-size and mesh-scale gates (`nV >= 10,000`, `r >= 5 * median_edge_length`)
* per-case interior/local-disc filtering (boundary-intersecting cases are skipped and reported)

Tests explicitly verify accuracy at realistic 5% surface area patches.

---

## Performance

Runtime depends primarily on vertex count and whether MSD is computed.

Typical performance on fsaverage6 (~41k vertices):

| Metric                          | Time                          |
| ------------------------------- | ----------------------------- |
| Radius + Perimeter              | seconds–minutes               |
| MSD (full one-to-all distances) | minutes                       |
| Full hemisphere                 | practical on workstation CPUs |

Performance scales approximately linearly with number of vertices.

---

## Algorithmic background: the heat method

Traditional shortest-path algorithms traverse mesh edges sequentially. These methods become inefficient when computing distances from every vertex.

The heat method instead solves a pair of sparse linear systems:

Step 1: simulate heat diffusion from a source vertex
Step 2: compute the normalized temperature gradient field
Step 3: integrate the gradient field to recover geodesic distance

This produces accurate geodesic distance fields efficiently using optimized sparse solvers.

Advantages:

* robust to mesh irregularities
* highly parallelizable
* efficient for repeated distance solves
* ideal for cortical surface analysis

Reference:

Crane et al., Geodesics in Heat (SIGGRAPH 2013)

---

## Scientific interpretation

These metrics describe intrinsic wiring geometry of cortex.

### Mean Separation Distance

Represents global wiring cost.

Higher MSD:

* spatially isolated regions
* association cortex

Lower MSD:

* spatially central regions
* sensory cortex

---

### Radius function

Represents spatial scale of local cortical neighborhoods.

Smaller radius:

* denser cortical packing
* shorter intrinsic wiring cost

Larger radius:

* expanded cortical areas
* longer intrinsic wiring distances

---

### Perimeter function

Represents boundary extent between cortical regions.

Related to inter-area connection costs and cortical topology.

---

## Optimization summary

Major performance contributors:

| Optimization                   | Impact                        |
| ------------------------------ | ----------------------------- |
| Heat method geodesics          | dominant speed improvement    |
| Sparse linear algebra          | scalable distance solves      |
| Numba-compiled geometry        | 10×–100× speedup              |
| Submesh restriction            | reduces problem size          |
| Warm-started radius bracketing | fewer root-finding iterations |
| Precomputed geometry           | avoids redundant computation  |

---

## Current backend recommendation

Use:

```
fastcw_pp3d.py
```

This uses potpourri3d with robust heat method solver and is actively maintained.

The pycortex backend remains for compatibility only.

---

## References

Crane et al. (2013)
Geodesics in Heat

Ecker et al. (2013)
Intrinsic wiring cost in the cerebral cortex

Gao et al. (2015)
pycortex: interactive surface visualization

Sharp et al. (2019–)
potpourri3d geometry processing library

---

## Summary

This package provides a fast, scalable, and validated implementation of intrinsic cortical wiring metrics suitable for modern cortical surface datasets.
