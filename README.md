# Fast Cortical Wiring (beta)
A fast, Python-based implementation of the intrinsic cortical wiring cost metrics described in Ecker et al. (2013), optimized with Potpourri3d and/or Pycortex Functions for Heat-Based Geodesics.



## Overview

This repository provides a tool to replicate and extend the neuroimaging analysis from the paper **"Intrinsic gray-matter connectivity of the brain in adults with autism spectrum disorder"** by Ecker et al.. The original study introduced a method to quantify the "wiring cost" of the brain's cortex by measuring geodesic distances on the cortical surface.

This implementation calculates the three key metrics from the paper:
1.  **Mean Separation Distance (MSD):** The global "wiring cost" of a cortical location.
2.  **Radius Function:** A measure of local intra-areal wiring cost.
3.  **Perimeter Function:** A measure of local inter-areal wiring cost.

The primary advantage of this implementation is its performance. It leverages the "heat method" for geodesic distance computation via the `pycortex` library or 'Potpourri3d', the latter recently added, both of which is significantly faster than traditional approaches like Dijkstra's algorithm. It also uses `numba` for optional just-in-time (JIT) compilation of performance-critical geometric calculations, providing a substantial speedup.

## Key Features

* **Fast & Efficient:** Uses `pycortex`'s heat method for rapid geodesic distance calculation.
* **Optimized:** Critical geometric loops for area and perimeter calculation are JIT-compiled with `numba` for a 10-100x speedup.
* **Allows Cortical Masking:** Operates on a cortex-only submesh, allowing users to exclude the medial wall, for example.
* **Robust:** Implements robust geometric calculations to handle edge cases, such as vertices located exactly on an isoline.
* **Turnkey Usage:** Provides a simple command-line interface to process subjects from a FreeSurfer `subjects_dir`.
* ** Precomputed face geometry/length scales for clipping tolerances
* ** BFS vertex ordering + warm-started radius bracketing
* **Standard Outputs:** Generates results in both user-friendly CSV and FreeSurfer-compatible `.mgh` formats for easy visualization and statistical analysis. By Default, outputs appear in FreeSurfer subject surf folders.

### Performance: The Heat Method Advantage

This implementation's speed comes from using the **heat method** for geodesic distance computation (Crane et al., 2013), provided by the `pycortex` library (Gao et al., 2015).

Instead of traversing the mesh edge-by-edge like classic algorithms (e.g., Dijkstra's), the heat method simulates the diffusion of heat from a source vertex. This provides a robust and efficient way to model the "wavefronts" from which geodesic distances can be derived.  The algorithm works in three main steps:

1.  **Solve the Heat Flow**: A burst of heat is placed at the source vertex, and the algorithm calculates how it spreads across the mesh for a short time, establishing a smooth temperature gradient.
2.  **Evaluate the Gradient Field**: The algorithm determines the direction of fastest heat dissipation across the surface. This gradient field effectively creates a map of vectors pointing back toward the source along the shortest paths.
3.  **Solve the Poisson Equation**: Finally, this gradient field is integrated to recover the actual geodesic distance values for all points on the mesh simultaneously.

This "whole-surface" approach is significantly faster than traditional methods for the "one-to-all" distance calculations required by this analysis. While Dijkstra's algorithm is efficient for finding a single shortest path, it becomes slow when repeated for every target vertex. The heat method solves for the entire distance field at once using efficient linear algebra, making it ideal for calculating the MSD and the numerous local metrics across the cortex.

## Scientific Background

The brain's cortex is a highly folded sheet. The "wiring cost" between two points can be estimated by the shortest path between them *along this folded surface*, known as the geodesic distance. This script calculates metrics based on these distances to characterize the intrinsic connectivity of the cortex, as proposed by Ecker et al. (2013).

* **Mean Separation Distance (MSD):** For each point (vertex) on the cortical surface, the MSD is its average geodesic distance to all other points on the surface. It represents the global wiring cost or "isolation" of that point.
* **Radius Function (Intra-areal cost):** The geodesic radius required to draw a circle on the cortical surface that encloses a fixed percentage (e.g., 5%) of the total cortical area. Smaller radii imply lower costs for local, within-area connections.
* **Perimeter Function (Inter-areal cost):** The perimeter of the geodesic circle defined by the radius function. This is related to the cost of making connections to adjacent areas just outside the local patch.

## Requirements

The script is built for Python 3. The primary dependencies are:
* `numpy`
* `nibabel`
* `pandas`
* `matplotlib`
* `tqdm`
* **`pycortex`** (Required for geodesic distance computation)
* **`numba`** (Optional, but highly recommended for performance)

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/FastCorticalWiring.git](https://github.com/your-username/FastCorticalWiring.git)
    cd FastCorticalWiring
    ```

2.  Install the required Python packages. It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

    **`requirements.txt`:**
    ```
    numpy
    nibabel
    pandas
    matplotlib
    tqdm
    pycortex
    numba
    ```

## Usage

The script is run from the command line and requires a path to a FreeSurfer subjects directory and a subject ID.

### Advanced Information


# Cortical Wiring Analysis

Fast computation of intrinsic cortical wiring costs using pycortex (jit-optimized)

## Usage

```bash
python cortical_wiring_analysis.py <subject_dir> <subject_id> [OPTIONS]
```

## Required Arguments

| Argument | Description |
|----------|-------------|
| `subject_dir` | FreeSurfer subjects directory |
| `subject_id` | Subject ID |

## Optional Arguments

### Output Options
| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `{subject_dir}/{subject_id}/surf/` | Output directory for results |
| `--overwrite` | `False` | Overwrite existing output files (default: exit if files exist) |
| `--visualize` | `False` | Generate visualization PNG files |

### Surface Processing
| Flag | Default | Description |
|------|---------|-------------|
| `--hemispheres` | `lh rh` | Hemispheres to process (space-separated) |
| `--surf-type` | `pial` | Surface type: `pial`, `white`, `inflated`, or custom surface name (e.g., `pialsurface6`) |
| `--custom-label` | `None` | Custom cortex label name for non-standard surfaces (e.g., `cortex6` for `{hemi}.cortex6.label`) |

### Computation Options
| Flag | Default | Description |
|------|---------|-------------|
| `--compute-msd` / `--no-compute-msd` | `True` | Enable/disable MSD (Mean Separation Distance) computation. MSD is the area-weighted average geodesic distance from each vertex to all other cortical vertices, quantifying global connectivity cost. |
| `--scale` | `0.05` | Scale for local measures (proportion of cortex area). Determines target area for radius/perimeter functions - e.g., 0.05 = 5% of total cortical area. This 5% value was used in the original Ecker et al. 2013 paper. |
| `--area-tol` | `0.01` | Relative tolerance for area binary search convergence (1% = 0.01). Controls precision when finding the radius that encompasses exactly the target area. Default is usually sufficient. |
| `--eps` | `1e-6` | Numerical tolerance for geometric computations, particularly for determining when vertices lie exactly on geodesic isolines. Default handles typical mesh precision well. |

## Examples

### Basic usage with default settings:
```bash
python cortical_wiring_analysis.py /path/to/freesurfer/subjects fsaverage
```

### Process only left hemisphere with custom output:
```bash
python cortical_wiring_analysis.py /path/to/subjects subject01 \
    --hemispheres lh \
    --output-dir /path/to/results \
    --visualize
```

### Use custom surface with specific parameters:
```bash
python cortical_wiring_analysis.py /path/to/subjects subject01 \
    --surf-type pialsurface6 \
    --custom-label cortex6 \
    --scale 0.1 \
    --overwrite
```

### Disable MSD computation for faster processing:
```bash
python cortical_wiring_analysis.py /path/to/subjects subject01 \
    --no-compute-msd \
    --area-tol 0.05
```


![fast!](./log2o.png)
