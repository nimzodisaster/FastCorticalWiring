# FastCorticalWiring

A fast, Python-based implementation of the intrinsic cortical wiring cost metrics described in Ecker et al. (2013), optimized with Pycortex and Numba.

## Overview

[cite_start]This repository provides a tool to replicate and extend the neuroimaging analysis from the paper **"Intrinsic gray-matter connectivity of the brain in adults with autism spectrum disorder"** by Ecker et al.[cite: 5]. [cite_start]The original study introduced a method to quantify the "wiring cost" of the brain's cortex by measuring geodesic distances on the cortical surface[cite: 14, 73].

This implementation calculates the three key metrics from the paper:
1.  [cite_start]**Mean Separation Distance (MSD):** The global "wiring cost" of a cortical location[cite: 75].
2.  [cite_start]**Radius Function:** A measure of local intra-areal wiring cost[cite: 77].
3.  [cite_start]**Perimeter Function:** A measure of local inter-areal wiring cost[cite: 77].

The primary advantage of this implementation is its performance. It leverages the "heat method" for geodesic distance computation via the `pycortex` library, which is significantly faster than traditional approaches like Dijkstra's algorithm. It also uses `numba` for optional just-in-time (JIT) compilation of performance-critical geometric calculations, providing a substantial speedup.

## Key Features

* **Fast & Efficient:** Uses `pycortex`'s heat method for rapid geodesic distance calculation.
* **Optimized:** Critical geometric loops for area and perimeter calculation are JIT-compiled with `numba` for a 10-100x speedup.
* **Memory-Aware:** Operates on a cortex-only submesh, excluding the medial wall to drastically reduce computation time and memory usage.
* **Robust:** Implements robust geometric calculations to handle edge cases, such as vertices located exactly on an isoline.
* **Easy to Use:** Provides a simple command-line interface to process subjects from a FreeSurfer `subjects_dir`.
* **Standard Outputs:** Generates results in both user-friendly CSV and FreeSurfer-compatible `.mgh` formats for easy visualization and statistical analysis.

## Scientific Background

The brain's cortex is a highly folded sheet. [cite_start]The "wiring cost" between two points can be estimated by the shortest path between them *along this folded surface*, known as the geodesic distance[cite: 14, 73]. This script calculates metrics based on these distances to characterize the intrinsic connectivity of the cortex.

* [cite_start]**Mean Separation Distance (MSD):** For each point (vertex) on the cortical surface, the MSD is its average geodesic distance to all other points on the surface[cite: 75]. It represents the global wiring cost or "isolation" of that point.
* [cite_start]**Radius Function (Intra-areal cost):** The geodesic radius required to draw a circle on the cortical surface that encloses a fixed percentage (e.g., 5%) of the total cortical area[cite: 76, 283]. [cite_start]Smaller radii imply lower costs for local, within-area connections[cite: 227].
* [cite_start]**Perimeter Function (Inter-areal cost):** The perimeter of the geodesic circle defined by the radius function[cite: 77]. [cite_start]This is related to the cost of making connections to adjacent areas just outside the local patch[cite: 284].

**Reference Paper:**
> Ecker, C., Ronan, L., Feng, Y., Daly, E., Murphy, C., Ginestet, C. E., ... & Murphy, D. G. (2013). Intrinsic gray-matter connectivity of the brain in adults with autism spectrum disorder. [cite_start]*Proceedings of the National Academy of Sciences*, *110*(32), 13222-13227[cite: 5].

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

### Basic Example

To process both hemispheres of `subject01` using the pial surface and save the results to the subject's default `surf/` directory:
```bash
python main.py /path/to/subjects_dir subject01
