#!/usr/bin/env python3
"""
Fast implementation of intrinsic cortical wiring cost measures from Ecker et al. 2013
using pycortex for geodesic distance computation via the heat method.

SCIENTIFIC BACKGROUND:
This code implements the analysis from Ecker et al. 2013 which quantifies how "expensive" 
it would be to wire different regions of the brain cortex. The key insight is that regions 
with high local curvature require more "wiring" (neural connections) to connect nearby areas,
which can be quantified using geodesic distances on the cortical surface.

KEY METRICS COMPUTED:
1. Mean Separation Distance (MSD): Average geodesic distance from each vertex to all others
2. Radius Function: Geodesic radius needed to encompass a fixed area around each vertex  
3. Perimeter Function: Perimeter of that geodesic disc (related to wiring cost)

PERFORMANCE OPTIMIZATIONS:
- Cortex-only submesh (drops medial wall faces) with index maps
- Robust polygon handling for geodesic disc area/perimeter computation
- Consistent epsilon + vertex-on-isoline handling; APARC -1 exclusion
- Candidate face filtering for area/perimeter (iso-band only)
- Optional Numba JIT for the face-clipping loops (big speedup)
- Optional sampling for MSD; precomputed normals/areas; proper CLI flags
"""

import os
import numpy as np
import nibabel as nib  # For reading FreeSurfer brain surface files
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm  # Progress bars
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# OPTIONAL DEPENDENCIES: Numba JIT and pycortex
# ============================================================================

# Try to import Numba for Just-In-Time compilation of critical loops
# This can provide 10-100x speedup for the geometric intersection computations
try:
    from numba import njit  # imported only to set flag; used inside guarded block
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

# Import pycortex for geodesic distance computation using the heat method
# This is a fast algorithm for computing geodesic distances on meshes
try:
    import cortex
    from cortex.polyutils import Surface
    PYCORTEX_AVAILABLE = True
except ImportError as e:
    PYCORTEX_AVAILABLE = False
    raise ImportError(
        "This script requires pycortex for fast geodesic distance computation.\n"
        "Install with: pip install pycortex"
    ) from e

# ============================================================================
# NUMBA JIT KERNELS (High-performance geometric computations)
# ============================================================================

if NUMBA_AVAILABLE:
    
    @njit(fastmath=True, cache=True)
    def _sign_eps(x, eps):
        """
        Robust sign function with epsilon tolerance.
        
        Returns:
        -1 if x is clearly negative (< -eps)
         1 if x is clearly positive (> eps)  
         0 if x is near zero (within eps, i.e., on the isoline)
         
        This prevents classification inconsistencies due to floating-point precision.
        """
        if x < -eps:
            return -1
        if x > eps:
            return 1
        return 0

    @njit(fastmath=True, cache=True)
    def _edge_intersection_point(pi, pj, di, dj, r, abs_tol):
        """
        Find where the geodesic distance isoline d=r intersects edge (pi,pj).
        
        Args:
            pi, pj: 3D coordinates of edge endpoints
            di, dj: geodesic distances at endpoints
            r: target radius (isoline level)
            abs_tol: absolute tolerance for endpoint detection
            
        Returns:
            (px, py, pz, valid): intersection point coordinates and validity flag
            
        ALGORITHM:
        Uses linear interpolation along the edge. If distance varies linearly
        from di to dj, then d(t) = di + t*(dj-di), and we solve d(t) = r
        to get t = (r-di)/(dj-di). The 3D position is pi + t*(pj-pi).
        
        ROBUSTNESS FEATURE:
        If an endpoint is very close to the target distance (within abs_tol),
        return that endpoint directly to avoid numerical issues.
        """
        # Check if endpoints are on the isoline (within tolerance)
        if abs(di - r) <= abs_tol:
            return pi[0], pi[1], pi[2], True
        if abs(dj - r) <= abs_tol:
            return pj[0], pj[1], pj[2], True
            
        # Linear interpolation parameter
        denom = dj - di
        if abs(denom) < 1e-20:  # Nearly constant distance along edge
            return 0.0, 0.0, 0.0, False
            
        t = (r - di) / denom
        
        # Clamp to edge bounds [0,1]
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
            
        # Compute 3D intersection point
        return pi[0] + (pj[0]-pi[0]) * t, \
               pi[1] + (pj[1]-pi[1]) * t, \
               pi[2] + (pj[2]-pi[2]) * t, True

    @njit(fastmath=True, cache=True)
    def _farthest_pair(P, m):
        """
        Find the pair of points with maximum distance among first m points in P.
        
        This is used when we have >2 intersection points due to numerical issues
        (e.g., vertex exactly on isoline causes multiple edges to "intersect").
        The farthest pair represents the actual isoline chord across the triangle.
        
        Returns:
            (i, j, distance): indices and distance of farthest pair
        """
        best_i = 0
        best_j = 1
        best_d = -1.0
        
        for i in range(m):
            for j in range(i+1, m):
                dx = P[j,0]-P[i,0]
                dy = P[j,1]-P[i,1]
                dz = P[j,2]-P[i,2]
                d2 = dx*dx + dy*dy + dz*dz
                if d2 > best_d:
                    best_d = d2
                    best_i = i
                    best_j = j
        return best_i, best_j, best_d**0.5

    @njit(fastmath=True, cache=True)
    def _triangle_area_with_unit_normal(a, b, c, n):
        """
        Compute area of triangle (a,b,c) using precomputed unit normal n.
        
        Standard formula: Area = 0.5 * ||(b-a) × (c-a)||
        Since n is the unit normal: Area = 0.5 * |n · ((b-a) × (c-a))|
        
        This is more numerically stable than computing the cross product norm.
        """
        # Edge vectors from vertex a
        v1x = b[0]-a[0]
        v1y = b[1]-a[1]
        v1z = b[2]-a[2]
        v2x = c[0]-a[0]
        v2y = c[1]-a[1]
        v2z = c[2]-a[2]
        
        # Cross product (b-a) × (c-a)
        cx = v1y*v2z - v1z*v2y
        cy = v1z*v2x - v1x*v2z
        cz = v1x*v2y - v1y*v2x
        
        # Dot with unit normal and take absolute value
        return 0.5 * abs(n[0]*cx + n[1]*cy + n[2]*cz)

    @njit(fastmath=True, cache=True)
    def _area_inside_radius_kernel(V, F, unit_normals, face_areas, distances, r, eps, cand_idx):
        """
        Core computational kernel for computing area inside geodesic radius r.
        
        ALGORITHM:
        For each triangle face, determines how much of its area lies within
        geodesic distance r from the source vertex. Uses polygon clipping:
        
        1. Classify vertices as inside/outside/on the isoline
        2. Find intersection points where isoline crosses triangle edges
        3. Compute area of clipped polygon using triangulation
        
        CASES HANDLED:
        - nin=3: Entire triangle inside → add full face area
        - nin=1: One vertex inside → triangle formed by inside vertex + 2 intersections
        - nin=2: Two vertices inside → quadrilateral → split into 2 triangles
        - nin=0: No vertices inside → no contribution
        
        ROBUSTNESS FEATURES:
        - Scale-aware tolerance for deduplication (abs_tol + rel_tol * face_size)
        - Consistent vertex classification using _sign_eps
        - Farthest-pair selection when >2 intersections found
        """
        area_sum = 0.0
        abs_tol = eps
        rel_tol = 1e-9  # Relative tolerance factor
        
        for k in range(cand_idx.shape[0]):
            f_idx = cand_idx[k]
            i0, i1, i2 = F[f_idx, 0], F[f_idx, 1], F[f_idx, 2]
            d0, d1, d2 = distances[i0], distances[i1], distances[i2]
            
            # Skip faces with invalid distances
            if not (np.isfinite(d0) and np.isfinite(d1) and np.isfinite(d2)):
                continue
                
            v0 = V[i0]; v1 = V[i1]; v2 = V[i2]
            
            # Compute face length scale for adaptive tolerance
            # This ensures tolerance scales with mesh resolution
            e01x = v1[0]-v0[0]; e01y = v1[1]-v0[1]; e01z = v1[2]-v0[2]
            e12x = v2[0]-v1[0]; e12y = v2[1]-v1[1]; e12z = v2[2]-v1[2]
            e20x = v0[0]-v2[0]; e20y = v0[1]-v2[1]; e20z = v0[2]-v2[2]
            L01 = (e01x*e01x + e01y*e01y + e01z*e01z)**0.5
            L12 = (e12x*e12x + e12y*e12y + e12z*e12z)**0.5
            L20 = (e20x*e20x + e20y*e20y + e20z*e20z)**0.5
            L = L01
            if L12 > L: L = L12
            if L20 > L: L = L20
            tol = abs_tol + rel_tol * L
            tol2 = tol * tol

            # Classify vertices relative to isoline using robust sign function
            s0 = _sign_eps(d0 - r, eps)
            s1 = _sign_eps(d1 - r, eps)
            s2 = _sign_eps(d2 - r, eps)
            
            # Count vertices inside radius (≤ 0 means inside or on boundary)
            b0 = (s0 <= 0)
            b1 = (s1 <= 0)
            b2 = (s2 <= 0)
            nin = (1 if b0 else 0) + (1 if b1 else 0) + (1 if b2 else 0)
            
            if nin == 0:
                continue  # No area contribution
            if nin == 3:
                area_sum += face_areas[f_idx]  # Full triangle inside
                continue
                
            # Partial triangle case: find isoline intersections with edges
            P = np.zeros((3,3))  # Up to 3 intersection points
            m = 0  # Number of intersections found
            
            # Check edge 0-1
            # XOR condition handles case where exactly one endpoint is on isoline
            if (s0 * s1 < 0) or ((s0 == 0) ^ (s1 == 0)):
                px,py,pz,ok = _edge_intersection_point(v0, v1, d0, d1, r, abs_tol)
                if ok:
                    P[m,0],P[m,1],P[m,2] = px,py,pz; m += 1
                    
            # Check edge 1-2
            if (s1 * s2 < 0) or ((s1 == 0) ^ (s2 == 0)):
                px,py,pz,ok = _edge_intersection_point(v1, v2, d1, d2, r, abs_tol)
                if ok:
                    # Deduplicate against previous intersections
                    dup = False
                    for t in range(m):
                        dx=px-P[t,0]; dy=py-P[t,1]; dz=pz-P[t,2]
                        if dx*dx+dy*dy+dz*dz <= tol2:
                            dup = True; break
                    if not dup:
                        P[m,0],P[m,1],P[m,2] = px,py,pz; m += 1
                        
            # Check edge 2-0
            if (s2 * s0 < 0) or ((s2 == 0) ^ (s0 == 0)):
                px,py,pz,ok = _edge_intersection_point(v2, v0, d2, d0, r, abs_tol)
                if ok:
                    dup = False
                    for t in range(m):
                        dx=px-P[t,0]; dy=py-P[t,1]; dz=pz-P[t,2]
                        if dx*dx+dy*dy+dz*dz <= tol2:
                            dup = True; break
                    if not dup:
                        P[m,0],P[m,1],P[m,2] = px,py,pz; m += 1

            n = unit_normals[f_idx]
            
            # Compute area based on number of inside vertices and intersections
            if nin == 1 and m >= 2:
                # One vertex inside: form triangle with inside vertex + 2 intersections
                a = v0 if b0 else (v1 if b1 else v2)
                i,j,_ = _farthest_pair(P, m)  # Use farthest pair if >2 intersections
                P1 = P[i]; P2 = P[j]
                area_sum += _triangle_area_with_unit_normal(a, P1, P2, n)
                
            elif nin == 2:
                # Two vertices inside: form quadrilateral, split into triangles
                if b0 and b1:
                    a = v0; bv = v1
                elif b1 and b2:
                    a = v1; bv = v2
                else:
                    a = v2; bv = v0
                    
                if m >= 2:
                    # Quadrilateral case: split along diagonal a-P2
                    i,j,_ = _farthest_pair(P, m)
                    P1 = P[i]; P2 = P[j]
                    area_sum += _triangle_area_with_unit_normal(a, bv, P2, n)
                    area_sum += _triangle_area_with_unit_normal(a, P2, P1, n)
                elif m == 1:
                    # Degenerate case: triangle with two inside vertices + 1 intersection
                    P1 = P[0]
                    area_sum += _triangle_area_with_unit_normal(a, bv, P1, n)
                    
        return area_sum

    @njit(fastmath=True, cache=True)
    def _perimeter_at_radius_kernel(V, F, distances, r, eps, band_idx):
        """
        Core kernel for computing perimeter of geodesic isoline at radius r.
        
        ALGORITHM:
        For each triangle that intersects the isoline (band_idx), find where
        the isoline crosses the triangle edges and sum the lengths of those
        chord segments.
        
        The perimeter represents the "boundary length" of the geodesic disc,
        which is related to the wiring cost in the Ecker et al. model.
        
        ROBUSTNESS:
        - Uses same scale-aware tolerance as area computation
        - Farthest-pair selection when >2 intersections found
        - Proper deduplication of intersection points
        """
        perim_sum = 0.0
        abs_tol = eps
        rel_tol = 1e-9
        
        for k in range(band_idx.shape[0]):
            f_idx = band_idx[k]
            i0, i1, i2 = F[f_idx, 0], F[f_idx, 1], F[f_idx, 2]
            d0, d1, d2 = distances[i0], distances[i1], distances[i2]
            
            if not (np.isfinite(d0) and np.isfinite(d1) and np.isfinite(d2)):
                continue
                
            v0 = V[i0]; v1 = V[i1]; v2 = V[i2]
            
            # Scale-aware tolerance computation (same as area kernel)
            e01x = v1[0]-v0[0]; e01y = v1[1]-v0[1]; e01z = v1[2]-v0[2]
            e12x = v2[0]-v1[0]; e12y = v2[1]-v1[1]; e12z = v2[2]-v1[2]
            e20x = v0[0]-v2[0]; e20y = v0[1]-v2[1]; e20z = v0[2]-v2[2]
            L01 = (e01x*e01x + e01y*e01y + e01z*e01z)**0.5
            L12 = (e12x*e12x + e12y*e12y + e12z*e12z)**0.5
            L20 = (e20x*e20x + e20y*e20y + e20z*e20z)**0.5
            L = L01
            if L12 > L: L = L12
            if L20 > L: L = L20
            tol = abs_tol + rel_tol * L
            tol2 = tol * tol

            # Vertex classification
            s0 = _sign_eps(d0 - r, eps)
            s1 = _sign_eps(d1 - r, eps)
            s2 = _sign_eps(d2 - r, eps)
            
            # Find intersection points (same logic as area kernel)
            P = np.zeros((3,3))
            m = 0
            
            if (s0 * s1 < 0) or ((s0 == 0) ^ (s1 == 0)):
                px,py,pz,ok = _edge_intersection_point(v0, v1, d0, d1, r, abs_tol)
                if ok:
                    P[m,0],P[m,1],P[m,2] = px,py,pz; m += 1
                    
            if (s1 * s2 < 0) or ((s1 == 0) ^ (s2 == 0)):
                px,py,pz,ok = _edge_intersection_point(v1, v2, d1, d2, r, abs_tol)
                if ok:
                    dup = False
                    for t in range(m):
                        dx=px-P[t,0]; dy=py-P[t,1]; dz=pz-P[t,2]
                        if dx*dx+dy*dy+dz*dz <= tol2:
                            dup = True; break
                    if not dup:
                        P[m,0],P[m,1],P[m,2] = px,py,pz; m += 1
                        
            if (s2 * s0 < 0) or ((s2 == 0) ^ (s0 == 0)):
                px,py,pz,ok = _edge_intersection_point(v2, v0, d2, d0, r, abs_tol)
                if ok:
                    dup = False
                    for t in range(m):
                        dx=px-P[t,0]; dy=py-P[t,1]; dz=pz-P[t,2]
                        if dx*dx+dy*dy+dz*dz <= tol2:
                            dup = True; break
                    if not dup:
                        P[m,0],P[m,1],P[m,2] = px,py,pz; m += 1
                        
            # Add chord length to perimeter
            if m >= 2:
                i,j,Lij = _farthest_pair(P, m)
                perim_sum += Lij
                
        return perim_sum
        
else:
    # Pure-Python fallback utilities used by non-JIT code paths
    def _sign_eps(x, eps):
        """Non-JIT version of sign function for Python fallback."""
        return -1 if x < -eps else (1 if x > eps else 0)


# ============================================================================
# MAIN ANALYSIS CLASS
# ============================================================================

class FastCorticalWiringAnalysis:
    """
    Compute intrinsic cortical wiring costs with geodesics from pycortex.
    
    This class implements the full analysis pipeline:
    1. Load FreeSurfer surface and create cortex-only submesh
    2. Compute geodesic distances using pycortex heat method
    3. Calculate wiring cost metrics (MSD, radius function, perimeter function)
    4. Save results and generate visualizations
    
    KEY OPTIMIZATION: Works on a cortex-only submesh to exclude medial wall,
    reducing computation time and memory usage significantly.
    """

    def __init__(self, subject_dir, subject_id, hemi='lh', surf_type='pial', eps=1e-6, custom_label=None):
        """
        Initialize the analysis for a specific subject and hemisphere.
        
        Args:
            subject_dir: FreeSurfer subjects directory path
            subject_id: Subject identifier
            hemi: Hemisphere ('lh' or 'rh')
            surf_type: Surface type ('pial', 'white', 'inflated') or custom name
            eps: Numerical tolerance for geometric computations
            custom_label: Custom label name for cortex mask (required for custom surf_type)
        """
        self.subject_dir = subject_dir
        self.subject_id = subject_id
        self.hemi = hemi
        self.surf_type = surf_type
        self.eps = float(eps)
        self.custom_label = custom_label

        # Load surface geometry from FreeSurfer files
        self.vertices_full, self.faces_full = self._load_surface()
        self.n_vertices_full = self.vertices_full.shape[0]
        self.n_faces_full = self.faces_full.shape[0]
        
        # Create cortex mask (exclude medial wall and unknown regions)
        self.cortex_mask_full = self._load_cortex_mask()
        self.n_cortex_vertices = int(np.sum(self.cortex_mask_full))
        print(f"Loaded surface: {self.n_vertices_full} vertices, {self.n_faces_full} faces")
        print(f"Cortical vertices (excluding medial wall): {self.n_cortex_vertices}")
        
        # Build cortex-only submesh for computational efficiency
        # This dramatically reduces computation time by working only on cortical regions
        (
            self.vertices,           # Cortex-only vertex coordinates
            self.faces,             # Cortex-only face indices (re-indexed)
            self.sub_to_orig,       # Map from submesh to original vertex indices
            self.orig_to_sub,       # Map from original to submesh vertex indices
        ) = self._build_cortex_submesh(self.vertices_full, self.faces_full, self.cortex_mask_full)
        
        self.n_vertices = self.vertices.shape[0]
        self.n_faces = self.faces.shape[0]
        
        # Create pycortex Surface object for geodesic distance computation
        self.surface = Surface(self.vertices, self.faces)
        
        # Precompute face geometry for efficient area/perimeter calculations
        self.face_normals, self.face_areas, self.face_unit_normals = self._precompute_face_geometry(self.vertices, self.faces)
        
        # Compute vertex areas (for normalization and total cortex area)
        self.vertex_areas_sub = self._compute_vertex_areas(self.vertices, self.faces)  # Areas in submesh
        self.vertex_areas = np.zeros(self.n_vertices_full, dtype=np.float64)  # Areas in full mesh
        self.vertex_areas[self.sub_to_orig] = self.vertex_areas_sub
        
        # Initialize result arrays (will be filled by computation methods)
        self.msd = np.full(self.n_vertices_full, np.nan, dtype=np.float32)
        self.radius_function = np.full(self.n_vertices_full, np.nan, dtype=np.float32)
        self.perimeter_function = np.full(self.n_vertices_full, np.nan, dtype=np.float32)

    # ========================================================================
    # DATA LOADING AND PREPROCESSING
    # ========================================================================
    
    def _load_surface(self):
        """
        Load brain surface geometry from FreeSurfer files.
        
        Returns vertices (N,3 coordinates) and faces (M,3 triangle indices).
        FreeSurfer stores surfaces as triangle meshes in a custom binary format.
        """
        surf_path = os.path.join(self.subject_dir, self.subject_id, 'surf', f'{self.hemi}.{self.surf_type}')
        if not os.path.exists(surf_path):
            raise FileNotFoundError(f"Surface file not found: {surf_path}")
        vertices, faces = nib.freesurfer.read_geometry(surf_path)
        return vertices.astype(np.float64), faces.astype(np.int32)

    def _load_cortex_mask(self):
        """
        Create boolean mask identifying cortical vertices (excluding medial wall).
        
        Tries three strategies in order:
        1. Use explicit cortex label file (most accurate) - custom name if provided
        2. Use aparc annotation to exclude non-cortical regions  
        3. Use all vertices (fallback)
        
        The medial wall represents the interhemispheric connection area and
        is excluded from cortical wiring analysis.
        """
        # Strategy 1: Use explicit cortex label file (custom name or default)
        if self.custom_label:
            label_name = f'{self.hemi}.{self.custom_label}.label'
            print(f"Using custom cortex label: {label_name}")
        else:
            label_name = f'{self.hemi}.cortex.label'
        
        label_path = os.path.join(self.subject_dir, self.subject_id, 'label', label_name)
        if os.path.exists(label_path):
            try:
                # Try nibabel's built-in parser first
                cortex_idx = nib.freesurfer.read_label(label_path)
                cortex_mask = np.zeros(self.vertices_full.shape[0], dtype=bool)
                # Filter valid indices and set mask
                cortex_idx = cortex_idx[(cortex_idx >= 0) & (cortex_idx < cortex_mask.shape[0])]
                cortex_mask[cortex_idx] = True
                print(f"Loaded cortex mask: {int(np.sum(cortex_mask))} cortical vertices")
                return cortex_mask
            except Exception:
                # Fallback to manual parsing if nibabel fails
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                n_vertices_in_label = int(lines[1].strip())
                cortex_mask = np.zeros(self.vertices_full.shape[0], dtype=bool)
                for i in range(2, 2 + n_vertices_in_label):
                    vertex_idx = int(lines[i].split()[0])
                    if 0 <= vertex_idx < cortex_mask.shape[0]:
                        cortex_mask[vertex_idx] = True
                print(f"Loaded cortex mask: {int(np.sum(cortex_mask))} cortical vertices")
                return cortex_mask
        
        # Strategy 2: Use aparc annotation to exclude non-cortical regions
        print("Warning: Cortex label not found. Using aparc annotation...")
        annot_path = os.path.join(self.subject_dir, self.subject_id, 'label', f'{self.hemi}.aparc.annot')
        if os.path.exists(annot_path):
            labels, ctab, names = nib.freesurfer.read_annot(annot_path)
            # Exclude known non-cortical regions
            exclude_names = {b'unknown', b'Unknown', b'corpuscallosum', b'Medial_wall'}
            exclude_indices = {i for i, name in enumerate(names) if any(excl in name for excl in exclude_names)}
            cortex_mask = np.ones(self.vertices_full.shape[0], dtype=bool)
            for idx in exclude_indices:
                cortex_mask[labels == idx] = False
            cortex_mask[labels == -1] = False  # Exclude unlabeled vertices
            print(f"Created cortex mask from aparc: {int(np.sum(cortex_mask))} cortical vertices")
            return cortex_mask
        
        # Strategy 3: Fallback - use all vertices
        print("Warning: No cortex label or aparc annotation found. Using all vertices.")
        return np.ones(self.vertices_full.shape[0], dtype=bool)

    def _build_cortex_submesh(self, vertices, faces, cortex_mask):
        """
        Build a submesh containing only cortical vertices and faces.
        
        This is a key optimization: by working on a smaller submesh, we can
        dramatically reduce computation time for geodesic distance calculations.
        
        Returns:
            vertices_sub: Coordinates of cortical vertices only
            faces_sub: Triangle faces with indices remapped to submesh
            sub_to_orig: Mapping from submesh vertex index to original index
            orig_to_sub: Mapping from original vertex index to submesh index (-1 if not cortical)
        """
        # Create vertex index mappings
        sub_to_orig = np.where(cortex_mask)[0].astype(np.int32)  # Cortical vertex indices
        orig_to_sub = np.full(vertices.shape[0], -1, dtype=np.int32)
        orig_to_sub[sub_to_orig] = np.arange(sub_to_orig.shape[0], dtype=np.int32)
        
        # Keep only faces where all vertices are cortical
        keep = cortex_mask[faces].all(axis=1)
        faces_kept_orig = faces[keep]
        
        # Remap face indices to submesh vertex indices
        faces_sub = orig_to_sub[faces_kept_orig]
        
        # Extract cortical vertex coordinates
        verts_sub = vertices[sub_to_orig]
        
        return verts_sub, faces_sub, sub_to_orig, orig_to_sub

    def _precompute_face_geometry(self, V, F):
        """
        Precompute face normals, areas, and unit normals for efficient geometry operations.
        
        This avoids recomputing these values repeatedly during area/perimeter calculations.
        Includes robust handling of near-degenerate faces using scale-aware tolerances.
        
        Returns:
            face_normals: Raw cross-product normals (not normalized)
            face_areas: Triangle areas  
            face_unit_normals: Unit normal vectors (robust for tiny triangles)
        """
        # Get triangle vertices
        p0 = V[F[:, 0]]
        p1 = V[F[:, 1]] 
        p2 = V[F[:, 2]]
        
        # Compute edge vectors for scale estimation
        e01 = p1 - p0
        e12 = p2 - p1
        e20 = p0 - p2
        
        # Face normals via cross product
        normals = np.cross(e01, p2 - p0)
        norm_n = np.linalg.norm(normals, axis=1)
        
        # Scale-aware tiny threshold based on edge lengths
        # This prevents division by zero for very small triangles
        L = np.maximum.reduce([np.linalg.norm(e01, axis=1),
                               np.linalg.norm(e12, axis=1),
                               np.linalg.norm(e20, axis=1)])
        tiny = np.finfo(np.float64).eps * (L * L) * 10.0
        
        # Compute areas and unit normals
        areas = 0.5 * norm_n
        denom = np.maximum(norm_n, tiny)  # Avoid division by zero
        unit = (normals.T / denom).T
        
        # Handle degenerate cases
        bad = norm_n < tiny
        areas[bad] = 0.0
        unit[bad] = 0.0
        
        return normals, areas, unit

    def _compute_vertex_areas(self, V, F):
        """
        Compute vertex areas using the standard approach of 1/3 of adjacent face areas.
        
        Each vertex gets 1/3 of the area of each triangle it belongs to.
        This is equivalent to the barycentric dual cell area.
        """
        vertex_areas = np.zeros(V.shape[0], dtype=np.float64)
        
        # Get face areas (already computed)
        face_areas = self.face_areas
        
        # Add 1/3 of each face area to each of its vertices
        for i, face in enumerate(F):
            area_contrib = face_areas[i] / 3.0
            vertex_areas[face[0]] += area_contrib
            vertex_areas[face[1]] += area_contrib
            vertex_areas[face[2]] += area_contrib
            
        return vertex_areas

    # ========================================================================
    # GEODESIC DISTANCE COMPUTATION
    # ========================================================================
    
    def _compute_geodesic_distances_from_subvertex(self, sub_idx):
        """
        Compute geodesic distances from a single vertex in the submesh.
        
        Uses pycortex's heat method implementation, which is much faster than
        Dijkstra's algorithm for dense distance queries on meshes.
        
        The heat method works by:
        1. Solve heat diffusion equation to steady state
        2. Compute gradient of heat function  
        3. Solve Poisson equation to recover distances
        """
        d = self.surface.geodesic_distance([int(sub_idx)])
        return d.astype(np.float64)

    def compute_geodesic_distances_from_vertex(self, source_idx):
        """
        Public interface for computing geodesic distances from any original vertex index.
        
        Handles the submesh mapping automatically and returns distances in the full mesh.
        Returns infinite distance for non-cortical vertices.
        """
        if not self.cortex_mask_full[source_idx]:
            # Non-cortical vertex: return infinite distances
            out = np.full(self.n_vertices_full, np.inf, dtype=np.float64)
            return out
            
        # Map to submesh, compute distances, map back to full mesh
        sub_idx = self.orig_to_sub[source_idx]
        d_sub = self._compute_geodesic_distances_from_subvertex(sub_idx)
        out = np.full(self.n_vertices_full, np.inf, dtype=np.float64)
        out[self.sub_to_orig] = d_sub
        return out

    # ========================================================================
    # HELPER FOR PYTHON FALLBACK: EDGE-ISOLINE INTERSECTIONS
    # ========================================================================
    
    def _edge_intersections(self, r, d, v_face):
        """
        Python fallback for finding edge-isoline intersections (when Numba unavailable).
        
        Returns intersection points where the level-set d=r crosses triangle edges.
        Uses the same robust logic as the Numba kernels: scale-aware tolerance
        and endpoint detection.
        """
        eps = self.eps
        P = []
        
        # Compute face length scale for tolerance
        e01 = v_face[1] - v_face[0]
        e12 = v_face[2] - v_face[1]
        e20 = v_face[0] - v_face[2]
        L = max(np.linalg.norm(e01), np.linalg.norm(e12), np.linalg.norm(e20))
        abs_tol = eps
        rel_tol = 1e-9
        tol = abs_tol + rel_tol * L
        
        # Check each edge for intersections
        edges = ((0,1),(1,2),(2,0))
        for a,b in edges:
            s0 = _sign_eps(d[a] - r, eps)
            s1 = _sign_eps(d[b] - r, eps)
            # Edge crosses isoline if signs differ or exactly one endpoint is on isoline
            cross = (s0 * s1 < 0) or ((s0 == 0) ^ (s1 == 0))
            if not cross:
                continue
                
            di, dj = d[a], d[b]
            
            # Handle endpoints on isoline
            if abs(di - r) <= abs_tol:
                p = v_face[a]
            elif abs(dj - r) <= abs_tol:
                p = v_face[b]
            else:
                # Linear interpolation
                denom = (dj - di)
                if abs(denom) < 1e-20:
                    continue
                t = (r - di) / denom
                t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
                p = v_face[a] + t * (v_face[b] - v_face[a])
            
            # Add to list if not a duplicate
            if not any(np.linalg.norm(p - q) <= tol for q in P):
                P.append(p)
                
        return P

    # ========================================================================
    # AREA AND PERIMETER COMPUTATION (with candidate-face filtering)
    # ========================================================================
    
    def _area_inside_radius(self, radius, distances_sub, dmin=None, dmax=None):
        """
        Compute the area of cortical surface within geodesic radius from a source vertex.
        
        ALGORITHM:
        Uses polygon clipping to determine what portion of each triangle lies within
        the geodesic disc. Key optimization: only processes candidate faces that
        potentially intersect the disc (prefiltered by dmin <= r).
        
        This implements the geometric core of the Ecker et al. wiring cost model.
        """
        r = float(radius)
        eps = self.eps
        V, F = self.vertices, self.faces
        
        # Precompute min/max distances per face for candidate filtering
        if dmin is None or dmax is None:
            df = distances_sub[F]
            dmin = df.min(axis=1)
            dmax = df.max(axis=1)
        
        # Candidate face filtering: only process faces that might contribute
        # This eliminates most faces and dramatically speeds up computation
        cand_idx = np.flatnonzero(dmin <= (r + eps)).astype(np.int32)
        
        if NUMBA_AVAILABLE:
            # Use optimized JIT kernel
            return float(_area_inside_radius_kernel(V, F, self.face_unit_normals, self.face_areas, distances_sub, r, eps, cand_idx))
        
        # Pure Python fallback (slower but works without Numba)
        area = 0.0
        normals, areas = self.face_unit_normals, self.face_areas
        for f_idx in cand_idx:
            face = F[f_idx]
            d = distances_sub[face]
            if not np.all(np.isfinite(d)):
                continue
                
            v_face = V[face]
            inside = d <= r + eps
            n_in = int(np.sum(inside))
            
            if n_in == 0:
                continue
            if n_in == 3:
                area += areas[f_idx]
                continue
                
            # Partial triangle cases
            if n_in == 1:
                i_inside = int(np.where(inside)[0][0])
                P = self._edge_intersections(r, d, v_face)
                if len(P) < 2:
                    continue
                a = v_face[i_inside]
                b, c = P[0], P[1]
                area += 0.5 * abs(np.dot(normals[f_idx], np.cross(b - a, c - a)))
            else:  # n_in == 2
                idx_in = np.where(inside)[0]
                a = v_face[idx_in[0]]
                b = v_face[idx_in[1]]
                P = self._edge_intersections(r, d, v_face)
                if len(P) == 0:
                    continue
                if len(P) == 1:
                    area += 0.5 * abs(np.dot(normals[f_idx], np.cross(b - a, P[0] - a)))
                else:
                    # Split quadrilateral into two triangles
                    p1, p2 = P[0], P[1]
                    area += 0.5 * abs(np.dot(normals[f_idx], np.cross(b - a, p2 - a)))
                    area += 0.5 * abs(np.dot(normals[f_idx], np.cross(p2 - a, p1 - a)))
        return float(area)

    def _perimeter_at_radius(self, radius, distances_sub, dmin=None, dmax=None):
        """
        Compute the perimeter of the geodesic isoline at given radius.
        
        The perimeter represents the "boundary length" of the geodesic disc,
        which correlates with wiring cost in neural connectivity models.
        
        Uses candidate filtering to only process faces in the "band" around the isoline.
        """
        r = float(radius)
        eps = self.eps
        V, F = self.vertices, self.faces
        
        if dmin is None or dmax is None:
            df = distances_sub[F]
            dmin = df.min(axis=1)
            dmax = df.max(axis=1)
        
        # Band filtering: only faces that intersect the isoline
        band_idx = np.flatnonzero((dmin <= (r + eps)) & (dmax >= (r - eps))).astype(np.int32)
        
        if NUMBA_AVAILABLE:
            return float(_perimeter_at_radius_kernel(V, F, distances_sub, r, eps, band_idx))
        
        # Python fallback
        perim = 0.0
        for f_idx in band_idx:
            face = F[f_idx]
            d = distances_sub[face]
            if not np.all(np.isfinite(d)):
                continue
                
            v_face = V[face]
            P = self._edge_intersections(r, d, v_face)
            
            if len(P) == 2:
                perim += float(np.linalg.norm(P[1] - P[0]))
            elif len(P) > 2:
                # Greedy pairing for degenerate cases (not as robust as farthest-pair)
                used = [False]*len(P)
                for i in range(len(P)):
                    if used[i]:
                        continue
                    best_j = -1
                    best_d = 1e30
                    for j in range(i+1, len(P)):
                        if used[j]:
                            continue
                        dd = np.linalg.norm(P[j]-P[i])
                        if dd < best_d:
                            best_d = dd
                            best_j = j
                    if best_j >= 0:
                        used[i] = used[best_j] = True
                        perim += float(best_d)
        return float(perim)

    def _find_radius_for_area(self, distances_sub, target_area, tol=0.01, max_iter=30, dmin=None, dmax=None):
        """
        Find the geodesic radius that encompasses exactly the target area.
        
        Uses binary search (bisection method) to solve the inverse problem:
        Given target area A, find radius r such that area_inside_radius(r) ≈ A.
        
        This is the key computation for the radius function in Ecker et al. 2013.
        """
        valid = np.isfinite(distances_sub)
        if not np.any(valid):
            return np.nan
            
        if dmin is None or dmax is None:
            df = distances_sub[self.faces]
            dmin = df.min(axis=1)
            dmax = df.max(axis=1)
        
        # Initialize search bounds
        r_min, r_max = 0.0, float(np.max(distances_sub[valid]))
        
        # Check if target area is achievable
        area_max = self._area_inside_radius(r_max, distances_sub, dmin=dmin, dmax=dmax)
        if area_max + 1e-12 < target_area:
            return np.nan  # Target area too large
        
        # Binary search for optimal radius
        for _ in range(max_iter):
            r_mid = 0.5 * (r_min + r_max)
            area = self._area_inside_radius(r_mid, distances_sub, dmin=dmin, dmax=dmax)
            
            # Check convergence
            if target_area > 0 and abs(area - target_area) / target_area < tol:
                return r_mid
                
            # Update search bounds
            if area < target_area:
                r_min = r_mid
            else:
                r_max = r_mid
                
        return 0.5 * (r_min + r_max)

    # ========================================================================
    # PUBLIC COMPUTATION METHODS
    # ========================================================================


    def compute_all_wiring_costs(self, compute_msd=True, scale=0.05, area_tol=0.01):
        """
        Compute all wiring cost metrics (MSD, radius, perimeter) in a single pass.

        This optimized method iterates through each cortical vertex only once. In each
        iteration, it computes the geodesic distance vector and then calculates all
        derived metrics (MSD, radius, perimeter) from that vector before discarding it.
        This avoids re-computing the expensive geodesic distances in separate loops.

        Args:
            compute_msd (bool): If True, compute Mean Separation Distance.
            scale (float): The proportion of total cortical area to use for local measures.
            area_tol (float): Relative tolerance for the area binary search.
        """
        if compute_msd:
            print("Computing Mean Separation Distances and Local Wiring Costs...")
        else:
            print(f"Computing Local Wiring Costs at scale {scale*100:.2f}%...")

        # Calculate target area once for local measures
        total_area = float(np.sum(self.vertex_areas_sub))
        target_area = total_area * float(scale)
        print(f"Target area for local measures: {target_area:.2f} mm² ({scale*100:.2f}% of {total_area:.2f} mm²)")

        # Single loop over all cortical vertices
        for sub_idx in tqdm(range(self.n_vertices), desc="Computing wiring costs"):
            # 1. Compute geodesic distances FROM this vertex ONCE
            d_sub = self._compute_geodesic_distances_from_subvertex(sub_idx)
            orig_idx = self.sub_to_orig[sub_idx]

            # 2. Compute MSD from the distance vector (if requested)
            if compute_msd:
            valid = (d_sub > self.eps) & np.isfinite(d_sub)
            if np.any(valid):
                w = self.vertex_areas_sub[valid]      # submesh areas align with d_sub
                msd_val = float((d_sub[valid] * w).sum() / w.sum())
            else:
                msd_val = np.nan
            self.msd[orig_idx] = np.float32(msd_val)

            # 3. Compute Local Wiring Costs from the SAME distance vector
            # Precompute min/max distances per face for efficiency
            df = d_sub[self.faces]
            dmin = df.min(axis=1)
            dmax = df.max(axis=1)

            # Find radius that gives target area
            r = self._find_radius_for_area(d_sub, target_area, tol=area_tol, dmin=dmin, dmax=dmax)
            if not np.isfinite(r):
                # If radius can't be found, local costs are NaN. MSD may still be valid.
                self.radius_function[orig_idx] = np.nan
                self.perimeter_function[orig_idx] = np.nan
                continue

            # Compute perimeter at that radius
            perim = self._perimeter_at_radius(r, d_sub, dmin=dmin, dmax=dmax)

            # Store results in full mesh arrays
            self.radius_function[orig_idx] = np.float32(r)
            self.perimeter_function[orig_idx] = np.float32(perim)

        # Print summary statistics at the end
        if compute_msd:
            valid_msd = self.msd[np.isfinite(self.msd)]
            if valid_msd.size:
                print(f"MSD stats: min={np.min(valid_msd):.2f}, max={np.max(valid_msd):.2f}, mean={np.mean(valid_msd):.2f}")

        vr = self.radius_function[np.isfinite(self.radius_function)]
        vp = self.perimeter_function[np.isfinite(self.perimeter_function)]
        if vr.size:
            print(f"Radius stats: min={np.min(vr):.2f}, max={np.max(vr):.2f}, mean={np.mean(vr):.2f}")
        if vp.size:
            print(f"Perimeter stats: min={np.min(vp):.2f}, max={np.max(vp):.2f}, mean={np.mean(vp):.2f}")

        return self.msd, self.radius_function, self.perimeter_function
            
    # ========================================================================
    # INPUT/OUTPUT AND VISUALIZATION
    # ========================================================================
    
    def save_results(self, output_dir):
        """
        Save computed metrics to CSV and FreeSurfer format files.
        
        Outputs:
        - CSV file: All results in tabular format for analysis
        - .mgh files: FreeSurfer overlay format for visualization in tools like FreeView
        
        Adheres to FreeSurfer naming conventions (e.g., lh.pial.radius.mgh) when
        saving to the default subject surf directory.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # --- Save to CSV (always includes subject ID) ---
        csv_output_file = os.path.join(output_dir, f'{self.subject_id}_{self.hemi}_{self.surf_type}_wiring_costs.csv')
        results = pd.DataFrame({
            'vertex_id': np.arange(self.n_vertices_full),
            'is_cortex': self.cortex_mask_full,
            'msd': self.msd,
            'radius_function': self.radius_function,
            'perimeter_function': self.perimeter_function
        })
        results.to_csv(csv_output_file, index=False)
        print(f"Results saved to: {csv_output_file}")
        
        # --- Determine MGH filename pattern based on the output directory ---
        standard_surf_dir = os.path.join(self.subject_dir, self.subject_id, 'surf')
        if os.path.abspath(output_dir) == os.path.abspath(standard_surf_dir):
            # Standard FreeSurfer naming (e.g., lh.pial.radius.mgh)
            print("Using standard FreeSurfer naming convention for MGH files.")
            mgh_filename_template = f"{self.hemi}.{self.surf_type}.{{metric}}.mgh"
        else:
            # Non-standard directory, include subject_id to avoid name collisions
            print("Using subject-specific naming convention for MGH files in custom directory.")
            mgh_filename_template = f"{self.subject_id}_{self.hemi}_{self.surf_type}.{{metric}}.mgh"
            
        # --- Robustly save to .mgh format using a template header ---
        template_affine = None
        template_header = None
        
        try:
            template_path = os.path.join(self.subject_dir, self.subject_id, 'mri', 'orig.mgz')
            if not os.path.exists(template_path):
                template_path = os.path.join(self.subject_dir, self.subject_id, 'mri', 'T1.mgz')
            
            print(f"Using header template from: {template_path}")
            template_img = nib.load(template_path)
            template_affine = template_img.affine
            template_header = template_img.header
        
        except FileNotFoundError:
            print("Warning: Could not find mri/orig.mgz or mri/T1.mgz for header template.")
            print("Falling back to a generic header.")
            template_affine = np.array([[-1., 0., 0., 0.], [0., 0., 1., 0.], [0.,-1., 0., 0.], [0., 0., 0., 1.]])
            template_header = nib.MGHHeader()
            template_header.set_data_shape([self.n_vertices_full, 1, 1])

        def _robust_mgh_write(path, data_array, affine, header):
            if not np.any(np.isfinite(data_array)):
                print(f"Skipping save for {os.path.basename(path)} as it contains no valid data.")
                return
            payload = data_array.astype(np.float32).reshape((self.n_vertices_full, 1, 1))
            payload[~np.isfinite(payload)] = 0
            output_header = header.copy()
            output_header.set_data_shape(payload.shape)
            output_header.set_data_dtype(np.float32)
            mgh_image = nib.MGHImage(payload, affine, output_header)
            nib.save(mgh_image, path)
            print(f"Saved MGH overlay to: {path}")

        # Save each metric using the chosen filename template
        if np.any(np.isfinite(self.msd)):
            filename = mgh_filename_template.format(metric='msd')
            _robust_mgh_write(os.path.join(output_dir, filename), self.msd, template_affine, template_header)
        if np.any(np.isfinite(self.radius_function)):
            filename = mgh_filename_template.format(metric='radius')
            _robust_mgh_write(os.path.join(output_dir, filename), self.radius_function, template_affine, template_header)
        if np.any(np.isfinite(self.perimeter_function)):
            filename = mgh_filename_template.format(metric='perimeter')
            _robust_mgh_write(os.path.join(output_dir, filename), self.perimeter_function, template_affine, template_header)
    
    def get_output_files(self, output_dir):
        """
        Get list of expected output file paths, respecting the naming convention.
        """
        # Determine the MGH filename pattern based on the output directory
        standard_surf_dir = os.path.join(self.subject_dir, self.subject_id, 'surf')
        if os.path.abspath(output_dir) == os.path.abspath(standard_surf_dir):
            mgh_template = f"{self.hemi}.{self.surf_type}.{{metric}}.mgh"
        else:
            mgh_template = f"{self.subject_id}_{self.hemi}_{self.surf_type}.{{metric}}.mgh"

        files = [
            os.path.join(output_dir, f'{self.subject_id}_{self.hemi}_{self.surf_type}_wiring_costs.csv')
        ]
        
        if np.any(np.isfinite(self.msd)):
            files.append(os.path.join(output_dir, mgh_template.format(metric='msd')))
        if np.any(np.isfinite(self.radius_function)):
            files.append(os.path.join(output_dir, mgh_template.format(metric='radius')))
        if np.any(np.isfinite(self.perimeter_function)):
            files.append(os.path.join(output_dir, mgh_template.format(metric='perimeter')))
            
        return files
    
    def check_output_files_exist(self, output_dir):
        """
        Check if output files already exist, respecting the naming convention.
        """
        # Determine the MGH filename pattern based on the output directory
        standard_surf_dir = os.path.join(self.subject_dir, self.subject_id, 'surf')
        if os.path.abspath(output_dir) == os.path.abspath(standard_surf_dir):
            mgh_template = f"{self.hemi}.{self.surf_type}.{{metric}}.mgh"
        else:
            mgh_template = f"{self.subject_id}_{self.hemi}_{self.surf_type}.{{metric}}.mgh"
            
        # For existence check, assume all metrics will be computed
        expected_files = [
            os.path.join(output_dir, f'{self.subject_id}_{self.hemi}_{self.surf_type}_wiring_costs.csv'),
            os.path.join(output_dir, mgh_template.format(metric='msd')),
            os.path.join(output_dir, mgh_template.format(metric='radius')),
            os.path.join(output_dir, mgh_template.format(metric='perimeter'))
        ]
        
        existing_files = [f for f in expected_files if os.path.exists(f)]
        return len(existing_files) > 0, existing_files
        
       

    def visualize_results(self, measure='msd', output_file=None):
        """
        Create 3D scatter plot and histogram visualization of computed measures.
        
        Shows spatial distribution of wiring costs across the cortical surface
        and the statistical distribution of values.
        """
        if measure == 'msd':
            data = self.msd
            title = 'Mean Separation Distances'
        elif measure == 'radius':
            data = self.radius_function
            title = 'Radius Function'
        elif measure == 'perimeter':
            data = self.perimeter_function
            title = 'Perimeter Function'
        else:
            raise ValueError(f"Unknown measure: {measure}")
        
        if data is None:
            print(f"Measure {measure} has not been computed yet")
            return
        
        # Prepare visualization data (mask non-cortical vertices)
        vis_data = data.copy()
        vis_data[~self.cortex_mask_full] = np.nan
        
        fig = plt.figure(figsize=(12, 8))
        
        # 3D scatter plot of cortical surface
        ax = fig.add_subplot(121, projection='3d')
        cortex_vertices = self.vertices_full[self.cortex_mask_full]
        cortex_data = vis_data[self.cortex_mask_full]
        p = ax.scatter(cortex_vertices[:, 0], cortex_vertices[:, 1], cortex_vertices[:, 2], 
                      c=cortex_data, cmap='coolwarm', s=1)
        ax.set_title(f'{title} - {self.hemi} hemisphere')
        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Z')
        plt.colorbar(p, ax=ax, fraction=0.046, pad=0.04)
        
        # Histogram of values
        ax2 = fig.add_subplot(122)
        valid_data = vis_data[np.isfinite(vis_data)]
        ax2.hist(valid_data, bins=50, edgecolor='black')
        ax2.set_xlabel(measure.upper())
        ax2.set_ylabel('Number of vertices')
        ax2.set_title(f'Distribution of {title}')
        if valid_data.size:
            ax2.axvline(np.mean(valid_data), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(valid_data):.2f}')
            ax2.legend()
        
        plt.tight_layout()
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {output_file}")
        else:
            plt.show()


# ============================================================================
# DRIVER FUNCTIONS
# ============================================================================

def process_subject(subject_dir, subject_id, output_dir=None, hemispheres=['lh', 'rh'],
                   surf_type='pial', custom_label=None, compute_msd=True,
                   scale=0.05, area_tol=0.01, eps=1e-6, overwrite=False, visualize=False):
    """
    Process a single subject through the complete cortical wiring analysis pipeline.
    
    This is the main entry point that handles both hemispheres and all analysis steps.
    """
    # Set default output directory to subject's surf directory
    if output_dir is None:
        output_dir = os.path.join(subject_dir, subject_id, 'surf')
    
    print("="*60)
    print(f"Processing subject: {subject_id}")
    print(f"Surface type: {surf_type}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    for hemi in hemispheres:
        print(f"\n--- Processing {hemi} hemisphere ---")
        
        # Initialize analysis for this hemisphere
        analysis = FastCorticalWiringAnalysis(subject_dir, subject_id, hemi=hemi, 
                                            surf_type=surf_type, eps=eps, custom_label=custom_label)
        
        # Check for existing output files before processing
        if not overwrite:
            files_exist, existing_files = analysis.check_output_files_exist(output_dir)
            if files_exist:
                print(f"ERROR: Output files already exist for {subject_id} {hemi} {surf_type}:")
                for f in existing_files:
                    print(f"  - {f}")
                print("Use --overwrite to overwrite existing files, or specify a different output directory.")
                continue  # Skip to next hemisphere
        
        # --- EFFICIENT COMPUTATION ---
        # Compute all metrics in a single pass to avoid redundant geodesic calculations.
        analysis.compute_all_wiring_costs(
            compute_msd=compute_msd,
            scale=scale,
            area_tol=area_tol
        )
        
        # Save results
        analysis.save_results(output_dir)
        
        # Generate visualizations if requested
        if visualize:
            if compute_msd and np.any(np.isfinite(analysis.msd)):
                viz_file = os.path.join(output_dir, f'{subject_id}_{hemi}_{surf_type}_msd.png')
                analysis.visualize_results('msd', viz_file)
            if np.any(np.isfinite(analysis.radius_function)):
                viz_file = os.path.join(output_dir, f'{subject_id}_{hemi}_{surf_type}_radius.png')
                analysis.visualize_results('radius', viz_file)
            if np.any(np.isfinite(analysis.perimeter_function)):
                viz_file = os.path.join(output_dir, f'{subject_id}_{hemi}_{surf_type}_perimeter.png')
                analysis.visualize_results('perimeter', viz_file)



def main():
    """Command-line interface for the cortical wiring analysis."""
    parser = argparse.ArgumentParser(
        description='Fast computation of intrinsic cortical wiring costs using pycortex (jit-optimized)'
    )
    parser.add_argument('subject_dir', help='FreeSurfer subjects directory')
    parser.add_argument('subject_id', help='Subject ID')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory (default: {subject_dir}/{subject_id}/surf/)')
    parser.add_argument('--hemispheres', nargs='+', default=['lh', 'rh'], help='Hemispheres to process')
    parser.add_argument('--surf-type', default='pial', 
                       help='Surface type: pial, white, inflated, or custom surface name (e.g., pialsurface6)')
    parser.add_argument('--custom-label', default=None,
                       help='Custom cortex label name for non-standard surfaces (e.g., cortex6 for {hemi}.cortex6.label)')
    parser.add_argument('--overwrite', action='store_true', default=False,
                       help='Overwrite existing output files (default: False - exit if files exist)')
    parser.add_argument('--visualize', action='store_true', default=False,
                       help='Generate visualization PNG files (default: False)')
    parser.add_argument('--compute-msd', dest='compute_msd', action='store_true', default=True, help='Compute MSDs (default: True)')
    parser.add_argument('--no-compute-msd', dest='compute_msd', action='store_false', help='Disable MSD computation')
    parser.add_argument('--scale', type=float, default=0.05, help='Scale for local measures (proportion of cortex area, e.g., 0.05)')
    parser.add_argument('--area-tol', type=float, default=0.01, help='Relative tolerance for area binary search (e.g., 0.01)')
    parser.add_argument('--eps', type=float, default=1e-6, help='Numerical tolerance for isoline tests')
    args = parser.parse_args()

    if not PYCORTEX_AVAILABLE:
        print("ERROR: pycortex is required! Install with: pip install pycortex")
        return
    
    # Validation: custom surfaces should have custom labels for proper masking
    if args.surf_type not in ['pial', 'white', 'inflated'] and args.custom_label is None:
        print("WARNING: Using custom surface without custom cortex label.")
        print("This may result in incorrect cortical masking if the mesh has been resampled.")
        print("Consider using --custom-label to specify the appropriate cortex label.")

    process_subject(args.subject_dir, args.subject_id, args.output_dir,
                    hemispheres=args.hemispheres, surf_type=args.surf_type, custom_label=args.custom_label,
                    compute_msd=args.compute_msd,
                    scale=args.scale, area_tol=args.area_tol, eps=args.eps,
                    overwrite=args.overwrite, visualize=args.visualize)

    print("Analysis complete!")
    

if __name__ == "__main__":
    main()
