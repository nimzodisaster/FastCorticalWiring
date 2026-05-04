#!/usr/bin/env python3
"""
Shared implementation of intrinsic cortical wiring cost measures from Ecker et al. 2013.

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
- Engine-adapter geodesics on a cortex-only submesh
- Robust polygon handling for geodesic disc area/perimeter computation
- Consistent epsilon + vertex-on-isoline handling; APARC -1 exclusion
- Candidate face filtering for area/perimeter (iso-band only)
- Precomputed face geometry/length scales for clipping tolerances
- BFS vertex ordering + warm-started radius bracketing
- Optional Numba JIT for the face-clipping loops
"""

import numpy as np
from collections import deque
from tqdm import tqdm  # Progress bars
import warnings

from distance_engines import create_distance_engine
warnings.filterwarnings('ignore')


def classify_nonmanifold_vertices(vertices, faces):
    """
    Classify boundary and non-manifold vertices from a triangle mesh.

    Args:
        vertices: Vertex coordinates with shape (N, 3). Used for indexing bounds.
        faces: Triangle indices with shape (M, 3).

    Returns:
        dict with keys:
            boundary_verts: set of vertex ids on boundary edges (edge has exactly one face)
            non_manifold_verts: set of vertex ids on non-manifold edges (edge has >2 faces)
            also_boundary: set intersection of boundary_verts and non_manifold_verts
            interior: set of non_manifold_verts not on boundary
    """
    V = np.asarray(vertices)
    F = np.asarray(faces)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"Invalid vertex array shape {V.shape}; expected (N, 3).")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"Invalid face array shape {F.shape}; expected (M, 3).")
    if F.size == 0:
        return {
            "boundary_verts": set(),
            "non_manifold_verts": set(),
            "also_boundary": set(),
            "interior": set(),
        }

    edge_faces = {}
    for fi, tri in enumerate(F):
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        for a, b in ((i0, i1), (i1, i2), (i2, i0)):
            edge = (a, b) if a < b else (b, a)
            edge_faces.setdefault(edge, []).append(fi)

    boundary_verts = set()
    non_manifold_verts = set()
    for (a, b), face_list in edge_faces.items():
        count = len(face_list)
        if count == 1:
            boundary_verts.add(a)
            boundary_verts.add(b)
        elif count > 2:
            non_manifold_verts.add(a)
            non_manifold_verts.add(b)

    also_boundary = non_manifold_verts & boundary_verts
    interior = non_manifold_verts - boundary_verts
    return {
        "boundary_verts": boundary_verts,
        "non_manifold_verts": non_manifold_verts,
        "also_boundary": also_boundary,
        "interior": interior,
    }

# ============================================================================
# OPTIONAL DEPENDENCIES: Numba JIT
# ============================================================================

# Try to import Numba for Just-In-Time compilation of critical loops
# This can provide 10-100x speedup for the geometric intersection computations
try:
    from numba import njit  # imported only to set flag; used inside guarded block
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

# ============================================================================
# NUMBA JIT KERNELS (Optimized Workspace Allocations)
# ============================================================================

if NUMBA_AVAILABLE:
    @njit(fastmath=True, cache=True)
    def _sign_eps(x, eps):
        if x < -eps:
            return -1
        if x > eps:
            return 1
        return 0

    @njit(fastmath=True, cache=True)
    def _edge_intersection_point(pi, pj, di, dj, r, abs_tol):
        if abs(di - r) <= abs_tol:
            return pi[0], pi[1], pi[2], True
        if abs(dj - r) <= abs_tol:
            return pj[0], pj[1], pj[2], True

        denom = dj - di
        if abs(denom) < 1e-20:
            return 0.0, 0.0, 0.0, False

        t = (r - di) / denom
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0

        return (
            pi[0] + (pj[0] - pi[0]) * t,
            pi[1] + (pj[1] - pi[1]) * t,
            pi[2] + (pj[2] - pi[2]) * t,
            True,
        )

    @njit(fastmath=True, cache=True)
    def _farthest_pair(P, m):
        best_i = 0
        best_j = 1
        best_d = -1.0

        for i in range(m):
            for j in range(i + 1, m):
                dx = P[j, 0] - P[i, 0]
                dy = P[j, 1] - P[i, 1]
                dz = P[j, 2] - P[i, 2]
                d2 = dx * dx + dy * dy + dz * dz
                if d2 > best_d:
                    best_d = d2
                    best_i = i
                    best_j = j
        return best_i, best_j, best_d ** 0.5

    @njit(fastmath=True, cache=True)
    def _triangle_area_with_unit_normal(a, b, c, n):
        v1x = b[0] - a[0]
        v1y = b[1] - a[1]
        v1z = b[2] - a[2]
        v2x = c[0] - a[0]
        v2y = c[1] - a[1]
        v2z = c[2] - a[2]

        cx = v1y * v2z - v1z * v2y
        cy = v1z * v2x - v1x * v2z
        cz = v1x * v2y - v1y * v2x
        return 0.5 * abs(n[0] * cx + n[1] * cy + n[2] * cz)

    @njit(fastmath=True, cache=True)
    def _area_inside_radius_band_kernel(V, F, unit_normals, face_areas, face_L, distances, r, eps, band_idx):
        area_sum = 0.0
        abs_tol = eps
        rel_tol = 1e-9
        
        # PRE-ALLOCATE WORKSPACES: Avoids heap allocation inside the tight loop
        P = np.zeros((3, 3), dtype=np.float64)
        P1 = np.zeros(3, dtype=np.float64)
        P2 = np.zeros(3, dtype=np.float64)
        
        for k in range(band_idx.shape[0]):
            f_idx = band_idx[k]
            i0, i1, i2 = F[f_idx, 0], F[f_idx, 1], F[f_idx, 2]
            d0, d1, d2 = distances[i0], distances[i1], distances[i2]
            
            if not (np.isfinite(d0) and np.isfinite(d1) and np.isfinite(d2)):
                continue
                
            v0 = V[i0]; v1 = V[i1]; v2 = V[i2]
            
            L = face_L[f_idx]
            tol = abs_tol + rel_tol * L
            tol2 = tol * tol

            s0 = _sign_eps(d0 - r, eps)
            s1 = _sign_eps(d1 - r, eps)
            s2 = _sign_eps(d2 - r, eps)
            
            b0 = (s0 <= 0)
            b1 = (s1 <= 0)
            b2 = (s2 <= 0)
            nin = (1 if b0 else 0) + (1 if b1 else 0) + (1 if b2 else 0)
            
            if nin == 0:
                continue
            if nin == 3:
                area_sum += face_areas[f_idx]
                continue
                
            # Reset per-face workspace index for this triangle's intersection points.
            m = 0
            
            if (s0 * s1 < 0) or ((s0 == 0) ^ (s1 == 0)):
                px, py, pz, ok = _edge_intersection_point(v0, v1, d0, d1, r, abs_tol)
                if ok:
                    P[m, 0], P[m, 1], P[m, 2] = px, py, pz
                    m += 1
                    
            if (s1 * s2 < 0) or ((s1 == 0) ^ (s2 == 0)):
                px, py, pz, ok = _edge_intersection_point(v1, v2, d1, d2, r, abs_tol)
                if ok:
                    dup = False
                    for t in range(m):
                        dx = px - P[t, 0]
                        dy = py - P[t, 1]
                        dz = pz - P[t, 2]
                        if dx*dx + dy*dy + dz*dz <= tol2:
                            dup = True
                            break
                    if not dup:
                        P[m, 0], P[m, 1], P[m, 2] = px, py, pz
                        m += 1
                        
            if (s2 * s0 < 0) or ((s2 == 0) ^ (s0 == 0)):
                px, py, pz, ok = _edge_intersection_point(v2, v0, d2, d0, r, abs_tol)
                if ok:
                    dup = False
                    for t in range(m):
                        dx = px - P[t, 0]
                        dy = py - P[t, 1]
                        dz = pz - P[t, 2]
                        if dx*dx + dy*dy + dz*dz <= tol2:
                            dup = True
                            break
                    if not dup:
                        P[m, 0], P[m, 1], P[m, 2] = px, py, pz
                        m += 1

            n = unit_normals[f_idx]
            
            if nin == 1 and m >= 2:
                a = v0 if b0 else (v1 if b1 else v2)
                i, j, _ = _farthest_pair(P, m)
                # Overwrite P1/P2 workspaces
                P1[0], P1[1], P1[2] = P[i, 0], P[i, 1], P[i, 2]
                P2[0], P2[1], P2[2] = P[j, 0], P[j, 1], P[j, 2]
                area_sum += _triangle_area_with_unit_normal(a, P1, P2, n)
                
            elif nin == 2:
                if not b0:
                    vo = v0; vi1 = v1; vi2 = v2
                    do = d0; di1 = d1; di2 = d2
                elif not b1:
                    vo = v1; vi1 = v2; vi2 = v0
                    do = d1; di1 = d2; di2 = d0
                else:
                    vo = v2; vi1 = v0; vi2 = v1
                    do = d2; di1 = d0; di2 = d1

                p1x, p1y, p1z, ok1 = _edge_intersection_point(vo, vi1, do, di1, r, abs_tol)
                p2x, p2y, p2z, ok2 = _edge_intersection_point(vo, vi2, do, di2, r, abs_tol)

                if ok1 and ok2:
                    P1[0], P1[1], P1[2] = p1x, p1y, p1z
                    P2[0], P2[1], P2[2] = p2x, p2y, p2z
                    outside_area = _triangle_area_with_unit_normal(vo, P1, P2, n)
                    inside_area = face_areas[f_idx] - outside_area
                    if inside_area < 0.0: inside_area = 0.0
                    if inside_area > face_areas[f_idx]: inside_area = face_areas[f_idx]
                    area_sum += inside_area
                elif m >= 2:
                    i, j, _ = _farthest_pair(P, m)
                    P1[0], P1[1], P1[2] = P[i, 0], P[i, 1], P[i, 2]
                    P2[0], P2[1], P2[2] = P[j, 0], P[j, 1], P[j, 2]
                    outside_area = _triangle_area_with_unit_normal(vo, P1, P2, n)
                    inside_area = face_areas[f_idx] - outside_area
                    if inside_area < 0.0: inside_area = 0.0
                    if inside_area > face_areas[f_idx]: inside_area = face_areas[f_idx]
                    area_sum += inside_area
                    
        return area_sum

    @njit(fastmath=True, cache=True)
    def _perimeter_at_radius_kernel(V, F, face_L, distances, r, eps, band_idx):
        perim_sum = 0.0
        abs_tol = eps
        rel_tol = 1e-9
        
        # PRE-ALLOCATE WORKSPACE
        P = np.zeros((3, 3), dtype=np.float64)
        
        for k in range(band_idx.shape[0]):
            f_idx = band_idx[k]
            i0, i1, i2 = F[f_idx, 0], F[f_idx, 1], F[f_idx, 2]
            d0, d1, d2 = distances[i0], distances[i1], distances[i2]
            
            if not (np.isfinite(d0) and np.isfinite(d1) and np.isfinite(d2)):
                continue
                
            v0 = V[i0]; v1 = V[i1]; v2 = V[i2]
            L = face_L[f_idx]
            tol = abs_tol + rel_tol * L
            tol2 = tol * tol

            s0 = _sign_eps(d0 - r, eps)
            s1 = _sign_eps(d1 - r, eps)
            s2 = _sign_eps(d2 - r, eps)
            
            m = 0
            
            if (s0 * s1 < 0) or ((s0 == 0) ^ (s1 == 0)):
                px, py, pz, ok = _edge_intersection_point(v0, v1, d0, d1, r, abs_tol)
                if ok:
                    P[m, 0], P[m, 1], P[m, 2] = px, py, pz
                    m += 1
                    
            if (s1 * s2 < 0) or ((s1 == 0) ^ (s2 == 0)):
                px, py, pz, ok = _edge_intersection_point(v1, v2, d1, d2, r, abs_tol)
                if ok:
                    dup = False
                    for t in range(m):
                        dx = px - P[t, 0]; dy = py - P[t, 1]; dz = pz - P[t, 2]
                        if dx*dx + dy*dy + dz*dz <= tol2:
                            dup = True; break
                    if not dup:
                        P[m, 0], P[m, 1], P[m, 2] = px, py, pz
                        m += 1
                        
            if (s2 * s0 < 0) or ((s2 == 0) ^ (s0 == 0)):
                px, py, pz, ok = _edge_intersection_point(v2, v0, d2, d0, r, abs_tol)
                if ok:
                    dup = False
                    for t in range(m):
                        dx = px - P[t, 0]; dy = py - P[t, 1]; dz = pz - P[t, 2]
                        if dx*dx + dy*dy + dz*dz <= tol2:
                            dup = True; break
                    if not dup:
                        P[m, 0], P[m, 1], P[m, 2] = px, py, pz
                        m += 1
                        
            if m >= 2:
                i, j, Lij = _farthest_pair(P, m)
                perim_sum += Lij
                
        return perim_sum       
else:
    import sys
    import logging

    # Set up a loud banner that cuts through standard terminal output
    LOUD_WARNING = """
    ================================================================================
    CRITICAL PERFORMANCE WARNING: NUMBA IS NOT INSTALLED OR FAILED TO LOAD
    ================================================================================
    The JIT-optimized geometric kernels could not be initialized. 
    This analysis is falling back to pure-Python polygon clipping. 
    
    Processing a standard FreeSurfer mesh (~150,000 vertices) in pure Python 
    will take an EXORBITANT amount of time (potentially days instead of 1-2 hours).
    
    It is highly recommended that you terminate this script and install Numba:
        pip install numba
    ================================================================================
    """
    
    # Print directly to stderr so it bypasses standard output redirection
    print(LOUD_WARNING, file=sys.stderr)
    
    # Also log it just in case they are piping stderr to a file
    logging.warning("Numba unavailable. Falling back to slow pure-Python operations.")

    # Pure-Python fallback utilities used by non-JIT code paths
    def _sign_eps(x, eps):
        """Non-JIT version of sign function for Python fallback."""
        return -1 if x < -eps else (1 if x > eps else 0)


# ============================================================================
# MAIN ANALYSIS CLASS
# ============================================================================

class FastCorticalWiringAnalysis:
    """
    Compute intrinsic cortical wiring costs with geodesics from potpourri3d.
    
    The computational core is format-agnostic. Provide already-loaded arrays:
    - vertices: (N, 3) float coordinates
    - faces: (M, 3) triangle indices
    - cortex_mask: (N,) boolean mask
    """

    DEFAULT_SCALES = (
        0.00060000,
        0.00100000,
        0.00200000,
        0.00400000,
        0.0080000,
        0.01600000,
    )

    def __init__(
        self,
        vertices,
        faces,
        cortex_mask,
        engine_type="potpourri",
        engine_kwargs=None,
        eps=1e-6,
        metadata=None,
        compute_anisotropy=False,
        strict_anisotropy=False,
        allow_interior_nonmanifold=False,
    ):
        """
        Initialize analysis from in-memory mesh arrays.
        
        Args:
            vertices: Vertex coordinates with shape (N, 3)
            faces: Triangle indices with shape (M, 3)
            cortex_mask: Boolean mask of cortical vertices with shape (N,)
            engine_type: Distance backend identifier ('potpourri', 'pycortex', or 'pygeodesic')
            engine_kwargs: Optional backend-specific kwargs
            eps: Numerical tolerance for geometric computations
            metadata: Optional metadata dict for provenance/naming
            compute_anisotropy: Initialize potpourri3d vector heat log-map solver
            strict_anisotropy: Raise if log-map anisotropy cannot be initialized
            allow_interior_nonmanifold: Continue even when interior non-manifold
                vertices are detected in the cortical submesh.

        Notes:
            use_robust=True is required for non-manifold interior geometry.
            FreeSurfer pial surfaces decimated with PyVista may contain
            non-manifold edges confined to medial wall boundaries; those can be
            safe to ignore for geodesic distance use in this pipeline.
        """
        V, F, M = self._validate_mesh_inputs(vertices, faces, cortex_mask)
        self.engine_type = str(engine_type).lower()
        self.engine_kwargs = dict(engine_kwargs or {})
        self.metadata = dict(metadata) if metadata is not None else {}
        self.subject_dir = self.metadata.get("subject_dir")
        self.subject_id = self.metadata.get("subject_id", "surface")
        self.hemi = self.metadata.get("hemi", "lh")
        self.surf_type = self.metadata.get("surf_type", "surface")
        self.eps = float(eps)
        self.custom_label = self.metadata.get("custom_label")
        self.compute_anisotropy = bool(compute_anisotropy)
        self.strict_anisotropy = bool(strict_anisotropy)
        self.allow_interior_nonmanifold = bool(allow_interior_nonmanifold)

        # Input mesh in original vertex space
        self.vertices_full = V
        self.faces_full = F
        self.n_vertices_full = self.vertices_full.shape[0]
        self.n_faces_full = self.faces_full.shape[0]
        
        # Cortex mask in original vertex space
        self.cortex_mask_full = M
        self.n_cortex_vertices = int(np.sum(self.cortex_mask_full))
        print(f"Loaded mesh: {self.n_vertices_full} vertices, {self.n_faces_full} faces")
        print(f"Cortical vertices: {self.n_cortex_vertices}")
        
        # Build cortex-only submesh for computational efficiency
        # This dramatically reduces computation time by working only on cortical regions
        (
            self.vertices,           # Cortex-only vertex coordinates
            self.faces,             # Cortex-only face indices (re-indexed)
            self.sub_to_orig,       # Map from submesh to original vertex indices
            self.orig_to_sub,       # Map from original to submesh vertex indices
        ) = self._build_cortex_submesh(self.vertices_full, self.faces_full, self.cortex_mask_full)

        manifold_report = classify_nonmanifold_vertices(self.vertices, self.faces)
        n_boundary = len(manifold_report["boundary_verts"])
        n_non_manifold = len(manifold_report["non_manifold_verts"])
        n_also_boundary = len(manifold_report["also_boundary"])
        n_interior = len(manifold_report["interior"])
        print(f"Boundary vertices: {n_boundary}")
        print(f"Non-manifold vertices: {n_non_manifold}")
        print(f"Non-manifold also on boundary: {n_also_boundary}")
        print(f"Interior non-manifold vertices: {n_interior}")
        if n_interior == 0:
            if n_non_manifold > 0:
                print("All non-manifold vertices are on the medial wall boundary.")
            else:
                print("No non-manifold vertices found.")
        else:
            print(
                f"{n_interior} non-manifold vertices are in the interior — "
                "geodesic distances may be corrupted."
            )
            if self.engine_type == "potpourri":
                if self.engine_kwargs.get("use_robust", False):
                    print("Robust potpourri3d mode is already enabled for this surface.")
                else:
                    self.engine_kwargs["use_robust"] = True
                    print(
                        "Switching this surface to robust potpourri3d mode "
                        "(use_robust=True)."
                    )
            elif not self.allow_interior_nonmanifold:
                warnings.warn(
                    "Interior non-manifold vertices detected in cortical submesh. "
                    "Automatic use_robust fallback is only available for the potpourri engine; "
                    f"continuing with engine_type={self.engine_type!r}.",
                    RuntimeWarning,
                )
        
        self.n_vertices = self.vertices.shape[0]
        self.n_faces = self.faces.shape[0]
        if self.n_vertices == 0:
            raise ValueError("Cortex mask produced an empty cortical submesh.")
        if self.n_faces == 0:
            raise ValueError("No valid cortical faces remain after applying cortex mask.")
        
        # Create distance engine on the cortical submesh
        V = np.ascontiguousarray(self.vertices, dtype=np.float64)
        F = np.ascontiguousarray(self.faces, dtype=np.int32)
        self.vertices = V
        self.faces = F

        # Identify physical submesh boundaries (free edges appear in exactly one face).
        edges = np.vstack(
            (
                self.faces[:, [0, 1]],
                self.faces[:, [1, 2]],
                self.faces[:, [2, 0]],
            )
        ).astype(np.int32, copy=False)
        edges.sort(axis=1)
        unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
        boundary_edges = unique_edges[edge_counts == 1]
        if boundary_edges.size:
            self.boundary_indices = np.unique(boundary_edges.reshape(-1)).astype(np.int32, copy=False)
        else:
            self.boundary_indices = np.empty((0,), dtype=np.int32)

        self.distance_engine = create_distance_engine(self.engine_type, V, F, self.engine_kwargs)
        print(f"Geodesic backend: {self.distance_engine.name}")

        self._vector_heat_solver = None
        self._vector_heat_unavailable_reason = None
        if self.compute_anisotropy:
            self._initialize_vector_heat_solver(strict=self.strict_anisotropy)

        # Cache face columns and reusable scratch buffers for per-face distance bounds.
        self._f0 = np.ascontiguousarray(self.faces[:, 0])
        self._f1 = np.ascontiguousarray(self.faces[:, 1])
        self._f2 = np.ascontiguousarray(self.faces[:, 2])
        self._d0_buf = np.empty(self.n_faces, dtype=np.float64)
        self._d1_buf = np.empty(self.n_faces, dtype=np.float64)
        self._d2_buf = np.empty(self.n_faces, dtype=np.float64)
        self._dmin_buf = np.empty(self.n_faces, dtype=np.float64)
        self._dmax_buf = np.empty(self.n_faces, dtype=np.float64)

        # Cache submesh adjacency and BFS traversal order for reuse across runs.
        self._adj = self._build_vertex_adjacency()
        self._bfs_order, _ = self._bfs_order_all(start=0)
        
        # Precompute face geometry for efficient area/perimeter calculations
        self.face_normals, self.face_areas, self.face_unit_normals, self.face_L = self._precompute_face_geometry(self.vertices, self.faces)
        self.vertex_normals_sub = self._compute_vertex_normals(self.vertices, self.faces)
        
        # Compute vertex areas (for normalization and total cortex area)
        self.vertex_areas_sub = self._compute_vertex_areas(self.vertices, self.faces)  # Areas in submesh
        self.vertex_areas = np.zeros(self.n_vertices_full, dtype=np.float64)  # Areas in full mesh
        self.vertex_areas[self.sub_to_orig] = self.vertex_areas_sub
        
        # Initialize result arrays (will be filled by computation methods)
        self.msd = np.full(self.n_vertices_full, np.nan, dtype=np.float32)
        self.active_scales = tuple(self.DEFAULT_SCALES)
        self.radius_function = {
            float(scale): np.full(self.n_vertices_full, np.nan, dtype=np.float32) for scale in self.active_scales
        }
        self.perimeter_function = {
            float(scale): np.full(self.n_vertices_full, np.nan, dtype=np.float32) for scale in self.active_scales
        }
        self.anisotropy_function = {
            float(scale): np.full(self.n_vertices_full, np.nan, dtype=np.float32) for scale in self.active_scales
        }
        self.dist_to_boundary = np.full(self.n_vertices_full, np.nan, dtype=np.float32)
        self.sampled_radii = None
        self.sampled_areas = None
        self.n_samples_per_vertex = None
        self.n_samples_between_scales = 0
        self.boundary_cap_fraction = None

    def _initialize_vector_heat_solver(self, strict=False):
        """Initialize potpourri3d's vector heat solver for log-map anisotropy."""
        if self._vector_heat_solver is not None:
            return
        if self._vector_heat_unavailable_reason is not None and not strict:
            return
        try:
            import potpourri3d as pp3d
        except Exception as exc:
            self._vector_heat_unavailable_reason = (
                "potpourri3d is required for log-map anisotropy. "
                "Install a version with MeshVectorHeatSolver.compute_log_map."
            )
            if strict:
                raise ImportError(self._vector_heat_unavailable_reason) from exc
            warnings.warn(self._vector_heat_unavailable_reason, RuntimeWarning)
            return

        solver_cls = getattr(pp3d, "MeshVectorHeatSolver", None)
        if solver_cls is None or not hasattr(solver_cls, "compute_log_map"):
            self._vector_heat_unavailable_reason = (
                "Installed potpourri3d does not expose MeshVectorHeatSolver.compute_log_map; "
                "upgrade potpourri3d to use log-map anisotropy."
            )
            if strict:
                raise RuntimeError(self._vector_heat_unavailable_reason)
            warnings.warn(self._vector_heat_unavailable_reason, RuntimeWarning)
            return

        self._vector_heat_solver = solver_cls(self.vertices, self.faces)

    @staticmethod
    def normalize_scales(scale):
        """Normalize scale input into an ordered tuple of unique valid fractions."""
        if scale is None:
            raw = list(FastCorticalWiringAnalysis.DEFAULT_SCALES)
        elif np.isscalar(scale):
            raw = [scale]
        else:
            raw = list(scale)

        if not raw:
            raise ValueError("At least one scale must be provided.")

        normalized = []
        seen = set()
        for item in raw:
            value = float(item)
            if not np.isfinite(value):
                raise ValueError(f"Invalid scale {item!r}: must be finite.")
            if value <= 0.0 or value > 1.0:
                raise ValueError(f"Invalid scale {item!r}: expected 0 < scale <= 1.")
            if value not in seen:
                seen.add(value)
                normalized.append(value)

        normalized.sort()
        return tuple(normalized)

    @staticmethod
    def scale_token(scale):
        """Stable string token for metric names and filenames."""
        return format(float(scale), "g")

    @staticmethod
    def _validate_mesh_inputs(vertices, faces, cortex_mask):
        """Validate and normalize input arrays."""
        V = np.asarray(vertices, dtype=np.float64)
        F_raw = np.asarray(faces)
        M = np.asarray(cortex_mask)

        if V.ndim != 2 or V.shape[1] != 3:
            raise ValueError(f"Invalid vertex array shape {V.shape}; expected (N, 3).")
        if F_raw.ndim != 2 or F_raw.shape[1] != 3:
            raise ValueError(f"Invalid face array shape {F_raw.shape}; expected (M, 3).")
        if F_raw.shape[0] == 0:
            raise ValueError("Face array is empty.")
        if M.ndim != 1:
            raise ValueError(f"Invalid cortex mask shape {M.shape}; expected (N,).")
        if M.shape[0] != V.shape[0]:
            raise ValueError(f"Mask length mismatch: mask has {M.shape[0]} entries, vertices has {V.shape[0]}.")

        if not np.issubdtype(F_raw.dtype, np.integer):
            if np.any(np.abs(F_raw - np.rint(F_raw)) > 0):
                raise ValueError("Face indices must be integers.")
            F = np.rint(F_raw).astype(np.int32)
        else:
            F = F_raw.astype(np.int32, copy=False)

        if np.any(F < 0):
            raise ValueError("Face indices contain negative values.")
        if np.any(F >= V.shape[0]):
            bad = int(np.max(F))
            raise ValueError(
                f"Face indices out of range: max face index {bad}, but only {V.shape[0]} vertices provided."
            )

        M = M.astype(bool, copy=False)
        if not np.any(M):
            raise ValueError("Cortex mask is empty (no True vertices).")

        return V, F, M

    @classmethod
    def from_freesurfer(
        cls,
        subject_dir,
        subject_id,
        hemi="lh",
        surf_type="pial",
        engine_type="potpourri",
        engine_kwargs=None,
        eps=1e-6,
        custom_label=None,
        mask_path=None,
        no_mask=False,
        allow_interior_nonmanifold=False,
    ):
        """
        Compatibility constructor for positional FreeSurfer workflows.
        """
        from io_utils import load_surface_and_mask

        vertices, faces, cortex_mask, metadata = load_surface_and_mask(
            standard="freesurfer",
            subject_dir=subject_dir,
            subject_id=subject_id,
            hemi=hemi,
            surf_type=surf_type,
            custom_label=custom_label,
            mask_path=mask_path,
            no_mask=no_mask,
        )
        return cls(
            vertices,
            faces,
            cortex_mask,
            engine_type=engine_type,
            engine_kwargs=engine_kwargs,
            eps=eps,
            metadata=metadata,
            allow_interior_nonmanifold=allow_interior_nonmanifold,
        )
        
    def _build_cortex_submesh(self, vertices, faces, cortex_mask):
            """
            Build a submesh containing only cortical vertices and valid faces.
            Filters out degenerate triangles and orphaned vertices to ensure 
            numerical stability in the Laplacian solver.
            """
            # 1. Keep only faces where all vertices are cortical
            keep_faces = cortex_mask[faces].all(axis=1)
            faces_kept_orig = faces[keep_faces]
            if faces_kept_orig.shape[0] == 0:
                return (
                    np.empty((0, 3), dtype=np.float64),
                    np.empty((0, 3), dtype=np.int32),
                    np.empty((0,), dtype=np.int32),
                    np.full(vertices.shape[0], -1, dtype=np.int32),
                )
            
            # 2. Filter out degenerate faces (area near zero)
            p0 = vertices[faces_kept_orig[:, 0]]
            p1 = vertices[faces_kept_orig[:, 1]]
            p2 = vertices[faces_kept_orig[:, 2]]
            
            # Cross product magnitude gives 2x the face area
            cross = np.cross(p1 - p0, p2 - p0)
            areas = 0.5 * np.linalg.norm(cross, axis=1)
            
            # Use a strict tolerance to drop sliver triangles
            valid_area_mask = areas > 1e-12
            faces_kept_orig = faces_kept_orig[valid_area_mask]
            if faces_kept_orig.shape[0] == 0:
                return (
                    np.empty((0, 3), dtype=np.float64),
                    np.empty((0, 3), dtype=np.int32),
                    np.empty((0,), dtype=np.int32),
                    np.full(vertices.shape[0], -1, dtype=np.int32),
                )
            
            # 3. Re-evaluate which vertices are actually used 
            # (Dropping degenerate faces might leave some vertices entirely unreferenced)
            used_vertices = np.unique(faces_kept_orig)
            
            # 4. Create contiguous vertex index mappings
            sub_to_orig = used_vertices.astype(np.int32)
            orig_to_sub = np.full(vertices.shape[0], -1, dtype=np.int32)
            orig_to_sub[sub_to_orig] = np.arange(sub_to_orig.shape[0], dtype=np.int32)
            
            # Remap face indices to submesh vertex indices
            faces_sub = orig_to_sub[faces_kept_orig]
            
            # Extract contiguous cortical vertex coordinates
            verts_sub = vertices[sub_to_orig]
            
            return verts_sub, faces_sub, sub_to_orig, orig_to_sub

    def _precompute_face_geometry(self, V, F):
        """
        Precompute face normals, areas, unit normals, and max edge length per face.
        
        This avoids recomputing these values repeatedly during area/perimeter calculations.
        Includes robust handling of near-degenerate faces using scale-aware tolerances.
        
        Returns:
            face_normals: Raw cross-product normals (not normalized)
            face_areas: Triangle areas  
            face_unit_normals: Unit normal vectors (robust for tiny triangles)
            face_L: Maximum edge length per face
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
        
        return normals, areas, unit, L

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

    def _compute_vertex_normals(self, V, F):
        """
        Compute area-weighted vertex normals from adjacent face normals.

        Degenerate vertices (zero accumulated normal norm) receive [0, 0, 1].
        """
        n_vertices = int(V.shape[0])
        normals = np.zeros((n_vertices, 3), dtype=np.float64)
        face_normals = self.face_normals
        for f_idx, face in enumerate(F):
            n = face_normals[f_idx]
            normals[face[0]] += n
            normals[face[1]] += n
            normals[face[2]] += n

        norms = np.linalg.norm(normals, axis=1)
        good = norms > 0.0
        normals[good] /= norms[good][:, None]
        normals[~good] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return normals

    def _tangent_frame_at_subvertex(self, sub_idx):
        """Return an orthonormal tangent frame (e1, e2, n) at a submesh vertex."""
        n = np.asarray(self.vertex_normals_sub[int(sub_idx)], dtype=np.float64)
        n_norm = float(np.linalg.norm(n))
        if not np.isfinite(n_norm) or n_norm <= 0.0:
            n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            n = n / n_norm

        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(ref, n))) > 0.95:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        e1 = np.cross(n, ref)
        e1_norm = float(np.linalg.norm(e1))
        if e1_norm <= 0.0 or not np.isfinite(e1_norm):
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            e1 = np.cross(n, ref)
            e1_norm = float(np.linalg.norm(e1))
        if e1_norm <= 0.0 or not np.isfinite(e1_norm):
            return (
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
                np.array([0.0, 1.0, 0.0], dtype=np.float64),
                np.array([0.0, 0.0, 1.0], dtype=np.float64),
            )

        e1 = e1 / e1_norm
        e2 = np.cross(n, e1)
        e2_norm = float(np.linalg.norm(e2))
        if e2_norm <= 0.0 or not np.isfinite(e2_norm):
            return (
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
                np.array([0.0, 1.0, 0.0], dtype=np.float64),
                np.array([0.0, 0.0, 1.0], dtype=np.float64),
            )
        e2 = e2 / e2_norm
        return e1, e2, n

    # ========================================================================
    # GEODESIC DISTANCE COMPUTATION
    # ========================================================================
    
    def _compute_geodesic_distances_from_subvertex(self, sub_idx):
        """
        Compute geodesic distances from a single vertex in the submesh.
        """
        d = self.distance_engine.compute_distance(int(sub_idx))
        return np.ascontiguousarray(d, dtype=np.float64)

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
    
    def _edge_intersections(self, r, d, v_face, f_idx):
        """
        Python fallback for finding edge-isoline intersections (when Numba unavailable).
        
        Returns intersection points where the level-set d=r crosses triangle edges.
        Uses the same robust logic as the Numba kernels: scale-aware tolerance
        and endpoint detection.
        """
        eps = self.eps
        P = []
        
        # Use precomputed face length scale for tolerance
        L = self.face_L[f_idx]
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
    
    def _area_band_python(self, radius, distances_sub, band_idx):
        """Python fallback for partially clipped faces in the isoline band."""
        r = float(radius)
        eps = self.eps
        V, F = self.vertices, self.faces
        area = 0.0
        unit_normals, areas = self.face_unit_normals, self.face_areas

        for f_idx in band_idx:
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

            n = unit_normals[f_idx]
            if n_in == 1:
                i_inside = int(np.where(inside)[0][0])
                P = self._edge_intersections(r, d, v_face, f_idx)
                if len(P) < 2:
                    continue
                a = v_face[i_inside]
                b, c = P[0], P[1]
                area += 0.5 * abs(np.dot(n, np.cross(b - a, c - a)))
            else:
                idx_out = int(np.where(~inside)[0][0])
                idx_in = np.where(inside)[0]
                vo = v_face[idx_out]
                vi1 = v_face[idx_in[0]]
                vi2 = v_face[idx_in[1]]
                do = d[idx_out]
                di1 = d[idx_in[0]]
                di2 = d[idx_in[1]]

                p1, p2 = None, None
                if abs(do - r) <= eps:
                    p1 = vo
                elif abs(di1 - r) <= eps:
                    p1 = vi1
                else:
                    den = di1 - do
                    if abs(den) > 1e-20:
                        t = (r - do) / den
                        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
                        p1 = vo + t * (vi1 - vo)

                if abs(do - r) <= eps:
                    p2 = vo
                elif abs(di2 - r) <= eps:
                    p2 = vi2
                else:
                    den = di2 - do
                    if abs(den) > 1e-20:
                        t = (r - do) / den
                        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
                        p2 = vo + t * (vi2 - vo)

                if p1 is None or p2 is None:
                    P = self._edge_intersections(r, d, v_face, f_idx)
                    if len(P) < 2:
                        continue
                    p1, p2 = P[0], P[1]

                outside_area = 0.5 * abs(np.dot(n, np.cross(p1 - vo, p2 - vo)))
                inside_area = areas[f_idx] - outside_area
                inside_area = max(0.0, min(float(areas[f_idx]), float(inside_area)))
                area += inside_area

        return float(area)

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
        
        fully_in_mask = dmax <= (r - eps)
        band_mask = (dmin <= (r + eps)) & ~fully_in_mask

        inside_area = float(self.face_areas[fully_in_mask].sum())
        band_idx = np.flatnonzero(band_mask).astype(np.int32, copy=False)

        if band_idx.size == 0:
            return inside_area
        
        if NUMBA_AVAILABLE:
            band_area = float(
                _area_inside_radius_band_kernel(
                    V,
                    F,
                    self.face_unit_normals,
                    self.face_areas,
                    self.face_L,
                    distances_sub,
                    r,
                    eps,
                    band_idx,
                )
            )
            return inside_area + band_area
        
        return inside_area + self._area_band_python(r, distances_sub, band_idx)

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
        band_idx = np.flatnonzero((dmin <= (r + eps)) & (dmax >= (r - eps))).astype(np.int32, copy=False)
        
        if NUMBA_AVAILABLE:
            return float(_perimeter_at_radius_kernel(V, F, self.face_L, distances_sub, r, eps, band_idx))
        
        # Python fallback
        perim = 0.0
        for f_idx in band_idx:
            face = F[f_idx]
            d = distances_sub[face]
            if not np.all(np.isfinite(d)):
                continue
                
            v_face = V[face]
            P = self._edge_intersections(r, d, v_face, f_idx)
            
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

    def _compute_log_map_at_subvertex(self, sub_idx):
        """
        Compute tangent-plane log-map coordinates from a source submesh vertex.

        Returns None when log-map anisotropy is not enabled or unavailable.
        """
        if self._vector_heat_solver is None:
            return None
        log_map = self._vector_heat_solver.compute_log_map(int(sub_idx))
        log_map = np.asarray(log_map, dtype=np.float64)
        if log_map.shape != (self.n_vertices, 2):
            raise ValueError(
                f"Vector heat log map returned shape {log_map.shape}; expected ({self.n_vertices}, 2)."
            )
        return np.ascontiguousarray(log_map)

    def _anisotropy_at_radius(self, log_map_xy, distances_sub, radius):
        """
        Eigenvalue ratio of the area-weighted second-moment tensor of a geodesic disk.

        Returns 1 - lambda_min / lambda_max in [0, 1]. Values near 0 indicate an
        isotropic disk in tangent coordinates; values near 1 indicate elongation.
        """
        if log_map_xy is None:
            return np.nan
        r = float(radius)
        if not np.isfinite(r) or r <= 0.0:
            return np.nan

        d = np.asarray(distances_sub, dtype=np.float64)
        xy_all = np.asarray(log_map_xy, dtype=np.float64)
        if xy_all.ndim != 2 or xy_all.shape[1] != 2 or xy_all.shape[0] != d.shape[0]:
            return np.nan

        interior = np.isfinite(d) & (d <= r + self.eps)
        interior &= np.isfinite(xy_all[:, 0]) & np.isfinite(xy_all[:, 1])
        if int(np.sum(interior)) < 3:
            return np.nan

        xy = xy_all[interior]
        w = self.vertex_areas_sub[interior]
        wsum = float(np.sum(w))
        if not np.isfinite(wsum) or wsum <= 0.0:
            return np.nan

        mean_xy = (xy * w[:, None]).sum(axis=0) / wsum
        centered = xy - mean_xy
        cov = (centered[:, :, None] * centered[:, None, :] * w[:, None, None]).sum(axis=0) / wsum
        eigvals = np.linalg.eigvalsh(cov)
        if eigvals.shape[0] != 2 or eigvals[1] <= 0.0 or not np.all(np.isfinite(eigvals)):
            return np.nan
        ratio = max(0.0, min(1.0, float(eigvals[0] / eigvals[1])))
        return 1.0 - ratio

    def _disk_anisotropy_from_vertices(self, sub_idx, distances_sub, radius):
        """
        Legacy tangent-projection anisotropy helper retained for tests/backcompat.

        This is an extrinsic approximation and returns 1 - lambda_min/lambda_max,
        so round disks are near 0 and elongated disks approach 1. The production
        anisotropy metric uses _anisotropy_at_radius with intrinsic log-map
        coordinates when vector heat is enabled.
        """
        r = float(radius)
        if not np.isfinite(r) or r <= 0.0:
            return np.nan
        d = np.asarray(distances_sub, dtype=np.float64)
        interior = np.isfinite(d) & (d <= r + self.eps)
        if int(np.sum(interior)) < 3:
            return np.nan

        e1, e2, _ = self._tangent_frame_at_subvertex(sub_idx)
        origin = self.vertices[int(sub_idx)]
        delta = self.vertices[interior] - origin
        xy = np.column_stack((delta @ e1, delta @ e2))
        w = self.vertex_areas_sub[interior]
        wsum = float(np.sum(w))
        if not np.isfinite(wsum) or wsum <= 0.0:
            return np.nan

        mean_xy = (xy * w[:, None]).sum(axis=0) / wsum
        centered = xy - mean_xy
        cov = (centered[:, :, None] * centered[:, None, :] * w[:, None, None]).sum(axis=0) / wsum
        eigvals = np.linalg.eigvalsh(cov)
        if eigvals[1] <= 0.0 or not np.all(np.isfinite(eigvals)):
            return np.nan
        ratio = max(0.0, min(1.0, float(eigvals[0] / eigvals[1])))
        return 1.0 - ratio

    def _ensure_sample_storage_capacity(self, needed_width):
        """Grow NaN-padded sampled radius/area storage when a row exceeds capacity."""
        needed_width = int(needed_width)
        if needed_width <= 0:
            return
        if self.sampled_radii is None or self.sampled_areas is None:
            width = max(needed_width, 1)
            self.sampled_radii = np.full((self.n_vertices_full, width), np.nan, dtype=np.float32)
            self.sampled_areas = np.full((self.n_vertices_full, width), np.nan, dtype=np.float32)
            self.n_samples_per_vertex = np.zeros(self.n_vertices_full, dtype=np.int32)
            return
        current = int(self.sampled_radii.shape[1])
        if needed_width <= current:
            return
        new_width = max(needed_width, current * 2)
        pad = new_width - current
        self.sampled_radii = np.pad(
            self.sampled_radii,
            ((0, 0), (0, pad)),
            mode="constant",
            constant_values=np.nan,
        )
        self.sampled_areas = np.pad(
            self.sampled_areas,
            ((0, 0), (0, pad)),
            mode="constant",
            constant_values=np.nan,
        )

    def _store_vertex_area_samples(self, orig_idx, samples, max_radius=None):
        """Deduplicate, sort, boundary-filter, and store sampled (radius, area) pairs."""
        if self.sampled_radii is None or self.sampled_areas is None or self.n_samples_per_vertex is None:
            self._ensure_sample_storage_capacity(1)

        clean = []
        for r, a in samples:
            r_f = float(r)
            a_f = float(a)
            if not (np.isfinite(r_f) and np.isfinite(a_f)):
                continue
            if r_f < 0.0:
                continue
            if max_radius is not None and np.isfinite(max_radius) and r_f > float(max_radius):
                continue
            clean.append((r_f, a_f))

        if not clean:
            self.n_samples_per_vertex[int(orig_idx)] = 0
            return 0

        clean.sort(key=lambda item: item[0])
        radii = np.asarray([item[0] for item in clean], dtype=np.float64)
        total_range = float(np.max(radii) - np.min(radii)) if radii.size else 0.0
        tol = max(1e-12, 1e-9 * total_range)

        dedup = []
        for r, a in clean:
            if dedup and abs(r - dedup[-1][0]) <= tol:
                dedup[-1] = (r, a)
            else:
                dedup.append((r, a))

        n = len(dedup)
        self._ensure_sample_storage_capacity(n)
        row = int(orig_idx)
        self.sampled_radii[row, :] = np.nan
        self.sampled_areas[row, :] = np.nan
        if n:
            self.sampled_radii[row, :n] = np.asarray([x[0] for x in dedup], dtype=np.float32)
            self.sampled_areas[row, :n] = np.asarray([x[1] for x in dedup], dtype=np.float32)
        self.n_samples_per_vertex[row] = np.int32(n)
        return n

    def _find_radius_for_area(
        self,
        distances_sub,
        target_area,
        tol=0.01,
        max_iter=30,
        dmin=None,
        dmax=None,
        r_init=None,
        r_lower=None,
        r_upper=None,
        delta0=None,
        expand_factor=1.6,
        max_expand=25,
        neighbor_range=None,
        r_euclid=None,
    ):
        """
        Find geodesic radius r such that area_inside_radius(r) ≈ target_area.

        Optional warm-start around r_init plus bracket expansion are used to
        accelerate convergence while preserving exact bisection solving.
        """
        history = []

        def record(r_val, area_val):
            r_f = float(r_val)
            a_f = float(area_val)
            if np.isfinite(r_f) and np.isfinite(a_f):
                history.append((r_f, a_f))
            return a_f

        valid = np.isfinite(distances_sub)
        if not np.any(valid):
            return np.nan, 0, history

        # Precompute per-face min/max distance if not provided (used for fast face rejection)
        if dmin is None or dmax is None:
            df = distances_sub[self.faces]
            dmin = df.min(axis=1)
            dmax = df.max(axis=1)

        # Global maximum reachable distance (used for fallback/upper caps)
        d_global_max = float(np.max(distances_sub[valid]))
        if not np.isfinite(d_global_max) or d_global_max <= 0:
            return np.nan, 0, history

        # Quick feasibility check: even at max radius, can we reach target_area?
        area_at_max = record(
            d_global_max,
            self._area_inside_radius(d_global_max, distances_sub, dmin=dmin, dmax=dmax),
        )
        if area_at_max + 1e-12 < target_area:
            return np.nan, 0, history  # target too large for this (sub)mesh / disconnected distances

        # Helper to compute area (kept local to avoid attribute lookups in tight loops)
        def area_inside(r):
            return record(r, self._area_inside_radius(r, distances_sub, dmin=dmin, dmax=dmax))

        # If target_area is tiny, return ~0
        if target_area <= 0:
            record(0.0, 0.0)
            return 0.0, 0, history

        def clamp_radius(x):
            if x is None:
                return None
            try:
                val = float(x)
            except Exception:
                return None
            if not np.isfinite(val):
                return None
            return min(d_global_max, max(0.0, val))

        # ---------------------------------------------------------------------
        # Bracketing
        # ---------------------------------------------------------------------
        lower = clamp_radius(r_lower)
        upper = clamp_radius(r_upper)
        seed = clamp_radius(r_init)

        # Choose a small initial delta0 if not provided.
        if delta0 is None or not np.isfinite(delta0) or delta0 <= 0:
            delta0 = 0.01 * d_global_max
        delta0 = float(max(delta0, 1e-12))

        if lower is None and upper is None:
            if seed is None:
                r_min, r_max = 0.0, d_global_max
            else:
                r_min = max(0.0, seed - delta0)
                r_max = min(d_global_max, seed + delta0)
        else:
            r_min = lower if lower is not None else (max(0.0, seed - delta0) if seed is not None else 0.0)
            r_max = upper if upper is not None else (min(d_global_max, seed + delta0) if seed is not None else d_global_max)

        if r_min > r_max:
            r_min, r_max = r_max, r_min
        if r_max <= r_min + 1e-12:
            if seed is None:
                seed = 0.5 * (r_min + r_max)
            r_min = max(0.0, seed - delta0)
            r_max = min(d_global_max, seed + delta0)
        if r_max <= r_min + 1e-12:
            r_min, r_max = 0.0, d_global_max

        a_min = area_inside(r_min) if r_min > 0 else record(0.0, 0.0)
        a_max = area_inside(r_max)

        # Symmetric bracket recovery:
        # - contract lower bound stepwise if it is already above target
        # - expand upper bound stepwise if it is below target
        n_contract = 0
        while a_min > target_area + 1e-12 and r_min > 0.0 and n_contract < max_expand:
            new_r_min = max(0.0, r_min / expand_factor)
            if new_r_min >= r_min - 1e-12:
                break
            r_min = new_r_min
            a_min = area_inside(r_min) if r_min > 0 else record(0.0, 0.0)
            n_contract += 1

        # Compute adaptive expansion step size.
        # Use neighbor variability as the primary scale, with a fallback
        # based on the Euclidean radius estimate when neighbor data is absent.
        if neighbor_range is not None and neighbor_range > 0.0:
            _expand_step = neighbor_range * 2.0
        elif r_euclid is not None and r_euclid > 0.0:
            _expand_step = 0.05 * r_euclid
        else:
            _expand_step = 0.01 * d_global_max

        # Minimum step to avoid getting stuck in flat regions
        _expand_step = max(_expand_step, 1e-4)

        n_expand = 0
        while a_max + 1e-12 < target_area and r_max < d_global_max and n_expand < max_expand:
            new_r_max = min(d_global_max, r_max + _expand_step)
            if new_r_max <= r_max + 1e-12:
                break
            r_max = new_r_max
            a_max = area_inside(r_max)
            n_expand += 1

        # Fallback: if additive expansion exhausted max_expand without bracketing,
        # switch to a single multiplicative jump to global bounds.
        if a_max + 1e-12 < target_area:
            r_max = d_global_max
            a_max = area_inside(r_max)

        # Final conservative fallback if bracket still does not contain target.
        if a_min > target_area + 1e-12 or a_max + 1e-12 < target_area:
            r_min, r_max = 0.0, d_global_max

        _bisection_count = 0
        for _ in range(max_iter):
            r_mid = 0.5 * (r_min + r_max)
            a_mid = area_inside(r_mid)
            _bisection_count += 1

            # Convergence: relative area error
            if abs(a_mid - target_area) / target_area < tol:
                return r_mid, _bisection_count, history

            if a_mid < target_area:
                r_min = r_mid
            else:
                r_max = r_mid

        r_final = 0.5 * (r_min + r_max)
        if not history or abs(history[-1][0] - r_final) > 0.0:
            area_inside(r_final)
        return r_final, _bisection_count, history

    def _build_vertex_adjacency(self):
        """
        Build submesh vertex adjacency list from triangle faces.
        """
        n = self.vertices.shape[0]
        adj = [set() for _ in range(n)]
        for a, b, c in self.faces:
            ia = int(a); ib = int(b); ic = int(c)
            adj[ia].add(ib); adj[ia].add(ic)
            adj[ib].add(ia); adj[ib].add(ic)
            adj[ic].add(ia); adj[ic].add(ib)
        return [sorted(list(s)) for s in adj]

    def _bfs_order_all(self, start=0):
        """
        BFS traversal order over all components of the submesh graph.
        """
        adj = getattr(self, "_adj", None)
        if adj is None:
            adj = self._build_vertex_adjacency()
        n = len(adj)
        visited = np.zeros(n, dtype=bool)
        order = []

        def bfs_from(s):
            q = deque([s])
            visited[s] = True
            while q:
                v = q.popleft()
                order.append(v)
                for nb in adj[v]:
                    if not visited[nb]:
                        visited[nb] = True
                        q.append(nb)

        try:
            s0_candidate = int(start)
        except Exception:
            s0_candidate = 0
        s0 = s0_candidate if 0 <= s0_candidate < n else 0
        bfs_from(s0)

        for v in range(n):
            if not visited[v]:
                bfs_from(v)

        return order, adj

    # ========================================================================
    # PUBLIC COMPUTATION METHODS
    # ========================================================================


    def compute_all_wiring_costs(
        self,
        compute_msd=True,
        scale=None,
        area_tol=0.01,
        vertex_subset=None,
        n_samples_between_scales=10,
        compute_anisotropy=None,
        boundary_cap_fraction=0.5,
    ):
        """
        Compute all wiring cost metrics (MSD, radius, perimeter, anisotropy) in a single pass.

        This optimized method iterates through each cortical vertex only once. In each
        iteration, it computes the geodesic distance vector and then calculates all
        derived metrics (MSD, radius, perimeter, anisotropy) from that vector before discarding it.
        This avoids re-computing the expensive geodesic distances in separate loops.

        Args:
            compute_msd (bool): Deprecated and ignored; MSD is always computed.
            scale: Single scale or iterable of scales as proportions of total cortical area.
            area_tol (float): Relative tolerance for the area binary search.
            vertex_subset: Optional iterable of original vertex indices to compute.
            n_samples_between_scales: Number of supplementary log-spaced area samples
                between adjacent solved target-scale radii. Use 0 for bisection
                history only; values above ~5 increase storage with diminishing returns.
            compute_anisotropy: Optional override for log-map anisotropy computation.
            boundary_cap_fraction: Optional multiplier on distance-to-boundary used
                to skip supplementary samples near mesh boundaries. Set to None
                to disable. Converged target-scale samples are always retained.
        """
        scales = self.normalize_scales(scale)
        n_samples_between_scales = int(n_samples_between_scales)
        if n_samples_between_scales < 0:
            raise ValueError("n_samples_between_scales must be >= 0.")
        if boundary_cap_fraction is None:
            boundary_cap_value = None
        else:
            boundary_cap_value = float(boundary_cap_fraction)
            if not np.isfinite(boundary_cap_value) or boundary_cap_value < 0.0:
                raise ValueError("boundary_cap_fraction must be finite and >= 0, or None.")
        do_anisotropy = self.compute_anisotropy if compute_anisotropy is None else bool(compute_anisotropy)
        if do_anisotropy and self._vector_heat_solver is None:
            self._initialize_vector_heat_solver(strict=self.strict_anisotropy)

        self.active_scales = scales
        self.n_samples_between_scales = n_samples_between_scales
        self.boundary_cap_fraction = boundary_cap_value
        self.msd[:] = np.nan
        self.dist_to_boundary = np.full(self.n_vertices_full, np.nan, dtype=np.float32)
        self.radius_function = {
            float(s): np.full(self.n_vertices_full, np.nan, dtype=np.float32) for s in scales
        }
        self.perimeter_function = {
            float(s): np.full(self.n_vertices_full, np.nan, dtype=np.float32) for s in scales
        }
        self.anisotropy_function = {
            float(s): np.full(self.n_vertices_full, np.nan, dtype=np.float32) for s in scales
        }
        initial_sample_width = max(1, len(scales) * 80 + max(0, len(scales) - 1) * n_samples_between_scales)
        self.sampled_radii = np.full((self.n_vertices_full, initial_sample_width), np.nan, dtype=np.float32)
        self.sampled_areas = np.full((self.n_vertices_full, initial_sample_width), np.nan, dtype=np.float32)
        self.n_samples_per_vertex = np.zeros(self.n_vertices_full, dtype=np.int32)

        print("Computing Mean Separation Distances and Local Wiring Costs...")

        # Calculate target area once for local measures
        total_area = float(np.sum(self.vertex_areas_sub))
        target_areas = {float(s): total_area * float(s) for s in scales}
        print("Target areas for local measures:")
        for s in scales:
            target_area = target_areas[float(s)]
            print(f"  - {s*100:.2f}%: {target_area:.2f} mm² of {total_area:.2f} mm²")

        order_sub = list(self._bfs_order)
        adj = self._adj
        if vertex_subset is not None:
            subset_sub_set = set()
            for raw_idx in vertex_subset:
                try:
                    orig_idx = int(raw_idx)
                except Exception:
                    continue
                if orig_idx < 0 or orig_idx >= self.n_vertices_full:
                    continue
                if not self.cortex_mask_full[orig_idx]:
                    continue
                sub_idx = int(self.orig_to_sub[orig_idx])
                if sub_idx >= 0:
                    subset_sub_set.add(sub_idx)
            order_sub = [v for v in order_sub if v in subset_sub_set]
            print(f"Restricting computation to {len(order_sub)} specified vertices.")

        r_sub_by_scale = {float(s): np.full(self.n_vertices, np.nan, dtype=np.float32) for s in scales}
        r_euclid_by_scale = {
            float(s): (float(np.sqrt(target_areas[float(s)] / np.pi)) if target_areas[float(s)] > 0 else 0.0)
            for s in scales
        }

        import time as _time

        _t_geodesic = 0.0
        _t_msd      = 0.0
        _t_dminmax  = 0.0
        _t_radius   = 0.0
        _t_perim    = 0.0
        _t_logmap   = 0.0
        _t_anisotropy = 0.0
        _t_samples  = 0.0
        _n_iters    = 0
        _n_total    = len(order_sub)

        # Single loop over all cortical vertices (BFS order for warm-start locality)
        for sub_idx in tqdm(order_sub, desc="Computing wiring costs"):

            # --- Phase 1: Geodesic solve ---
            _t0 = _time.perf_counter()
            d_sub = self._compute_geodesic_distances_from_subvertex(sub_idx)
            _dt_geodesic = _time.perf_counter() - _t0
            _t_geodesic += _dt_geodesic

            _t0 = _time.perf_counter()
            log_map_xy = self._compute_log_map_at_subvertex(sub_idx) if do_anisotropy else None
            _dt_logmap = _time.perf_counter() - _t0
            _t_logmap += _dt_logmap

            orig_idx = self.sub_to_orig[sub_idx]

            if self.boundary_indices.size:
                b_dists = d_sub[self.boundary_indices]
                b_finite = b_dists[np.isfinite(b_dists)]
                min_b_dist = float(np.min(b_finite)) if b_finite.size else np.nan
            else:
                min_b_dist = np.inf
            self.dist_to_boundary[orig_idx] = np.float32(min_b_dist)

            # --- Phase 2: MSD ---
            _t0 = _time.perf_counter()
            valid = (d_sub > self.eps) & np.isfinite(d_sub)
            if np.any(valid):
                w = self.vertex_areas_sub[valid]
                wsum = float(np.sum(w))
                if np.isfinite(wsum) and wsum > 0.0:
                    msd_val = float((d_sub[valid] * w).sum() / wsum)
                else:
                    msd_val = np.nan
            else:
                msd_val = np.nan
            self.msd[orig_idx] = np.float32(msd_val)
            _dt_msd = _time.perf_counter() - _t0
            _t_msd += _dt_msd

            # --- Phase 3: dmin/dmax buffers ---
            _t0 = _time.perf_counter()
            np.take(d_sub, self._f0, out=self._d0_buf)
            np.take(d_sub, self._f1, out=self._d1_buf)
            np.take(d_sub, self._f2, out=self._d2_buf)
            np.minimum(self._d0_buf, self._d1_buf, out=self._dmin_buf)
            np.minimum(self._dmin_buf, self._d2_buf, out=self._dmin_buf)
            np.maximum(self._d0_buf, self._d1_buf, out=self._dmax_buf)
            np.maximum(self._dmax_buf, self._d2_buf, out=self._dmax_buf)
            _dt_dminmax = _time.perf_counter() - _t0
            _t_dminmax += _dt_dminmax

            # --- Phase 4 & 5: Radius bisection and perimeter (per scale) ---
            r_prev_scale = np.nan
            _is_first_vertex = (_n_iters == 0)
            _dt_radius_iter = 0.0
            _dt_perim_iter = 0.0
            _dt_anisotropy_iter = 0.0
            _bisection_iters_by_scale = []
            _area_samples_for_vertex = []
            for s in scales:
                scale_key = float(s)
                r_sub = r_sub_by_scale[scale_key]
                solved_neighbor_r = [float(r_sub[u]) for u in adj[sub_idx] if np.isfinite(r_sub[u])]

                neighbor_lower = None
                neighbor_upper = None
                if solved_neighbor_r:
                    neighbors = np.asarray(solved_neighbor_r, dtype=np.float64)
                    neighbor_eps = max(self.eps * 10.0, 1e-6)
                    neighbor_lower = max(0.0, float(np.min(neighbors)) - neighbor_eps)
                    neighbor_upper = float(np.max(neighbors)) + neighbor_eps
                    r_init = 0.5 * (neighbor_lower + neighbor_upper)
                    _neighbor_range = float(np.max(neighbors)) - float(np.min(neighbors))
                else:
                    r_init = r_euclid_by_scale[scale_key]
                    _neighbor_range = 0.0

                if np.isfinite(r_prev_scale):
                    if neighbor_lower is None:
                        r_lower = float(r_prev_scale)
                    else:
                        r_lower = max(neighbor_lower, float(r_prev_scale))
                else:
                    r_lower = neighbor_lower
                r_upper = neighbor_upper

                _t0 = _time.perf_counter()
                if _is_first_vertex:
                    _cold_r_euclid = r_euclid_by_scale[scale_key]
                    _cold_delta0 = 0.4 * _cold_r_euclid
                    _r_out = self._find_radius_for_area(
                        d_sub,
                        target_areas[scale_key],
                        tol=area_tol,
                        dmin=self._dmin_buf,
                        dmax=self._dmax_buf,
                        r_init=_cold_r_euclid,
                        r_lower=max(0.0, _cold_r_euclid - _cold_delta0),
                        r_upper=_cold_r_euclid + _cold_delta0,
                        delta0=_cold_delta0,
                        max_iter=50,
                    )
                else:
                    _r_out = self._find_radius_for_area(
                        d_sub,
                        target_areas[scale_key],
                        tol=area_tol,
                        dmin=self._dmin_buf,
                        dmax=self._dmax_buf,
                        r_init=r_init,
                        r_lower=r_lower,
                        r_upper=r_upper,
                        neighbor_range=_neighbor_range,
                        r_euclid=r_euclid_by_scale[scale_key],
                    )
                if isinstance(_r_out, tuple):
                    if len(_r_out) >= 3:
                        r, _bcount, _history = _r_out[:3]
                    elif len(_r_out) == 2:
                        r, _bcount = _r_out
                        _history = []
                    else:
                        r, _bcount, _history = _r_out[0], 0, []
                else:
                    # Compatibility for tests/mocks that still return a scalar radius.
                    r, _bcount, _history = _r_out, 0, []
                _dt_radius = _time.perf_counter() - _t0
                _t_radius += _dt_radius
                _dt_radius_iter += _dt_radius
                if _history:
                    _area_samples_for_vertex.extend(_history)
                if not np.isfinite(r):
                    _bisection_iters_by_scale.append(0)
                    r_sub[sub_idx] = np.nan
                    self.radius_function[scale_key][orig_idx] = np.nan
                    self.perimeter_function[scale_key][orig_idx] = np.nan
                    self.anisotropy_function[scale_key][orig_idx] = np.nan
                    continue

                _bisection_iters_by_scale.append(int(_bcount))
                r_prev_scale = float(r)
                _area_samples_for_vertex.append((float(r), float(target_areas[scale_key])))

                _t0 = _time.perf_counter()
                perim = self._perimeter_at_radius(r, d_sub, dmin=self._dmin_buf, dmax=self._dmax_buf)
                _dt_perim = _time.perf_counter() - _t0
                _t_perim += _dt_perim
                _dt_perim_iter += _dt_perim

                _t0 = _time.perf_counter()
                if log_map_xy is not None:
                    anisotropy_val = self._anisotropy_at_radius(log_map_xy, d_sub, r)
                elif do_anisotropy:
                    anisotropy_val = np.nan
                else:
                    anisotropy_val = self._disk_anisotropy_from_vertices(sub_idx, d_sub, r)
                _dt_anis = _time.perf_counter() - _t0
                _t_anisotropy += _dt_anis
                _dt_anisotropy_iter += _dt_anis
                r_sub[sub_idx] = np.float32(r)
                self.radius_function[scale_key][orig_idx] = np.float32(r)
                self.perimeter_function[scale_key][orig_idx] = np.float32(perim)
                self.anisotropy_function[scale_key][orig_idx] = np.float32(anisotropy_val)

            _t0 = _time.perf_counter()
            if boundary_cap_value is not None and np.isfinite(self.dist_to_boundary[orig_idx]):
                boundary_cap = boundary_cap_value * float(self.dist_to_boundary[orig_idx])
            else:
                boundary_cap = np.inf
            if n_samples_between_scales > 0:
                solved_radii_for_vertex = [float(r_sub_by_scale[float(s)][sub_idx]) for s in scales]
                for i in range(len(solved_radii_for_vertex) - 1):
                    r_lo = solved_radii_for_vertex[i]
                    r_hi = solved_radii_for_vertex[i + 1]
                    if not (np.isfinite(r_lo) and np.isfinite(r_hi) and r_hi > r_lo and r_lo > 0.0):
                        continue
                    extra_rs = np.exp(
                        np.linspace(
                            np.log(r_lo),
                            np.log(r_hi),
                            n_samples_between_scales + 2,
                        )
                    )[1:-1]
                    for r_extra in extra_rs:
                        if float(r_extra) > boundary_cap:
                            continue
                        a_extra = self._area_inside_radius(
                            float(r_extra),
                            d_sub,
                            dmin=self._dmin_buf,
                            dmax=self._dmax_buf,
                        )
                        _area_samples_for_vertex.append((float(r_extra), float(a_extra)))

            self._store_vertex_area_samples(orig_idx, _area_samples_for_vertex, max_radius=None)
            _dt_samples_iter = _time.perf_counter() - _t0
            _t_samples += _dt_samples_iter

            _n_iters += 1
            _dt_total_iter = (
                _dt_geodesic
                + _dt_logmap
                + _dt_msd
                + _dt_dminmax
                + _dt_radius_iter
                + _dt_perim_iter
                + _dt_anisotropy_iter
                + _dt_samples_iter
            )
            tqdm.write(
                f"[{_n_iters}/{_n_total}] "
                f"geodesic={1000.0 * _dt_geodesic:.2f}ms "
                f"log_map={1000.0 * _dt_logmap:.2f}ms "
                f"msd={1000.0 * _dt_msd:.2f}ms "
                f"dminmax={1000.0 * _dt_dminmax:.2f}ms "
                f"radius={1000.0 * _dt_radius_iter:.2f}ms "
                f"perim={1000.0 * _dt_perim_iter:.2f}ms "
                f"anisotropy={1000.0 * _dt_anisotropy_iter:.2f}ms "
                f"samples={1000.0 * _dt_samples_iter:.2f}ms "
                f"bisection_iters={'+'.join(str(x) for x in _bisection_iters_by_scale)} "
                f"total={1000.0 * _dt_total_iter:.2f}ms"
            )

        if self.n_samples_per_vertex is not None and self.sampled_radii is not None:
            max_used = int(np.max(self.n_samples_per_vertex)) if self.n_samples_per_vertex.size else 0
            final_width = max(1, max_used)
            self.sampled_radii = self.sampled_radii[:, :final_width]
            self.sampled_areas = self.sampled_areas[:, :final_width]

        _t_total = (
            _t_geodesic
            + _t_logmap
            + _t_msd
            + _t_dminmax
            + _t_radius
            + _t_perim
            + _t_anisotropy
            + _t_samples
        )
        if _t_total > 0 and _n_iters > 0:
            _per = lambda t: f"{t:.1f}s ({100.0 * t / _t_total:.1f}%)"
            _timing_lines = [
                f"Timing breakdown over {_n_iters} vertices:",
                f"  geodesic solve : {_per(_t_geodesic)}",
                f"  log map        : {_per(_t_logmap)}",
                f"  MSD            : {_per(_t_msd)}",
                f"  dmin/dmax      : {_per(_t_dminmax)}",
                f"  radius bisect  : {_per(_t_radius)}  [{len(scales)} scales]",
                f"  perimeter      : {_per(_t_perim)}  [{len(scales)} scales]",
                f"  anisotropy     : {_per(_t_anisotropy)}  [{len(scales)} scales]",
                f"  samples        : {_per(_t_samples)}",
                f"  total          : {_t_total:.1f}s",
                f"  per vertex     : {1000.0 * _t_total / _n_iters:.2f} ms/vertex",
                f"  geodesic frac  : {100.0 * _t_geodesic / _t_total:.1f}%",
                f"  geometry frac  : {100.0 * (_t_msd + _t_dminmax + _t_radius + _t_perim + _t_anisotropy + _t_samples) / _t_total:.1f}%",
            ]
            for line in _timing_lines:
                print(line)

        # Print summary statistics at the end
        valid_msd = self.msd[np.isfinite(self.msd)]
        if valid_msd.size:
            print(f"MSD stats: min={np.min(valid_msd):.2f}, max={np.max(valid_msd):.2f}, mean={np.mean(valid_msd):.2f}")

        for s in scales:
            scale_key = float(s)
            vr = self.radius_function[scale_key][np.isfinite(self.radius_function[scale_key])]
            vp = self.perimeter_function[scale_key][np.isfinite(self.perimeter_function[scale_key])]
            va = self.anisotropy_function[scale_key][np.isfinite(self.anisotropy_function[scale_key])]
            if vr.size:
                print(
                    f"Radius stats @ {s*100:.2f}%: "
                    f"min={np.min(vr):.2f}, max={np.max(vr):.2f}, mean={np.mean(vr):.2f}"
                )
            if vp.size:
                print(
                    f"Perimeter stats @ {s*100:.2f}%: "
                    f"min={np.min(vp):.2f}, max={np.max(vp):.2f}, mean={np.mean(vp):.2f}"
                )
            if va.size:
                print(
                    f"Anisotropy stats @ {s*100:.2f}%: "
                    f"min={np.min(va):.3f}, max={np.max(va):.3f}, mean={np.mean(va):.3f}"
                )

        return self.msd, self.radius_function, self.perimeter_function
            
    # ========================================================================
    # INPUT/OUTPUT
    # ========================================================================
    
    def get_metric_arrays(self):
        """Return computed scalar metric arrays; sampled radius/area pairs are saved separately."""
        out = {
            "msd": self.msd,
            "dist_to_boundary": self.dist_to_boundary,
        }
        for scale in self.active_scales:
            scale_key = float(scale)
            token = self.scale_token(scale_key)
            out[f"radius_{token}"] = self.radius_function[scale_key]
            out[f"perimeter_{token}"] = self.perimeter_function[scale_key]
            out[f"anisotropy_{token}"] = self.anisotropy_function[scale_key]
        return out
