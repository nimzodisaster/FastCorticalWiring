#!/usr/bin/env python3
"""Geodesic distance engine adapters for FastCorticalWiring."""

from abc import ABC, abstractmethod
import importlib

import numpy as np
import scipy.sparse as sp

_BACKEND_CHECK_IMPORT_ERRORS = []


def _resolve_verify_suitesparse():
    candidates = (
        "backend_check",
        ".backend_check",
    )
    for mod_name in candidates:
        try:
            if mod_name.startswith("."):
                mod = importlib.import_module(mod_name, package=__package__)
            else:
                mod = importlib.import_module(mod_name)
        except Exception as exc:
            _BACKEND_CHECK_IMPORT_ERRORS.append((mod_name, exc))
            continue
        fn = getattr(mod, "verify_suitesparse", None)
        if callable(fn):
            return fn
        _BACKEND_CHECK_IMPORT_ERRORS.append((mod_name, AttributeError("verify_suitesparse() missing")))
    return None


_VERIFY_SUITESPARSE = _resolve_verify_suitesparse()


def verify_suitesparse(**kwargs) -> bool:
    if _VERIFY_SUITESPARSE is None:
        return False
    return bool(_VERIFY_SUITESPARSE(**kwargs))


class BaseDistanceEngine(ABC):
    """Minimal interface expected by the shared wiring pipeline."""
    supports_batching = False

    def __init__(self, vertices, faces, **engine_kwargs):
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(faces, dtype=np.int32)
        self.engine_kwargs = dict(engine_kwargs or {})

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def compute_distance(self, source_idx: int) -> np.ndarray:
        raise NotImplementedError

    def compute_distance_batch(self, source_indices) -> np.ndarray:
        """Compute one-to-all distances for multiple source vertices.

        Returns an array with shape (n_vertices, n_sources). Engines with no
        native batched solver inherit this repeated single-source fallback.
        """
        sources = np.atleast_1d(np.asarray(source_indices, dtype=np.int64)).reshape(-1)
        if sources.size == 0:
            return np.empty((self.vertices.shape[0], 0), dtype=np.float64)
        columns = [self.compute_distance(int(src)) for src in sources]
        return np.ascontiguousarray(np.column_stack(columns), dtype=np.float64)


class PotpourriDistanceEngine(BaseDistanceEngine):
    """potpourri3d heat-method solver adapter."""

    def __init__(self, vertices, faces, **engine_kwargs):
        super().__init__(vertices, faces, **engine_kwargs)
        allow_eigen_fallback = bool(self.engine_kwargs.pop("allow_eigen_fallback", False))
        try:
            import potpourri3d as pp3d
        except Exception as exc:
            raise ImportError("potpourri3d is required for the 'potpourri' engine.") from exc

        if not allow_eigen_fallback and _VERIFY_SUITESPARSE is None:
            lines = ["\nBackend checker could not be imported in this environment."]
            for mod_name, err in _BACKEND_CHECK_IMPORT_ERRORS:
                lines.append(f"  - {mod_name}: {err}")
            lines.append(
                "Ensure backend_check.py is present and importable alongside distance_engines.py."
            )
            raise RuntimeError("\n".join(lines))

        if not allow_eigen_fallback and not verify_suitesparse(verbose=True):
            raise RuntimeError(
                "\n"
                "==================== FASTCW BACKEND CHECK FAILED ====================\n"
                "potpourri3d does not appear to be compiled with SuiteSparse support.\n"
                "Running with Eigen fallback can be orders of magnitude slower.\n"
                "\n"
                "Fix:\n"
                "  Reinstall with source build:\n"
                "    python -m pip install --no-binary potpourri3d potpourri3d\n"
                "\n"
                "Bypass explicitly (accept slow fallback):\n"
                "  --allow-eigen-fallback\n"
                "====================================================================\n"
            )

        kwargs = {"use_robust": False}
        kwargs.update(self.engine_kwargs)
        self._solver = pp3d.MeshHeatMethodDistanceSolver(self.vertices, self.faces, **kwargs)

    @property
    def name(self) -> str:
        return "potpourri"

    def compute_distance(self, source_idx: int) -> np.ndarray:
        d = self._solver.compute_distance(int(source_idx))
        return np.ascontiguousarray(np.asarray(d, dtype=np.float64))


class BatchHeatDistanceEngine(BaseDistanceEngine):
    """Batched heat-method distance solver using robust-laplacian + CHOLMOD.

    The mesh operators come from robust_laplacian.mesh_laplacian(), which uses
    intrinsic operators for numerical stability and compatibility with the
    clean-mesh potpourri3d heat-method path. No public IDT toggle is exposed:
    the validation target is direct agreement with potpourri3d use_robust=False
    on representative clean cortical meshes.
    """

    supports_batching = True

    def __init__(self, vertices, faces, **engine_kwargs):
        super().__init__(vertices, faces, **engine_kwargs)
        self.t_coef = float(self.engine_kwargs.pop("t_coef", 1.0))
        self.poisson_regularization = float(self.engine_kwargs.pop("poisson_regularization", 1e-8))
        self.debug_diagnostics = bool(self.engine_kwargs.pop("debug_diagnostics", False))
        if self.t_coef <= 0.0 or not np.isfinite(self.t_coef):
            raise ValueError("t_coef must be finite and > 0 for batch_heat.")
        if self.poisson_regularization < 0.0 or not np.isfinite(self.poisson_regularization):
            raise ValueError("poisson_regularization must be finite and >= 0 for batch_heat.")
        if self.engine_kwargs:
            unsupported = ", ".join(sorted(self.engine_kwargs.keys()))
            raise ValueError(f"Unsupported engine kwargs for batch_heat engine: {unsupported}")

        try:
            import robust_laplacian
        except Exception as exc:
            raise ImportError(
                "batch_heat requires robust-laplacian. Install with: pip install robust-laplacian"
            ) from exc
        try:
            from sksparse.cholmod import cholesky
        except Exception as exc:
            raise ImportError(
                "batch_heat requires scikit-sparse with SuiteSparse/CHOLMOD. "
                "Install with: pip install scikit-sparse"
            ) from exc

        self.n_vertices = int(self.vertices.shape[0])
        self.n_faces = int(self.faces.shape[0])
        if self.n_vertices == 0:
            raise ValueError("batch_heat requires at least one vertex.")
        if self.n_faces == 0:
            raise ValueError("batch_heat requires at least one triangle face.")

        L, M = robust_laplacian.mesh_laplacian(self.vertices, self.faces)
        self.L = sp.csc_matrix(L, dtype=np.float64)
        if sp.issparse(M):
            self.M = sp.csc_matrix(M, dtype=np.float64)
        else:
            mass = np.asarray(M, dtype=np.float64).reshape(-1)
            self.M = sp.diags(mass, format="csc", dtype=np.float64)
        self.mass_diag = np.asarray(self.M.diagonal(), dtype=np.float64)
        if self.mass_diag.shape != (self.n_vertices,):
            raise ValueError("robust_laplacian returned an invalid mass matrix shape.")
        if not np.all(np.isfinite(self.mass_diag)):
            raise ValueError("robust_laplacian returned non-finite vertex masses.")

        h = self._mean_edge_length()
        self.t = float(self.t_coef * h * h)
        if not np.isfinite(self.t) or self.t <= 0.0:
            raise ValueError("Unable to compute a positive heat time step for batch_heat.")

        A_heat = (self.M + self.t * self.L).tocsc()
        if self.poisson_regularization > 0.0:
            A_poisson = (self.L + self.poisson_regularization * sp.eye(self.n_vertices, format="csc")).tocsc()
        else:
            A_poisson = self.L.tocsc()
        self._factor_heat = cholesky(A_heat)
        self._factor_poisson = cholesky(A_poisson)

        self._precompute_face_operators()

    @property
    def name(self) -> str:
        return "batch_heat"

    def _mean_edge_length(self) -> float:
        edges = np.vstack(
            (
                self.faces[:, [0, 1]],
                self.faces[:, [1, 2]],
                self.faces[:, [2, 0]],
            )
        )
        edges.sort(axis=1)
        unique_edges = np.unique(edges, axis=0)
        lengths = np.linalg.norm(self.vertices[unique_edges[:, 1]] - self.vertices[unique_edges[:, 0]], axis=1)
        lengths = lengths[np.isfinite(lengths) & (lengths > 0.0)]
        if lengths.size == 0:
            raise ValueError("batch_heat cannot compute mean edge length from degenerate mesh.")
        return float(np.mean(lengths))

    def _precompute_face_operators(self):
        f0 = self.faces[:, 0]
        f1 = self.faces[:, 1]
        f2 = self.faces[:, 2]
        p0 = self.vertices[f0]
        p1 = self.vertices[f1]
        p2 = self.vertices[f2]
        e01 = p1 - p0
        e02 = p2 - p0
        normals = np.cross(e01, e02)
        norm_len = np.linalg.norm(normals, axis=1)
        areas = 0.5 * norm_len
        valid = np.isfinite(areas) & (areas > 0.0)
        if not np.all(valid):
            raise ValueError("batch_heat does not support zero-area or non-finite triangles.")
        unit_normals = normals / norm_len[:, None]
        inv_2a = 1.0 / (2.0 * areas)

        grad_b1 = np.cross(e02, unit_normals) * inv_2a[:, None]
        grad_b2 = np.cross(unit_normals, e01) * inv_2a[:, None]
        grad_b0 = -grad_b1 - grad_b2
        self._face_areas = np.ascontiguousarray(areas, dtype=np.float64)
        self._grad_b0 = np.ascontiguousarray(grad_b0, dtype=np.float64)
        self._grad_b1 = np.ascontiguousarray(grad_b1, dtype=np.float64)
        self._grad_b2 = np.ascontiguousarray(grad_b2, dtype=np.float64)

    def _validate_sources(self, source_indices) -> np.ndarray:
        sources = np.atleast_1d(np.asarray(source_indices, dtype=np.int64)).reshape(-1)
        if sources.size == 0:
            return sources
        bad = sources[(sources < 0) | (sources >= self.n_vertices)]
        if bad.size:
            raise ValueError(f"source_idx out of range: {int(bad[0])} for mesh with {self.n_vertices} vertices")
        return sources

    def _solve_factor(self, factor, rhs) -> np.ndarray:
        out = factor.solve_A(rhs)
        if sp.issparse(out):
            out = out.toarray()
        out = np.asarray(out, dtype=np.float64)
        if out.ndim == 1:
            out = out[:, None]
        return np.ascontiguousarray(out, dtype=np.float64)

    def compute_distance_batch(self, source_indices) -> np.ndarray:
        sources = self._validate_sources(source_indices)
        k = int(sources.size)
        if k == 0:
            return np.empty((self.n_vertices, 0), dtype=np.float64)

        rhs = np.zeros((self.n_vertices, k), dtype=np.float64)
        rhs[sources, np.arange(k)] = 1.0
        u = self._solve_factor(self._factor_heat, rhs)
        if u.shape != (self.n_vertices, k):
            raise ValueError(f"batch_heat heat solve returned shape {u.shape}; expected {(self.n_vertices, k)}")

        u0 = u[self.faces[:, 0], :]
        u1 = u[self.faces[:, 1], :]
        u2 = u[self.faces[:, 2], :]
        grad = (
            (u1 - u0)[:, None, :] * self._grad_b1[:, :, None]
            + (u2 - u0)[:, None, :] * self._grad_b2[:, :, None]
        )
        grad_norm = np.linalg.norm(grad, axis=1)
        X = np.zeros_like(grad)
        denom = grad_norm[:, None, :]
        valid = np.isfinite(denom) & (denom > 1e-30)
        np.divide(-grad, denom, out=X, where=valid)

        face_weight = self._face_areas[:, None]
        div0 = face_weight * np.einsum("fck,fc->fk", X, self._grad_b0)
        div1 = face_weight * np.einsum("fck,fc->fk", X, self._grad_b1)
        div2 = face_weight * np.einsum("fck,fc->fk", X, self._grad_b2)
        div = np.zeros((self.n_vertices, k), dtype=np.float64)
        np.add.at(div, self.faces[:, 0], div0)
        np.add.at(div, self.faces[:, 1], div1)
        np.add.at(div, self.faces[:, 2], div2)

        phi = self._solve_factor(self._factor_poisson, div)
        if phi.shape != (self.n_vertices, k):
            raise ValueError(f"batch_heat poisson solve returned shape {phi.shape}; expected {(self.n_vertices, k)}")
        if self.debug_diagnostics:
            print(f"\n=== BatchHeat diagnostic ===")
            print(f"phi shape: {phi.shape}")
            print(f"per-column min: {phi.min(axis=0)}")
            print(f"per-column max: {phi.max(axis=0)}")
            print(f"argmin (should equal sources): {phi.argmin(axis=0)}")
            print(f"sources:                       {sources}")
            print(f"argmin == sources: {np.all(phi.argmin(axis=0) == sources)}")
            print(f"phi at sources: {phi[sources, np.arange(k)]}")

            # Compare to potpourri for the first source in this batch.
            import potpourri3d as pp3d

            pot = pp3d.MeshHeatMethodDistanceSolver(self.vertices, self.faces, use_robust=False)
            d_pot = pot.compute_distance(int(sources[0]))
            print(f"potpourri d at source: {d_pot[sources[0]]}")
            print(f"potpourri d range: {d_pot.min()} to {d_pot.max()}")
            print(f"batch_heat d range (col 0): {phi[:, 0].min()} to {phi[:, 0].max()}")
            print(f"=== end diagnostic ===\n")
        phi -= np.nanmin(phi, axis=0, keepdims=True)
        return np.ascontiguousarray(phi, dtype=np.float64)

    def compute_distance(self, source_idx: int) -> np.ndarray:
        return self.compute_distance_batch([int(source_idx)])[:, 0]


class PotpourriFmmDistanceEngine(BaseDistanceEngine):
    """potpourri3d Fast Marching Method solver adapter."""

    def __init__(self, vertices, faces, **engine_kwargs):
        super().__init__(vertices, faces, **engine_kwargs)
        try:
            import potpourri3d as pp3d
        except ImportError as exc:
            raise ImportError("potpourri3d is required for the 'potpourri_fmm' engine.") from exc

        self._solver = pp3d.MeshFastMarchingDistanceSolver(
            self.vertices,
            self.faces,
            **self.engine_kwargs,
        )

    @property
    def name(self) -> str:
        return "potpourri_fmm"

    def compute_distance(self, source_idx: int) -> np.ndarray:
        src = int(source_idx)
        n_vertices = self.vertices.shape[0]

        if src < 0 or src >= n_vertices:
            raise ValueError(f"source_idx out of range: {src} for mesh with {n_vertices} vertices")

        try:
            # Older potpourri3d API accepted a single vertex index directly.
            d = self._solver.compute_distance(src)
        except TypeError:
            # Newer API expects sources as curve points in barycentric form.
            # A vertex source is encoded as (vertex_index, []).
            source_curves = [[(src, [])]]
            try:
                d = self._solver.compute_distance(source_curves, [], False)
            except TypeError:
                d = self._solver.compute_distance(source_curves, sign=False)

        d = np.asarray(d, dtype=np.float64).reshape(-1)

        if d.shape[0] != n_vertices:
            raise ValueError(
                f"potpourri_fmm returned invalid distance shape {d.shape}; expected ({n_vertices},)"
            )
        return np.ascontiguousarray(d)


class PycortexDistanceEngine(BaseDistanceEngine):
    """pycortex geodesic solver adapter."""

    def __init__(self, vertices, faces, **engine_kwargs):
        super().__init__(vertices, faces, **engine_kwargs)
        try:
            from cortex.polyutils import Surface
        except Exception as exc:
            raise ImportError(
                "pycortex backend requested but pycortex is not installed. "
                "Install with: pip install pycortex"
            ) from exc

        if self.engine_kwargs:
            # pycortex Surface currently does not consume kwargs; keep localized and explicit.
            unsupported = ", ".join(sorted(self.engine_kwargs.keys()))
            raise ValueError(f"Unsupported engine kwargs for pycortex engine: {unsupported}")
        self._surface = Surface(self.vertices, self.faces)

    @property
    def name(self) -> str:
        return "pycortex"

    def compute_distance(self, source_idx: int) -> np.ndarray:
        d = self._surface.geodesic_distance([int(source_idx)])
        d = np.asarray(d, dtype=np.float64)
        if d.ndim != 1 or d.shape[0] != self.vertices.shape[0]:
            raise ValueError(
                f"pycortex engine returned invalid distance shape {d.shape}; "
                f"expected ({self.vertices.shape[0]},)"
            )
        return np.ascontiguousarray(d)


class PyGeodesicDistanceEngine(BaseDistanceEngine):
    """Exact discrete geodesic solver adapter via pygeodesic."""

    def __init__(self, vertices, faces, **engine_kwargs):
        super().__init__(vertices, faces, **engine_kwargs)
        try:
            import pygeodesic.geodesic as geodesic
        except Exception as exc:
            raise ImportError(
                "pygeodesic backend requested but pygeodesic is not installed. "
                "Install with: pip install pygeodesic"
            ) from exc

        if self.engine_kwargs:
            unsupported = ", ".join(sorted(self.engine_kwargs.keys()))
            raise ValueError(f"Unsupported engine kwargs for pygeodesic engine: {unsupported}")

        self._solver = geodesic.PyGeodesicAlgorithmExact(self.vertices, self.faces)

    @property
    def name(self) -> str:
        return "pygeodesic"

    def compute_distance(self, source_idx: int) -> np.ndarray:
        src = int(source_idx)
        n_vertices = self.vertices.shape[0]
        if src < 0 or src >= n_vertices:
            raise ValueError(f"source_idx out of range: {src} for mesh with {n_vertices} vertices")

        source_indices = np.asarray([src], dtype=np.int32)
        distances, _ = self._solver.geodesicDistances(source_indices, None)
        d = np.asarray(distances, dtype=np.float64)
        if d.ndim != 1 or d.shape[0] != n_vertices:
            raise ValueError(
                f"pygeodesic returned invalid distance shape {d.shape}; expected ({n_vertices},)"
            )
        return np.ascontiguousarray(d)


ENGINE_REGISTRY = {
    "batch_heat": BatchHeatDistanceEngine,
    "potpourri": PotpourriDistanceEngine,
    "potpourri_fmm": PotpourriFmmDistanceEngine,
    "pycortex": PycortexDistanceEngine,
    "pygeodesic": PyGeodesicDistanceEngine,
}


def create_distance_engine(engine_type, vertices, faces, engine_kwargs=None):
    """Construct a distance engine by name."""
    key = str(engine_type).strip().lower()
    if key not in ENGINE_REGISTRY:
        valid = ", ".join(sorted(ENGINE_REGISTRY.keys()))
        raise ValueError(f"Unknown engine_type '{engine_type}'. Valid options: {valid}")
    cls = ENGINE_REGISTRY[key]
    kwargs = dict(engine_kwargs or {})
    return cls(vertices, faces, **kwargs)
