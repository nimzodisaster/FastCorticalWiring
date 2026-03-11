#!/usr/bin/env python3
"""Geodesic distance engine adapters for FastCorticalWiring."""

from abc import ABC, abstractmethod

import numpy as np

_BACKEND_CHECK_IMPORT_ERROR = None
try:
    from backend_check import verify_suitesparse
except Exception as exc_abs:
    try:
        from .backend_check import verify_suitesparse  # type: ignore
    except Exception as exc_rel:
        _BACKEND_CHECK_IMPORT_ERROR = (exc_abs, exc_rel)

        def verify_suitesparse() -> bool:
            return False


class BaseDistanceEngine(ABC):
    """Minimal interface expected by the shared wiring pipeline."""

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


class PotpourriDistanceEngine(BaseDistanceEngine):
    """potpourri3d heat-method solver adapter."""

    def __init__(self, vertices, faces, **engine_kwargs):
        super().__init__(vertices, faces, **engine_kwargs)
        allow_eigen_fallback = bool(self.engine_kwargs.pop("allow_eigen_fallback", False))
        try:
            import potpourri3d as pp3d
        except Exception as exc:
            raise ImportError("potpourri3d is required for the 'potpourri' engine.") from exc

        if not allow_eigen_fallback and not verify_suitesparse():
            import_note = ""
            if _BACKEND_CHECK_IMPORT_ERROR is not None:
                import_note = (
                    "\nBackend checker import failed in this environment.\n"
                    f"Absolute import error: {_BACKEND_CHECK_IMPORT_ERROR[0]}\n"
                    f"Relative import error: {_BACKEND_CHECK_IMPORT_ERROR[1]}\n"
                    "Ensure backend_check.py is present alongside distance_engines.py.\n"
                )
            raise RuntimeError(
                "\n"
                "==================== FASTCW BACKEND CHECK FAILED ====================\n"
                "potpourri3d does not appear to be compiled with SuiteSparse support.\n"
                "Running with Eigen fallback can be orders of magnitude slower.\n"
                f"{import_note}"
                "\n"
                "Fix:\n"
                "  Reinstall with source build:\n"
                "    python -m pip install --no-binary potpourri3d potpourri3d\n"
                "\n"
                "Bypass explicitly (accept slow fallback):\n"
                "  --allow-eigen-fallback\n"
                "====================================================================\n"
            )

        kwargs = {"use_robust": True}
        kwargs.update(self.engine_kwargs)
        self._solver = pp3d.MeshHeatMethodDistanceSolver(self.vertices, self.faces, **kwargs)

    @property
    def name(self) -> str:
        return "potpourri"

    def compute_distance(self, source_idx: int) -> np.ndarray:
        d = self._solver.compute_distance(int(source_idx))
        return np.ascontiguousarray(np.asarray(d, dtype=np.float64))


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
