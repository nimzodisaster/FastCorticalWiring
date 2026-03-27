#!/usr/bin/env python3
"""Unit tests for interior-disk covariance anisotropy metric."""

import unittest
from unittest import mock

import numpy as np

import core_analysis
from core_analysis import FastCorticalWiringAnalysis


class _DummyDistanceEngine:
    def __init__(self, vertices):
        self._n = int(vertices.shape[0])

    @property
    def name(self):
        return "dummy"

    def compute_distance(self, source_idx):
        src = int(source_idx)
        d = np.abs(np.arange(self._n, dtype=np.float64) - float(src))
        return np.ascontiguousarray(d, dtype=np.float64)


def _dummy_engine_factory(_engine_type, vertices, _faces, _engine_kwargs):
    return _DummyDistanceEngine(vertices)


def _make_planar_mesh_with_center():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],
        ],
        dtype=np.int32,
    )
    mask = np.ones(vertices.shape[0], dtype=bool)
    return vertices, faces, mask


def _ellipse_points(a, b, n=60):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = a * np.cos(t)
    y = b * np.sin(t)
    return np.column_stack([x, y, np.zeros_like(x)]).astype(np.float64)


def _rotate_z(points, angle_rad):
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return points @ r.T


class AnisotropyMetricTests(unittest.TestCase):
    def setUp(self):
        vertices, faces, mask = _make_planar_mesh_with_center()
        with mock.patch("core_analysis.create_distance_engine", side_effect=_dummy_engine_factory):
            self.analysis = FastCorticalWiringAnalysis(
                vertices,
                faces,
                mask,
                engine_type="potpourri",
                eps=1e-6,
                metadata={"subject_id": "synthetic", "hemi": "lh", "surf_type": "planar"},
            )

    def _set_synthetic_disk(self, pts):
        # Seed vertex at origin + interior vertices from synthetic point cloud.
        seed = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        v = np.vstack([seed, np.asarray(pts, dtype=np.float64)])
        self.analysis.vertices = v
        self.analysis.vertex_areas_sub = np.ones(v.shape[0], dtype=np.float64)
        self.analysis.vertex_normals_sub = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (v.shape[0], 1))
        d = np.linalg.norm(v - seed[0], axis=1)
        return d

    def test_flat_circular_patch_has_low_anisotropy(self):
        pts = _ellipse_points(1.0, 1.0, n=80)
        d = self._set_synthetic_disk(pts)
        anis = self.analysis._disk_anisotropy_from_vertices(0, d, radius=1.0)
        self.assertTrue(np.isfinite(anis))
        self.assertLess(abs(float(anis)), 0.05)

    def test_elongated_region_has_higher_anisotropy(self):
        pts_circle = _ellipse_points(1.0, 1.0, n=80)
        d_circle = self._set_synthetic_disk(pts_circle)
        a_circle = self.analysis._disk_anisotropy_from_vertices(0, d_circle, radius=1.0)

        pts_elong = _ellipse_points(2.0, 0.5, n=80)
        d_elong = self._set_synthetic_disk(pts_elong)
        # Use radius large enough to include the elongated interior points.
        a_elong = self.analysis._disk_anisotropy_from_vertices(0, d_elong, radius=2.1)

        self.assertTrue(np.isfinite(a_circle))
        self.assertTrue(np.isfinite(a_elong))
        self.assertGreater(float(a_elong), float(a_circle) + 0.2)

    def test_rotation_invariance(self):
        pts = _ellipse_points(2.0, 0.5, n=80)
        d = self._set_synthetic_disk(pts)
        a0 = self.analysis._disk_anisotropy_from_vertices(0, d, radius=2.1)

        pts_rot = _rotate_z(pts, np.deg2rad(37.0))
        d_rot = self._set_synthetic_disk(pts_rot)
        a1 = self.analysis._disk_anisotropy_from_vertices(0, d_rot, radius=2.1)

        self.assertTrue(np.isfinite(a0))
        self.assertTrue(np.isfinite(a1))
        self.assertAlmostEqual(float(a0), float(a1), places=6)

    def test_consistent_with_or_without_numba_flag(self):
        pts = _ellipse_points(1.8, 0.6, n=80)
        d = self._set_synthetic_disk(pts)
        old = core_analysis.NUMBA_AVAILABLE
        try:
            core_analysis.NUMBA_AVAILABLE = True
            a_numba = self.analysis._disk_anisotropy_from_vertices(0, d, radius=2.0)
            core_analysis.NUMBA_AVAILABLE = False
            a_py = self.analysis._disk_anisotropy_from_vertices(0, d, radius=2.0)
        finally:
            core_analysis.NUMBA_AVAILABLE = old

        self.assertTrue(np.isfinite(a_numba))
        self.assertTrue(np.isfinite(a_py))
        self.assertAlmostEqual(float(a_numba), float(a_py), places=12)

    def test_metric_arrays_expose_anisotropy_key(self):
        metrics = self.analysis.get_metric_arrays()
        for scale in self.analysis.active_scales:
            token = FastCorticalWiringAnalysis.scale_token(scale)
            self.assertIn(f"anisotropy_{token}", metrics)


if __name__ == "__main__":
    unittest.main()
