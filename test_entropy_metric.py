#!/usr/bin/env python3
"""Unit tests for global/local entropy metrics."""

import unittest
from unittest import mock

import numpy as np

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


def _make_simple_mesh():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    mask = np.ones(vertices.shape[0], dtype=bool)
    return vertices, faces, mask


class EntropyMetricTests(unittest.TestCase):
    def setUp(self):
        vertices, faces, mask = _make_simple_mesh()
        with mock.patch("core_analysis.create_distance_engine", side_effect=_dummy_engine_factory):
            self.analysis = FastCorticalWiringAnalysis(
                vertices,
                faces,
                mask,
                engine_type="potpourri",
                eps=1e-6,
                metadata={"subject_id": "synthetic", "hemi": "lh", "surf_type": "simple"},
            )

    def test_weighted_entropy_is_higher_for_spread_distribution(self):
        w = np.ones(8, dtype=np.float64)
        concentrated = np.array([0.9, 0.91, 0.89, 0.9, 0.91, 0.9, 0.89, 0.9], dtype=np.float64)
        spread = np.array([0.1, 0.4, 0.8, 1.1, 1.6, 2.0, 2.4, 2.8], dtype=np.float64)
        h_conc = self.analysis._weighted_entropy_from_normalized_distances(concentrated, w)
        h_spread = self.analysis._weighted_entropy_from_normalized_distances(spread, w)
        self.assertTrue(np.isfinite(h_conc))
        self.assertTrue(np.isfinite(h_spread))
        self.assertGreater(float(h_spread), float(h_conc))

    def test_weighted_entropy_handles_empty_or_zero_weight(self):
        h_empty = self.analysis._weighted_entropy_from_normalized_distances(
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
        h_zero = self.analysis._weighted_entropy_from_normalized_distances(
            np.array([0.2, 0.3], dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64),
        )
        self.assertTrue(np.isnan(h_empty))
        self.assertTrue(np.isnan(h_zero))

    def test_local_entropy_uses_interior_disk_distances(self):
        seen = {}

        def _capture(d_norm, weights, **_kwargs):
            seen["d_norm"] = np.asarray(d_norm, dtype=np.float64)
            seen["weights"] = np.asarray(weights, dtype=np.float64)
            return 0.123

        self.analysis._weighted_entropy_from_normalized_distances = _capture
        d = np.array([0.0, 0.2, 0.9, 1.1], dtype=np.float64)
        out = self.analysis._local_entropy_from_distances(d, radius=1.0)
        self.assertAlmostEqual(float(out), 0.123, places=6)
        self.assertIn("d_norm", seen)
        # Interior disk semantics: include distances <= r only.
        self.assertEqual(seen["d_norm"].shape[0], 3)
        self.assertTrue(np.all(seen["d_norm"] <= 1.0 + 1e-12))
        self.assertTrue(np.allclose(seen["d_norm"], np.array([0.0, 0.2, 0.9], dtype=np.float64)))

    def test_global_entropy_normalizes_by_msd(self):
        seen = {}

        def _capture(d_norm, weights, **_kwargs):
            seen["d_norm"] = np.asarray(d_norm, dtype=np.float64)
            return 0.5

        self.analysis._weighted_entropy_from_normalized_distances = _capture
        d = np.array([0.0, 2.0, 4.0, np.inf], dtype=np.float64)
        out = self.analysis._global_entropy_from_distances(d, msd_val=2.0)
        self.assertAlmostEqual(float(out), 0.5, places=6)
        self.assertTrue(np.allclose(seen["d_norm"], np.array([0.0, 1.0, 2.0], dtype=np.float64)))


if __name__ == "__main__":
    unittest.main()
