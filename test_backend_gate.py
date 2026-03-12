#!/usr/bin/env python3
"""Tests for SuiteSparse gate behavior in the potpourri engine."""

import types
import unittest
from unittest import mock

import numpy as np

import distance_engines


class BackendGateTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        self.faces = np.array([[0, 1, 2]], dtype=np.int32)

    def _fake_pp3d_module(self):
        mod = types.SimpleNamespace()

        class _Solver:
            def __init__(self, _v, _f, **_kwargs):
                pass

            def compute_distance(self, source_idx):
                src = int(source_idx)
                return np.array([float(src), 1.0, 2.0], dtype=np.float64)

        mod.MeshHeatMethodDistanceSolver = _Solver
        return mod

    def test_potpourri_raises_when_suitesparse_check_fails(self):
        fake_pp3d = self._fake_pp3d_module()
        with mock.patch.dict("sys.modules", {"potpourri3d": fake_pp3d}):
            with mock.patch("distance_engines.verify_suitesparse", return_value=False):
                with self.assertRaises(RuntimeError):
                    distance_engines.PotpourriDistanceEngine(self.vertices, self.faces)

    def test_potpourri_allows_bypass_flag(self):
        fake_pp3d = self._fake_pp3d_module()
        with mock.patch.dict("sys.modules", {"potpourri3d": fake_pp3d}):
            with mock.patch("distance_engines.verify_suitesparse", return_value=False):
                engine = distance_engines.PotpourriDistanceEngine(
                    self.vertices,
                    self.faces,
                    allow_eigen_fallback=True,
                )
        d = engine.compute_distance(0)
        self.assertEqual(d.shape, (3,))


if __name__ == "__main__":
    unittest.main()
