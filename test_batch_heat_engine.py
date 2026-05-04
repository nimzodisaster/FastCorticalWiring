#!/usr/bin/env python3
"""Tests for batched distance-engine interfaces."""

import types
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

import distance_engines


class _LinearDistanceEngine(distance_engines.BaseDistanceEngine):
    @property
    def name(self):
        return "linear"

    def compute_distance(self, source_idx):
        src = int(source_idx)
        return np.abs(np.arange(self.vertices.shape[0], dtype=np.float64) - float(src))


class BatchDistanceInterfaceTests(unittest.TestCase):
    def test_base_compute_distance_batch_falls_back_to_single_source(self):
        vertices = np.zeros((5, 3), dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        engine = _LinearDistanceEngine(vertices, faces)

        d_batch = engine.compute_distance_batch([0, 2, 4])

        self.assertFalse(engine.supports_batching)
        self.assertEqual(d_batch.shape, (5, 3))
        np.testing.assert_allclose(d_batch[:, 0], engine.compute_distance(0))
        np.testing.assert_allclose(d_batch[:, 1], engine.compute_distance(2))
        np.testing.assert_allclose(d_batch[:, 2], engine.compute_distance(4))


class BatchHeatEngineTests(unittest.TestCase):
    def _fake_dependency_modules(self):
        robust_laplacian = types.SimpleNamespace()

        def mesh_laplacian(_vertices, _faces):
            L = sp.csc_matrix(
                np.array(
                    [
                        [2.0, -1.0, -1.0],
                        [-1.0, 2.0, -1.0],
                        [-1.0, -1.0, 2.0],
                    ],
                    dtype=np.float64,
                )
            )
            M = sp.eye(3, format="csc", dtype=np.float64)
            return L, M

        robust_laplacian.mesh_laplacian = mesh_laplacian

        cholmod = types.SimpleNamespace()
        cholmod.factors = []

        class _Factor:
            def __init__(self, matrix):
                self.matrix = np.asarray(matrix.toarray(), dtype=np.float64)
                self.solve_rhs = []

            def solve_A(self, rhs):
                rhs_arr = np.asarray(rhs, dtype=np.float64)
                self.solve_rhs.append(rhs_arr.copy())
                return np.linalg.solve(self.matrix, rhs_arr)

        def cholesky(matrix):
            factor = _Factor(matrix)
            cholmod.factors.append(factor)
            return factor

        cholmod.cholesky = cholesky
        sksparse = types.SimpleNamespace(cholmod=cholmod)
        return {
            "robust_laplacian": robust_laplacian,
            "sksparse": sksparse,
            "sksparse.cholmod": cholmod,
        }

    def _make_engine(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        with mock.patch.dict("sys.modules", self._fake_dependency_modules()):
            return distance_engines.create_distance_engine("batch_heat", vertices, faces)

    def test_registry_constructs_batch_heat_engine_with_mocked_dependencies(self):
        engine = self._make_engine()
        self.assertEqual(engine.name, "batch_heat")
        self.assertTrue(engine.supports_batching)

    def test_single_source_delegates_to_batch_column(self):
        engine = self._make_engine()
        d_single = engine.compute_distance(1)
        d_batch = engine.compute_distance_batch([1])
        self.assertEqual(d_batch.shape, (3, 1))
        np.testing.assert_allclose(d_single, d_batch[:, 0])

    def test_batch_shape_and_per_column_gauge(self):
        engine = self._make_engine()
        d_batch = engine.compute_distance_batch([0, 2])
        self.assertEqual(d_batch.shape, (3, 2))
        np.testing.assert_allclose(np.min(d_batch, axis=0), np.zeros(2), atol=1e-12)
        self.assertTrue(np.all(np.isfinite(d_batch)))

    def test_heat_rhs_uses_unit_impulses_not_mass_weighting(self):
        modules = self._fake_dependency_modules()
        robust_laplacian = modules["robust_laplacian"]

        def mesh_laplacian(_vertices, _faces):
            L = sp.csc_matrix(
                np.array(
                    [
                        [2.0, -1.0, -1.0],
                        [-1.0, 2.0, -1.0],
                        [-1.0, -1.0, 2.0],
                    ],
                    dtype=np.float64,
                )
            )
            M = sp.diags([0.25, 2.0, 4.0], format="csc", dtype=np.float64)
            return L, M

        robust_laplacian.mesh_laplacian = mesh_laplacian
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        with mock.patch.dict("sys.modules", modules):
            engine = distance_engines.create_distance_engine("batch_heat", vertices, faces)
            engine.compute_distance_batch([0, 2])

        heat_rhs = modules["sksparse.cholmod"].factors[0].solve_rhs[0]
        expected = np.array(
            [
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(heat_rhs, expected, atol=0.0)

    def test_planar_triangle_gradient_formula(self):
        engine = distance_engines.BatchHeatDistanceEngine.__new__(
            distance_engines.BatchHeatDistanceEngine
        )
        engine.vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        engine.faces = np.array([[0, 1, 2]], dtype=np.int32)
        engine._precompute_face_operators()

        # For u(x, y) = x + y, the face gradient should be exactly (1, 1, 0).
        u = np.array([0.0, 1.0, 1.0], dtype=np.float64)
        grad = (
            (u[1] - u[0]) * engine._grad_b1[0]
            + (u[2] - u[0]) * engine._grad_b2[0]
        )
        np.testing.assert_allclose(grad, np.array([1.0, 1.0, 0.0]), atol=1e-12)

    def test_missing_dependencies_raise_clear_import_error(self):
        vertices = np.zeros((3, 3), dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        with mock.patch.dict("sys.modules", {"robust_laplacian": None}):
            with self.assertRaises(ImportError):
                distance_engines.BatchHeatDistanceEngine(vertices, faces)


if __name__ == "__main__":
    unittest.main()
