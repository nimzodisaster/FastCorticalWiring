#!/usr/bin/env python3
"""Tests for vertex-subsetting behavior and subset input loading."""

import os
import tempfile
import unittest
from unittest import mock

import numpy as np

import fastcw
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


class _BatchRecordingDistanceEngine(_DummyDistanceEngine):
    supports_batching = True

    def __init__(self, vertices):
        super().__init__(vertices)
        self.batch_calls = []

    def compute_distance_batch(self, source_indices):
        sources = [int(src) for src in source_indices]
        self.batch_calls.append(sources)
        return np.column_stack([self.compute_distance(src) for src in sources])


class VertexSubsetAnalysisTests(unittest.TestCase):
    def test_normalize_scales_sorts_ascending(self):
        scales = FastCorticalWiringAnalysis.normalize_scales([0.05, 0.001, 0.01, 0.005])
        self.assertEqual(scales, (0.001, 0.005, 0.01, 0.05))

    def test_interior_nonmanifold_auto_enables_potpourri_robust_mode(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 1],
                [1, 3, 2],
                [0, 1, 4],
                [0, 4, 5],
                [0, 5, 1],
                [1, 5, 4],
                [0, 1, 6],
                [0, 6, 7],
                [0, 7, 1],
                [1, 7, 6],
            ],
            dtype=np.int32,
        )
        cortex_mask = np.ones(vertices.shape[0], dtype=bool)
        engine_calls = []

        def capture_engine(engine_type, vertices_arg, faces_arg, engine_kwargs):
            engine_calls.append(
                {
                    "engine_type": engine_type,
                    "engine_kwargs": dict(engine_kwargs or {}),
                }
            )
            return _DummyDistanceEngine(vertices_arg)

        with mock.patch("core_analysis.create_distance_engine", side_effect=capture_engine):
            analysis = FastCorticalWiringAnalysis(
                vertices,
                faces,
                cortex_mask,
                engine_type="potpourri",
                engine_kwargs={},
                eps=1e-6,
                metadata={"subject_id": "synthetic", "hemi": "lh", "surf_type": "nonmanifold"},
            )

        self.assertTrue(analysis.engine_kwargs["use_robust"])
        self.assertEqual(len(engine_calls), 1)
        self.assertTrue(engine_calls[0]["engine_kwargs"]["use_robust"])

    def test_compute_all_wiring_costs_respects_vertex_subset(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 2.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        cortex_mask = np.array([True, True, True, True, False], dtype=bool)

        with mock.patch("core_analysis.create_distance_engine", side_effect=_dummy_engine_factory):
            analysis = FastCorticalWiringAnalysis(
                vertices,
                faces,
                cortex_mask,
                engine_type="potpourri",
                eps=1e-6,
                metadata={"subject_id": "synthetic", "hemi": "lh", "surf_type": "unit_square"},
            )

        analysis._find_radius_for_area = lambda *args, **kwargs: 1.0
        analysis._perimeter_at_radius = lambda *args, **kwargs: 2.0
        analysis._disk_anisotropy_from_vertices = lambda *args, **kwargs: 0.25

        analysis.compute_all_wiring_costs(
            compute_msd=True,
            scale=0.2,
            area_tol=0.1,
            vertex_subset=[0, 2, 4, 99, -1],
        )

        scale_key = FastCorticalWiringAnalysis.normalize_scales(0.2)[0]
        for idx in (0, 2):
            self.assertTrue(np.isfinite(analysis.msd[idx]))
            self.assertTrue(np.isfinite(analysis.radius_function[scale_key][idx]))
            self.assertTrue(np.isfinite(analysis.perimeter_function[scale_key][idx]))
            self.assertTrue(np.isfinite(analysis.anisotropy_function[scale_key][idx]))
            n_samples = int(analysis.n_samples_per_vertex[idx])
            self.assertGreaterEqual(n_samples, 1)
            self.assertTrue(np.any(np.isclose(analysis.sampled_radii[idx, :n_samples], 1.0)))

        for idx in (1, 3, 4):
            self.assertTrue(np.isnan(analysis.msd[idx]))
            self.assertTrue(np.isnan(analysis.radius_function[scale_key][idx]))
            self.assertTrue(np.isnan(analysis.perimeter_function[scale_key][idx]))
            self.assertTrue(np.isnan(analysis.anisotropy_function[scale_key][idx]))

    def test_requested_intrinsic_anisotropy_does_not_use_extrinsic_fallback(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        cortex_mask = np.ones(vertices.shape[0], dtype=bool)

        with mock.patch("core_analysis.create_distance_engine", side_effect=_dummy_engine_factory):
            analysis = FastCorticalWiringAnalysis(
                vertices,
                faces,
                cortex_mask,
                engine_type="potpourri",
                eps=1e-6,
                metadata={"subject_id": "synthetic", "hemi": "lh", "surf_type": "unit_square"},
            )

        analysis._find_radius_for_area = lambda *args, **kwargs: 1.0
        analysis._perimeter_at_radius = lambda *args, **kwargs: 2.0
        analysis._disk_anisotropy_from_vertices = lambda *args, **kwargs: 0.25

        analysis.compute_all_wiring_costs(
            compute_msd=True,
            scale=0.2,
            area_tol=0.1,
            vertex_subset=[0],
            compute_anisotropy=True,
        )

        scale_key = FastCorticalWiringAnalysis.normalize_scales(0.2)[0]
        self.assertTrue(np.isnan(analysis.anisotropy_function[scale_key][0]))

    def test_multiscale_solving_uses_sorted_scales_and_cold_start_bounds(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        cortex_mask = np.array([True, True, True, True], dtype=bool)

        with mock.patch("core_analysis.create_distance_engine", side_effect=_dummy_engine_factory):
            analysis = FastCorticalWiringAnalysis(
                vertices,
                faces,
                cortex_mask,
                engine_type="potpourri",
                eps=1e-6,
                metadata={"subject_id": "synthetic", "hemi": "lh", "surf_type": "unit_square"},
            )

        # Keep geometry calls cheap and deterministic for this control-flow test.
        analysis._perimeter_at_radius = lambda *args, **kwargs: 1.0

        calls = []

        def fake_find_radius(distances_sub, target_area, **kwargs):
            calls.append({"target_area": float(target_area), "r_lower": kwargs.get("r_lower")})
            return float(len(calls))

        analysis._find_radius_for_area = fake_find_radius

        analysis.compute_all_wiring_costs(
            compute_msd=False,
            scale=[0.2, 0.05, 0.1],
            area_tol=0.1,
            vertex_subset=[0],
        )

        self.assertEqual(len(calls), 3)
        target_areas = [c["target_area"] for c in calls]
        self.assertEqual(target_areas, sorted(target_areas))
        self.assertIsNotNone(calls[0]["r_lower"])
        self.assertIsNotNone(calls[1]["r_lower"])
        self.assertIsNotNone(calls[2]["r_lower"])
        self.assertLess(calls[0]["r_lower"], calls[1]["r_lower"])
        self.assertLess(calls[1]["r_lower"], calls[2]["r_lower"])

    def test_compute_all_wiring_costs_consumes_distance_batches_in_order(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        cortex_mask = np.ones(vertices.shape[0], dtype=bool)
        engine_holder = {}

        def factory(_engine_type, vertices_arg, _faces, _engine_kwargs):
            engine = _BatchRecordingDistanceEngine(vertices_arg)
            engine_holder["engine"] = engine
            return engine

        with mock.patch("core_analysis.create_distance_engine", side_effect=factory):
            analysis = FastCorticalWiringAnalysis(
                vertices,
                faces,
                cortex_mask,
                engine_type="potpourri",
                eps=1e-6,
                metadata={"subject_id": "synthetic", "hemi": "lh", "surf_type": "unit_square"},
            )

        analysis._find_radius_for_area = lambda *args, **kwargs: 1.0
        analysis._perimeter_at_radius = lambda *args, **kwargs: 2.0
        analysis._disk_anisotropy_from_vertices = lambda *args, **kwargs: 0.25

        analysis.compute_all_wiring_costs(
            compute_msd=True,
            scale=0.2,
            area_tol=0.1,
            batch_size=2,
            n_samples_between_scales=0,
        )

        expected_batches = [
            analysis._bfs_order[i : i + 2] for i in range(0, len(analysis._bfs_order), 2)
        ]
        self.assertEqual(engine_holder["engine"].batch_calls, expected_batches)


class _StubAnalysis:
    last_compute_kwargs = None
    DEFAULT_SCALES = FastCorticalWiringAnalysis.DEFAULT_SCALES

    @staticmethod
    def normalize_scales(scale):
        return FastCorticalWiringAnalysis.normalize_scales(scale)

    @staticmethod
    def scale_token(scale):
        return FastCorticalWiringAnalysis.scale_token(scale)

    def __init__(
        self,
        _vertices,
        _faces,
        cortex_mask,
        engine_type=None,
        engine_kwargs=None,
        eps=None,
        metadata=None,
        allow_interior_nonmanifold=False,
    ):
        self.engine_type = engine_type
        self.engine_kwargs = engine_kwargs
        self.eps = eps
        self.metadata = dict(metadata or {})
        self.allow_interior_nonmanifold = bool(allow_interior_nonmanifold)
        n = int(cortex_mask.shape[0])
        self.cortex_mask_full = np.asarray(cortex_mask, dtype=bool)
        self.n_vertices_full = n
        self.msd = np.full(n, np.nan, dtype=np.float32)
        self.active_scales = tuple(self.DEFAULT_SCALES)
        self.radius_function = {
            float(s): np.full(n, np.nan, dtype=np.float32) for s in self.active_scales
        }
        self.perimeter_function = {
            float(s): np.full(n, np.nan, dtype=np.float32) for s in self.active_scales
        }
        self.anisotropy_function = {
            float(s): np.full(n, np.nan, dtype=np.float32) for s in self.active_scales
        }

    def compute_all_wiring_costs(self, **kwargs):
        _StubAnalysis.last_compute_kwargs = dict(kwargs)
        scales = FastCorticalWiringAnalysis.normalize_scales(kwargs.get("scale"))
        self.active_scales = scales
        n = self.n_vertices_full
        self.radius_function = {
            float(s): np.full(n, np.nan, dtype=np.float32) for s in self.active_scales
        }
        self.perimeter_function = {
            float(s): np.full(n, np.nan, dtype=np.float32) for s in self.active_scales
        }
        self.anisotropy_function = {
            float(s): np.full(n, np.nan, dtype=np.float32) for s in self.active_scales
        }
        return self.msd, self.radius_function, self.perimeter_function

    def get_metric_arrays(self):
        out = {"msd": self.msd}
        for scale in self.active_scales:
            token = FastCorticalWiringAnalysis.scale_token(scale)
            out[f"radius_{token}"] = self.radius_function[float(scale)]
            out[f"perimeter_{token}"] = self.perimeter_function[float(scale)]
            out[f"anisotropy_{token}"] = self.anisotropy_function[float(scale)]
        return out


class VertexListLoadingTests(unittest.TestCase):
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
        self.cortex_mask = np.array([True, True, True], dtype=bool)
        self.metadata = {
            "legacy_mode": False,
            "surface_path": "/tmp/mock.surf.gii",
            "output_basename": "mock_surface",
            "hemi": "lh",
            "surf_type": "pial",
        }

    def _run_single(self, vertex_list, sample_frac=None, sample_count=None, sample_method=None):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("io_utils.load_surface_and_mask", return_value=(self.vertices, self.faces, self.cortex_mask, self.metadata)):
                with mock.patch("fastcw.FastCorticalWiringAnalysis", _StubAnalysis):
                    with mock.patch("fastcw._save_analysis_outputs", return_value=[]):
                        return fastcw._run_single_surface(
                            standard="freesurfer",
                            surface_path="/tmp/mock.surf",
                            mask_path=None,
                            output_dir=tmpdir,
                            output_basename=None,
                            subject_dir=None,
                            subject_id=None,
                            hemi="lh",
                            surf_type="pial",
                            custom_label=None,
                            no_mask=False,
                            output_format="csv",
                            engine_type="potpourri",
                            engine_kwargs={},
                            compute_msd=False,
                            scale=0.05,
                            area_tol=0.01,
                            eps=1e-6,
                            overwrite=True,
                            sample_frac=sample_frac,
                            sample_count=sample_count,
                            sample_method=sample_method,
                            vertex_list=vertex_list,
                        )

    def test_vertex_list_single_index_is_normalized_to_list(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.write("2\n")
            path = tmp.name
        try:
            self._run_single(path)
            self.assertEqual(_StubAnalysis.last_compute_kwargs["vertex_subset"], [2])
        finally:
            os.unlink(path)

    def test_vertex_list_multiple_indices(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.write("0\n2\n1\n")
            path = tmp.name
        try:
            self._run_single(path)
            self.assertEqual(_StubAnalysis.last_compute_kwargs["vertex_subset"], [0, 2, 1])
        finally:
            os.unlink(path)

    def test_missing_vertex_list_raises_hard_error(self):
        missing = "/tmp/does_not_exist_vertex_subset.txt"
        with self.assertRaises(FileNotFoundError):
            self._run_single(missing)

    def test_vertex_list_overrides_sampling_settings(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.write("1\n")
            path = tmp.name
        try:
            self._run_single(path, sample_frac=0.5, sample_method="random")
            self.assertEqual(_StubAnalysis.last_compute_kwargs["vertex_subset"], [1])
        finally:
            os.unlink(path)


class MetricNameRegistrationTests(unittest.TestCase):
    def test_metric_names_for_scales_include_anisotropy(self):
        names = fastcw._metric_names_for_scales((0.05,))
        self.assertIn("msd", names)
        self.assertIn("radius_0.05", names)
        self.assertIn("perimeter_0.05", names)
        self.assertIn("anisotropy_0.05", names)


class NamingSuffixTests(unittest.TestCase):
    def test_resolve_naming_appends_sampling_suffix(self):
        metadata = {
            "legacy_mode": False,
            "surface_path": "/tmp/mock.surf.gii",
            "output_basename": "mock_surface",
        }
        csv_filename, scalar_stem = fastcw._resolve_naming(
            metadata,
            output_dir="/tmp",
            output_basename=None,
            suffix="_sample-stratified-frac40p",
        )
        self.assertEqual(csv_filename, "mock_surface_sample-stratified-frac40p_wiring_costs.csv")
        self.assertEqual(scalar_stem, "mock_surface_sample-stratified-frac40p.{metric}")

    def test_resolve_naming_appends_subset_suffix(self):
        metadata = {
            "legacy_mode": False,
            "surface_path": "/tmp/mock.surf.gii",
            "output_basename": "mock_surface",
        }
        csv_filename, scalar_stem = fastcw._resolve_naming(
            metadata,
            output_dir="/tmp",
            output_basename=None,
            suffix="_subset",
        )
        self.assertEqual(csv_filename, "mock_surface_subset_wiring_costs.csv")
        self.assertEqual(scalar_stem, "mock_surface_subset.{metric}")


if __name__ == "__main__":
    unittest.main()
