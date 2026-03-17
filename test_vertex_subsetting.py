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


class VertexSubsetAnalysisTests(unittest.TestCase):
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

        for idx in (1, 3, 4):
            self.assertTrue(np.isnan(analysis.msd[idx]))
            self.assertTrue(np.isnan(analysis.radius_function[scale_key][idx]))
            self.assertTrue(np.isnan(analysis.perimeter_function[scale_key][idx]))


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
    ):
        self.engine_type = engine_type
        self.engine_kwargs = engine_kwargs
        self.eps = eps
        self.metadata = dict(metadata or {})
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
        return self.msd, self.radius_function, self.perimeter_function

    def get_metric_arrays(self):
        out = {"msd": self.msd}
        for scale in self.active_scales:
            token = FastCorticalWiringAnalysis.scale_token(scale)
            out[f"radius_{token}"] = self.radius_function[float(scale)]
            out[f"perimeter_{token}"] = self.perimeter_function[float(scale)]
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
