#!/usr/bin/env python3
"""Tests for strict FreeSurfer mask resolution behavior."""

import unittest
from unittest import mock

import numpy as np

import io_utils


class MaskFailFastTests(unittest.TestCase):
    def test_freesurfer_raises_when_no_mask_is_found(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        with mock.patch("io_utils._load_freesurfer_surface", return_value=(vertices, faces)):
            with mock.patch("os.path.isdir", return_value=False):
                with self.assertRaises(FileNotFoundError):
                    io_utils.load_surface_and_mask(
                        standard="freesurfer",
                        surface_path="/tmp/subj/surf/lh.pial",
                        mask_path=None,
                        subject_dir=None,
                        subject_id=None,
                        hemi="lh",
                        surf_type="pial",
                        custom_label=None,
                        no_mask=False,
                    )

    def test_freesurfer_no_mask_flag_still_allows_full_mesh(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        with mock.patch("io_utils._load_freesurfer_surface", return_value=(vertices, faces)):
            v, f, cortex_mask, metadata = io_utils.load_surface_and_mask(
                standard="freesurfer",
                surface_path="/tmp/subj/surf/lh.pial",
                mask_path=None,
                subject_dir=None,
                subject_id=None,
                hemi="lh",
                surf_type="pial",
                custom_label=None,
                no_mask=True,
            )

        self.assertEqual(v.shape[0], 3)
        self.assertEqual(f.shape[0], 1)
        self.assertTrue(np.all(cortex_mask))
        self.assertEqual(metadata.get("mask_source"), "all_vertices_no_mask")


if __name__ == "__main__":
    unittest.main()
