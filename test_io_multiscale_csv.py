#!/usr/bin/env python3
"""Tests for multi-scale CSV export layout."""

import os
import tempfile
import unittest

import numpy as np

from io_utils import save_results_csv


class MultiScaleCsvTests(unittest.TestCase):
    def test_csv_contains_dynamic_scale_columns(self):
        cortex_mask = np.array([True, False, True], dtype=bool)
        msd = np.array([1.0, np.nan, 2.0], dtype=np.float32)
        global_entropy = np.array([1.1, np.nan, 1.3], dtype=np.float32)
        radius = {
            0.001: np.array([0.1, np.nan, 0.2], dtype=np.float32),
            0.05: np.array([1.1, np.nan, 1.2], dtype=np.float32),
        }
        perimeter = {
            0.001: np.array([0.3, np.nan, 0.4], dtype=np.float32),
            0.05: np.array([1.3, np.nan, 1.4], dtype=np.float32),
        }
        anisotropy = {
            0.001: np.array([0.05, np.nan, 0.07], dtype=np.float32),
            0.05: np.array([0.15, np.nan, 0.17], dtype=np.float32),
        }
        local_entropy = {
            0.001: np.array([0.45, np.nan, 0.47], dtype=np.float32),
            0.05: np.array([0.85, np.nan, 0.87], dtype=np.float32),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = save_results_csv(
                tmpdir,
                "results.csv",
                cortex_mask,
                msd,
                global_entropy,
                radius,
                perimeter,
                anisotropy,
                local_entropy,
            )
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

        self.assertGreaterEqual(len(lines), 2)
        self.assertEqual(
            lines[0],
            "vertex_id,is_cortex,msd,global_entropy,radius_0.001,perimeter_0.001,anisotropy_0.001,local_entropy_0.001,radius_0.05,perimeter_0.05,anisotropy_0.05,local_entropy_0.05",
        )
        row0 = lines[1].split(",")
        self.assertEqual(row0[0], "0")
        self.assertEqual(row0[1], "1")
        self.assertAlmostEqual(float(row0[2]), 1.0, places=6)
        self.assertAlmostEqual(float(row0[3]), 1.1, places=6)
        self.assertAlmostEqual(float(row0[4]), 0.1, places=6)
        self.assertAlmostEqual(float(row0[5]), 0.3, places=6)
        self.assertAlmostEqual(float(row0[6]), 0.05, places=6)
        self.assertAlmostEqual(float(row0[7]), 0.45, places=6)
        self.assertAlmostEqual(float(row0[8]), 1.1, places=6)
        self.assertAlmostEqual(float(row0[9]), 1.3, places=6)
        self.assertAlmostEqual(float(row0[10]), 0.15, places=6)
        self.assertAlmostEqual(float(row0[11]), 0.85, places=6)


if __name__ == "__main__":
    unittest.main()
