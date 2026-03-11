#!/usr/bin/env python3
"""CLI parsing smoke checks for engine terminology."""

import subprocess
import sys
import unittest


class EngineCliParsingTests(unittest.TestCase):
    def test_help_lists_engine_choices_without_legacy(self):
        proc = subprocess.run(
            [sys.executable, "fastcw.py", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        text = proc.stdout + proc.stderr
        self.assertIn("--engine {potpourri,potpourri_fmm,pycortex,pygeodesic}", text)
        self.assertNotIn("{potpourri,legacy,pygeodesic}", text)
        self.assertNotIn("--visualize", text)
        self.assertIn("--scale SCALE [SCALE ...]", text)
        self.assertIn("--allow-eigen-fallback", text)

    def test_engine_pycortex_is_accepted_by_parser(self):
        proc = subprocess.run(
            [sys.executable, "fastcw.py", "--engine", "pycortex", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)

    def test_engine_legacy_is_rejected_by_parser(self):
        proc = subprocess.run(
            [sys.executable, "fastcw.py", "--engine", "legacy", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        text = proc.stdout + proc.stderr
        self.assertIn("invalid choice", text)

    def test_subset_flags_are_mutually_exclusive(self):
        proc = subprocess.run(
            [
                sys.executable,
                "fastcw.py",
                "--sample-vertices",
                "100",
                "--vertex-list",
                "verts.txt",
                "subjects_dir",
                "subject_id",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        text = proc.stdout + proc.stderr
        self.assertIn("not allowed with argument", text)

    def test_scale_accepts_single_value(self):
        proc = subprocess.run(
            [sys.executable, "fastcw.py", "--scale", "0.05", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)

    def test_scale_accepts_multiple_values(self):
        proc = subprocess.run(
            [sys.executable, "fastcw.py", "--scale", "0.01", "0.05", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
