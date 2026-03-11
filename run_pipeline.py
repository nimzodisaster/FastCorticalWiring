#!/usr/bin/env python3
"""Parallel native-space FastCW runner (no resampling)."""

import argparse
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path


ENV = {}


def run_cmd(cmd, subject_id, log_file=None, dry_run=False):
    """Execute a command or print it in dry-run mode."""
    if dry_run:
        logging.info(f"[DRY RUN - {subject_id}] {' '.join(cmd)}")
        return True

    try:
        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT, env=ENV)
        else:
            subprocess.run(cmd, check=True, capture_output=True, text=True, env=ENV)
        return True
    except subprocess.CalledProcessError as exc:
        error_msg = exc.stderr if exc.stderr else "Check log file for details."
        logging.error(f"[{subject_id}] Command failed: {' '.join(cmd[:3])}... \nError: {error_msg}")
        return False


def build_fastcw_cmd(args, subject):
    """Build FastCW command for a single subject in native surface space."""
    cmd = [
        args.venv_python,
        args.fastcw_path,
        "--surf-type",
        args.surf_type,
        "--hemispheres",
        *args.hemispheres,
    ]

    if args.custom_label:
        cmd.extend(["--custom-label", args.custom_label])
    if args.mask:
        cmd.extend(["--mask", args.mask])
    if args.no_mask:
        cmd.append("--no-mask")
    if args.no_compute_msd:
        cmd.append("--no-compute-msd")
    if args.overwrite:
        cmd.append("--overwrite")
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    if args.output_format:
        cmd.extend(["--output-format", args.output_format])
    if args.engine:
        cmd.extend(["--engine", args.engine])
    if args.sample_vertices is not None:
        cmd.extend(["--sample-vertices", str(args.sample_vertices)])
    if args.vertex_list:
        cmd.extend(["--vertex-list", args.vertex_list])

    cmd.extend(["--scale", *[str(s) for s in args.scale], "--area-tol", str(args.area_tol), "--eps", str(args.eps)])
    for item in args.engine_kw:
        cmd.extend(["--engine-kw", item])

    cmd.extend([args.subjects_dir, subject])
    return cmd


def process_subject(subject, args):
    """Run FastCW for one subject in native space."""
    logging.info(f"STARTING: {subject}")

    log_path = Path(args.log_dir) / f"{subject}.log"
    if args.overwrite and log_path.exists() and not args.dry_run:
        log_path.unlink()

    cmd = build_fastcw_cmd(args, subject)
    if run_cmd(cmd, subject, log_file=log_path, dry_run=args.dry_run):
        logging.info(f"FINISHED: {subject}")
    else:
        logging.error(f"FAILED: {subject}")


def main():
    parser = argparse.ArgumentParser(description="Parallel native-space FastCW pipeline (no resampling)")
    parser.add_argument("subject_list", help="Path to text file containing subject IDs (one per line)")

    parser.add_argument("--subjects-dir", default=os.environ.get("SUBJECTS_DIR"), help="Path to FreeSurfer SUBJECTS_DIR")
    parser.add_argument(
        "--fastcw-path",
        default=str(Path(__file__).parent.resolve() / "fastcw.py"),
        help="Path to the FastCW CLI script (default: fastcw.py)",
    )
    parser.add_argument("--venv-python", default=sys.executable, help="Path to Python executable")

    parser.add_argument("--surf-type", default="pial", help="Native surface type for FastCW (e.g., pial, white)")
    parser.add_argument("--custom-label", default=None, help="Optional custom cortex label name (without hemi/suffix)")
    parser.add_argument("--hemispheres", nargs="+", default=["lh", "rh"], help="Hemispheres to process")
    parser.add_argument("--mask", default=None, help="Optional explicit mask path passed to FastCW")
    parser.add_argument("--no-mask", action="store_true", help="Run FastCW without cortical masking")
    parser.add_argument("--output-dir", default=None, help="Optional output directory")
    parser.add_argument("--output-format", default=None, help="Optional FastCW output format override")
    parser.add_argument("--engine", default=None, help="Optional FastCW geodesic engine")
    parser.add_argument("--engine-kw", action="append", default=[], help="FastCW engine-specific key=value (repeatable)")
    parser.add_argument("--sample-vertices", type=int, default=None, help="Subset by FPS sample size")
    parser.add_argument("--vertex-list", default=None, help="Subset by vertex-index list file")
    parser.add_argument("--no-compute-msd", action="store_true", help="Disable MSD computation")
    parser.add_argument(
        "--scale",
        nargs="+",
        type=float,
        default=[0.001, 0.005, 0.01, 0.05],
        help="One or more scales for local measures",
    )
    parser.add_argument("--area-tol", type=float, default=0.01, help="Relative tolerance for area search")
    parser.add_argument("--eps", type=float, default=1e-6, help="Numerical tolerance for isoline tests")

    parser.add_argument("-j", "--jobs", type=int, default=14, help="Number of parallel processes")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--log-dir", default="logs_fastcw", help="Directory for per-subject logs")
    args = parser.parse_args()

    Path(args.log_dir).mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if not args.subjects_dir:
        logging.error("SUBJECTS_DIR is not set and was not provided via --subjects-dir.")
        sys.exit(1)
    if not Path(args.subject_list).exists():
        logging.error(f"Subject list '{args.subject_list}' not found.")
        sys.exit(1)
    if not Path(args.fastcw_path).exists():
        logging.error(f"FastCW script '{args.fastcw_path}' not found.")
        sys.exit(1)

    global ENV
    ENV = os.environ.copy()
    ENV.update(
        {
            "SUBJECTS_DIR": args.subjects_dir,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "MKL_DYNAMIC": "FALSE",
        }
    )

    with open(args.subject_list, "r", encoding="utf-8") as f:
        subjects = [line.strip() for line in f if line.strip()]

    mode = "DRY RUN" if args.dry_run else "EXECUTION"
    logging.info(f"Initializing {mode} for {len(subjects)} subjects using {args.jobs} workers.")
    logging.info(f"Native-space run: surf-type={args.surf_type}, hemispheres={','.join(args.hemispheres)}")

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        worker_func = partial(process_subject, args=args)
        executor.map(worker_func, subjects)

    logging.info("Pipeline complete.")


if __name__ == "__main__":
    main()
