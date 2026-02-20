#!/usr/bin/env python3
import os
import subprocess
import logging
import argparse
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial

ENV = {}

def run_cmd(cmd, subject_id, log_file=None, dry_run=False):
    """Executes a shell command or prints it if in dry-run mode."""
    if dry_run:
        logging.info(f"[DRY RUN - {subject_id}] {' '.join(cmd)}")
        return True

    try:
        if log_file:
            with open(log_file, "a") as f:
                subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT, env=ENV)
        else:
            subprocess.run(cmd, check=True, capture_output=True, text=True, env=ENV)
        return True
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else "Check log file for details."
        logging.error(f"[{subject_id}] Command failed: {' '.join(cmd[:2])}... \nError: {error_msg}")
        return False

def process_subject(subject, args):
    """
    Pipeline: 
    1. Downsample lh/rh native surfaces to target ico order
    2. Downsample lh/rh native cortex labels to target ico order
    3. Run fastcw_pp3d.py on the downsampled surfaces
    """
    logging.info(f"STARTING: {subject}")
    subject_path = Path(args.subjects_dir) / subject
    
    for hemi in ['lh', 'rh']:
        tval = subject_path / "surf" / f"{hemi}.{args.trg_surf}"
        if not args.overwrite and tval.exists() and not args.dry_run:
            continue
            
        surf_cmd = [
            "mri_surf2surf", "--hemi", hemi, "--srcsubject", subject,
            "--sval-xyz", args.src_surf, "--trgsubject", args.trg_subject, "--trgicoorder", str(args.ico_order),
            "--tval-xyz", str(subject_path / "mri/brain.mgz"),
            "--tval", str(tval)
        ]
        if not run_cmd(surf_cmd, subject, dry_run=args.dry_run): return

    for hemi in ['lh', 'rh']:
        label_out = subject_path / "label" / f"{hemi}.{args.trg_label}.label"
        if not args.overwrite and label_out.exists() and not args.dry_run:
            continue

        label_cmd = [
            "mri_label2label", 
            "--srclabel", str(subject_path / f"label/{hemi}.{args.src_label}.label"),
            "--srcsubject", subject, "--trglabel", str(label_out),
            "--trgsubject", "ico", "--regmethod", "surface", 
            "--hemi", hemi, "--trgicoorder", str(args.ico_order)
        ]
        if not run_cmd(label_cmd, subject, dry_run=args.dry_run): return

    log_path = Path(args.log_dir) / f"{subject}.log"
    if args.overwrite and log_path.exists() and not args.dry_run: 
        log_path.unlink()

    fastcw_cmd = [
        args.venv_python, args.fastcw_path,
        "--surf-type", args.trg_surf,
        "--custom-label", args.trg_label
    ]
    
    if args.overwrite:
        fastcw_cmd.append("--overwrite")
        
    fastcw_cmd.extend([args.subjects_dir, subject])

    if run_cmd(fastcw_cmd, subject, log_file=log_path, dry_run=args.dry_run):
        logging.info(f"FINISHED: {subject}")
    else:
        logging.error(f"FAILED: {subject} during FastCW phase.")

def main():
    parser = argparse.ArgumentParser(description="Parallel FreeSurfer Downsampling & FastCW Pipeline")
    parser.add_argument("subject_list", help="Path to text file containing subject IDs (one per line)")
    
    # Core Paths
    parser.add_argument("--subjects-dir", default=os.environ.get("SUBJECTS_DIR"), 
                        help="Path to FreeSurfer SUBJECTS_DIR")
    parser.add_argument("--fs-home", default=os.environ.get("FREESURFER_HOME", "/usr/local/freesurfer"), 
                        help="Path to FreeSurfer installation")
    parser.add_argument("--fastcw-path", default=str(Path(__file__).parent.resolve() / "fastcw_pp3d.py"), 
                        help="By default the path to the fastcw_pp3d.py script. If using legacy fastcw.py script, use this flag with fastcw.py")
    parser.add_argument("--venv-python", default=sys.executable, 
                        help="Path to the Python executable")

    # Pipeline Parameters
    parser.add_argument("--src-surf", default="pial", help="Source surface name")
    parser.add_argument("--trg-surf", default="pialsurface6", help="Target surface name")
    parser.add_argument("--src-label", default="cortex", help="Source label name")
    parser.add_argument("--trg-label", default="cortex6", help="Target label name")
    parser.add_argument("--trg-subject", default="fsaverage6", help="Target subject for registration")
    parser.add_argument("--ico-order", type=int, default=6, help="Icosahedron order for downsampling")
    
    # Execution Parameters
    parser.add_argument("-j", "--jobs", type=int, default=14, help="Number of parallel processes")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--log-dir", default="logs_fastcw", help="Directory to store subject logs")
    
    args = parser.parse_args()

    Path(args.log_dir).mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Validation Checks
    if not args.subjects_dir:
        logging.error("SUBJECTS_DIR is not set in the environment and was not provided via --subjects-dir.")
        sys.exit(1)

    if not Path(args.subject_list).exists():
        logging.error(f"Subject list '{args.subject_list}' not found.")
        sys.exit(1)

    if not Path(args.fastcw_path).exists():
        logging.error(f"FastCW script '{args.fastcw_path}' not found. Please verify the path.")
        sys.exit(1)

    # Build isolated execution environment
    global ENV
    ENV = os.environ.copy()
    ENV.update({
        "FREESURFER_HOME": args.fs_home,
        "SUBJECTS_DIR": args.subjects_dir,
        "PATH": f"{args.fs_home}/bin:{ENV.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{args.fs_home}/lib/gomp/lib:{ENV.get('LD_LIBRARY_PATH', '')}",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE"
    })

    with open(args.subject_list, "r") as f:
        subjects = [line.strip() for line in f if line.strip()]

    mode = "DRY RUN" if args.dry_run else "EXECUTION"
    logging.info(f"Initializing {mode} for {len(subjects)} subjects using {args.jobs} workers.")
    logging.info(f"Mapping {args.src_surf} -> {args.trg_surf} and {args.src_label} -> {args.trg_label}")

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        worker_func = partial(process_subject, args=args)
        executor.map(worker_func, subjects)

    logging.info("Pipeline complete.")

if __name__ == "__main__":
    main()
