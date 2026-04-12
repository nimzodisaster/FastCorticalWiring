#!/usr/bin/env python3
"""
Compute Schaefer parcel aggregations in native space for traditional FreeSurfer
cortical metrics (thickness, area, volume, curv, crv, sulc).

Expects standard FreeSurfer morphometry files in each subject's surf/ directory:
    {hemi}.{metric}  (e.g., lh.thickness)
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import nibabel as nib
except ImportError as exc:
    raise SystemExit("This script requires nibabel.") from exc


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

DEFAULT_SUBJECTS_DIR = Path("/mnt/gold/master_fs_bibsnet/SUBJECTS")
DEFAULT_SUBJECTLIST = Path("./subjectlist.csv")
DEFAULT_ANNOT_NAME = "Schaefer2018_500Parcels_7Networks_order"
DEFAULT_OUTPUT_DIR = Path("./fs_traditional_schaefer_csv")
DEFAULT_HEMIS = ("lh", "rh")
DEFAULT_METRICS = ("thickness", "area", "volume", "curv", "crv", "sulc")


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def check_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required FreeSurfer binary not found on PATH: {name}")

def run_cmd(cmd: Sequence[str], env: Optional[dict] = None) -> None:
    subprocess.run(list(cmd), check=True, env=env)

def read_subject_ids(subjectlist_csv: Path) -> List[str]:
    if not subjectlist_csv.exists():
        raise FileNotFoundError(f"Subject list not found: {subjectlist_csv}")

    lines = subjectlist_csv.read_text(encoding="utf-8").splitlines()
    rows = [line.split(",") if "," in line else line.split("\t") if "\t" in line else [line] 
            for line in lines if line.strip()]

    if not rows:
        raise ValueError(f"Subject list is empty: {subjectlist_csv}")

    header = [cell.strip().lower() for cell in rows[0]]
    target_headers = {"subject", "subject_id", "subjectid", "fsid", "freesurfer_subject_id", "freesurfer"}
    
    subject_col = next((i for i, name in enumerate(header) if name in target_headers), 0)
    has_header = header[subject_col] in target_headers

    subject_ids = [row[subject_col].strip() for row in rows[1 if has_header else 0:] if subject_col < len(row) and row[subject_col].strip()]

    if not subject_ids:
        raise ValueError(f"No subject IDs could be read from: {subjectlist_csv}")
    return subject_ids

def read_scalar_overlay(path: Path) -> np.ndarray:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".nii.gz") or path.suffix.lower() in {".nii", ".mgh", ".mgz"}:
        return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float64).reshape(-1)
    try:
        return np.asarray(nib.freesurfer.read_morph_data(str(path)), dtype=np.float64).reshape(-1)
    except Exception as exc:
        raise ValueError(f"Unsupported scalar file or failed to read: {path}") from exc

def sanitize_label(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._]+", "_", name.strip())
    return re.sub(r"_+", "_", cleaned).strip("_") or "unnamed"


# -----------------------------------------------------------------------------
# FreeSurfer Schaefer transfer utilities
# -----------------------------------------------------------------------------

def read_annot(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    labels, ctab, names = nib.freesurfer.read_annot(str(path), orig_ids=False)
    decoded = [n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n) for n in names]
    return np.asarray(labels, dtype=np.int32), np.asarray(ctab), decoded

def parcel_names_from_fsaverage(subjects_dir: Path, annot_stem: str, hemi: str) -> Dict[int, str]:
    annot_path = subjects_dir / "fsaverage" / "label" / f"{hemi}.{annot_stem}.annot"
    labels, _, names = read_annot(annot_path)
    code_to_name: Dict[int, str] = {}
    for code in np.unique(labels):
        code_int = int(code)
        if code_int >= 0:
            code_to_name[code_int] = names[code_int] if code_int < len(names) else f"label_{code_int}"
    return code_to_name

def ensure_fsaverage_seg(subjects_dir: Path, annot_stem: str, hemi: str, overwrite: bool = False) -> Path:
    fsavg_dir = subjects_dir / "fsaverage"
    annot_path = fsavg_dir / "label" / f"{hemi}.{annot_stem}.annot"
    seg_path = fsavg_dir / "label" / f"{hemi}.{annot_stem}.seg.mgh"
    ctab_path = fsavg_dir / "label" / f"{hemi}.{annot_stem}.seg.ctab"

    if not annot_path.exists():
        raise FileNotFoundError(str(annot_path))
    if seg_path.exists() and ctab_path.exists() and not overwrite:
        return seg_path

    env = dict(**os.environ)
    env["SUBJECTS_DIR"] = str(subjects_dir)
    cmd = [
        "mri_annotation2label", "--subject", "fsaverage", "--hemi", hemi,
        "--seg", str(seg_path), "--ctab", str(ctab_path),
        "--annotation", annot_stem, "--segbase", "0"
    ]
    run_cmd(cmd, env=env)
    return seg_path

def ensure_subject_native_seg(subjects_dir: Path, subject_id: str, annot_stem: str, hemi: str, overwrite: bool = False) -> Path:
    fsavg_seg = ensure_fsaverage_seg(subjects_dir, annot_stem, hemi, overwrite=False)
    out_seg = subjects_dir / subject_id / "surf" / f"{hemi}.{annot_stem}.from_fsaverage.seg.mgh"
    sphere_reg = subjects_dir / subject_id / "surf" / f"{hemi}.sphere.reg"
    
    if not sphere_reg.exists():
        raise FileNotFoundError(str(sphere_reg))
    if out_seg.exists() and not overwrite:
        return out_seg

    env = dict(**os.environ)
    env["SUBJECTS_DIR"] = str(subjects_dir)
    cmd = [
        "mri_surf2surf", "--mapmethod", "nnf", "--hemi", hemi,
        "--srcsubject", "fsaverage", "--srcsurfreg", "sphere.reg", "--sval", str(fsavg_seg),
        "--trgsubject", subject_id, "--trgsurfreg", "sphere.reg", "--tval", str(out_seg)
    ]
    run_cmd(cmd, env=env)
    return out_seg

def parcellate_metric(metric_values: np.ndarray, vertex_labels: np.ndarray, code_to_name: Dict[int, str], hemi: str, metric_name: str) -> Dict[str, float]:
    if metric_values.shape[0] != vertex_labels.shape[0]:
        raise ValueError(f"Length mismatch: metric={metric_values.shape[0]}, labels={vertex_labels.shape[0]}")
    
    out: Dict[str, float] = {}
    is_extensive = metric_name in ("area", "volume")
    
    for code in sorted(code_to_name):
        mask = vertex_labels == code
        vals = metric_values[mask]
        vals = vals[np.isfinite(vals)]
        
        if not vals.size:
            value = np.nan
        else:
            value = float(np.nansum(vals)) if is_extensive else float(np.nanmean(vals))
            
        clean_name = sanitize_label(code_to_name[code])
        col_name = clean_name if "LH" in clean_name.upper() or "RH" in clean_name.upper() else f"{hemi}_{clean_name}"
        out[col_name] = value
    return out


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def run(
    *,
    subjects_dir: Path,
    subjectlist_csv: Path,
    metrics: Sequence[str],
    annot_stem: str,
    output_dir: Path,
    hemis: Sequence[str],
    overwrite_seg_cache: bool = False,
    overwrite_csv: bool = False,
) -> int:
    check_binary("mri_annotation2label")
    check_binary("mri_surf2surf")

    output_dir.mkdir(parents=True, exist_ok=True)
    subject_ids = read_subject_ids(subjectlist_csv)
    hemi_code_to_name = {hemi: parcel_names_from_fsaverage(subjects_dir, annot_stem, hemi) for hemi in hemis}
    
    for hemi in hemis:
        ensure_fsaverage_seg(subjects_dir, annot_stem, hemi, overwrite=overwrite_seg_cache)

    csv_paths = {m: output_dir / f"fs_traditional.{m}.csv" for m in metrics}
    for path in csv_paths.values():
        if path.exists() and not overwrite_csv:
            raise FileExistsError(f"Output CSV already exists: {path}")

    all_metric_rows: Dict[str, List[Dict[str, object]]] = {m: [] for m in metrics}

    for idx, subject_id in enumerate(subject_ids, start=1):
        print(f"[{idx}/{len(subject_ids)}] {subject_id}")
        per_subject_rows: Dict[str, Dict[str, object]] = {m: {"subject_id": subject_id} for m in metrics}

        for hemi in hemis:
            native_seg = ensure_subject_native_seg(
                subjects_dir=subjects_dir,
                subject_id=subject_id,
                annot_stem=annot_stem,
                hemi=hemi,
                overwrite=overwrite_seg_cache,
            )
            vertex_labels = np.rint(read_scalar_overlay(native_seg)).astype(np.int32)

            for metric in metrics:
                metric_path = subjects_dir / subject_id / "surf" / f"{hemi}.{metric}"
                if not metric_path.exists():
                    raise FileNotFoundError(f"Missing source metric overlay: {metric_path}")
                
                values = read_scalar_overlay(metric_path)
                per_subject_rows[metric].update(
                    parcellate_metric(
                        metric_values=values,
                        vertex_labels=vertex_labels,
                        code_to_name=hemi_code_to_name[hemi],
                        hemi=hemi,
                        metric_name=metric
                    )
                )

        for metric in metrics:
            all_metric_rows[metric].append(per_subject_rows[metric])

    for metric, rows in all_metric_rows.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        ordered_cols = ["subject_id"] + sorted([c for c in df.columns if c != "subject_id"])
        df = df[ordered_cols]
        out_csv = csv_paths[metric]
        df.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv} ({df.shape[0]} subjects x {df.shape[1]-1} parcels)")

    return 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects-dir", type=Path, default=DEFAULT_SUBJECTS_DIR)
    p.add_argument("--subjectlist", type=Path, default=DEFAULT_SUBJECTLIST)
    p.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    p.add_argument("--annot-stem", default=DEFAULT_ANNOT_NAME)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--hemis", nargs="+", choices=["lh", "rh"], default=list(DEFAULT_HEMIS))
    p.add_argument("--overwrite-seg-cache", action="store_true", default=False)
    p.add_argument("--overwrite-csv", action="store_true", default=False)
    return p.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    return run(
        subjects_dir=args.subjects_dir,
        subjectlist_csv=args.subjectlist,
        metrics=args.metrics,
        annot_stem=str(args.annot_stem),
        output_dir=args.output_dir,
        hemis=args.hemis,
        overwrite_seg_cache=bool(args.overwrite_seg_cache),
        overwrite_csv=bool(args.overwrite_csv),
    )

if __name__ == "__main__":
    raise SystemExit(main())
