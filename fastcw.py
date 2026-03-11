#!/usr/bin/env python3
"""CLI and output orchestration for FastCorticalWiring."""

import argparse
import os

import numpy as np

from core_analysis import FastCorticalWiringAnalysis
from io_utils import farthest_point_sampling, infer_output_basename

METRIC_NAMES = ("msd", "radius", "perimeter")


def _resolve_output_kinds(output_format, standard):
    fmt = str(output_format).lower()
    std = str(standard).lower()
    if fmt == "auto":
        return ("csv", "mgh") if std == "freesurfer" else ("csv", "gii")
    if fmt == "csv":
        return ("csv",)
    if fmt == "mgh":
        if std != "freesurfer":
            raise ValueError("--output-format mgh requires --standard freesurfer.")
        return ("csv", "mgh")
    if fmt == "gii":
        if std != "fslr":
            raise ValueError("--output-format gii requires --standard fslr.")
        return ("csv", "gii")
    raise ValueError(f"Unsupported output format: {output_format}")


def _resolve_naming(metadata, output_dir, output_basename=None):
    output_dir_abs = os.path.abspath(output_dir)
    positional_fs_mode = bool(metadata.get("legacy_mode", False))
    subject_dir = metadata.get("subject_dir")
    subject_id = metadata.get("subject_id")
    hemi = metadata.get("hemi")
    surf_type = metadata.get("surf_type")

    if positional_fs_mode and subject_dir and subject_id and hemi and surf_type:
        standard_surf_dir = os.path.abspath(os.path.join(subject_dir, subject_id, "surf"))
        csv_filename = f"{subject_id}_{hemi}_{surf_type}_wiring_costs.csv"
        if output_dir_abs == standard_surf_dir:
            scalar_stem = f"{hemi}.{surf_type}.{{metric}}"
        else:
            scalar_stem = f"{subject_id}_{hemi}_{surf_type}.{{metric}}"
        viz_base = f"{subject_id}_{hemi}_{surf_type}"
        return csv_filename, scalar_stem, viz_base

    base = output_basename or metadata.get("output_basename") or infer_output_basename(metadata.get("surface_path"))
    csv_filename = f"{base}_wiring_costs.csv"
    scalar_stem = f"{base}.{{metric}}"
    return csv_filename, scalar_stem, base


def _expected_output_files(output_dir, output_kinds, csv_filename, scalar_stem):
    out = []
    if "csv" in output_kinds:
        out.append(os.path.join(output_dir, csv_filename))
    if "mgh" in output_kinds:
        out.extend([os.path.join(output_dir, f"{scalar_stem.format(metric=m)}.mgh") for m in METRIC_NAMES])
    if "gii" in output_kinds:
        out.extend([os.path.join(output_dir, f"{scalar_stem.format(metric=m)}.shape.gii") for m in METRIC_NAMES])
    return out


def _save_analysis_outputs(analysis, output_dir, output_kinds, csv_filename, scalar_stem):
    from io_utils import save_results_csv, save_results_gifti, save_results_mgh

    metrics = analysis.get_metric_arrays()
    written = []

    if "csv" in output_kinds:
        csv_path = save_results_csv(
            output_dir,
            csv_filename,
            analysis.cortex_mask_full,
            analysis.msd,
            analysis.radius_function,
            analysis.perimeter_function,
        )
        written.append(csv_path)

    if "mgh" in output_kinds:
        template_mgz = analysis.metadata.get("mgh_template_path")
        written.extend(
            save_results_mgh(
                output_dir,
                f"{scalar_stem}.mgh",
                metrics,
                n_vertices=analysis.n_vertices_full,
                template_mgz_path=template_mgz,
            )
        )

    if "gii" in output_kinds:
        written.extend(
            save_results_gifti(
                output_dir,
                f"{scalar_stem}.shape.gii",
                metrics,
                n_vertices=analysis.n_vertices_full,
            )
        )

    return written


def _run_single_surface(
    *,
    standard,
    surface_path,
    mask_path,
    output_dir,
    output_basename,
    subject_dir,
    subject_id,
    hemi,
    surf_type,
    custom_label,
    no_mask,
    output_format,
    engine_type,
    engine_kwargs,
    compute_msd,
    scale,
    area_tol,
    eps,
    overwrite,
    visualize,
    sample_vertices=None,
    vertex_list=None,
):
    from io_utils import load_surface_and_mask

    vertices, faces, cortex_mask, metadata = load_surface_and_mask(
        standard=standard,
        surface_path=surface_path,
        mask_path=mask_path,
        subject_dir=subject_dir,
        subject_id=subject_id,
        hemi=hemi,
        surf_type=surf_type,
        custom_label=custom_label,
        no_mask=no_mask,
    )

    if output_dir is None:
        if metadata.get("legacy_mode") and subject_dir and subject_id:
            output_dir = os.path.join(subject_dir, subject_id, "surf")
        elif metadata.get("surface_path"):
            output_dir = os.path.dirname(os.path.abspath(metadata["surface_path"])) or os.getcwd()
        else:
            output_dir = os.getcwd()

    output_kinds = _resolve_output_kinds(output_format, standard)
    csv_filename, scalar_stem, viz_base = _resolve_naming(metadata, output_dir, output_basename=output_basename)
    expected_files = _expected_output_files(output_dir, output_kinds, csv_filename, scalar_stem)
    existing_files = [f for f in expected_files if os.path.exists(f)]
    if existing_files and not overwrite:
        print("ERROR: Output files already exist:")
        for path in existing_files:
            print(f"  - {path}")
        print("Use --overwrite or a different --output-dir.")
        return []

    vertex_subset = None
    if vertex_list:
        if not os.path.exists(vertex_list):
            raise FileNotFoundError(f"--vertex-list file not found: {vertex_list}")
        loaded = np.loadtxt(vertex_list, dtype=int)
        vertex_subset = np.atleast_1d(loaded).astype(np.int64).tolist()
    elif sample_vertices is not None:
        n_sample = int(sample_vertices)
        if n_sample <= 0:
            raise ValueError("--sample-vertices must be a positive integer.")
        print(f"Applying Farthest Point Sampling for {n_sample} vertices...")
        vertex_subset = farthest_point_sampling(vertices, n_sample, cortex_mask)

    analysis = FastCorticalWiringAnalysis(
        vertices,
        faces,
        cortex_mask,
        engine_type=engine_type,
        engine_kwargs=engine_kwargs,
        eps=eps,
        metadata=metadata,
    )
    analysis.compute_all_wiring_costs(
        compute_msd=compute_msd,
        scale=scale,
        area_tol=area_tol,
        vertex_subset=vertex_subset,
    )
    written = _save_analysis_outputs(analysis, output_dir, output_kinds, csv_filename, scalar_stem)

    if visualize:
        if compute_msd and np.any(np.isfinite(analysis.msd)):
            analysis.visualize_results("msd", os.path.join(output_dir, f"{viz_base}_msd.png"))
        if np.any(np.isfinite(analysis.radius_function)):
            analysis.visualize_results("radius", os.path.join(output_dir, f"{viz_base}_radius.png"))
        if np.any(np.isfinite(analysis.perimeter_function)):
            analysis.visualize_results("perimeter", os.path.join(output_dir, f"{viz_base}_perimeter.png"))

    return written


def process_subject(
    subject_dir,
    subject_id,
    output_dir=None,
    hemispheres=["lh", "rh"],
    surf_type="pial",
    custom_label=None,
    compute_msd=True,
    scale=0.05,
    area_tol=0.01,
    eps=1e-6,
    overwrite=False,
    visualize=False,
    output_format="auto",
    engine_type="potpourri",
    engine_kwargs=None,
    mask_path=None,
    no_mask=False,
    sample_vertices=None,
    vertex_list=None,
):
    """Positional FreeSurfer workflow compatibility entry point."""
    if output_dir is None:
        output_dir = os.path.join(subject_dir, subject_id, "surf")

    print("=" * 60)
    print(f"Processing subject: {subject_id}")
    print(f"Surface type: {surf_type}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    all_written = []
    resolved_engine_kwargs = dict(engine_kwargs or {})
    for hemi in hemispheres:
        print(f"\n--- Processing {hemi} hemisphere ---")
        written = _run_single_surface(
            standard="freesurfer",
            surface_path=None,
            mask_path=mask_path,
            output_dir=output_dir,
            output_basename=None,
            subject_dir=subject_dir,
            subject_id=subject_id,
            hemi=hemi,
            surf_type=surf_type,
            custom_label=custom_label,
            no_mask=no_mask,
            output_format=output_format,
            engine_type=engine_type,
            engine_kwargs=resolved_engine_kwargs,
            compute_msd=compute_msd,
            scale=scale,
            area_tol=area_tol,
            eps=eps,
            overwrite=overwrite,
            visualize=visualize,
            sample_vertices=sample_vertices,
            vertex_list=vertex_list,
        )
        all_written.extend(written)

    return all_written


def _coerce_engine_kw_value(raw):
    """Parse basic scalar types from CLI key=value strings."""
    text = str(raw).strip()
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        if "." in text or "e" in lower:
            return float(text)
        return int(text)
    except ValueError:
        return text


def parse_engine_kwargs(values):
    """Parse repeated --engine-kw KEY=VALUE args into dict."""
    out = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Invalid --engine-kw '{item}'. Expected key=value.")
        key, raw_val = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --engine-kw '{item}'. Empty key.")
        out[key] = _coerce_engine_kw_value(raw_val)
    return out


def run_cli(default_engine="potpourri"):
    """Command-line interface for cortical wiring analysis."""

    parser = argparse.ArgumentParser(
        description="Fast computation of intrinsic cortical wiring costs with pluggable geodesic engines"
    )

    parser.add_argument("subject_dir", nargs="?", help="FreeSurfer subjects directory (positional mode)")
    parser.add_argument("subject_id", nargs="?", help="Subject ID (positional mode)")

    parser.add_argument("--standard", choices=["freesurfer", "fslr"], default="freesurfer", help="Surface standard")
    parser.add_argument("--surface", default=None, help="Input surface path (.surf.gii for fsLR, FreeSurfer surface for freesurfer)")
    parser.add_argument("--mask", default=None, help="Mask path (.label/.annot for freesurfer, .shape.gii/.func.gii for fsLR)")
    parser.add_argument("--output-format", choices=["auto", "csv", "mgh", "gii"], default="auto")
    parser.add_argument("--output-basename", default=None, help="Optional output basename override")
    parser.add_argument("--no-mask", action="store_true", default=False, help="Analyze all vertices without masking")
    parser.add_argument(
        "--engine",
        choices=["potpourri", "potpourri_fmm", "pycortex", "pygeodesic"],
        default=str(default_engine).lower(),
        help="Geodesic distance engine",
    )
    parser.add_argument("--engine-kw", action="append", default=[], help="Engine-specific option as key=value (repeatable)")

    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--hemispheres", nargs="+", default=["lh", "rh"], help="Hemispheres for positional FreeSurfer mode")
    parser.add_argument("--hemi", default="lh", help="Hemisphere label for explicit surface mode naming")
    parser.add_argument(
        "--surf-type",
        default="pial",
        help="Surface type: pial, white, inflated, or custom surface name (positional FreeSurfer mode)",
    )
    parser.add_argument(
        "--custom-label",
        default=None,
        help="Custom cortex label name for non-standard FreeSurfer surfaces (e.g., cortex6 for {hemi}.cortex6.label)",
    )
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing output files")
    parser.add_argument("--visualize", action="store_true", default=False, help="Generate visualization PNG files")
    parser.add_argument("--compute-msd", dest="compute_msd", action="store_true", default=True, help="Compute MSDs")
    parser.add_argument("--no-compute-msd", dest="compute_msd", action="store_false", help="Disable MSD computation")
    parser.add_argument("--scale", type=float, default=0.05, help="Scale for local measures as proportion of total area")
    parser.add_argument("--area-tol", type=float, default=0.01, help="Relative tolerance for area binary search")
    parser.add_argument("--eps", type=float, default=1e-6, help="Numerical tolerance for isoline tests")
    subset_group = parser.add_mutually_exclusive_group()
    subset_group.add_argument(
        "--sample-vertices",
        type=int,
        default=None,
        help="Spatially uniform sample of N cortical vertices",
    )
    subset_group.add_argument(
        "--vertex-list",
        type=str,
        default=None,
        help="Path to text file with specific vertex indices",
    )
    args = parser.parse_args()

    engine_kwargs = parse_engine_kwargs(args.engine_kw)
    using_explicit_surface = args.surface is not None
    using_positional_fs_mode = (args.subject_dir is not None) and (args.subject_id is not None) and (not using_explicit_surface)

    if using_explicit_surface:
        if args.standard == "fslr" and args.hemispheres != ["lh", "rh"]:
            print("WARNING: --hemispheres is ignored when --surface is provided.")
        if args.standard == "freesurfer" and args.surf_type not in ["pial", "white", "inflated"] and args.custom_label is None:
            print("WARNING: Using custom FreeSurfer surface without --custom-label.")
        written = _run_single_surface(
            standard=args.standard,
            surface_path=args.surface,
            mask_path=args.mask,
            output_dir=args.output_dir,
            output_basename=args.output_basename,
            subject_dir=args.subject_dir,
            subject_id=args.subject_id,
            hemi=args.hemi,
            surf_type=args.surf_type,
            custom_label=args.custom_label,
            no_mask=args.no_mask,
            output_format=args.output_format,
            engine_type=args.engine,
            engine_kwargs=engine_kwargs,
            compute_msd=args.compute_msd,
            scale=args.scale,
            area_tol=args.area_tol,
            eps=args.eps,
            overwrite=args.overwrite,
            visualize=args.visualize,
            sample_vertices=args.sample_vertices,
            vertex_list=args.vertex_list,
        )
        if written:
            print("Wrote outputs:")
            for path in written:
                print(f"  - {path}")
        print("Analysis complete!")
        return

    if using_positional_fs_mode:
        if args.standard != "freesurfer":
            raise ValueError("Positional FreeSurfer mode only supports --standard freesurfer.")
        if args.surf_type not in ["pial", "white", "inflated"] and args.custom_label is None:
            print("WARNING: Using custom surface without custom cortex label.")
            print("This may result in incorrect cortical masking if the mesh has been resampled.")
            print("Consider using --custom-label to specify the appropriate cortex label.")

        process_subject(
            args.subject_dir,
            args.subject_id,
            args.output_dir,
            hemispheres=args.hemispheres,
            surf_type=args.surf_type,
            custom_label=args.custom_label,
            compute_msd=args.compute_msd,
            scale=args.scale,
            area_tol=args.area_tol,
            eps=args.eps,
            overwrite=args.overwrite,
            visualize=args.visualize,
            output_format=args.output_format,
            engine_type=args.engine,
            engine_kwargs=engine_kwargs,
            mask_path=args.mask,
            no_mask=args.no_mask,
            sample_vertices=args.sample_vertices,
            vertex_list=args.vertex_list,
        )
        print("Analysis complete!")
        return

    raise ValueError(
        "Specify either positional arguments <subject_dir> <subject_id> "
        "or explicit --surface with --standard."
    )


if __name__ == "__main__":
    run_cli()
