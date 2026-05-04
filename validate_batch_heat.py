#!/usr/bin/env python3
"""Validate batch_heat distances against potpourri heat-method distances."""

import argparse
import sys

import numpy as np

from core_analysis import FastCorticalWiringAnalysis
from distance_engines import BatchHeatDistanceEngine, PotpourriDistanceEngine
from io_utils import load_surface_and_mask


def _build_submesh(vertices, faces, cortex_mask):
    analysis = FastCorticalWiringAnalysis.__new__(FastCorticalWiringAnalysis)
    return analysis._build_cortex_submesh(vertices, faces, cortex_mask)[:2]


def _error_stats(reference, candidate, rel_tol=1e-6):
    finite = np.isfinite(reference) & np.isfinite(candidate)
    abs_err = np.abs(candidate[finite] - reference[finite])
    denom = np.maximum(np.abs(reference[finite]), 1e-12)
    rel_err = abs_err / denom
    if rel_err.size == 0:
        raise RuntimeError("No finite distances were available for validation.")
    return {
        "n": int(rel_err.size),
        "max_abs": float(np.max(abs_err)),
        "max_rel": float(np.max(rel_err)),
        "p99_rel": float(np.percentile(rel_err, 99.0)),
        "frac_rel_lt_tol": float(np.mean(rel_err < float(rel_tol))),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate batch_heat against potpourri heat-method distances."
    )
    parser.add_argument("--standard", choices=["freesurfer", "fslr"], default="freesurfer")
    parser.add_argument("--surface", default=None, help="Input surface path")
    parser.add_argument("--mask", default=None, help="Input cortex mask path")
    parser.add_argument("--subject-dir", default=None)
    parser.add_argument("--subject-id", default=None)
    parser.add_argument("--hemi", default="lh")
    parser.add_argument("--surf-type", default="pial")
    parser.add_argument("--custom-label", default=None)
    parser.add_argument("--no-mask", action="store_true", default=False)
    parser.add_argument("--n-sources", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--allow-eigen-fallback", action="store_true", default=False)
    parser.add_argument("--rel-tol", type=float, default=1e-6)
    parser.add_argument("--required-fraction", type=float, default=0.99)
    parser.add_argument(
        "--batch-heat-debug",
        action="store_true",
        default=False,
        help="Print one-off batch_heat internals for sign/scale debugging.",
    )
    args = parser.parse_args(argv)

    if args.n_sources <= 0:
        raise ValueError("--n-sources must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    vertices, faces, cortex_mask, _metadata = load_surface_and_mask(
        standard=args.standard,
        surface_path=args.surface,
        mask_path=args.mask,
        subject_dir=args.subject_dir,
        subject_id=args.subject_id,
        hemi=args.hemi,
        surf_type=args.surf_type,
        custom_label=args.custom_label,
        no_mask=args.no_mask,
    )
    vertices_sub, faces_sub = _build_submesh(vertices, faces, cortex_mask)
    if vertices_sub.shape[0] == 0 or faces_sub.shape[0] == 0:
        raise RuntimeError("Cortex mask produced an empty validation submesh.")

    rng = np.random.default_rng(args.seed)
    n_sources = min(int(args.n_sources), int(vertices_sub.shape[0]))
    sources = rng.choice(vertices_sub.shape[0], size=n_sources, replace=False).astype(np.int64)

    potpourri = PotpourriDistanceEngine(
        vertices_sub,
        faces_sub,
        use_robust=False,
        allow_eigen_fallback=args.allow_eigen_fallback,
    )
    batch_heat = BatchHeatDistanceEngine(
        vertices_sub,
        faces_sub,
        debug_diagnostics=args.batch_heat_debug,
    )

    print(f"Validation submesh: {vertices_sub.shape[0]} vertices, {faces_sub.shape[0]} faces")
    print(f"Sources: {n_sources}, batch_size={args.batch_size}")

    all_ref = []
    all_batch = []
    all_single = []
    for start in range(0, n_sources, args.batch_size):
        batch_sources = sources[start : start + args.batch_size]
        ref = np.column_stack([potpourri.compute_distance(int(src)) for src in batch_sources])
        cand_batch = batch_heat.compute_distance_batch(batch_sources)
        cand_single = np.column_stack([batch_heat.compute_distance(int(src)) for src in batch_sources])
        all_ref.append(ref)
        all_batch.append(cand_batch)
        all_single.append(cand_single)

    ref = np.column_stack(all_ref)
    cand_batch = np.column_stack(all_batch)
    cand_single = np.column_stack(all_single)

    batch_stats = _error_stats(ref, cand_batch, rel_tol=args.rel_tol)
    single_stats = _error_stats(ref, cand_single, rel_tol=args.rel_tol)
    batch_vs_single = _error_stats(cand_single, cand_batch, rel_tol=args.rel_tol)

    print("batch_heat batched vs potpourri:")
    print(batch_stats)
    print("batch_heat single-source vs potpourri:")
    print(single_stats)
    print("batch_heat batched vs single-source:")
    print(batch_vs_single)

    passed = (
        batch_stats["frac_rel_lt_tol"] >= float(args.required_fraction)
        and single_stats["frac_rel_lt_tol"] >= float(args.required_fraction)
    )
    if not passed:
        print(
            "FAIL: relative-error fraction below target "
            f"tol={args.rel_tol:g}, required_fraction={args.required_fraction:g}",
            file=sys.stderr,
        )
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
