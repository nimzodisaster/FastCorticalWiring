#!/usr/bin/env python3
"""I/O helpers for FastCorticalWiring.

This module isolates file-format handling from solver code.
"""

import os
import warnings

import numpy as np

try:
    import nibabel as nib
except Exception:
    nib = None


def _require_nibabel():
    if nib is None:
        raise ImportError("nibabel is required for neuroimaging I/O. Install with: pip install nibabel")


def infer_output_basename(surface_path, fallback="surface"):
    """Infer a stable output basename from a surface filename."""
    if not surface_path:
        return fallback
    name = os.path.basename(surface_path)
    for suffix in (".surf.gii", ".shape.gii", ".func.gii", ".gii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    root, _ = os.path.splitext(name)
    return root if root else fallback


def _normalize_region_name(name):
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="ignore")
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _read_freesurfer_label_mask(label_path, n_vertices):
    _require_nibabel()
    cortex_mask = np.zeros(int(n_vertices), dtype=bool)
    try:
        cortex_idx = nib.freesurfer.read_label(label_path)
        cortex_idx = np.asarray(cortex_idx, dtype=np.int64)
    except Exception:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) < 2:
            raise ValueError(f"Malformed label file: {label_path}")
        n_vertices_in_label = int(lines[1].strip())
        cortex_idx = []
        for i in range(2, 2 + n_vertices_in_label):
            if i >= len(lines):
                break
            fields = lines[i].split()
            if not fields:
                continue
            cortex_idx.append(int(fields[0]))
        cortex_idx = np.asarray(cortex_idx, dtype=np.int64)

    valid = (cortex_idx >= 0) & (cortex_idx < cortex_mask.size)
    cortex_mask[cortex_idx[valid]] = True
    return cortex_mask


def _read_freesurfer_annot_mask(annot_path, n_vertices):
    _require_nibabel()
    labels, ctab, names = nib.freesurfer.read_annot(annot_path, orig_ids=True)
    labels = np.asarray(labels)
    if labels.shape[0] != int(n_vertices):
        raise ValueError(
            f"Annotation vertex count mismatch: annot has {labels.shape[0]}, mesh has {n_vertices}"
        )

    if ctab.ndim == 2 and ctab.shape[1] >= 5 and ctab.shape[0] == len(names):
        codes = ctab[:, 4]
    else:
        codes = np.arange(len(names), dtype=np.int32)

    name_to_code = {}
    for i, raw_name in enumerate(names):
        name_to_code[_normalize_region_name(raw_name)] = int(codes[i])

    exclude_tokens = ("unknown", "corpuscallosum", "medialwall")
    exclude_codes = {
        code for norm_name, code in name_to_code.items() if any(tok in norm_name for tok in exclude_tokens)
    }

    cortex_mask = np.ones(int(n_vertices), dtype=bool)
    for code in exclude_codes:
        cortex_mask[labels == code] = False
    cortex_mask[labels == -1] = False
    return cortex_mask


def _load_freesurfer_surface(surface_path):
    _require_nibabel()
    if not os.path.exists(surface_path):
        raise FileNotFoundError(f"Surface file not found: {surface_path}")
    vertices, faces = nib.freesurfer.read_geometry(surface_path)
    return vertices.astype(np.float64), faces.astype(np.int32)


def _load_fslr_surface(surface_path):
    _require_nibabel()
    if not os.path.exists(surface_path):
        raise FileNotFoundError(f"Surface file not found: {surface_path}")

    img = nib.load(surface_path)
    if not isinstance(img, nib.gifti.GiftiImage):
        raise ValueError(f"Expected GIFTI surface file, got: {surface_path}")

    pointset_code = nib.nifti1.intent_codes["NIFTI_INTENT_POINTSET"]
    triangle_code = nib.nifti1.intent_codes["NIFTI_INTENT_TRIANGLE"]

    pointsets = [da for da in img.darrays if int(da.intent) == int(pointset_code)]
    triangles = [da for da in img.darrays if int(da.intent) == int(triangle_code)]
    if len(pointsets) != 1 or len(triangles) != 1:
        raise ValueError(
            f"Malformed GIFTI surface '{surface_path}': expected exactly one POINTSET and one TRIANGLE array"
        )

    vertices = np.asarray(pointsets[0].data, dtype=np.float64)
    faces = np.asarray(triangles[0].data, dtype=np.int32)
    return vertices, faces


def _load_fslr_mask(mask_path, n_vertices):
    _require_nibabel()
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    img = nib.load(mask_path)
    if not isinstance(img, nib.gifti.GiftiImage):
        raise ValueError(f"Expected GIFTI mask file, got: {mask_path}")
    if len(img.darrays) == 0:
        raise ValueError(f"GIFTI mask contains no data arrays: {mask_path}")

    data = np.asarray(img.darrays[0].data)
    data = np.squeeze(data)
    if data.ndim != 1:
        raise ValueError(f"Mask must be a 1D scalar GIFTI array, got shape {tuple(data.shape)}: {mask_path}")
    if data.shape[0] != int(n_vertices):
        raise ValueError(
            f"Mask vertex count mismatch: mask has {data.shape[0]}, mesh has {n_vertices} ({mask_path})"
        )
    return data > 0


def load_surface_and_mask(
    *,
    standard,
    surface_path=None,
    mask_path=None,
    subject_dir=None,
    subject_id=None,
    hemi="lh",
    surf_type="pial",
    custom_label=None,
    no_mask=False,
):
    """Load mesh + cortex mask for FreeSurfer or fsLR workflows."""
    standard = str(standard).lower()
    metadata = {
        "standard": standard,
        "subject_dir": subject_dir,
        "subject_id": subject_id,
        "hemi": hemi,
        "surf_type": surf_type,
        "custom_label": custom_label,
    }

    if standard == "freesurfer":
        legacy_mode = False
        if surface_path is not None:
            surf_dir = os.path.dirname(os.path.abspath(surface_path))
            if os.path.basename(surf_dir) == "surf":
                subject_root = os.path.dirname(surf_dir)
                inferred_subject_id = os.path.basename(subject_root)
                inferred_subject_dir = os.path.dirname(subject_root)
                if subject_id is None:
                    subject_id = inferred_subject_id
                if subject_dir is None:
                    subject_dir = inferred_subject_dir

                metadata["subject_dir"] = subject_dir
                metadata["subject_id"] = subject_id

        if surface_path is None:
            if subject_dir is None or subject_id is None:
                raise ValueError("FreeSurfer loading requires either --surface or both subject_dir and subject_id.")
            legacy_mode = True
            surface_path = os.path.join(subject_dir, subject_id, "surf", f"{hemi}.{surf_type}")

        vertices, faces = _load_freesurfer_surface(surface_path)
        n_vertices = vertices.shape[0]
        metadata["surface_path"] = surface_path
        metadata["legacy_mode"] = legacy_mode
        metadata["output_basename"] = f"{subject_id}_{hemi}_{surf_type}" if subject_id else infer_output_basename(surface_path)

        if no_mask:
            cortex_mask = np.ones(n_vertices, dtype=bool)
            metadata["mask_source"] = "all_vertices_no_mask"
        elif mask_path:
            mask_ext = os.path.splitext(mask_path)[1].lower()
            if mask_ext == ".label":
                cortex_mask = _read_freesurfer_label_mask(mask_path, n_vertices)
                metadata["mask_source"] = f"explicit_label:{mask_path}"
            elif mask_ext == ".annot":
                cortex_mask = _read_freesurfer_annot_mask(mask_path, n_vertices)
                metadata["mask_source"] = f"explicit_annot:{mask_path}"
            else:
                raise ValueError(
                    f"Unsupported FreeSurfer mask file extension '{mask_ext}'. Use .label or .annot."
                )
        else:
            label_dir = None
            if subject_dir and subject_id:
                label_dir = os.path.join(subject_dir, subject_id, "label")
            elif surface_path:
                maybe_label = os.path.join(os.path.dirname(os.path.dirname(surface_path)), "label")
                if os.path.isdir(maybe_label):
                    label_dir = maybe_label

            cortex_mask = None
            if label_dir:
                if custom_label:
                    candidate = os.path.join(label_dir, f"{hemi}.{custom_label}.label")
                    if os.path.exists(candidate):
                        cortex_mask = _read_freesurfer_label_mask(candidate, n_vertices)
                        metadata["mask_source"] = f"custom_label:{candidate}"
                if cortex_mask is None:
                    candidate = os.path.join(label_dir, f"{hemi}.cortex.label")
                    if os.path.exists(candidate):
                        cortex_mask = _read_freesurfer_label_mask(candidate, n_vertices)
                        metadata["mask_source"] = f"default_label:{candidate}"
                if cortex_mask is None:
                    candidate = os.path.join(label_dir, f"{hemi}.aparc.annot")
                    if os.path.exists(candidate):
                        warnings.warn("Cortex label not found; using aparc annotation fallback.", RuntimeWarning)
                        cortex_mask = _read_freesurfer_annot_mask(candidate, n_vertices)
                        metadata["mask_source"] = f"aparc_fallback:{candidate}"

            if cortex_mask is None:
                warnings.warn("No cortex mask found; falling back to all vertices.", RuntimeWarning)
                cortex_mask = np.ones(n_vertices, dtype=bool)
                metadata["mask_source"] = "all_vertices_fallback"

        if subject_dir and subject_id:
            mgh_template = os.path.join(subject_dir, subject_id, "mri", "orig.mgz")
            if not os.path.exists(mgh_template):
                mgh_template = os.path.join(subject_dir, subject_id, "mri", "T1.mgz")
            if os.path.exists(mgh_template):
                metadata["mgh_template_path"] = mgh_template

        return vertices, faces, cortex_mask.astype(bool), metadata

    if standard == "fslr":
        if surface_path is None:
            raise ValueError("fsLR loading requires --surface pointing to a .surf.gii file.")
        vertices, faces = _load_fslr_surface(surface_path)
        n_vertices = vertices.shape[0]
        metadata["surface_path"] = surface_path
        metadata["legacy_mode"] = False
        metadata["output_basename"] = infer_output_basename(surface_path)

        if no_mask:
            cortex_mask = np.ones(n_vertices, dtype=bool)
            metadata["mask_source"] = "all_vertices_no_mask"
        else:
            if not mask_path:
                raise ValueError("fsLR mode requires --mask (GIFTI scalar) or --no-mask.")
            cortex_mask = _load_fslr_mask(mask_path, n_vertices)
            metadata["mask_source"] = f"gifti_mask:{mask_path}"
        return vertices, faces, cortex_mask.astype(bool), metadata

    raise ValueError(f"Unsupported standard '{standard}'. Expected 'freesurfer' or 'fslr'.")


def save_results_csv(output_dir, csv_filename, cortex_mask, msd, radius_function, perimeter_function):
    """Write tabular results CSV."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, csv_filename)

    n_vertices = int(len(cortex_mask))
    if len(msd) != n_vertices or len(radius_function) != n_vertices or len(perimeter_function) != n_vertices:
        raise ValueError("All metric arrays must match cortex_mask length for CSV output.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("vertex_id,is_cortex,msd,radius_function,perimeter_function\n")
        for i in range(n_vertices):
            f.write(
                f"{i},{int(bool(cortex_mask[i]))},{float(msd[i])},{float(radius_function[i])},{float(perimeter_function[i])}\n"
            )
    return path


def save_results_mgh(output_dir, filename_template, metrics, n_vertices, template_mgz_path=None):
    """Write scalar overlays in FreeSurfer MGH format."""
    _require_nibabel()
    os.makedirs(output_dir, exist_ok=True)

    if template_mgz_path and os.path.exists(template_mgz_path):
        template_img = nib.load(template_mgz_path)
        template_affine = template_img.affine
        template_header = template_img.header
    else:
        template_affine = np.array(
            [[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        template_header = nib.MGHHeader()
        template_header.set_data_shape([int(n_vertices), 1, 1])

    written = []
    for metric_name, data_array in metrics.items():
        arr = np.asarray(data_array)
        if arr.shape[0] != int(n_vertices):
            raise ValueError(
                f"Metric '{metric_name}' length mismatch for MGH output: {arr.shape[0]} vs {n_vertices}"
            )
        if not np.any(np.isfinite(arr)):
            continue

        payload = arr.astype(np.float32).reshape((int(n_vertices), 1, 1))
        payload[~np.isfinite(payload)] = 0

        header = template_header.copy()
        header.set_data_shape(payload.shape)
        header.set_data_dtype(np.float32)

        out_path = os.path.join(output_dir, filename_template.format(metric=metric_name))
        nib.save(nib.MGHImage(payload, template_affine, header), out_path)
        written.append(out_path)

    return written


def save_results_gifti(output_dir, filename_template, metrics, n_vertices):
    """Write scalar overlays in GIFTI format (one file per metric)."""
    _require_nibabel()
    os.makedirs(output_dir, exist_ok=True)
    shape_intent = nib.nifti1.intent_codes["NIFTI_INTENT_SHAPE"]

    written = []
    for metric_name, data_array in metrics.items():
        arr = np.asarray(data_array, dtype=np.float32)
        if arr.shape[0] != int(n_vertices):
            raise ValueError(
                f"Metric '{metric_name}' length mismatch for GIFTI output: {arr.shape[0]} vs {n_vertices}"
            )
        if not np.any(np.isfinite(arr)):
            continue

        da = nib.gifti.GiftiDataArray(data=arr, intent=int(shape_intent))
        da.meta = nib.gifti.GiftiMetaData.from_dict({"Name": metric_name})
        img = nib.gifti.GiftiImage(darrays=[da])

        out_path = os.path.join(output_dir, filename_template.format(metric=metric_name))
        nib.save(img, out_path)
        written.append(out_path)

    return written

def farthest_point_sampling(vertices, k, valid_mask):
    """Spatially uniform sampling using Euclidean Farthest Point Sampling."""
    valid_indices = np.where(valid_mask)[0]
    if k >= len(valid_indices):
        return valid_indices.tolist()

    sampled_indices = np.zeros(k, dtype=int)
    distances = np.full(vertices.shape[0], np.inf)
    
    farthest = valid_indices[0]
    
    for i in range(k):
        sampled_indices[i] = farthest
        dist = np.sum((vertices - vertices[farthest]) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        
        masked_distances = distances.copy()
        masked_distances[~valid_mask] = -1.0
        farthest = np.argmax(masked_distances)
        
    return sampled_indices.tolist()
