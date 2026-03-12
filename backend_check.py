#!/usr/bin/env python3
import importlib.machinery
import importlib.util
import os


def _is_extension_binary(path: str) -> bool:
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    return path.endswith(suffixes) or path.endswith(".dylib") or ".so." in path


def find_potpourri_binary() -> str | None:
    for candidate in ("potpourri3d.potpourri3d_bindings", "potpourri3d_bindings"):
        try:
            spec = importlib.util.find_spec(candidate)
        except ModuleNotFoundError:
            continue

        if spec and getattr(spec, "origin", None) and _is_extension_binary(spec.origin):
            return spec.origin

    spec = importlib.util.find_spec("potpourri3d")
    if spec and spec.submodule_search_locations:
        pkg_dir = os.fspath(list(spec.submodule_search_locations)[0])
        for name in sorted(os.listdir(pkg_dir)):
            if "potpourri3d_bindings" in name and _is_extension_binary(name):
                return os.path.join(pkg_dir, name)

    return None


def verify_suitesparse(*, verbose: bool = False) -> bool:
    """
    Heuristic check only.
    Returns True if the compiled extension binary contains likely
    SuiteSparse-related symbol/name strings.
    """
    binary = find_potpourri_binary()
    if not binary or not os.path.exists(binary):
        if verbose:
            print(f"SuiteSparse check FAIL: binary not found (resolved={binary})")
        return False

    needles = (
        b"cholmod",
        b"CHOLMOD",
        b"suitesparse",
        b"SuiteSparse",
        b"umfpack",
        b"UMFPACK",
        b"spqr",
        b"SPQR",
        b"colamd",
        b"camd",
        b"ccolamd",
        b"amd",
    )

    try:
        with open(binary, "rb") as f:
            blob = f.read()
    except OSError:
        if verbose:
            print(f"SuiteSparse check FAIL: unable to read binary ({binary})")
        return False

    hits = sum(1 for needle in needles if needle in blob)
    ok = hits >= 2
    if verbose:
        binary_desc = binary if binary else "unresolved"
        status = "PASS" if ok else "FAIL"
        print(f"SuiteSparse check {status}: binary={binary_desc} matched_needles={hits}")
    return ok
