"""Mask -> geometric / geological attribute computation.

Single home for every attribute derived from an object's binary MASK (+ pixel resolution).
Keeping it here means future attributes have an organised place to land instead of being
scattered across the graph-extract and image-generation layers.

Scope boundary: this file owns **mask geometry only**. DB-sourced parameters (fault `throw`,
`fluid`, model counts, `fault_mode`, intersections) stay in the graph-extract layer and are
filtered by config there as usual -- they are handed to us by the generator, not measured
from pixels.

Currently computed:
  - area_pct           -- coverage of the section (closure / salt / onlap)
  - dip_deg            -- fault apparent dip (RANSAC-isolated trace, PCA angle)
  - bbox_from_mask     -- x_min / y_min / x_max / y_max
  - center_from_bbox   -- bbox midpoint (how the scene stores object centres)
  - centroid_from_mask -- true pixel centroid

Future (all computable from the same masks + Δx,Δz -- see the attribute table):
  perimeter, principal orientation, aspect ratio, compactness, convexity,
  fault length / curvature (skeleton), horizon dip / curvature.
"""
from pathlib import Path

import numpy as np
from PIL import Image

# Which mask-computed feature keys get copied onto a graph node.
MASK_FEATURE_KEYS = ("dip_deg", "area_pct")

# Dip estimation (RANSAC + inlier gate). PCA alone is a moment fit, so a contaminated fault
# mask (a crossing structure, a stray blob) drags the major axis flat and produces a
# geologically absurd near-horizontal dip. RANSAC finds the dominant collinear pixel set and
# the dip is fit on THOSE inliers; masks whose dominant line covers less than
# _DIP_MIN_INLIER_FRAC of the pixels are too contaminated to trust and get no dip at all.
_DIP_MIN_PIXELS = 8
_DIP_MIN_INLIER_FRAC = 0.5      # dominant line must contain the majority of the mask's pixels
_DIP_RANSAC_THRESHOLD = 1.5     # px perpendicular distance for a pixel to count as on-line
_DIP_RANSAC_ITERS = 300
_DIP_RANSAC_SEED = 0            # fixed -> graph generation stays reproducible


# --- Group 1: single-object mask geometry -------------------------------------------------

def bbox_from_mask(mask):
    """x_min / y_min / x_max / y_max of the mask's true pixels, or None if empty."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return {
        "x_min": int(x_min),
        "y_min": int(y_min),
        "x_max": int(x_max),
        "y_max": int(y_max),
    }


def center_from_bbox(bbox):
    """Object centre as the bbox midpoint (matches how the scene stores centres)."""
    if not bbox:
        return None
    return {"x": (bbox["x_min"] + bbox["x_max"]) / 2, "y": (bbox["y_min"] + bbox["y_max"]) / 2}


def centroid_from_mask(mask):
    """True pixel centroid (mean of the mask coordinates), or None if empty."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    y, x = coords.mean(axis=0)
    return {"x": float(x), "y": float(y)}


def area_pct(mask):
    """Coverage of the whole section as a percentage of all pixels (1 dp)."""
    return round(100.0 * float(mask.sum()) / mask.size, 1)


# --- Group 3: line orientation (apparent dip) ---------------------------------------------

def ransac_inliers(pts):
    """Largest set of pixels collinear within _DIP_RANSAC_THRESHOLD px of a line through two
    sampled points. Deterministic (fixed seed) so graph generation stays reproducible."""
    n = len(pts)
    rng = np.random.default_rng(_DIP_RANSAC_SEED)
    best_inliers, best_count = None, -1
    for _ in range(_DIP_RANSAC_ITERS):
        i, j = rng.choice(n, 2, replace=False)
        d = pts[j] - pts[i]
        norm = float(np.hypot(d[0], d[1]))
        if norm < 1e-6:
            continue
        d = d / norm
        rel = pts - pts[i]
        perp = np.abs(rel[:, 0] * (-d[1]) + rel[:, 1] * d[0])   # perpendicular distance to line
        inliers = perp < _DIP_RANSAC_THRESHOLD
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers
    return best_inliers


def dip_degrees(mask):
    """Apparent dip of the fault trace = angle of its dominant line from horizontal
    (0 = flat, 90 = vertical). Image y runs downward, so magnitude only.

    RANSAC-then-PCA: RANSAC isolates the dominant collinear pixels (rejecting a crossing
    structure or stray blob that would flatten a plain PCA fit), then the angle is fit by
    PCA on those inliers. A mask whose dominant line covers less than _DIP_MIN_INLIER_FRAC
    of the pixels is too contaminated to assign a dip and returns None. On a clean single-
    fault mask every pixel is an inlier, so this reduces exactly to the old PCA angle."""
    ys, xs = np.nonzero(mask)
    if xs.size < _DIP_MIN_PIXELS:
        return None
    pts = np.column_stack([xs, ys]).astype(float)
    inliers = ransac_inliers(pts)
    if inliers is None:
        return None
    n_in = int(inliers.sum())
    if n_in < _DIP_MIN_PIXELS or n_in / len(pts) < _DIP_MIN_INLIER_FRAC:
        return None  # too contaminated to trust any single line
    p = pts[inliers]
    cov = np.cov(np.vstack([p[:, 0] - p[:, 0].mean(), p[:, 1] - p[:, 1].mean()]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    if eigvals[-1] <= 0 or eigvals[0] / eigvals[-1] > 0.6:
        return None  # inliers too round to have a meaningful dip direction
    major = eigvecs[:, -1]
    return round(float(np.degrees(np.arctan2(abs(major[1]), abs(major[0])))), 1)


# --- Aggregator used by the graph-extract layer -------------------------------------------

def mask_features(mask_path, object_id, object_type):
    """Read visual geometry off the object's scene mask: apparent dip for faults, coverage for
    closures/salt/onlap. Skip the merged per-type mask (object_id == type), whose combined
    geometry is meaningless -- EXCEPT onlap, whose only object IS the aggregate (its coverage
    is the meaningful measure)."""
    if str(object_id) == str(object_type) and object_type != "onlap":
        return {}
    mask_path = Path(mask_path)
    if not mask_path.is_file():
        return {}
    try:
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    except (OSError, ValueError):
        return {}
    if not mask.any():
        return {}

    features = {}
    if object_type in {"closure", "salt", "onlap"}:
        features["area_pct"] = area_pct(mask)
    if object_type == "fault":
        dip = dip_degrees(mask)
        if dip is not None:
            features["dip_deg"] = dip
    return features
