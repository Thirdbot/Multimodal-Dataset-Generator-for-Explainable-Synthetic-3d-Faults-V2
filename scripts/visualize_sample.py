"""Inspect Synthoseis build arrays in 3D with PyVista.

The script manually reads Zarr v3 chunk metadata, reconstructs a NumPy volume,
and displays it as either a volume or label-like mesh. It can optionally overlay
one highlighted array such as a fault or closure mask.
"""

import argparse
import json
from math import ceil
from pathlib import Path

import numpy as np
from numcodecs import blosc


ROOT = Path(__file__).resolve().parent.parent
BUILDS_ROOT = ROOT / "builds"


def _build_path_from_arg(sample):
    """Resolve a build folder argument or use the latest build folder."""
    if sample:
        build_path = Path(sample)
        if not build_path.is_absolute():
            build_path = ROOT / build_path
        return build_path

    samples = sorted(
        path for path in BUILDS_ROOT.iterdir()
        if path.is_dir() and path.name.startswith("seismic__")
    )
    if not samples:
        raise FileNotFoundError(f"no sample folder found in {BUILDS_ROOT}")
    return samples[-1]


def _list_zarr_groups(sample_path):
    """Find all .zarr groups under one generated sample folder."""
    return sorted(
        path for path in sample_path.rglob("*.zarr")
        if path.is_dir()
    )


def _read_json(path):
    return json.loads(Path(path).read_text())


def _metadata_for_zarr_group(group_path):
    """Return array metadata for both direct arrays and consolidated groups."""
    root_meta = _read_json(group_path / "zarr.json")
    if root_meta.get("node_type") == "array":
        return {
            "group_path": group_path,
            "array_name": group_path.stem,
            "array_path": group_path,
            "metadata": root_meta,
        }

    consolidated = root_meta.get("consolidated_metadata", {}).get("metadata", {})
    if consolidated:
        array_name, metadata = next(iter(consolidated.items()))
        return {
            "group_path": group_path,
            "array_name": array_name,
            "array_path": group_path / array_name,
            "metadata": metadata,
        }

    raise ValueError(f"could not find array metadata in {group_path}")


def list_arrays(sample_path):
    """List readable Zarr arrays and skip groups without array metadata."""
    arrays = []
    for group_path in _list_zarr_groups(sample_path):
        try:
            item = _metadata_for_zarr_group(group_path)
        except Exception:
            continue
        arrays.append(item)
    return arrays


def _default_array(sample_path):
    """Pick a useful default array for visualization when none is requested."""
    arrays = list_arrays(sample_path)
    preferred = [
        "seismicCubes_RFC_fullstack",
        "seismicCubes_cumsum_fullstack",
        "geology/faulted_lithology.zarr",
        "reservoir_label",
        "sealed_label",
        "trap_label",
        "all_closure_segments",
    ]

    for token in preferred:
        for item in arrays:
            rel = item["group_path"].relative_to(sample_path).as_posix()
            if token in rel:
                return item

    if not arrays:
        raise FileNotFoundError(f"no .zarr arrays found in {sample_path}")
    return arrays[0]


def _resolve_array(sample_path, array_arg):
    """Resolve a user-provided relative Zarr path or fallback default array."""
    arrays = list_arrays(sample_path)
    if not arrays:
        raise FileNotFoundError(f"no .zarr arrays found in {sample_path}")

    if not array_arg:
        return _default_array(sample_path)

    target = array_arg.rstrip("/")
    for item in arrays:
        rel = item["group_path"].relative_to(sample_path).as_posix()
        if rel == target or item["group_path"].name == target:
            return item

    raise FileNotFoundError(f"array {array_arg} not found in {sample_path}")


def _numpy_dtype(zarr_dtype):
    if zarr_dtype == "float32":
        return np.dtype("<f4")
    if zarr_dtype == "float64":
        return np.dtype("<f8")
    if zarr_dtype == "int32":
        return np.dtype("<i4")
    if zarr_dtype == "uint8":
        return np.dtype("u1")
    if zarr_dtype == "int8":
        return np.dtype("i1")
    raise ValueError(f"unsupported zarr dtype: {zarr_dtype}")


def _decode_fill_value(fill_value, dtype):
    if fill_value == "NaN":
        return np.nan
    if fill_value is None:
        return 0
    return dtype.type(fill_value)


def load_array(item):
    """Decode a Zarr v3 array into an in-memory NumPy array."""
    metadata = item["metadata"]
    shape = tuple(metadata["shape"])
    chunk_shape = tuple(metadata["chunk_grid"]["configuration"]["chunk_shape"])
    dtype = _numpy_dtype(metadata["data_type"])
    fill_value = _decode_fill_value(metadata.get("fill_value"), dtype)
    array = np.full(shape, fill_value, dtype=dtype)

    chunk_counts = tuple(ceil(size / chunk) for size, chunk in zip(shape, chunk_shape))
    chunk_root = item["array_path"] / "c"

    for i in range(chunk_counts[0]):
        for j in range(chunk_counts[1]):
            for k in range(chunk_counts[2]):
                chunk_path = chunk_root / str(i) / str(j) / str(k)
                if not chunk_path.exists():
                    continue

                raw = chunk_path.read_bytes()
                chunk = np.frombuffer(blosc.decompress(raw), dtype=dtype)

                i0 = i * chunk_shape[0]
                j0 = j * chunk_shape[1]
                k0 = k * chunk_shape[2]
                i1 = min(i0 + chunk_shape[0], shape[0])
                j1 = min(j0 + chunk_shape[1], shape[1])
                k1 = min(k0 + chunk_shape[2], shape[2])

                subshape = (i1 - i0, j1 - j0, k1 - k0)
                chunk_full = chunk.reshape(chunk_shape, order="C")
                array[i0:i1, j0:j1, k0:k1] = chunk_full[:subshape[0], :subshape[1], :subshape[2]]

    return array


def _build_grid(array, spacing):
    """Wrap a NumPy array as a PyVista ImageData grid."""
    import pyvista as pv

    grid = pv.ImageData()
    grid.dimensions = array.shape
    grid.spacing = spacing
    grid.origin = (0.0, 0.0, 0.0)
    grid.point_data["values"] = np.ascontiguousarray(array).ravel(order="F")
    return grid


def _looks_like_label(item, array):
    """Heuristic for deciding whether an array should render as labels/mesh."""
    name = item["group_path"].name.lower()
    if "label" in name or "closure" in name or "trap" in name or "reservoir" in name or "sealed" in name:
        return True
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return False
    unique_count = np.unique(finite).size
    return unique_count <= 8


def _visualize(item, array, spacing, cmap, opacity, threshold, show_bounds):
    """Render one array as a volume or label-thresholded mesh."""
    import pyvista as pv

    grid = _build_grid(array, spacing)
    plotter = pv.Plotter()
    plotter.add_axes()
    if show_bounds:
        plotter.show_bounds(grid="front", location="outer")

    if _looks_like_label(item, array):
        label_threshold = threshold if threshold is not None else 0.5
        mesh = grid.threshold(value=label_threshold, scalars="values")
        plotter.add_mesh(mesh, cmap=cmap, opacity=opacity, show_scalar_bar=True)
    else:
        volume_opacity = opacity
        if isinstance(opacity, float):
            volume_opacity = "sigmoid"
        plotter.add_volume(grid, scalars="values", cmap=cmap, opacity=volume_opacity, shade=False)

    title = item["group_path"].name
    plotter.add_text(title, font_size=10)
    plotter.show()


def _add_highlight(plotter, sample_path, highlight_arg, spacing, cmap, opacity, threshold):
    """Overlay a second array on an existing PyVista plotter."""
    highlight_item = _resolve_array(sample_path, highlight_arg)
    highlight_array = load_array(highlight_item)
    highlight_grid = _build_grid(highlight_array, spacing)

    if _looks_like_label(highlight_item, highlight_array):
        highlight_threshold = threshold if threshold is not None else 0.5
        mesh = highlight_grid.threshold(value=highlight_threshold, scalars="values")
        plotter.add_mesh(
            mesh,
            scalars="values",
            cmap=cmap,
            opacity=opacity,
            show_scalar_bar=True,
            categories=True,
        )
    else:
        finite = highlight_array[np.isfinite(highlight_array)]
        if finite.size == 0:
            return
        contour_values = np.linspace(finite.min(), finite.max(), 6)[1:-1]
        mesh = highlight_grid.contour(isosurfaces=contour_values, scalars="values")
        plotter.add_mesh(
            mesh,
            scalars="values",
            cmap=cmap,
            opacity=opacity,
            show_scalar_bar=True,
        )


def _visualize_with_highlight(
    item,
    array,
    sample_path,
    highlight,
    spacing,
    cmap,
    opacity,
    threshold,
    highlight_cmap,
    highlight_opacity,
    highlight_threshold,
    show_bounds,
):
    """Render a base array and a highlighted companion array together."""
    import pyvista as pv

    grid = _build_grid(array, spacing)
    plotter = pv.Plotter()
    plotter.add_axes()
    if show_bounds:
        plotter.show_bounds(grid="front", location="outer")

    if _looks_like_label(item, array):
        label_threshold = threshold if threshold is not None else 0.5
        mesh = grid.threshold(value=label_threshold, scalars="values")
        plotter.add_mesh(mesh, cmap=cmap, opacity=opacity, show_scalar_bar=True)
    else:
        volume_opacity = opacity
        if isinstance(opacity, float):
            volume_opacity = "sigmoid"
        plotter.add_volume(grid, scalars="values", cmap=cmap, opacity=volume_opacity, shade=False)

    _add_highlight(
        plotter=plotter,
        sample_path=sample_path,
        highlight_arg=highlight,
        spacing=spacing,
        cmap=highlight_cmap,
        opacity=highlight_opacity,
        threshold=highlight_threshold if highlight_threshold is not None else threshold,
    )

    title = f"{item['group_path'].name} + {Path(highlight).name}"
    plotter.add_text(title, font_size=10)
    plotter.show()


def main():
    parser = argparse.ArgumentParser(description="Display one Synthoseis sample array in 3D with PyVista.")
    parser.add_argument("sample", nargs="?", default=None, help="sample folder path under builds/")
    parser.add_argument("--array", default=None, help="relative .zarr path inside sample folder")
    parser.add_argument("--spacing", nargs=3, type=float, default=(1.0, 1.0, 1.0), metavar=("DX", "DY", "DZ"))
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--opacity", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=None, help="threshold for label-like arrays")
    parser.add_argument("--highlight", default=None, help="relative .zarr path to highlight in color over the base volume")
    parser.add_argument("--highlight-cmap", default="autumn")
    parser.add_argument("--highlight-opacity", type=float, default=0.65)
    parser.add_argument("--highlight-threshold", type=float, default=None)
    parser.add_argument("--show-bounds", action="store_true")
    args = parser.parse_args()

    sample_path = _build_path_from_arg(args.sample)
    item = _resolve_array(sample_path, args.array)
    array = load_array(item)

    if args.highlight:
        _visualize_with_highlight(
            item=item,
            array=array,
            sample_path=sample_path,
            highlight=args.highlight,
            spacing=tuple(args.spacing),
            cmap=args.cmap,
            opacity=args.opacity,
            threshold=args.threshold,
            highlight_cmap=args.highlight_cmap,
            highlight_opacity=args.highlight_opacity,
            highlight_threshold=args.highlight_threshold,
            show_bounds=args.show_bounds,
        )
        return

    _visualize(
        item=item,
        array=array,
        spacing=tuple(args.spacing),
        cmap=args.cmap,
        opacity=args.opacity,
        threshold=args.threshold,
        show_bounds=args.show_bounds,
    )


if __name__ == "__main__":
    main()
