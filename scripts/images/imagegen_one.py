"""Generate the 2D view-images + per-view property graphs for one property graph, in an isolated
subprocess.

Invoked by scripts/watcher/process.py `_run_image_gen` (via `python -m scripts.images.imagegen_one
<graph>`) so the GIL-bound matplotlib/numpy work parallelizes across cores (IMAGE_GEN_CONCURRENCY
subprocesses) instead of stalling the watcher's event loop.
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")  # non-interactive / headless-safe

import sys
from pathlib import Path

from scripts.images.images_generator import generate_images_for_graph
from scripts.graph.properties_2d_graph import main as generate_properties_2d_graphs

if __name__ == "__main__":
    graph_path = sys.argv[1]
    sample_id = Path(graph_path).stem.removesuffix("_db_extract_properties_graph")
    generate_images_for_graph(graph_path)
    generate_properties_2d_graphs({sample_id})
