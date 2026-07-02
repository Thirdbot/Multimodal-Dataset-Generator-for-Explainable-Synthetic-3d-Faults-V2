"""Run the downstream dataset pipeline once.

This starts after properties graphs already exist. It creates 2D image evidence,
2D properties graphs, verified QA rows, and the final multimodal CSV.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.common.logger_color import logger
from scripts.images.images_generator import generate_images_for_all
from scripts.graph.properties_2d_graph import main as generate_properties_2d_graphs
from Verifier.generator_pipeline import generate_multimodal_dataset
from Dataset.DatasetMaker import main as build_dataset_csv


def main():
    """Run all non-watcher downstream stages once."""
    logger.info("[RUN ONCE] -> image generation")
    generated_images = generate_images_for_all()
    logger.info(f"[DONE] -> image generation: {len(generated_images or [])}")

    logger.info("[RUN ONCE] -> 2D properties graph generation")
    generate_properties_2d_graphs()

    logger.info("[RUN ONCE] -> verified QA generation")
    rows = generate_multimodal_dataset()
    logger.info(f"[DONE] -> verified QA rows: {len(rows or [])}")

    logger.info("[RUN ONCE] -> dataset CSV build")
    build_dataset_csv()

    logger.info("[DONE] -> one-shot downstream pipeline")


if __name__ == "__main__":
    main()
