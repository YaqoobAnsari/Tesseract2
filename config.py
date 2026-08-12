"""
config.py - Single source of truth for repository paths.

Every path the pipeline touches is defined here, derived from this file's
own location - nothing depends on the current working directory or on an
absolute install path. Import constants from here instead of joining path
strings locally:

    from config import INPUT_IMAGES_DIR, RESULTS_DIR

Directory layout (see README):
    data/floorplans/          input floorplan images
    data/mappings/            inter-floor transition mappings (manual + AUTO_*)
    Model_weights/            model checkpoints (not tracked)
    results/single_floor/     per-image pipeline outputs (Json / Plots / Time&Meta)
    results/multi_floor/      merged building graphs (Jsons / Plots / Time&Meta)
    results/evaluation/       outputs of the evaluation/ suite
"""

import os
import sys

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Inputs
DATA_DIR = os.path.join(BASE_PATH, "data")
INPUT_IMAGES_DIR = os.path.join(DATA_DIR, "floorplans")
MAPPINGS_DIR = os.path.join(DATA_DIR, "mappings")

# Models
MODELS_DIR = os.path.join(BASE_PATH, "Models")
MODEL_WEIGHTS_DIR = os.path.join(BASE_PATH, "Model_weights")

# Outputs
RESULTS_BASE_DIR = os.path.join(BASE_PATH, "results")
RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, "single_floor")
MULTIFLOOR_RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, "multi_floor")
EVAL_RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, "evaluation")


def ensure_import_paths():
    """Put the repository root, utils/, and the model packages on sys.path
    (idempotent). Call once at the top of any entry point."""
    for p in (
        BASE_PATH,
        os.path.join(BASE_PATH, "utils"),
        os.path.join(MODELS_DIR, "Text_Models"),
        os.path.join(MODELS_DIR, "Interpreter"),
        os.path.join(MODELS_DIR, "Door_Models"),
    ):
        if p not in sys.path:
            sys.path.insert(0, p)


ensure_import_paths()
