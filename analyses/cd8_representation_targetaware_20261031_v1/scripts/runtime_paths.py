"""Explicit data roots and interpreters for a relocated Linux reproduction.

Historical hashes refer to the original execution; this is a publication adapter.
No directory is created and no data are read by importing this module.
"""
import os
import sys
from pathlib import Path

SOURCE_ROOT = (
    Path(os.environ.get("MIF_SOURCE_ROOT", str(Path(__file__).resolve().parents[3])))
    .expanduser()
    .resolve()
)
ROOT = Path(os.environ.get("MIF_PROJECT_ROOT", str(SOURCE_ROOT))).expanduser().resolve()
RUN = (
    Path(os.environ.get("MIF_RUN_ROOT", str(Path(__file__).resolve().parents[1])))
    .expanduser()
    .resolve()
)
FEATURE_PYTHON = Path(os.environ.get("MIF_FEATURE_PYTHON", sys.executable))
NEURAL_PYTHON = Path(os.environ.get("MIF_NEURAL_PYTHON", sys.executable))
