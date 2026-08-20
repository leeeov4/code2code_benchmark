# benchmark/config.py

from pathlib import Path
import os


if not os.getenv("BENCHMARK_DATA_DIR"):
    raise ValueError(f"Set BENCHMARK_DATA_DIR.")

BASE_DIR = Path(os.getenv("BENCHMARK_DATA_DIR"))

DATA_PATH = {
    "codenet":       BASE_DIR / "data" / "codenet",
    "multiple":      BASE_DIR / "data" / "multiple",
    "xcodeeval":     BASE_DIR / "data" / "xcodeeval",
    "bigclonebench": BASE_DIR / "data" / "bigclonebench",
}

PROCESSED_PATH = {
    "codenet":       BASE_DIR / "processed" / "codenet",
    "multiple":      BASE_DIR / "processed" / "multiple",
    "xcodeeval":     BASE_DIR / "processed" / "xcodeeval",
    "bigclonebench": BASE_DIR / "processed" / "bigclonebench",
}

OUTPUT_PATH =  BASE_DIR / "output"

SEED        = 442