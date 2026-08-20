# benchmark/scripts/prepare_codenet.py

import os
import random
import shutil
import tarfile
import argparse
from pathlib import Path
from tqdm import tqdm
from ..config import SEED

DATASETS = {
    "python":   "Project_CodeNet_Python800",
    "java": "Project_CodeNet_Java250",
    "cpp":  "Project_CodeNet_C++1000",
}

MAIN_ARCHIVE    = "Project_CodeNet.tar.gz"
BENCHMARKS_PATH = "Project_CodeNet/derived/benchmarks"

N_PROBLEMS    = 250
N_SUBMISSIONS = 300


def extract_main_archive(archive_path: Path, out_dir: Path):
    """Extracts only the dataset archives from derived/benchmarks to the final destination."""
    print(f"[CodeNet] Extracting main archive: {archive_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        members = [
            m for m in tar.getmembers()
            if m.name.startswith(BENCHMARKS_PATH) and m.name.endswith(".tar.gz")
        ]
        print(f"[CodeNet] Found {len(members)} dataset archives.")
        tar.extractall(path=out_dir, members=members)


def extract_dataset_archive(archive_path: Path, out_dir: Path):
    """Extracts a single dataset archive directly into out_dir."""
    print(f"[CodeNet] Extracting {archive_path.name}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=out_dir)
    archive_path.unlink()  # rimuovi il tar.gz dopo l'estrazione


def trim_dataset(dataset_dir: Path, seed: int):
    """Removes excess problems and submissions directly in dataset_dir."""
    rng      = random.Random(seed)
    problems = [p for p in dataset_dir.iterdir() if p.is_dir()]

    # Remove excess problems
    if len(problems) > N_PROBLEMS:
        to_keep   = set(p.name for p in rng.sample(problems, N_PROBLEMS))
        to_remove = [p for p in problems if p.name not in to_keep]
        print(f"[CodeNet] Removing {len(to_remove)} excess problems from {dataset_dir.name}...")
        for problem_dir in tqdm(to_remove, desc="Removing problems"):
            shutil.rmtree(problem_dir)
        problems = [p for p in dataset_dir.iterdir() if p.is_dir()]

    # Remove excess submissions for each problem
    for problem_dir in tqdm(problems, desc=f"Trimming {dataset_dir.name}"):
        submissions = [s for s in problem_dir.iterdir() if s.is_file()]
        if len(submissions) > N_SUBMISSIONS:
            to_keep   = set(s.name for s in rng.sample(submissions, N_SUBMISSIONS))
            to_remove = [s for s in submissions if s.name not in to_keep]
            for submission in to_remove:
                submission.unlink()


def main():
    parser = argparse.ArgumentParser(description="Prepare CodeNet dataset.")
    parser.add_argument(
        "--archive",
        required=True,
        help="Path to Project_CodeNet.tar.gz"
    )
    args = parser.parse_args()

    data_dir = Path(os.environ["BENCHMARK_DATA_DIR"]) / "data" / "codenet"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract dataset archives from the main archive
    extract_main_archive(Path(args.archive), data_dir)

    # Step 2: Extract and trim each dataset
    benchmarks_dir = data_dir / BENCHMARKS_PATH
    for lang, dataset_name in DATASETS.items():
        archive_path = benchmarks_dir / f"{dataset_name}.tar.gz"
        if not archive_path.exists():
            print(f"[CodeNet] WARN: {archive_path} non trovato, skip.")
            continue

        dataset_dir = data_dir / dataset_name
        if dataset_dir.exists():
            print(f"[CodeNet] {dataset_name} già presente, skip.")
            continue

        extract_dataset_archive(archive_path, data_dir)
        trim_dataset(dataset_dir, seed=SEED)

    # Remove now-empty benchmarks directory
    shutil.rmtree(data_dir / "Project_CodeNet", ignore_errors=True)

    print(f"[CodeNet] Datasets ready in {data_dir}")


if __name__ == "__main__":
    main()

"""
Il flusso ora è:
Project_CodeNet.tar.gz
    → estrai *.tar.gz direttamente in $BENCHMARK_DATA_DIR/data/
    → per ogni dataset: estrai in-place, rimuovi problemi in eccesso, rimuovi submission in eccesso
    → rimuovi directory Project_CodeNet/ residua
"""