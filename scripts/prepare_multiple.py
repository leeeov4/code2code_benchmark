# benchmark/scripts/prepare_multiple.py

import os
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm


LANGUAGES = ["cpp", "cs", "java", "js"]

HF_BASE = "hf://datasets/bigcode/MultiPL-E-completions/"
HF_PATH = "data/humaneval.{lang}.StarCoder2_15b_16k.0.2.reworded-00000-of-00001.parquet"


def download_language(language: str, out_dir: Path):
    out_path = out_dir / f"{language}.parquet"

    if out_path.exists():
        print(f"[MultiPL-E] {language}.parquet already exists, skipping.")
        return

    remote = HF_BASE + HF_PATH.format(lang=language)
    print(f"[MultiPL-E] Downloading {language}...")
    df = pd.read_parquet(remote)
    df.to_parquet(out_path)
    print(f"[MultiPL-E] Saved {out_path} ({len(df)} rows).")


def main():
    parser = argparse.ArgumentParser(description="Prepare MultiPL-E dataset.")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=LANGUAGES,
        choices=LANGUAGES,
        help="Languages to download (default: all)."
    )
    args = parser.parse_args()

    out_dir = Path(os.environ["BENCHMARK_DATA_DIR"]) / "data" / "multiple"
    out_dir.mkdir(parents=True, exist_ok=True)

    for lang in tqdm(args.languages, desc="Downloading MultiPL-E"):
        download_language(lang, out_dir)

    print(f"[MultiPL-E] Datasets ready in {out_dir}")


if __name__ == "__main__":
    main()