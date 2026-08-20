# benchmark/analysis/embedding_time.py

import time
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from ..core.base_model import BaseModel
from ..datasets.bigclonebench import BigCloneBench
from ..config import OUTPUT_PATH

WARMUP_STEPS = 20
TARGET_MB    = 10


class EmbeddingTimeAnalysis:

    def __init__(self, model: BaseModel):
        self.model  = model
        self.output = Path(OUTPUT_PATH)

    def run(self, device: str):
        snippets = self._load_snippets()
        print(f"[Timing] Snippet disponibili: {len(snippets)}")

        # warmup
        self._warmup(snippets[0].code)

        # misurazione
        timings   = []
        tot_bytes = 0

        for snippet in tqdm(snippets):
            code = snippet.code

            if device == "cuda":
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            self.model.encode(code)

            if device == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()

            timings.append(t_end - t_start)
            tot_bytes += len(code.encode("utf-8"))

            if tot_bytes / 1024 / 1024 >= TARGET_MB:
                break

        self._save(timings, tot_bytes, device)

    # ------------------------------------------------------------------ #
    #  Warmup                                                              #
    # ------------------------------------------------------------------ #

    def _warmup(self, code: str):
        print(f"[Timing] Warmup ({WARMUP_STEPS} steps)...")
        for _ in range(WARMUP_STEPS):
            self.model.encode(code)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # ------------------------------------------------------------------ #
    #  Load snippets da BigCloneBench                                      #
    # ------------------------------------------------------------------ #

    def _load_snippets(self):
        # BigCloneBench è la sorgente
        # perché ci servono solo gli snippet, non le tipologie
        candidates = []
        for clone_type in ["type1", "type2", "type3"]:
            dataset = BigCloneBench(clone_type=clone_type)
            candidates.extend(dataset.load_candidates(language="java"))

        return candidates

    # ------------------------------------------------------------------ #
    #  Salvataggio                                                         #
    # ------------------------------------------------------------------ #

    def _save(self, timings: list, tot_bytes: int, device: str):
        mean     = float(np.mean(timings))
        std      = float(np.std(timings))
        total    = float(np.sum(timings))
        tot_mb   = tot_bytes / 1024 / 1024

        print(f"[Timing] Mean:      {mean:.6f}s")
        print(f"[Timing] Std:       {std:.6f}s")
        print(f"[Timing] Total:     {total:.6f}s")
        print(f"[Timing] Bytes:     {tot_bytes} ({tot_mb:.2f} MB)")
        print(f"[Timing] Snippets:  {len(timings)}")

        out_path = self.output / "timings" / f"{self.model}_{device}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w") as f:
            f.write(f"{mean}\n")
            f.write(f"{std}\n")
            f.write(f"{total}\n")
            f.write(f"{tot_bytes}\n")
            f.write(f"{len(timings)}\n")

        print(f"[Timing] Risultati salvati → {out_path}")