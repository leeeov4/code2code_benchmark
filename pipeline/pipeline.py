# benchmark/pipeline/pipeline.py

import pickle
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from ..core.base_dataset import BaseDataset
from ..core.base_model import BaseModel
from ..core.code_snippet import CodeSnippet

from tqdm import tqdm

class Pipeline:

    def __init__(self, dataset: BaseDataset, model: BaseModel,
                 output_dir: str, top_k: int = None):
        self.dataset = dataset
        self.model = model
        self.output_dir = Path(output_dir)
        self.top_k = top_k

    # ------------------------------------------------------------------ #
    #  Stage 1 — Embeddings                                                #
    # ------------------------------------------------------------------ #

    def run_stage1_embeddings(self, language: str, version: str = "original"):
        if not self.dataset.is_ready(language):
            raise RuntimeError(
                f"Dataset non pronto per {language}. "
                f"Esegui prima select_queries o extract_and_serialize."
            )

        out_path = self._embeddings_path(language, version)
        if out_path.exists():
            raise FileExistsError(f"Embeddings già presenti: {out_path}")

        candidates = self.dataset.load_candidates(language, version)

        codes = [s.code for s in candidates]
        ids = [s.id for s in candidates]

        embeddings = self.model.encode_batch(codes)
        
        result = {"ids": ids, "embeddings": embeddings}
        self._save_pickle(result, out_path)
        print(f"[Stage 1] Salvati {len(ids)} embedding → {out_path}")

    # ------------------------------------------------------------------ #
    #  Stage 2 — Retrieval                                                 #
    # ------------------------------------------------------------------ #

    def run_stage2_retrieval(self, language: str,
                             query_version: str = "original",
                             candidate_version: str = "original"):
        out_path = self._scores_path(language, query_version, candidate_version)
        if out_path.exists():
            raise FileExistsError(f"Scores già presenti: {out_path}")

        # Load candidates embedding
        candidate_embeddings = self._load_pickle(
            self._embeddings_path(language, candidate_version)
        )
        candidate_ids = candidate_embeddings["ids"]
        
        candidate_matrix = candidate_embeddings["embeddings"]

        # Load queries
        queries = self.dataset.load_queries(language, query_version)
        query_codes = [q.code for q in queries]
        query_embeddings = self.model.encode_batch(query_codes, is_query=True)

        scores = {}
        for query, q_emb in zip(queries, query_embeddings):

            # candidates to exclude (queries and inherited clones for bigclonebench)
            excluded = self.dataset.get_excluded_candidates(query.id, language)
            if self.dataset.is_symmetric():
                excluded.add(query.id)

            q_tensor = q_emb.unsqueeze(0)
            sims = F.cosine_similarity(q_tensor, candidate_matrix)
            
            ranked = [
                (cid, sims[i].item())
                for i, cid in enumerate(candidate_ids)
                if cid not in excluded
            ]
            ranked.sort(key=lambda x: x[1], reverse=True)

            if self.top_k is not None:
                ranked = ranked[:self.top_k]

            scores[query.id] = ranked
            break

        self._save_pickle(scores, out_path)
        print(f"[Stage 2] Salvati scores per {len(scores)} query → {out_path}")

    # ------------------------------------------------------------------ #
    #  Stage 3 — Metriche                                                  #
    # ------------------------------------------------------------------ #

    def run_stage3_metrics(self, language: str,
                           query_version: str = "original",
                           candidate_version: str = "original"):
        scores = self._load_pickle(
            self._scores_path(language, query_version, candidate_version)
        )
        queries = self.dataset.load_queries(language, query_version)

        ground_truths = {
            q.id: set(self.dataset.get_ground_truth(q.id, language))
            for q in tqdm(queries)
        }

        k_values = self.dataset.K_VALUES
        results = {f"precision@{k}": [] for k in k_values}
        results.update({f"ndcg@{k}": [] for k in k_values})

        for query in tqdm(queries):
            
            ranked = scores[query.id]
            gt = ground_truths[query.id]
            
            for k in k_values:
                if len(gt) < k:          # ← skip se ground truth insufficiente
                    continue 
                
                results[f"precision@{k}"].append(self._precision_at_k(ranked, gt, k))
                results[f"ndcg@{k}"].append(self._ndcg_at_k(ranked, gt, k))

        summary = {}
        for k in k_values:
            n = len(results[f"precision@{k}"])
            summary[f"precision@{k}"]        = sum(results[f"precision@{k}"]) / n if n > 0 else None
            summary[f"ndcg@{k}"]             = sum(results[f"ndcg@{k}"])      / n if n > 0 else None
            summary[f"num_queries@{k}"]      = n
        


        out_path = self._metrics_path(language, query_version, candidate_version)
        self._save_json(summary, out_path)
        print(f"[Stage 3] Metriche salvate → {out_path}")
        return summary

    # ------------------------------------------------------------------ #
    #  Metrics                                                           #
    # ------------------------------------------------------------------ #

    def _precision_at_k(self, ranked: list, gt: set, k: int) -> float:
        top_k = [cid for cid, _ in ranked[:k]]
        hits = sum(1 for cid in top_k if cid in gt)
        return hits / k if k > 0 else 0.0

    def _ndcg_at_k(self, ranked: list, gt: set, k: int) -> float:
        top_k = [cid for cid, _ in ranked[:k]]
        dcg = sum(
            1.0 / math.log2(i + 2)
            for i, cid in enumerate(top_k)
            if cid in gt
        )

        relevant = [cid for cid in top_k if cid in gt]

        ideal = sum(
            1.0 / math.log2(i + 2)
            for i in range(len(relevant))
        )
        
        return dcg / ideal if ideal > 0 else 0.0

    # ------------------------------------------------------------------ #
    #  Path helpers                                                        #
    # ------------------------------------------------------------------ #

    def _embeddings_path(self, language: str, version: str) -> Path:
        return self.dataset._base_path(self.output_dir, language) / "embeddings" / f"{self.model}_{version}.pkl"

    def _scores_path(self, language: str, query_version: str, candidate_version: str) -> Path:
        return self.dataset._base_path(self.output_dir, language) / "scores" / \
               f"{self.model}_{query_version}_{candidate_version}.pkl"

    def _metrics_path(self, language: str, query_version: str, candidate_version: str) -> Path:
        return self.dataset._base_path(self.output_dir, language) / "metrics" / \
               f"{self.model}_{query_version}_{candidate_version}.json"

    # ------------------------------------------------------------------ #
    #  I/O                                                                 #
    # ------------------------------------------------------------------ #

    def _save_pickle(self, data: dict, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def _load_pickle(self, path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def _save_json(self, data: dict, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)