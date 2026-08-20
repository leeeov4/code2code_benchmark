# benchmark/datasets/problem_dataset.py

import random
from abc import abstractmethod
from collections import defaultdict

from ..core.base_dataset import BaseDataset
from ..core.code_snippet import CodeSnippet


class ProblemDataset(BaseDataset):

    # ------------------------------------------------------------------ #
    #  Abstract interface                                                  #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def supported_languages(self) -> list[str]: ...

    @abstractmethod
    def _load_original_candidates(self, language: str) -> list[CodeSnippet]: ...

    # ------------------------------------------------------------------ #
    #  Ground truth                                                        #
    # ------------------------------------------------------------------ #

    def get_ground_truth(self, query_id: str, language: str) -> list[str]:
        problem_id = query_id.split("/")[0]
        candidates = self._load_original_candidates(language)
        return [
            s.id for s in candidates
            if s.id.split("/")[0] == problem_id
            and s.id != query_id
        ]
    
    def get_ground_truths(self, queries: list, language: str) -> dict[str, set[str]]:
        candidates = self._load_original_candidates(language)

        # Raggruppa per problem_id
        problem_map = {}
        for s in candidates:
            problem_id = s.id.split("/")[0]
            problem_map.setdefault(problem_id, []).append(s.id)

        ground_truths = {}

        for q in queries:
            problem_id = q.id.split("/")[0]
            ground_truths[q.id] = {
                sid for sid in problem_map.get(problem_id, [])
                if sid != q.id
            }

        return ground_truths

    # ------------------------------------------------------------------ #
    #  Structural properties                                               #
    # ------------------------------------------------------------------ #

    def is_symmetric(self) -> bool:
        return True

    def is_ready(self, language: str) -> bool:
        return self._queries_path(language, version="original").exists()

    # ------------------------------------------------------------------ #
    #  Query selection                                                     #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _do_select(self, language: str, seed: int) -> list[CodeSnippet]: ...

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def _group_by_problem(self, snippets: list[CodeSnippet]) -> dict[str, list[CodeSnippet]]:
        groups = defaultdict(list)
        for s in snippets:
            problem_id = s.id.split("/")[0]
            groups[problem_id].append(s)
        return groups