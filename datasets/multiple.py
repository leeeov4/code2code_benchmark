# benchmark/datasets/multiple.py

import pandas as pd
import random

from .problem_dataset import ProblemDataset
from ..core.code_snippet import CodeSnippet


class MultiPLE(ProblemDataset):

    name = "multiple"

    LANGUAGES = ["cpp", "cs", "java", "js"]

    MAX_K    = 20
    K_VALUES = [1, 10, 20]

    # ------------------------------------------------------------------ #
    #  Abstract interface                                                  #
    # ------------------------------------------------------------------ #

    def supported_languages(self) -> list[str]:
        return self.LANGUAGES

    def _load_original_candidates(self, language: str) -> list[CodeSnippet]:
        candidates = []
        df = self._load_dataframe(language)
        for problem_index in range(0,len(df["problem"])):
            #Consider only problems with at least one correct solution
            if "OK" in df["statuses"][problem_index]:
                for fun_index in range(0,len(df["completions"][problem_index])):
                    #Could be the query
                    candidates.append(self._to_snippet(df["completions"][problem_index][fun_index], problem_index, fun_index, language))
        return candidates

    def _to_snippet(self, source_code: str, problem_index: int, fun_index: int, language: str) -> CodeSnippet:
        return CodeSnippet(
                        id=f"{problem_index}/{fun_index}",
                        code=source_code,
                        language=language
                    )
                
    def load_ids(self, path):
        ids = set()
        with open(path, "r") as f:
            for line in f:
                line = line.split("\n")[0].split(",")
                ids.add(f"{line[0]}/{line[1]}")
        return ids

    def filter_candidates(self, candidates, ids):
        return [c for c in candidates if c.id in ids]


    def _do_select(self, language: str, seed: int) -> list[CodeSnippet]:
        queries = []
        random.seed(seed)

        df = self._load_dataframe(language)
        for problem_index in range(0,len(df["problem"])):
            #Consider only problems with at least one correct solution
            if "OK" in df["statuses"][problem_index]:
                found = False
                while not found:
                    fun_index = random.randint(0, len(df["completions"][problem_index])-1)

                    found = df["statuses"][problem_index][fun_index] == "OK"
                
                queries.append(self._to_snippet(df["completions"][problem_index][fun_index], problem_index, fun_index, language))
        return queries

    def get_ground_truth(self, query_id: str, language: str) -> list[str]:
        problem_id = query_id.split("/")[0]
        candidates = self._load_original_candidates(language)
        df = self._load_dataframe(language)
        return [
            s.id for s in candidates
            if s.id.split("/")[0] == problem_id
            and df["statuses"][int(s.id.split("/")[0])][int(s.id.split("/")[1])] == "OK"
            and s.id != query_id
        ]

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def _load_dataframe(self, language: str) -> pd.DataFrame:
        path = self.data_path / "parquets" / f"humaneval.{language}.StarCoder2_15b_16k.0.2.parquet"
        
        if not path.exists():
            raise FileNotFoundError(f"Dataframe not found: {path}")
        return pd.read_parquet(path)