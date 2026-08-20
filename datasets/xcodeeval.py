# benchmark/datasets/xcodeeval.py

import random
import pandas as pd

from .problem_dataset import ProblemDataset
from ..core.code_snippet import CodeSnippet


class xCodeEval(ProblemDataset):
    
    name = "xcodeeval"
    LANGUAGES = ["python", "java", "cpp", "cs", "js"]
    MAX_K = 50
    K_VALUES = [1, 10, 20, 50]

    xcode_lang = {"cpp": "C++", "cs": "C#", "java": "Java", "js": "Javascript", "python": "Python"}

    # ------------------------------------------------------------------ #
    #  Interfaccia                                                         #
    # ------------------------------------------------------------------ #

    def supported_languages(self) -> list[str]:
        return self.LANGUAGES

    def _load_original_candidates(self, language: str) -> list[CodeSnippet]:
        df = self._load_dataframe(language)
        snippets = []
        for row in df.itertuples(index=True):
            for i, code_dict in enumerate(row.positive_code):
                snippets.append(self._to_snippet(code_dict["source_code"], row.Index, i, language))

        return snippets

    def _do_select(self, language: str, seed: int) -> list[CodeSnippet]:
        """Submission di riferimento per ogni problema."""
        df = self._load_dataframe(language)
        snippets = []
        for row in df.itertuples(index=True):
            snippets.append(self._to_snippet(row.source_code, row.Index, -1, language)) #id fittizio impostato a -1
        
        return snippets


    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def _load_dataframe(self, language: str) -> pd.DataFrame:
        path = self.data_path / "retrieval_code_code" / "validation"
        
        json_files = sorted([
            path / f
            for f in path.iterdir()
            if f.name.endswith(".jsonl") and f.name.startswith(self.xcode_lang[language]+"_")
        ])

        temps = []
        for file in json_files:
            if not file.exists():
                raise FileNotFoundError(f"Dataframe non trovato: {file}")
            data = pd.read_json(file, lines=True) # read data frame from json file
            temps.append(data) # append the data frame to the list

        return pd.concat(temps, ignore_index=True)

    def _to_snippet(self, source_code: str, row_index: int, id: int, language: str) -> CodeSnippet:
        return CodeSnippet(
            id=f"{row_index}/{id}",
            code=source_code,
            language=language,
        )