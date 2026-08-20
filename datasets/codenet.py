# benchmark/datasets/codenet.py

import random
from pathlib import Path

from .problem_dataset import ProblemDataset
from ..core.code_snippet import CodeSnippet


class CodeNet(ProblemDataset):

    name       = "codenet"
    LANGUAGES  = ["py", "java", "cpp"]
    EXTENSIONS = {"py": ".py", "java": ".java", "cpp": ".cpp"}
    MAX_K      = 50
    K_VALUES   = [1, 10, 20, 50]

    DATASETS_NAME = {"py": "Project_CodeNet_Python800",
                  "java": "Project_CodeNet_Java250",
                  "cpp": "Project_CodeNet_C++1000"}


    # ------------------------------------------------------------------ #
    #  Interface                                                         #
    # ------------------------------------------------------------------ #

    def supported_languages(self) -> list[str]:
        return self.LANGUAGES

    def _load_original_candidates(self, language: str) -> list[CodeSnippet]:
        snippets = []
        lang_path = self.data_path / self.DATASETS_NAME[language]

        for problem_dir in lang_path.iterdir():

            if not problem_dir.is_dir():
                continue
            for submission_file in problem_dir.iterdir():
                if submission_file.suffix != self.EXTENSIONS[language]:
                    continue
                snippets.append(self._to_snippet(problem_dir.name, submission_file, language))

        return snippets

    def filter_candidates(self, candidates, ids):
        return [c for c in candidates if c.id in ids]

    def _do_select(self, language: str, seed: int) -> list[CodeSnippet]:
        """Una submission casuale per ogni problema."""
        rng        = random.Random(seed)
        candidates = self._load_original_candidates(language)
        groups     = self._group_by_problem(candidates)
        return [rng.choice(submissions) for submissions in groups.values()]

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def _to_snippet(self, problem_id: str, submission_file: Path, language: str) -> CodeSnippet:
        return CodeSnippet(
            id=f"{problem_id}/{submission_file.stem}",
            code=submission_file.read_text(encoding="utf-8"),
            language=language
        )