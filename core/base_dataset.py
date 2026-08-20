# benchmark/core/base_dataset.py

import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path

from .code_snippet import CodeSnippet
from ..config import DATA_PATH, PROCESSED_PATH


class BaseDataset(ABC):

    name: str

    def __init__(self):
        self.data_path      = Path(DATA_PATH[self.name])
        self.processed_path = Path(PROCESSED_PATH[self.name])

    # ------------------------------------------------------------------ #
    #  Abstract interface                                                #
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def name(self) -> str:
        """Chiave usata in config.py"""

    @abstractmethod
    def supported_languages(self) -> list[str]: ...

    @abstractmethod
    def _load_original_candidates(self, language: str) -> list[CodeSnippet]: ...

    @abstractmethod
    def get_ground_truth(self, query_id: str, language: str) -> list[str]: ...

    @abstractmethod
    def is_ready(self, language: str) -> bool: ...

    # ------------------------------------------------------------------ #
    #  Metrics                                                           #
    # ------------------------------------------------------------------ #

    MAX_K    = 50
    K_VALUES = [1, 10, 20, 50]

    # ------------------------------------------------------------------ #
    #  Candidates and queries loading                                    #
    # ------------------------------------------------------------------ #

    def load_candidates(self, language: str, version: str = "original") -> list[CodeSnippet]:
        if version == "original":
            return self._load_original_candidates(language)
        return self._load_from_file(self._version_path(language, version) / "candidates.json")

    def load_queries(self, language: str, version: str = "original") -> list[CodeSnippet]:
        return self._load_from_file(self._queries_path(language, version))

    # ------------------------------------------------------------------ #
    #  Query selection                                                   #
    # ------------------------------------------------------------------ #

    def select_queries(self, language: str, seed: int = 42):
        path = self._queries_path(language, version="original")
        if path.exists():
            raise FileExistsError(
                f"Query già selezionate per {language}. "
                f"Cancella {path} per rigenerare."
            )
        queries = self._do_select(language, seed)
        self._save_to_file(queries, path)

    def _do_select(self, language: str, seed: int) -> list[CodeSnippet]:
        raise NotImplementedError(
            f"{self.__class__.__name__} non supporta la selezione casuale delle query."
        )

    # ------------------------------------------------------------------ #
    #  Structural properties                                             #
    # ------------------------------------------------------------------ #

    def is_symmetric(self) -> bool:
        return False

    def get_excluded_candidates(self, query_id: str, language: str) -> set[str]:
        return set()

    # ------------------------------------------------------------------ #
    #  Path helpers                                                        #
    # ------------------------------------------------------------------ #

    def _queries_path(self, language: str, version: str) -> Path:
        return self.processed_path / language / version / "queries.json"

    def _version_path(self, language: str, version: str) -> Path:
        return self.processed_path / language / version
    
    def _base_path(self, output_dir: Path, language: str) -> Path:
        return output_dir / self.name / language

    # ------------------------------------------------------------------ #
    #  I/O                                                                 #
    # ------------------------------------------------------------------ #

    def _save_to_file(self, snippets: list[CodeSnippet], path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([asdict(s) for s in snippets], f, indent=2)

    def _load_from_file(self, path: Path) -> list[CodeSnippet]:
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {path}")
        with open(path, "r") as f:
            return [CodeSnippet(**s) for s in json.load(f)]