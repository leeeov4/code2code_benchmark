# benchmark/core/base_model.py

import torch
from abc import ABC, abstractmethod


class BaseModel(ABC):

    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def encode(self, code: str, is_query: bool = False) -> torch.Tensor:
        """Encode a single code snippet and return its embedding as a CPU tensor."""

    @abstractmethod
    def encode_batch(self, codes: list[str], batch_size: int = 32, is_query: bool = False) -> list[torch.Tensor]:
        """Encode a list of code snippets and return their embeddings as CPU tensors."""

    def __str__(self) -> str:
        return self.model_name