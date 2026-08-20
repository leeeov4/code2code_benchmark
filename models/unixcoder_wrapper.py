# benchmark/models/unixcoder_wrapper.py

import torch
from .unixcoder import UniXcoder

from ..core.base_model import BaseModel


class UniXcoderWrapper(BaseModel):

    MODEL_ID   = "microsoft/unixcoder-base"
    MODEL_NAME = "unixcoder"
    MAX_LENGTH = 1023

    def __init__(self, device: str = None):
        super().__init__(self.MODEL_NAME, device)

        self.model = UniXcoder(self.MODEL_ID).to(self.device)
        self.model.eval()

    def encode(self, code: str, is_query: bool = False) -> torch.Tensor:
        return self.encode_batch([code])[0]

    def encode_batch(self, codes: list[str], batch_size: int = 32,
                     is_query: bool = False) -> list[torch.Tensor]:
        embeddings = []

        for i in range(0, len(codes), batch_size):
            batch  = codes[i:i + batch_size]
            tokens = self.model.tokenize(batch, max_length=self.MAX_LENGTH, padding=True, mode="<encoder-only>")

            with torch.inference_mode():
                _, batch_embeddings = self.model(torch.tensor(tokens).to(self.device))

            embeddings.append(batch_embeddings.cpu())         

        embedding_matrix = torch.cat(embeddings, dim=0)
        return embedding_matrix
