
# benchmark/models/nomic_embed.py

import torch
from sentence_transformers import SentenceTransformer

from ..core.base_model import BaseModel


class NomicEmbed(BaseModel):

    MODEL_ID   = "nomic-ai/nomic-embed-code"
    MODEL_NAME = "nomic_embed"
    MAX_LENGTH = 8192

    def __init__(self, device: str = None):
        super().__init__(self.MODEL_NAME, device)

        self.tokenizer = None
        self.model     = SentenceTransformer(self.MODEL_ID, trust_remote_code=True)
        self.model.eval()

    def encode(self, code: str, is_query: bool = False) -> torch.Tensor:
        return self.encode_batch([code])[0]

    def encode_batch(self, codes: list[str], batch_size: int = 16, is_query: bool = False) -> list[torch.Tensor]:
        self.model.max_seq_length = self.MAX_LENGTH

        with torch.inference_mode():
            embeddings = self.model.encode(codes, convert_to_tensor=True, batch_size=batch_size)

        return embeddings.detach()