# benchmark/models/codex.py

import torch
from transformers import AutoTokenizer, AutoModel

from ..core.base_model import BaseModel


class Codex(BaseModel):

    MODEL_ID   = "Salesforce/SFR-Embedding-Code-400M_R"
    MODEL_NAME = "codex"
    MAX_LENGTH = 8192

    def __init__(self, device: str = None):
        super().__init__(self.MODEL_NAME, device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID, trust_remote_code=True, add_eos_token=True
        )
        self.model = AutoModel.from_pretrained(
            self.MODEL_ID, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    def encode(self, code: str, is_query: bool = False) -> torch.Tensor:
        return self.encode_batch([code])[0]

    def encode_batch(self, codes: list[str], batch_size: int = 32,
                     is_query: bool = False) -> list[torch.Tensor]:
        embeddings = []

        for i in range(0, len(codes), batch_size):
            batch  = codes[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                max_length=self.MAX_LENGTH,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model(**inputs)

            # CLS token
            batch_embeddings = outputs.last_hidden_state[:, 0]

            embeddings.append(batch_embeddings.cpu())         

        embedding_matrix = torch.cat(embeddings, dim=0)
        return embedding_matrix