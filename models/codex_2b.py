# benchmark/models/codex_2b.py

import torch
from transformers import AutoModel
import torch.nn.functional as F

from ..core.base_model import BaseModel

class Codex2B(BaseModel):

    MODEL_ID   = "Salesforce/SFR-Embedding-Code-2B_R"
    MAX_LENGTH = 8192
    MODEL_NAME = "codex_2b"
    QUERY_INSTRUCTION = "Given Code or Text, retrieval relevant content"

    def __init__(self, device: str = None):
        super().__init__(self.MODEL_NAME, device)

        self.model = AutoModel.from_pretrained(self.MODEL_ID, trust_remote_code=True, device_map="auto", low_cpu_mem_usage=True)
        #.to(self.device)
        self.model.eval()

    def encode(self, code: str, is_query: bool = False) -> torch.Tensor:
        return self.encode_batch([code], is_query=is_query)[0]

    def encode_batch(self, codes: list[str], batch_size: int = 16,
                     is_query: bool = False) -> list[torch.Tensor]:

        embeddings = []

        for i in range(0, len(codes), batch_size):
            batch = codes[i:i + batch_size]

            with torch.inference_mode():
                if is_query:
                    batch_embeddings = self.model.encode_queries(
                        batch,
                        max_length=self.MAX_LENGTH,
                        instruction=self.QUERY_INSTRUCTION,
                        convert_to_tensor=True,
                        device=self.device
                    )
                else:
                    batch_embeddings = self.model.encode_corpus(
                        batch,
                        max_length=self.MAX_LENGTH,
                        convert_to_tensor=True,
                        device=self.device
                    )

            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

            embeddings.append(batch_embeddings.cpu())         

        embedding_matrix = torch.cat(embeddings, dim=0)
        return embedding_matrix