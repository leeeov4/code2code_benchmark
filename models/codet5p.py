# benchmark/models/codet5p.py

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

from ..core.base_model import BaseModel

from tqdm import tqdm

class CodeT5P(BaseModel):

    MODEL_ID   = "Salesforce/codet5p-110m-embedding"
    MODEL_NAME = "codet5p"
    MAX_LENGTH = 512

    def __init__(self, device: str = None):
        super().__init__(self.MODEL_NAME, device)

        config = AutoConfig.from_pretrained(self.MODEL_ID, trust_remote_code=True)
        if not hasattr(config, "is_decoder"):
            config.is_decoder = False

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID, trust_remote_code=True)
        self.model     = AutoModel.from_pretrained(
            self.MODEL_ID, config=config, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    def encode(self, code: str, is_query: bool = False) -> torch.Tensor:
        return self.encode_batch([code])[0]

    def encode_batch(self, codes: list[str], batch_size: int = 32,
                     is_query: bool = False) -> list[torch.Tensor]:
        embeddings = []

        for i in range(0, len(codes), batch_size):
            batch = codes[i:i + batch_size]

            inputs = self.tokenizer.batch_encode_plus(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.MAX_LENGTH
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                batch_embeddings = self.model(**inputs)

            embeddings.append(batch_embeddings.cpu())

        embedding_matrix = torch.cat(embeddings, dim=0)
        return embedding_matrix